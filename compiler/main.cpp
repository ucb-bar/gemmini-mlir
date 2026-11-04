#include "compiler.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include <fstream>
#include <cstdlib>
#include <set>
using namespace backend;
namespace {
class GemminiDialect final : public mlir::Dialect {
public:
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiDialect)
 static llvm::StringRef getDialectNamespace(){return "gemmini";}
 explicit GemminiDialect(mlir::MLIRContext*ctx):Dialect("gemmini",ctx,mlir::TypeID::get<GemminiDialect>()) {allowUnknownOperations();}
};
std::string readFile(const std::string&p) {std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open "+p);return std::string(std::istreambuf_iterator<char>(in),{});}
void writeFile(const std::string&p,const std::string&s) {std::ofstream o(p,std::ios::binary);if(!o)throw std::runtime_error("cannot write "+p);o<<s;}
std::string serialize(const Object&o){return llvm::formatv("{0:2}",Value(clone(o))).str();}
std::string prefix(){if(auto p=std::getenv("MERLIN_MLIR_INSTALL"))return p;return DEFAULT_MLIR_INSTALL;}
std::string hash(const std::string&s){auto bytes=llvm::SHA256::hash(llvm::ArrayRef<uint8_t>((const uint8_t*)s.data(),s.size()));return llvm::toHex(bytes,true);}
mlir::Type tensorType(mlir::Builder&b,const Object&o){auto dt=str(o,"dtype");mlir::Type t;if(dt=="f32")t=b.getF32Type();else if(dt=="f64")t=b.getF64Type();else if(dt=="f16")t=b.getF16Type();else if(dt=="bf16")t=b.getBF16Type();else{unsigned bits=std::stoul(dt.substr(dt[0]=='u'?1:1));t=b.getIntegerType(bits);}return mlir::RankedTensorType::get(shape(o),t);}
void emitTarget(const Object&p, const Object&cb){mlir::DialectRegistry reg;reg.insert<GemminiDialect,mlir::func::FuncDialect>();mlir::MLIRContext ctx(reg);ctx.loadAllAvailableDialects();mlir::OpBuilder b(&ctx);auto loc=b.getUnknownLoc();auto mod=mlir::ModuleOp::create(loc);mod->setAttr("gemmini.command_buffer",b.getStringAttr(serialize(cb)));std::vector<mlir::Type> ins,outs;for(auto&v:arr(p,"inputs"))ins.push_back(tensorType(b,*v.getAsObject()));for(auto&v:arr(p,"outputs"))outs.push_back(tensorType(b,*v.getAsObject()));auto f=mlir::func::FuncOp::create(loc,"target_program",b.getFunctionType(ins,outs));mod.push_back(f);auto*block=f.addEntryBlock();b.setInsertionPointToStart(block);std::map<std::string,mlir::Value> values;unsigned i=0;for(auto&v:arr(p,"inputs"))values[str(*v.getAsObject(),"id")]=block->getArgument(i++);for(auto&v:arr(p,"expressions")){auto&e=*v.getAsObject();mlir::OperationState state(loc,"gemmini."+str(e,"operator"));for(auto&o:arr(e,"operands"))state.addOperands(values.at(o.getAsString()->str()));state.addTypes(tensorType(b,e));state.addAttribute("math",b.getStringAttr(serialize(e)));auto*operation=b.create(state);values[str(e,"id")]=operation->getResult(0);}std::vector<mlir::Value> results;for(auto&v:arr(p,"outputs"))results.push_back(values.at(str(*v.getAsObject(),"value")));b.create<mlir::func::ReturnOp>(loc,results);if(mlir::failed(mlir::verify(mod)))throw std::runtime_error("target dialect verification failed");mod.print(llvm::outs());llvm::outs()<<"\n";}
std::optional<int64_t> debugOrigin(mlir::Location loc,bool source){
 if(auto f=mlir::dyn_cast<mlir::FileLineColLoc>(loc)){
  auto name=f.getFilename().strref();
  if(name.starts_with("merlin_task_") && f.getLine()>=100){int64_t task=0;if(!name.drop_front(12).getAsInteger(10,task))return source?(int64_t)f.getLine()-100:task;}
  if(!source&&name=="merlin_task"&&f.getLine()>=100)return (int64_t)f.getLine()-100;
 }
 if(auto c=mlir::dyn_cast<mlir::CallSiteLoc>(loc)){if(auto v=debugOrigin(c.getCaller(),source))return v;return debugOrigin(c.getCallee(),source);}
 if(auto f=mlir::dyn_cast<mlir::FusedLoc>(loc)){for(auto l:f.getLocations())if(auto v=debugOrigin(l,source))return v;}
 if(auto n=mlir::dyn_cast<mlir::NameLoc>(loc))return debugOrigin(n.getChildLoc(),source);
 return {};
}
std::string llvmArtifact(const std::string&c,Object&cb,const Object&p){
 llvm::SmallString<128> tmp;auto ec=llvm::sys::fs::createUniqueDirectory("gemmini-compiler",tmp);if(ec)throw std::runtime_error("cannot create compiler scratch");std::string dir=tmp.str().str(),cp=dir+"/kernel.c",ll=dir+"/kernel.ll";writeFile(cp,c);
 // Keep induction control explicit by default, including kernels extracted
 // from larger programs: their address loops also need provable bounds.
 bool structured=compilerSettingBool("structured_control",true);
 if(auto *params=cb.getObject("params"))if(auto *plan=params->getObject("global_program_plan"))
  if(auto *tasks=plan->getArray("tasks"))for(auto &value:*tasks)
   if(auto *task=value.getAsObject())if(str(*task,"kind")=="host")structured=true;
 std::string clang=prefix()+"/bin/clang";if(auto p=std::getenv("MERLIN_CLANG"))clang=p;std::vector<std::string> args={clang,"-target","riscv64-unknown-elf","-march=rv64gc_zba_zbb_zbs_zfh","-Xclang","-target-feature","-Xclang","-b","-Xclang","-target-feature","-Xclang","-zba","-Xclang","-target-feature","-Xclang","-zbb","-Xclang","-target-feature","-Xclang","-zbs","-mabi=lp64d","-mcmodel=medany",structured?"-O1":"-O2","-ffreestanding","-fno-builtin","-fno-stack-protector","-fno-strict-aliasing","-fwrapv","-fno-vectorize","-fno-slp-vectorize","-ffp-contract=off","-gline-tables-only","-fno-discard-value-names","-S","-emit-llvm",cp,"-o",ll};if(structured){args.push_back("-Xclang");args.push_back("-disable-llvm-passes");}std::vector<llvm::StringRef> refs;for(auto&a:args)refs.push_back(a);std::string error;int rc=llvm::sys::ExecuteAndWait(clang,refs,std::nullopt,{},120,0,&error);if(rc)throw std::runtime_error("clang kernel lowering failed: "+error);
 if(structured){
  std::string optimized=dir+"/structured.ll";
  std::vector<std::string> options={prefix()+"/bin/opt","-S","--passes=always-inline,sroa,mem2reg,instcombine<no-verify-fixpoint>,simplifycfg,loop-simplify,globaldce",ll,"-o",optimized};
  std::vector<llvm::StringRef> optRefs;for(auto&a:options)optRefs.push_back(a);
  if(llvm::sys::ExecuteAndWait(options[0],optRefs,std::nullopt,{},120,0,&error))throw std::runtime_error("structured LLVM optimization failed: "+error);
  ll=optimized;
 }
 llvm::LLVMContext lc;llvm::SMDiagnostic diag;auto module=llvm::parseIRFile(ll,diag,lc);if(!module){diag.print("gemmini-compile",llvm::errs());throw std::runtime_error("cannot parse emitted LLVM IR");}
 // Wrapping integer control is sufficient for the generated bounded loops.
 // Dropping optional poison promises refines LLVM semantics and avoids
 // requiring downstream consumers to prove those stronger promises.
 if(compilerSettingBool("conservative_ir",false))
  for(auto &function:*module)for(auto &block:function)
   for(auto &instruction:block)instruction.dropPoisonGeneratingFlags();
 // The logical ABI deliberately carries no parameter attributes.
 for(auto&f:*module){if(f.getName()=="kernel_entry"){for(unsigned a=0;a<f.arg_size();++a)f.setAttributes(f.getAttributes().removeParamAttributes(lc,a));}}
 mlir::DialectRegistry reg;reg.insert<mlir::DLTIDialect>();mlir::registerLLVMDialectImport(reg);mlir::MLIRContext context(reg);auto imported=mlir::translateLLVMIRToModule(std::move(module),&context);if(!imported)throw std::runtime_error("LLVM to MLIR import failed");mlir::Builder builder(&context);
 auto*params=cb.getObject("params");auto*plan=params?params->getObject("global_program_plan"):nullptr;auto*tasks=plan?plan->getArray("tasks"):nullptr;int64_t taskCount=tasks?tasks->size():1;
 std::vector<int64_t> counts(taskCount,0);int64_t total=0;
 std::map<int64_t,std::string> sourceRegions;
 for(auto&v:arr(p,"expressions")){auto&e=*v.getAsObject();auto region=str(e,"region_id");if(region.empty())continue;if(auto*indices=e.getArray("source_op_indices"))for(auto&i:*indices)sourceRegions[*i.getAsInteger()]=region;}

 for(auto f:imported->getOps<mlir::LLVM::LLVMFuncOp>()){
  if(f.getName()=="kernel_complete")continue;if(f.isExternal())continue;
  if(f.getName()!="kernel_entry")throw std::runtime_error("ordinary helper was not inlined: "+f.getName().str());
  f.removeArgAttrsAttr();f.removeResAttrsAttr();
  f.walk([&](mlir::Operation*op){if(op==f.getOperation())return;auto t=debugOrigin(op->getLoc(),false).value_or(0);if(t<0||t>=taskCount)t=0;op->setAttr("merlin.global_task",builder.getI64IntegerAttr(t));if(tasks){auto*task=(*tasks)[t].getAsObject();auto*indices=task->getArray("source_op_indices");auto source=debugOrigin(op->getLoc(),true);
   if(!source&&indices&&indices->size()==1)source=*(*indices)[0].getAsInteger();
   if(source){op->setAttr("merlin.source_op_index",builder.getI64IntegerAttr(*source));if(sourceRegions.count(*source))op->setAttr("prov.region_id",builder.getStringAttr(sourceRegions.at(*source)));}
   if(indices&&indices->size()>1){std::vector<int64_t> sources;for(auto&i:*indices)sources.push_back(*i.getAsInteger());op->setAttr("merlin.source_op_indices",builder.getDenseI64ArrayAttr(sources));}
   if(!op->hasAttr("prov.region_id"))if(auto*r=task->getArray("source_region_ids"))if(r->size()==1)op->setAttr("prov.region_id",builder.getStringAttr(*(*r)[0].getAsString()));}counts[t]++;total++;});
 }
 if(plan){int64_t cursor=0;for(int64_t t=0;t<taskCount;t++){auto*task=(*tasks)[t].getAsObject();(*task)["instruction_start"]=cursor;cursor+=std::max<int64_t>(counts[t],1);(*task)["instruction_end"]=cursor;}(*plan)["schedule_instruction_count"]=cursor;(*plan)["prologue_instruction_range"]=Array{0,0};(*plan)["epilogue_instruction_range"]=Array{cursor,cursor};}
 // Debug locations were used above solely to recover explicit source/task
 // ownership. Do not require downstream consumers to understand the compiler's
 // debug metadata spelling. The selected runner supplies its target ABI and
 // object options; imported LLVM module flags are not executable operations.
 for(auto &op:llvm::make_early_inc_range(imported->getBody()->getOperations()))
  if(op.getName().getStringRef()=="llvm.module_flags")op.erase();
 imported->walk([&](mlir::Operation *op){
  op->setLoc(builder.getUnknownLoc());
  op->removeAttr("loop_annotation");
  if(auto assembly=mlir::dyn_cast<mlir::LLVM::InlineAsmOp>(op)){
   assembly.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::None);
   assembly.setAsmDialectAttr({});
  }
  for(auto &region:op->getRegions())for(auto &block:region)
   for(auto arg:block.getArguments())arg.setLoc(builder.getUnknownLoc());
 });
 if(mlir::failed(mlir::verify(*imported)))throw std::runtime_error("LLVM artifact verification failed");std::string result;llvm::raw_string_ostream out(result);imported->print(out,mlir::OpPrintingFlags().printGenericOpForm());out<<"\n";llvm::sys::fs::remove_directories(dir);return result;
}
}
int main(int argc,char**argv){try{std::string input,output,debugC;bool artifact=false,parse=false,dumpProgram=false;for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--verify-diagnostics")parse=true;else if(a=="--emit-target-artifact")artifact=true;else if(a=="--dump-program")dumpProgram=true;else if(a.rfind("--emit-command-buffer=",0)==0)output=a.substr(22);else if(a.rfind("--emit-c=",0)==0)debugC=a.substr(9);else if(a.rfind("--lower-program-to-",0)==0){}else if(a[0]=='-')throw std::runtime_error("unknown option: "+a);else input=a;}if(input.empty())throw std::runtime_error("missing input program");auto source=readFile(input);auto program=parseProgram(source);if(parse)return 0;if(dumpProgram){llvm::outs()<<serialize(program)<<"\n";return 0;}auto cb=lowerCommands(program,hash(source));std::string result;if(artifact||!output.empty()||!debugC.empty()){auto c=emitC(program,cb);if(!debugC.empty())writeFile(debugC,c);if(artifact||!output.empty())result=llvmArtifact(c,cb,program);}if(!output.empty())writeFile(output,serialize(cb)+"\n");if(artifact)llvm::outs()<<result;else if(output.empty()&&debugC.empty())emitTarget(program,cb);return 0;}catch(const std::exception&e){llvm::errs()<<"gemmini-compile: "<<e.what()<<"\n";return 1;}}
