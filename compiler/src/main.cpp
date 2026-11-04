#include "compiler.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
namespace mc {
class DeviceDialect:public mlir::Dialect {
 public: explicit DeviceDialect(mlir::MLIRContext*c):Dialect("gemmini",c,mlir::TypeID::get<DeviceDialect>()){allowUnknownOperations();allowUnknownTypes();}
 static llvm::StringRef getDialectNamespace(){return "gemmini";}
};
static std::string read(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("cannot read "+p);return std::string(std::istreambuf_iterator<char>(f),{});}
static void run(std::vector<std::string> args){std::vector<char*> av;for(auto&s:args)av.push_back(s.data());av.push_back(nullptr);pid_t pid=fork();if(pid==0){execvp(av[0],av.data());_exit(127);}int status=0;if(pid<0||waitpid(pid,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status))throw std::runtime_error("target compilation failed");}
static unsigned taskLine(mlir::Location loc){
 if(auto f=mlir::dyn_cast<mlir::FileLineColLoc>(loc)){if(f.getFilename()=="merlin_task")return f.getLine();}
 if(auto c=mlir::dyn_cast<mlir::CallSiteLoc>(loc)){if(auto l=taskLine(c.getCaller()))return l;return taskLine(c.getCallee());}
 if(auto f=mlir::dyn_cast<mlir::FusedLoc>(loc))for(auto x:f.getLocations())if(auto l=taskLine(x))return l;
 return 0;
}
static void writeBuffer(std::ostream&f,const Obj&cb){
 // Tensor dictionary order is part of the default pointer ABI. llvm::json's
 // formatter sorts object keys, so serialize this dictionary in frontend order.
 auto params=cb.getObject("params");auto order=params?params->getArray("compiler_tensor_order"):nullptr;
 f<<"{";bool comma=false;
 for(auto&v:cb){if(comma)f<<",";comma=true;f<<llvm::formatv("{0}",Val(v.first.str())).str()<<":";
  if(v.first=="tensors"&&order){f<<"{";bool tc=false;for(auto&n:*order){auto name=n.getAsString();if(!name)continue;auto t=v.second.getAsObject()->get(*name);if(!t)continue;if(tc)f<<",";tc=true;f<<llvm::formatv("{0}",Val(name->str())).str()<<":"<<llvm::formatv("{0}",*t).str();}f<<"}";}
  else f<<llvm::formatv("{0}",v.second).str();
 }f<<"}\n";
}
static void admit(Context&c){
 for(auto&v:*c.cb.getArray("commands")){auto&o=*v.getAsObject();auto a=o.getObject("attributes");
  if(a&&hasStage(*a,"requant")&&num(*a,"requant_shift",-1)!=0){
   // This device's floating scaled readout rounds ties to even. The interface
   // integer requant stage requires half-up rounding, so it cannot be replaced
   // by a power-of-two floating scale without changing negative ties.
   auto op=str(o,"opcode");c.cb["declined"]=Obj{{"op",op},{"reason","Integer requant with a nonzero shift requires round-half-up arithmetic; this target lowering provides round-to-nearest-even scaled readout only."}};
   c.cb["commands"]=Arr{};return;
  }
 }
}
// Task functions are fully inlined, but retain their merlin_task locations.
// Keep changes of known task ownership at basic-block boundaries as well.
static unsigned llvmTaskLine(llvm::DebugLoc loc){
 for(auto*at=loc.get();at;at=at->getInlinedAt())
  if(at->getFilename()=="merlin_task")return at->getLine();
 return 0;
}
static void splitTaskBlocks(llvm::Module&m){
 std::vector<std::pair<llvm::Instruction*,llvm::DebugLoc>> boundaries;
 for(auto&fn:m)for(auto&block:fn){
  unsigned previous=0;llvm::DebugLoc previousLoc;
  for(auto&inst:block){
   if(llvm::isa<llvm::PHINode>(inst))continue;
   auto loc=inst.getDebugLoc();unsigned task=llvmTaskLine(loc);
   if(!task)continue;
   if(previous&&task!=previous)boundaries.emplace_back(&inst,previousLoc);
   previous=task;previousLoc=loc;
  }
 }
 // Collect first so splitting cannot invalidate the original block traversal.
 // LLVM updates successor PHIs; the inserted edge belongs to the prior task.
 for(auto&boundary:boundaries){
  auto*block=boundary.first->getParent();
  block->splitBasicBlock(boundary.first->getIterator(),block->getName()+".task");
  block->getTerminator()->setDebugLoc(boundary.second);
 }
}
// Replace an equality stop with an ordered stop only for a fully bounded,
// positive affine induction. Every backedge must pass through this exit test.
static void normalizeBoundedInduction(llvm::CondBrInst*branch,llvm::Loop*loop,llvm::DominatorTree&dominators){
 auto*cmp=llvm::dyn_cast<llvm::ICmpInst>(branch->getCondition());
 if(!cmp||cmp->getPredicate()!=llvm::ICmpInst::ICMP_NE)return;
 llvm::Value*value=cmp->getOperand(0);
 auto*bound=llvm::dyn_cast<llvm::ConstantInt>(cmp->getOperand(1));
 if(!bound){bound=llvm::dyn_cast<llvm::ConstantInt>(value);value=cmp->getOperand(1);}
 if(!bound)return;
 auto*phi=llvm::dyn_cast<llvm::PHINode>(value);
 if(!phi)if(auto*add=llvm::dyn_cast<llvm::BinaryOperator>(value))if(add->getOpcode()==llvm::Instruction::Add){
  phi=llvm::dyn_cast<llvm::PHINode>(add->getOperand(0));
  if(!phi)phi=llvm::dyn_cast<llvm::PHINode>(add->getOperand(1));
 }
 if(!phi||phi->getParent()!=loop->getHeader()||phi->getNumIncomingValues()!=2)return;
 int outside=-1,inside=-1;
 for(int i=0;i<2;i++)if(loop->contains(phi->getIncomingBlock(i)))inside=i;else outside=i;
 if(outside<0||inside<0)return;
 auto*latch=loop->getLoopLatch();
 if(!latch||phi->getIncomingBlock(inside)!=latch||!dominators.dominates(branch->getParent(),latch))return;
 auto*start=llvm::dyn_cast<llvm::ConstantInt>(phi->getIncomingValue(outside));
 auto*add=llvm::dyn_cast<llvm::BinaryOperator>(phi->getIncomingValue(inside));
 if(!start||!add||add->getOpcode()!=llvm::Instruction::Add||(value!=phi&&value!=add))return;
 llvm::ConstantInt*step=nullptr;
 if(add->getOperand(0)==phi)step=llvm::dyn_cast<llvm::ConstantInt>(add->getOperand(1));
 else if(add->getOperand(1)==phi)step=llvm::dyn_cast<llvm::ConstantInt>(add->getOperand(0));
 if(!step)return;
 const auto&s=start->getValue();const auto&e=bound->getValue();const auto&d=step->getValue();
 if(s.isNegative()||e.isNegative()||d.isNegative()||d.isZero()||!s.ult(e))return;
 if(!(e-s).urem(d).isZero())return;
 // Conservatively keep even the final speculative increment representable
 // as a nonnegative signed integer, so no signed or unsigned wrap is possible.
 const auto max=llvm::APInt::getSignedMaxValue(e.getBitWidth());
 if(d.ugt(max-e))return;
 if(cmp->hasOneUse()){
  cmp->setPredicate(llvm::ICmpInst::ICMP_ULT);cmp->setOperand(0,value);cmp->setOperand(1,bound);
 }else{
  auto*ordered=new llvm::ICmpInst(branch->getIterator(),llvm::ICmpInst::ICMP_ULT,value,bound,cmp->getName()+".bounded");
  ordered->setDebugLoc(cmp->getDebugLoc());branch->setCondition(ordered);
 }
}
// Give natural-loop exit tests a consistent continuation-true orientation.
// Swapping successor order changes neither CFG edges nor loop membership.
static void normalizeLoopBranches(llvm::Module&m){
 for(auto&fn:m){
  if(fn.isDeclaration())continue;
  llvm::DominatorTree dominators(fn);llvm::LoopInfo loops(dominators);
  for(auto&block:fn){
   auto*branch=llvm::dyn_cast<llvm::CondBrInst>(block.getTerminator());
   if(!branch)continue;
   auto*loop=loops.getLoopFor(&block);
   if(!loop||loop->contains(branch->getSuccessor(0))==loop->contains(branch->getSuccessor(1)))continue;
   auto*cmp=llvm::dyn_cast<llvm::ICmpInst>(branch->getCondition());
   if(!cmp)continue;
   if(!loop->contains(branch->getSuccessor(0))){
   auto inverse=llvm::ICmpInst::getInversePredicate(cmp->getPredicate());
   if(cmp->hasOneUse())cmp->setPredicate(inverse);
   else{
    auto*continuation=new llvm::ICmpInst(branch->getIterator(),inverse,cmp->getOperand(0),cmp->getOperand(1),cmp->getName()+".continue");
    continuation->setDebugLoc(cmp->getDebugLoc());
    branch->setCondition(continuation);
   }
   // LLVM also swaps any branch-weight metadata with the successors.
   branch->swapSuccessors();
   }
   normalizeBoundedInduction(branch,loop,dominators);
  }
 }
}
// Expose a conventional header test for positive constant-trip posttested
// loops whose results are in memory. Reuse the existing IV; add no counter.
static void canonicalizeSimpleLoopHeaders(llvm::Module&m){
 struct Plan{llvm::BasicBlock*header;llvm::CondBrInst*latch;llvm::PHINode*iv;llvm::ConstantInt*bound;};
 std::vector<Plan> plans;
 for(auto&fn:m){
  if(fn.isDeclaration())continue;
  llvm::DominatorTree dt(fn);llvm::LoopInfo loops(dt);
  for(auto*loop:loops.getLoopsInPreorder()){
   auto*header=loop->getHeader();auto*latch=loop->getLoopLatch();auto*exit=loop->getExitBlock();
   if(!latch||!exit||header->isEHPad())continue;
   bool exitInputsSafe=true;
   for(auto&phi:exit->phis()){
    int edge=phi.getBasicBlockIndex(latch);
    if(edge<0){exitInputsSafe=false;break;}
    if(auto*value=llvm::dyn_cast<llvm::Instruction>(phi.getIncomingValue(edge)))
     if(!dt.dominates(value,header->getTerminator()))exitInputsSafe=false;
   }
   if(!exitInputsSafe)continue;
   if(loop->getExitingBlock()!=latch)continue;
   auto*branch=llvm::dyn_cast<llvm::CondBrInst>(latch->getTerminator());
   if(!branch||branch->getSuccessor(0)!=header||branch->getSuccessor(1)!=exit)continue;
   auto*cmp=llvm::dyn_cast<llvm::ICmpInst>(branch->getCondition());
   if(!cmp||cmp->getPredicate()!=llvm::ICmpInst::ICMP_ULT)continue;
   auto*bound=llvm::dyn_cast<llvm::ConstantInt>(cmp->getOperand(1));
   auto*iv=llvm::dyn_cast<llvm::PHINode>(cmp->getOperand(0));
   bool comparesCurrent=iv!=nullptr;
   auto*add=llvm::dyn_cast<llvm::BinaryOperator>(cmp->getOperand(0));
   if(comparesCurrent){
    int edge=iv->getBasicBlockIndex(latch);
    add=edge<0?nullptr:llvm::dyn_cast<llvm::BinaryOperator>(iv->getIncomingValue(edge));
   }
   if(!bound||!add||add->getOpcode()!=llvm::Instruction::Add)continue;
   auto*addIV=llvm::dyn_cast<llvm::PHINode>(add->getOperand(0));
   auto*step=llvm::dyn_cast<llvm::ConstantInt>(add->getOperand(1));
   if(!addIV||!step){addIV=llvm::dyn_cast<llvm::PHINode>(add->getOperand(1));step=llvm::dyn_cast<llvm::ConstantInt>(add->getOperand(0));}
   if(comparesCurrent&&addIV!=iv)continue;
   iv=addIV;
   if(!iv||!step||iv->getParent()!=header||iv->getNumIncomingValues()!=2)continue;
   int inside=-1,outside=-1;
   for(int i=0;i<2;i++)if(loop->contains(iv->getIncomingBlock(i)))inside=i;else outside=i;
   if(inside<0||outside<0||iv->getIncomingBlock(inside)!=latch||iv->getIncomingValue(inside)!=add)continue;
   auto*start=llvm::dyn_cast<llvm::ConstantInt>(iv->getIncomingValue(outside));
   if(!start)continue;
   const auto&a=start->getValue();const auto&originalBound=bound->getValue();const auto&d=step->getValue();
   if(a.isNegative()||originalBound.isNegative()||d.isNegative()||d.isZero())continue;
   auto maximum=llvm::APInt::getSignedMaxValue(originalBound.getBitWidth());
   auto b=originalBound;
   // A posttest of the current IV includes the first IV at or above its
   // bound. Moving that test before the body shifts the limit by one step.
   if(comparesCurrent){if(d.ugt(maximum-b))continue;b+=d;}
   if(!a.ult(b)||d.ugt(maximum-b))continue;
   bound=llvm::ConstantInt::get(m.getContext(),b);
   bool escapes=false;
   for(auto*block:loop->blocks())for(auto&inst:*block)for(auto*user:inst.users())
    if(auto*use=llvm::dyn_cast<llvm::Instruction>(user))if(!loop->contains(use->getParent()))escapes=true;
   if(escapes)continue;
   plans.push_back({header,branch,iv,bound});
  }
 }
 // Each plan changes only its distinct header and latch. Nested-loop changes
 // preserve enclosing membership and all previously proved induction facts.
 for(auto&plan:plans){
  auto*body=plan.header->splitBasicBlock(plan.header->getFirstNonPHIIt(),plan.header->getName()+".body");
  auto*old=plan.header->getTerminator();
  auto*test=new llvm::ICmpInst(old->getIterator(),llvm::ICmpInst::ICMP_ULT,plan.iv,plan.bound,"iv.continue");
  test->setDebugLoc(llvm::cast<llvm::Instruction>(plan.latch->getCondition())->getDebugLoc());
  auto*guard=llvm::CondBrInst::Create(test,body,plan.latch->getSuccessor(1),old->getIterator());
  guard->setDebugLoc(plan.latch->getDebugLoc());old->eraseFromParent();
  // The exit edge now leaves the new header, including when splitting a
  // single-block loop moved the old latch into body. Values were proved to
  // dominate the header before the transform.
  for(auto&phi:guard->getSuccessor(1)->phis()){
   int edge=phi.getBasicBlockIndex(plan.latch->getParent());
   phi.setIncomingBlock(edge,plan.header);
  }
  auto*oldTest=llvm::cast<llvm::Instruction>(plan.latch->getCondition());
  auto*backedge=llvm::UncondBrInst::Create(plan.header,plan.latch->getIterator());
  backedge->setDebugLoc(plan.latch->getDebugLoc());plan.latch->eraseFromParent();
  if(oldTest->use_empty())oldTest->eraseFromParent();
 }
}
// Collapse small pure triangles into selects, retaining ordered/unordered
// floating comparisons exactly. Never speculate memory, calls or FP narrowing.
static void selectPureTriangles(llvm::Module&m){
 for(auto&fn:m){
  bool changed;
  do{
   changed=false;
   for(auto&head:fn){
    auto*branch=llvm::dyn_cast<llvm::CondBrInst>(head.getTerminator());
    if(!branch)continue;
    unsigned owner=llvmTaskLine(branch->getDebugLoc());
    auto*condition=llvm::dyn_cast<llvm::Instruction>(branch->getCondition());
    if(!owner||!condition||llvmTaskLine(condition->getDebugLoc())!=owner)continue;
    for(unsigned side=0;side<2;side++){
     auto*arm=branch->getSuccessor(side);auto*join=branch->getSuccessor(1-side);
     if(arm==join||arm==&head||join==&head||arm->isEHPad()||arm->hasAddressTaken()||arm->getSinglePredecessor()!=&head||!arm->phis().empty())continue;
     auto*edge=llvm::dyn_cast<llvm::UncondBrInst>(arm->getTerminator());
     if(!edge||edge->getSuccessor(0)!=join||llvmTaskLine(edge->getDebugLoc())!=owner)continue;
     bool safe=true;unsigned count=0;
     for(auto&inst:*arm){
      if(&inst==edge)continue;
      if(++count>8||llvmTaskLine(inst.getDebugLoc())!=owner||inst.mayReadOrWriteMemory()||inst.mayHaveSideEffects()||llvm::isa<llvm::CallBase,llvm::FPToSIInst,llvm::FPToUIInst>(inst)||!llvm::isSafeToSpeculativelyExecute(&inst)){safe=false;break;}
     }
     if(!safe)continue;
     for(auto&phi:join->phis())if(llvmTaskLine(phi.getDebugLoc())!=owner||phi.getBasicBlockIndex(&head)<0||phi.getBasicBlockIndex(arm)<0)safe=false;
     if(!safe)continue;
     // Move definitions in their original order before replacing incoming PHIs.
     while(&arm->front()!=edge)arm->front().moveBefore(branch->getIterator());
     for(auto&phi:join->phis()){
      auto*direct=phi.getIncomingValueForBlock(&head);auto*through=phi.getIncomingValueForBlock(arm);
      auto*choice=llvm::SelectInst::Create(branch->getCondition(),side==0?through:direct,side==0?direct:through,phi.getName()+".choice",branch->getIterator());
      choice->setDebugLoc(phi.getDebugLoc());
      phi.setIncomingValue(phi.getBasicBlockIndex(&head),choice);
      phi.removeIncomingValue(arm,false);
     }
     auto*next=llvm::UncondBrInst::Create(join,branch->getIterator());
     next->setDebugLoc(branch->getDebugLoc());branch->eraseFromParent();
     arm->eraseFromParent();changed=true;break;
    }
    if(changed)break;
   }
  }while(changed);
 }
}
static mlir::OwningOpRef<mlir::ModuleOp> compile(Context&c,mlir::MLIRContext&context){
 char temp[]="/tmp/merlin-compile-XXXXXX";char*d=mkdtemp(temp);if(!d)throw std::runtime_error("temporary directory failed");std::string dir=d;
 {std::ofstream f(dir+"/kernel.c");f<<c.source;}
 std::string prefix=std::getenv("MERLIN_MLIR_INSTALL")?std::getenv("MERLIN_MLIR_INSTALL"):"/work/merlin/toolchains/llvm-a47bddcc/install";
 std::string clang=std::getenv("MERLIN_CLANG")?std::getenv("MERLIN_CLANG"):prefix+"/bin/clang";
 run({clang,"--target=riscv64-unknown-elf","-march=rv64gc_zba_zbb_zbs_zfh","-mabi=lp64d","-mcmodel=medany","-O2","-ffreestanding","-fno-builtin","-ffp-contract=off","-fno-vectorize","-fno-slp-vectorize","-gline-tables-only","-S","-emit-llvm",dir+"/kernel.c","-o",dir+"/kernel.ll"});
 llvm::LLVMContext llvmContext;llvm::SMDiagnostic diag;auto m=llvm::parseIRFile(dir+"/kernel.ll",diag,llvmContext);if(!m){diag.print("compiler",llvm::errs());throw std::runtime_error("LLVM parsing failed");}
 // Task functions were optimized in isolation: own all their instructions,
 // including address calculations and stores without surviving debug metadata.
 llvm::DIBuilder debug(*m,false);auto taskFile=debug.createFile("merlin_task","");
 std::vector<llvm::Function*> taskFunctions;
 for(auto&fn:*m)if(fn.getName().starts_with("merlin_task_fn_")){
  auto suffix=fn.getName().drop_front(15);unsigned task=0;suffix.split('_').first.getAsInteger(10,task);
  auto scope=debug.createLexicalBlockFile(fn.getSubprogram(),taskFile,0);
  for(auto&block:fn)for(auto&inst:block)inst.setDebugLoc(llvm::DILocation::get(llvmContext,task+1,1,scope));
  fn.removeFnAttr(llvm::Attribute::NoInline);taskFunctions.push_back(&fn);
 }
 debug.finalize();
 for(auto*fn:taskFunctions){
  while(!fn->use_empty()){
   auto*call=llvm::dyn_cast<llvm::CallBase>(*fn->user_begin());if(!call)throw std::runtime_error("unexpected task function use");
   llvm::InlineFunctionInfo info;auto result=llvm::InlineFunction(*call,info);
   if(!result.isSuccess())throw std::runtime_error("task inlining failed");
  }
  fn->eraseFromParent();
 }
 splitTaskBlocks(*m);
 normalizeLoopBranches(*m);
 canonicalizeSimpleLoopHeaders(*m);
 selectPureTriangles(*m);
 // The source interface has ordinary wrapping index/control arithmetic.
 // Do not export additional poison conditions inferred by the C optimizer.
 // Dropping these flags only broadens defined behavior; optimized values and
 // source task ownership remain intact.
 for(auto &fn:*m)for(auto &block:fn)for(auto &inst:block)inst.dropPoisonGeneratingFlags();
 auto module=mlir::translateLLVMIRToModule(std::move(m),&context,false);if(!module)throw std::runtime_error("LLVM import failed");
 mlir::Builder b(&context);
 for(auto fn:module->getOps<mlir::LLVM::LLVMFuncOp>()){
  // The public entry pointer attributes are deliberately empty.
  if(fn.getName()=="kernel_entry"){fn.removeArgAttrsAttr();fn.removeResAttrsAttr();}
  if(fn.getName()=="kernel_complete")continue;
  fn.walk([&](mlir::Operation*op){auto line=taskLine(op->getLoc());if(line){
   int64_t source=line-1;const Obj*record=nullptr;
   if(auto params=c.cb.getObject("params")){
    if(auto schedule=params->getArray("compiler_schedule"))for(auto&v:*schedule){auto&s=*v.getAsObject();if(num(s,"task_index")==line-1){source=num(s,"source_op_index");record=s.getObject("source_op");break;}}
    if(!record)if(auto commands=params->getArray("compiler_command_sources"))if(line<=commands->size()){record=(*commands)[line-1].getAsObject();if(record)source=num(*record,"source_op_index",source);}
   }
   op->setAttr("merlin.global_task",b.getI64IntegerAttr(line-1));op->setAttr("merlin.source_op_index",b.getI64IntegerAttr(source));
   if(record)if(auto prov=record->getObject("prov")){auto region=str(*prov,"region_id");if(!region.empty())op->setAttr("prov.region_id",b.getStringAttr(region));}
  }});
 }
 // Debug locations are used only to recover stable source ownership. Do not
 // expose clang's temporary paths or LLVM-version-specific debug metadata in
 // the normalized runtime artifact.
 module->walk([&](mlir::Operation*op){op->setLoc(b.getUnknownLoc());op->removeAttr("loop_annotation");op->removeAttr("tbaa");op->removeAttr("access_groups");});
 for(auto&op:llvm::make_early_inc_range(module->getBody()->getOperations()))if(op.getName().getStringRef()=="llvm.module_flags")op.erase();
 unlink((dir+"/kernel.c").c_str());unlink((dir+"/kernel.ll").c_str());rmdir(dir.c_str());return module;
}
}
int main(int argc,char**argv){try{
 std::string input,cbPath;bool artifact=false,verify=false,ownedInput=false;for(int i=1;i<argc;++i){std::string a=argv[i];if(a.rfind("--semantic-input=",0)==0)input=a.substr(17);if(a.rfind("--emit-command-buffer=",0)==0)cbPath=a.substr(22);if(a=="--emit-target-artifact")artifact=true;if(a=="--verify-diagnostics")verify=true;}
 if(input.empty()){
 std::string sourcePath;for(int i=1;i<argc;++i)if(argv[i][0]!='-')sourcePath=argv[i];if(sourcePath.empty())throw std::runtime_error("missing input source");
 auto package=std::filesystem::absolute(argv[0]).parent_path().parent_path();
 char temp[]="/tmp/merlin-reader-XXXXXX";int fd=mkstemp(temp);if(fd<0)throw std::runtime_error("reader tempfile failed");close(fd);input=temp;ownedInput=true;
 mc::run({"python",(package/"frontend.py").string(),"--dump-semantic="+input,sourcePath});
 }
 auto jsonText=mc::read(input);if(ownedInput)unlink(input.c_str());auto parsed=llvm::json::parse(jsonText);if(!parsed)throw std::runtime_error(llvm::toString(parsed.takeError()));auto doc=std::move(*parsed);auto*o=doc.getAsObject();if(!o)throw std::runtime_error("invalid semantic input");mc::Context c;c.sourceHash=mc::str(*o,"source_sha256");
 auto*model=o->getObject("model");if(!model)throw std::runtime_error("missing semantic model");
 if(mc::str(*o,"format")=="linalg"){for(auto&v:*model)c.inventory[v.first]=mc::copy(v.second);mc::lowerInventory(c);}else for(auto&v:*model)c.cb[v.first]=mc::copy(v.second);
 if(mc::str(c.cb,"abi_version")!="0.1")throw std::runtime_error("unsupported interface ABI version");c.cb["target"]="gemmini";
 mc::admit(c);
 if(verify)return 0;
 mlir::DialectRegistry registry;registry.insert<mc::DeviceDialect, mlir::DLTIDialect>();mlir::registerLLVMDialectImport(registry);mlir::MLIRContext context(registry);context.getOrLoadDialect<mc::DeviceDialect>();
 if(artifact){mc::emitKernel(c);auto module=mc::compile(c,context);module->print(llvm::outs(),mlir::OpPrintingFlags().printGenericOpForm());llvm::outs()<<"\n";}
 else if(cbPath.empty()){
  auto module=mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));mlir::OpBuilder b(&context);b.setInsertionPointToEnd(module.getBody());
  for(auto&v:*c.cb.getArray("commands")){auto&cmd=*v.getAsObject();mlir::OperationState st(b.getUnknownLoc(),"gemmini."+mc::str(cmd,"opcode"));st.addAttribute("descriptor",b.getStringAttr(llvm::formatv("{0}",v).str()));b.create(st);}module.print(llvm::outs());llvm::outs()<<"\n";
 }
 if(!cbPath.empty()){std::ofstream f(cbPath);if(!f)throw std::runtime_error("cannot write command buffer");mc::writeBuffer(f,c.cb);}
 return 0;
 }catch(const std::exception&e){llvm::errs()<<"compiler: "<<e.what()<<"\n";return 1;}}
