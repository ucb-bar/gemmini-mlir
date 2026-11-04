#include "compiler.h"
#include "host.h"
#include "math_support.h"
#include <set>
namespace mc {
std::string ctype(const std::string& d){
 if(d=="i8"||d=="si8")return "signed char";if(d=="ui8")return "unsigned char";
 if(d=="i16")return "short";if(d=="ui16")return "unsigned short";
 if(d=="i32")return "int";if(d=="ui32")return "unsigned int";
 if(d=="i64"||d=="index")return "long";if(d=="ui64")return "unsigned long";
 if(d=="f32")return "float";if(d=="f64")return "double";
 if(d=="f16")return "_Float16";if(d=="bf16")return "unsigned short";
 if(d=="i1")return "unsigned char";
 throw std::runtime_error("unsupported storage dtype "+d);
}
void emitKernel(Context&c){
 auto ts=c.cb.getObject("tensors"); if(!ts)throw std::runtime_error("missing tensors");
 std::vector<std::string> order;
 auto add=[&](std::string n){if(std::find(order.begin(),order.end(),n)==order.end())order.push_back(n);};
 if(auto abi=c.cb.getObject("kernel_abi")){for(auto&v:*abi->getArray("args"))add(str(*v.getAsObject(),"tensor"));}
 else {
  std::set<std::string> produced;
  auto commands=c.cb.getArray("commands");
  for(auto&v:*commands){auto&o=*v.getAsObject();if(auto p=o.getObject("operands"))if(p->getString("dst"))produced.insert(str(*p,"dst"));}
  // Source declaration order is transported explicitly by the input reader.
  std::vector<std::string> declared;
  if(auto params=c.cb.getObject("params"))if(auto names=params->getArray("compiler_tensor_order"))for(auto&v:*names)declared.push_back(v.getAsString()->str());
  if(declared.empty())for(auto&v:*ts)declared.push_back(v.first.str());
  for(auto&n:declared){auto&o=*ts->getObject(n);auto role=str(o,"role");if(!produced.count(n)&&role!="output"&&role!="intermediate")add(n);}
  for(auto&v:*commands){auto&o=*v.getAsObject();auto op=str(o,"opcode");if(op!="RES_PACK"&&op!="MATMUL_RESIDENT"&&op!="MATMUL"&&op!="EVICT"){auto p=o.getObject("operands");if(p&&p->getString("dst")&&ts->getObject(str(*p,"dst")))add(str(*p,"dst"));}}
  for(auto&v:*ts)if(str(*v.second.getAsObject(),"role")=="output")add(v.first.str());
 }
 c.tensors.clear();c.names.clear();
 std::map<std::string,std::string> aliases;
 if(auto params=c.cb.getObject("params"))if(auto list=params->getArray("compiler_tensor_aliases"))for(auto &v:*list){auto &a=*v.getAsObject();aliases[str(a,"tensor")]=str(a,"source");}
 for(auto&n:order){auto*o=ts->getObject(n);if(!o)throw std::runtime_error("ABI tensor missing "+n);Tensor t{n,str(*o,"dtype"),str(*o,"role"),"p"+std::to_string(c.tensors.size()),ints(*o,"shape")};c.names[n]=c.tensors.size();c.tensors.push_back(t);}
 // Compiler internal tensor storage may be outside the pointer ABI.
 for(auto&v:*ts)if(!c.names.count(v.first.str())){auto&o=*v.second.getAsObject();Tensor t{v.first.str(),str(o,"dtype"),str(o,"role"),"tmp"+std::to_string(c.tensors.size()),ints(o,"shape")};if(!aliases.count(t.name))c.globals<<"static "<<ctype(t.dtype)<<" "<<t.var<<"["<<t.count()<<"] __attribute__((aligned(64)));\n";c.names[t.name]=c.tensors.size();c.tensors.push_back(t);}
 for(auto &a:aliases){auto &view=c.tensor(a.first);auto &root=c.tensor(a.second);if(view.dtype!=root.dtype||view.count()!=root.count()||aliases.count(a.second))throw std::runtime_error("invalid tensor storage alias");view.var=root.var;}
 // Private contiguous source views are shape descriptors sharing declared
 // storage. They deliberately carry neither an ABI allocation nor a C global.
 if(auto params=c.cb.getObject("params"))if(auto views=params->getArray("compiler_internal_views"))for(auto&v:*views){
  auto&o=*v.getAsObject();auto name=str(o,"tensor"),source=str(o,"source");
  if(c.names.count(name))throw std::runtime_error("duplicate private view "+name);
  const auto&root=c.tensor(source);Tensor view{name,str(o,"dtype"),"intermediate",root.var,ints(o,"shape")};
  if(view.dtype!=root.dtype||view.count()!=root.count())throw std::runtime_error("invalid private contiguous view "+name);
  c.names[name]=c.tensors.size();c.tensors.push_back(std::move(view));
 }
 std::map<std::string,std::string> residents;
 std::map<std::string,Obj> accs;
 bool deviceWork=!c.cb.getArray("commands")->empty();
 if(auto params=c.cb.getObject("params"))if(auto schedule=params->getArray("compiler_schedule"))for(auto&v:*schedule){auto kind=str(*v.getAsObject(),"kind");if(kind!="source"&&kind!="alias")deviceWork=true;}
 auto resolve=[&](std::string n){return residents.count(n)?residents.at(n):n;};
 auto emitCommand=[&](const Obj&original){
  Obj cmd;for(auto&v:original)cmd[v.first]=copy(v.second);
  auto opcode=str(cmd,"opcode");auto*p=cmd.getObject("operands");if(!p)throw std::runtime_error("missing operands");
  if(opcode=="RES_PACK"){residents[str(*p,"dst")]=resolve(str(*p,"src"));return;}
  if(opcode=="EVICT")return;
  if(opcode=="MATMUL_RESIDENT"||opcode=="MATMUL"){Obj x;for(auto&v:*p)x[v.first]=copy(v.second);x["rhs"]=resolve(str(*p,"rhs"));accs[str(*p,"dst")]=std::move(x);return;}
  if(opcode=="COMMIT"){
   auto it=accs.find(str(*p,"src"));if(it==accs.end())throw std::runtime_error("unbound accumulator");
   auto&mm=it->second;auto& a=c.tensor(str(mm,"lhs"));auto&w=c.tensor(str(mm,"rhs"));
   if(a.dtype!="i8"||w.dtype!="i8")throw std::runtime_error("non-i8 contraction");
   auto*attrs=cmd.getObject("attributes");Obj empty;const Obj&attrsRef=attrs?*attrs:empty;
   auto bias=str(attrsRef,"bias",str(*p,"bias"));
   emitMatmul(c,a.var,w.var,c.ptr(str(*p,"dst")),a.count()/a.shape.back(),a.shape.back(),w.shape.back(),attrsRef,bias.empty()?"":c.ptr(bias));return;
  }
  if(opcode=="CONV2D"){(*p)["weight"]=resolve(str(*p,"weight"));emitConv(c,cmd);return;}
  if(opcode=="ROW_MAJOR_VIEW")return;
  if(!emitOther(c,cmd))throw std::runtime_error("unsupported command "+opcode);
 };
 auto*params=c.cb.getObject("params");auto*schedule=params?params->getArray("compiler_schedule"):nullptr;
 if(schedule){
  for(auto&v:*schedule){auto&s=*v.getAsObject();int task=num(s,"task_index",num(s,"source_op_index"));c.body<<"\n#line "<<(task+1)<<" \"merlin_task\"\n";
   if(str(s,"kind")=="source"){
    auto list=[&](const char*k){std::vector<std::string> ns;if(auto a=s.getArray(k))for(auto&x:*a)ns.push_back(x.getAsString().value_or("").str());return ns;};
    auto*op=s.getObject("source_op");if(!op||!emitHost(c,*op,list("input_tensors"),list("output_tensors"),list("result_tensors"),list("capture_tensors")))throw std::runtime_error("unhandled source operation");
   }else if(str(s,"kind")=="accelerator_readout"){
    auto&src=c.tensor(str(s,"source_tensor"));emitReadout(c,src.var,c.ptr(str(s,"result_tensor")),src.count()/src.shape.back(),src.shape.back(),*s.getObject("attributes"));
   }else if(auto a=s.getArray("compiler_commands")){for(auto&cmd:*a)emitCommand(*cmd.getAsObject());}
   else if(auto a=s.getArray("command_indices")){for(auto&i:*a)emitCommand(*(*c.cb.getArray("commands"))[*i.getAsInteger()].getAsObject());}
  }
 }else{int i=0;for(auto&v:*c.cb.getArray("commands")){c.body<<"\n#line "<<(++i)<<" \"merlin_task\"\n";emitCommand(*v.getAsObject());}}
 std::ostringstream fn,parameters,arguments,bindings;
 for(size_t i=0;i<order.size();++i){if(i){parameters<<",";arguments<<",";}parameters<<"void *arg"<<i;arguments<<"arg"<<i;auto&t=c.tensor(order[i]);bindings<<ctype(t.dtype)<<" *p"<<i<<"=("<<ctype(t.dtype)<<"*)arg"<<i<<";\n";}
 // Optimize each task independently. The LLVM importer assigns every operation
 // in these functions its source owner before inlining them into the ABI entry.
 // This also owns instructions whose debug locations clang drops or hoists.
 std::vector<std::pair<int,std::string>> bodies;
 std::string taskBodies=c.body.str();
 if(deviceWork||c.usesDma)taskBodies="#line 1 \"merlin_task\"\ngm_flush();\n"+taskBodies;
 std::istringstream lines(taskBodies);std::string line;int taskLine=0;
 while(std::getline(lines,line)){
  if(line.rfind("#line ",0)==0){taskLine=std::stoi(line.substr(6));continue;}
  if(bodies.empty()||bodies.back().first!=taskLine)bodies.push_back({taskLine,""});
  bodies.back().second+=line+"\n";
 }
 std::vector<std::string> calls;
 for(size_t i=0;i<bodies.size();++i){auto&part=bodies[i];if(part.second.find_first_not_of(" \n\t")==std::string::npos)continue;
  auto name="merlin_task_fn_"+std::to_string(part.first?part.first-1:0)+"_"+std::to_string(i);
  fn<<"#line "<<(part.first?part.first:1)<<" \"merlin_task\"\nstatic __attribute__((noinline)) void "<<name<<"("<<parameters.str()<<"){\n"<<bindings.str()<<part.second<<"}\n";
  calls.push_back(name);
 }
 fn<<"#line 1 \"compiler_wrapper\"\nvoid kernel_entry("<<parameters.str()<<"){\n";
 for(auto&name:calls)fn<<name<<"("<<arguments.str()<<");\n";
 fn<<"}\nvoid kernel_complete(void){__asm__ volatile(\"fence\"::: \"memory\");}\n";
 c.source=hostSupport()+deviceSupport()+c.globals.str()+fn.str();
}
}
