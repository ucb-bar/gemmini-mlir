#include "compiler.h"
#include <set>
#include <algorithm>
#include <cctype>
#include <limits>

namespace mc {
namespace {
Arr namesArray(const std::vector<std::string>&v){Arr a;for(auto&s:v)a.push_back(s);return a;}
std::string compact(std::string s){s.erase(std::remove_if(s.begin(),s.end(),[](unsigned char x){return std::isspace(x);}),s.end());return s;}
bool identityMaps(const Obj&op,size_t rank){
 auto maps=op.getArray("indexing_maps"),its=op.getArray("iterator_types");if(!maps||maps->size()!=2||!its||its->size()!=rank)return false;
 std::string dims;for(size_t i=0;i<rank;++i){if(i)dims+=",";dims+="d"+std::to_string(i);}auto expected="affine_map<("+dims+")->("+dims+")>";
 for(auto&v:*maps){auto s=v.getAsString();if(!s||compact(s->str())!=expected)return false;}
 for(auto&v:*its){auto s=v.getAsString();if(!s||*s!="parallel")return false;}return true;
}
bool isBiasAdd(const Obj&op,const Obj&matrix,const Obj&bias,const Obj&result){
 if(str(matrix,"dtype")!="i32"||str(bias,"dtype")!="i32"||str(result,"dtype")!="i32")return false;
 auto shape=ints(matrix,"shape"),bs=ints(bias,"shape");if(shape.size()<2||bs.size()!=1||bs[0]!=shape.back()||ints(result,"shape")!=shape)return false;
 auto sb=op.getObject("scalar_body");auto maps=op.getArray("indexing_maps"),its=op.getArray("iterator_types");if(!sb||!maps||maps->size()!=3||!its||its->size()!=shape.size())return false;
 for(auto&it:*its){auto s=it.getAsString();if(!s||*s!="parallel")return false;}
 auto ops=sb->getArray("operations"),ys=sb->getArray("yields");if(!ops||ops->size()!=1||!ys||ys->size()!=1)return false;auto y=(*ys)[0].getAsString();if(!y||*y!="op:0:0")return false;
 auto add=(*ops)[0].getAsObject();if(!add||str(*add,"op")!="arith.addi")return false;auto av=add->getArray("operands");if(!av||av->size()!=2)return false;auto a=(*av)[0].getAsString(),b=(*av)[1].getAsString();if(!a||!b||!((*a=="arg:0"&&*b=="arg:1")||(*a=="arg:1"&&*b=="arg:0")))return false;
 std::string dims;for(size_t i=0;i<shape.size();++i){if(i)dims+=",";dims+="d"+std::to_string(i);}auto ident="affine_map<("+dims+")->("+dims+")>",vector="affine_map<("+dims+")->(d"+std::to_string(shape.size()-1)+")>";
 auto m0=(*maps)[0].getAsString(),m1=(*maps)[1].getAsString(),m2=(*maps)[2].getAsString();return m0&&m1&&m2&&compact(m0->str())==ident&&compact(m1->str())==vector&&compact(m2->str())==ident;
}
// A full parallel output map initializes every element without observing its
// destination operand. Reductions and partial/indexed writes retain their seed.
bool unusedDestination(const Obj&op,size_t result){
 auto body=op.getObject("scalar_body");auto ins=op.getArray("ins"),maps=op.getArray("indexing_maps"),its=op.getArray("iterator_types"),rs=op.getArray("results");
 if(!body||!ins||!maps||!its||!rs||result>=rs->size()||ins->size()+result>=maps->size())return false;
 auto typ=(*rs)[result].getAsObject();if(!typ||ints(*typ,"shape").size()!=its->size())return false;
 std::string dims;for(size_t i=0;i<its->size();++i){auto it=(*its)[i].getAsString();if(!it||*it!="parallel")return false;if(i)dims+=",";dims+="d"+std::to_string(i);}
 auto map=(*maps)[ins->size()+result].getAsString();if(!map||compact(map->str())!="affine_map<("+dims+")->("+dims+")>")return false;
 auto ref="arg:"+std::to_string(ins->size()+result);
 if(auto operations=body->getArray("operations"))for(auto&v:*operations)if(auto o=v.getAsObject())if(auto operands=o->getArray("operands"))for(auto&a:*operands)if(auto r=a.getAsString())if(*r==ref)return false;
 if(auto yields=body->getArray("yields"))for(auto&v:*yields)if(auto r=v.getAsString())if(*r==ref)return false;
 return true;
}
bool contiguousView(const Obj&op){
 auto kind=str(op,"kind");if(kind!="tensor.collapse_shape"&&kind!="tensor.expand_shape"&&kind!="tensor.cast")return false;
 auto ins=op.getArray("ins"),rs=op.getArray("results");if(!ins||ins->size()!=1||!rs||rs->size()!=1)return false;
 auto a=(*ins)[0].getAsObject(),b=(*rs)[0].getAsObject();if(!a||!b||str(*a,"dtype")!=str(*b,"dtype"))return false;
 auto count=[](const Obj&t){int64_t n=1;for(auto d:ints(t,"shape")){if(d<=0||n>std::numeric_limits<int64_t>::max()/d)return int64_t(-1);n*=d;}return n;};
 auto n=count(*a);if(n<=0||n!=count(*b))return false;
 // Mathematical tensors are stored contiguously; these operations preserve
 // flat element order. Static casts additionally preserve the full shape.
 return kind!="tensor.cast"||ints(*a,"shape")==ints(*b,"shape");
}
std::string hostReason(const Obj&op){
 auto in=op.getArray("ins"),out=op.getArray("results");std::set<std::string>types;for(auto group:{in,out})if(group)for(auto&v:*group)if(auto r=v.getAsObject())types.insert(str(*r,"dtype"));
 if(str(op,"kind").find("tensor.")==0)return "Source "+str(op,"kind")+" carries an explicit tensor initialization, indexing, or shape transformation; scalar indexed loads and stores preserve its specified element order and element representation.";
 std::string kinds;if(auto body=op.getObject("scalar_body"))if(auto operations=body->getArray("operations"))for(auto&v:*operations)if(auto o=v.getAsObject()){auto k=str(*o,"op");if(!kinds.empty())kinds+=", ";kinds+=k;}
 if(types.count("f32")||types.count("f64")||types.count("f16")||types.count("bf16"))return "Source "+str(op,"kind")+" requires floating tensor values"+(kinds.empty()?std::string():" and scalar operations "+kinds)+"; signed i8 multiply/i32 accumulate and scaled signed-i8 readout cannot preserve this operation's floating result and intermediate rounding.";
 if(kinds.find("arith.xori")!=std::string::npos)return "Source bitwise XOR changes individual integer bits; accumulator addition, multiplication and scaled readout do not implement this bitwise truth table.";
 if(str(op,"kind").find("tensor.")==0)return "Source "+str(op,"kind")+" carries a tensor indexing or shape transformation; this task preserves the specified element order through scalar indexed loads and stores.";
 return "Source "+str(op,"kind")+(kinds.empty()?std::string():" scalar DAG "+kinds)+" has no equivalent supported integer contraction, column bias-add, or signed-i8 accumulator readout pattern; scalar execution preserves the explicit typed SSA operations.";
}
bool isClampI8(const Obj &op){
 auto body=op.getObject("scalar_body");if(!body)return false;auto ops=body->getArray("operations"),captures=body->getArray("captures"),yields=body->getArray("yields");if(!ops||!captures||!yields||yields->size()!=1||ops->size()!=3)return false;
 auto bound=[&](const std::string&r,int n){if(r.rfind("capture:",0)!=0)return false;int i=std::stoi(r.substr(8));if(i<0||size_t(i)>=captures->size())return false;auto c=(*captures)[i].getAsObject();return c&&real(*c,"const_value",1e99)==n;};
 auto operands=[&](size_t i){std::vector<std::string>r;auto o=(*ops)[i].getAsObject();if(o)if(auto a=o->getArray("operands"))for(auto&x:*a)if(auto t=x.getAsString())r.push_back(t->str());return r;};
 auto a=(*ops)[0].getAsObject(),b=(*ops)[1].getAsObject(),d=(*ops)[2].getAsObject();if(!a||!b||!d)return false;auto av=operands(0),bv=operands(1),dv=operands(2);
 if(str(*a,"op")!="arith.maxsi"||str(*b,"op")!="arith.minsi"||str(*d,"op")!="arith.trunci"||av.size()!=2||bv.size()!=2||dv.size()!=1)return false;
 if(!((av[0]=="arg:0"&&bound(av[1],-128))||(av[1]=="arg:0"&&bound(av[0],-128))))return false;
 if(!((bv[0]=="op:0:0"&&bound(bv[1],127))||(bv[1]=="op:0:0"&&bound(bv[0],127))))return false;
 auto y=(*yields)[0].getAsString();return dv[0]=="op:1:0"&&y&&*y=="op:2:0";
}
Arr shapeArray(const std::vector<int64_t>&v){Arr a;for(auto x:v)a.push_back(x);return a;}
std::string stringAt(const Arr *a,size_t i,const std::string &fallback){if(a&&i<a->size())if(auto s=(*a)[i].getAsString())return s->str();return fallback;}
std::string key(int64_t op,int64_t result){return std::to_string(op)+":"+std::to_string(result);}
std::string edge(const Obj&v,const std::vector<std::string>&args,const std::map<std::string,std::string>&values){
 auto s=v.getArray("source");if(!s||s->size()<2)return "";auto k=(*s)[0].getAsString();auto i=(*s)[1].getAsInteger();if(!k||!i)return "";
 if(*k=="arg"){if(*i>=0&&size_t(*i)<args.size())return args[*i];throw std::runtime_error("invalid function argument edge");}
 if(*k=="op"){auto p=values.find(key(*i,num(v,"result_index")));if(p!=values.end())return p->second;throw std::runtime_error("unresolved source value edge");}return "";
}
std::vector<std::string> resolve(const Arr*a,const std::vector<std::string>&args,const std::map<std::string,std::string>&values){std::vector<std::string>v;if(a)for(auto &x:*a){auto o=x.getAsObject();v.push_back(o?edge(*o,args,values):"");}return v;}
}

void lowerInventory(Context &c){
 auto &inv=c.inventory;auto ops=inv.getArray("ops"),args=inv.getArray("args"),returns=inv.getArray("returns");
 if(!ops||!args||!returns)throw std::runtime_error("source inventory lacks complete operations, arguments or ordered return bindings");
 Obj tensors;Arr commands,schedule,abiArgs,abiOutputs,sourceValues,tasks,entries,lanePlacement,aliases,internalViews;
 std::map<std::string,std::string>viewRoots;
 std::set<int64_t>deadInitializers;
 std::vector<std::string>argNames,outNames;std::map<std::string,std::string>values,returned;
 std::set<std::string>used;
 auto reserve=[&](std::string n){if(n.empty()||!used.insert(n).second)throw std::runtime_error("duplicate logical tensor name "+n);return n;};
 auto fresh=[&](std::string n){while(used.count(n))n="_"+n;used.insert(n);return n;};
 auto declare=[&](const std::string&name,const Obj&typ,const std::string&role){
   auto shape=ints(typ,"shape");for(auto d:shape)if(d<=0)throw std::runtime_error("dynamic or invalid source tensor extent");
   std::string dtype=str(typ,"dtype");if(dtype.empty())throw std::runtime_error("missing tensor element type");
   tensors[name]=Obj{{"shape",shapeArray(shape)},{"dtype",dtype},{"role",role}};
   abiArgs.push_back(Obj{{"tensor",name},{"access",role=="output"||role=="intermediate"?"write":"read"}});
 };
 for(size_t i=0;i<args->size();++i){auto a=(*args)[i].getAsObject();if(!a)throw std::runtime_error("invalid argument record");
  auto name=reserve(stringAt(inv.getArray("logical_inputs"),i,"arg"+std::to_string(i)));
  auto role=stringAt(inv.getArray("logical_input_roles"),i,"input");
  declare(name,*a,role);argNames.push_back(name);entries.push_back(name);
 }
 for(size_t i=0;i<returns->size();++i){auto r=(*returns)[i].getAsObject();if(!r)throw std::runtime_error("invalid return record");
   auto name=reserve(stringAt(inv.getArray("logical_outputs"),i,"out"+std::to_string(i)));outNames.push_back(name);abiOutputs.push_back(name);
   auto src=r->getArray("source");if(src&&src->size()>1&&(*src)[0].getAsString()&&*(*src)[0].getAsString()=="op"&&(*src)[1].getAsInteger()){
     auto k=key(*(*src)[1].getAsInteger(),num(*r,"result_index"));if(!returned.count(k))returned[k]=name;
   }
 }
 // Only private initialization results whose uses are proven unobserved may
 // disappear. Every input/capture/return and every read destination keeps them.
 std::set<std::string>observed;
 auto observe=[&](const Obj&r){auto src=r.getArray("source");if(src&&src->size()>1&&(*src)[0].getAsString()&&*(*src)[0].getAsString()=="op"&&(*src)[1].getAsInteger())observed.insert(key(*(*src)[1].getAsInteger(),num(r,"result_index")));};
 for(auto&v:*returns)if(auto r=v.getAsObject())observe(*r);
 for(auto&v:*ops)if(auto op=v.getAsObject()){
  if(auto ins=op->getArray("ins"))for(auto&x:*ins)if(auto r=x.getAsObject())observe(*r);
  if(auto outs=op->getArray("outs"))for(size_t i=0;i<outs->size();++i)if(!unusedDestination(*op,i))if(auto r=(*outs)[i].getAsObject())observe(*r);
  if(auto body=op->getObject("scalar_body"))if(auto captures=body->getArray("captures"))for(auto&x:*captures)if(auto r=x.getAsObject())observe(*r);
 }
 for(auto&v:*ops)if(auto op=v.getAsObject()){
  auto kind=str(*op,"kind");if(kind!="tensor.splat"&&kind!="tensor.empty"&&kind!="linalg.fill"&&kind!="arith.constant")continue;
  bool dead=true;auto rs=op->getArray("results");if(!rs||rs->empty())continue;for(size_t i=0;i<rs->size();++i)if(observed.count(key(num(*op,"id"),i)))dead=false;
  if(dead)deadInitializers.insert(num(*op,"id"));
 }
 for(auto &v:*ops){auto op=v.getAsObject();if(!op)throw std::runtime_error("invalid operation inventory");auto rs=op->getArray("results");if(!rs)continue;
   for(size_t ri=0;ri<rs->size();++ri){auto r=(*rs)[ri].getAsObject();if(!r)throw std::runtime_error("invalid result type");auto k=key(num(*op,"id"),ri);if(deadInitializers.count(num(*op,"id"))){values[k]="";continue;}auto it=returned.find(k);
     auto name=it==returned.end()?fresh("v"+std::to_string(num(*op,"id"))+"_"+std::to_string(ri)):it->second;
     declare(name,*r,it==returned.end()?"intermediate":"output");values[k]=name;sourceValues.push_back(Obj{{"op_index",num(*op,"source_op_index",num(*op,"id"))},{"result_index",int64_t(ri)},{"tensor",name}});
   }
 }
 // Retain the explicit storage boundary for an entire shape-only chain
 // ending in a published view. Internal descriptor reuse is optional; preserve
 // the established materialization path at this externally visible boundary.
 std::set<std::string>viewKeys,protectedViews;
 for(auto&v:*ops)if(auto op=v.getAsObject())if(contiguousView(*op)){
  auto k=key(num(*op,"id"),0);viewKeys.insert(k);if(returned.count(k))protectedViews.insert(k);
 }
 // Direct source SSA is topologically ordered, so a single reverse walk reaches
 // every view predecessor without crossing an arithmetic or layout permutation.
 for(size_t reverse=ops->size();reverse>0;--reverse){auto&op=*(*ops)[reverse-1].getAsObject();auto k=key(num(op,"id"),0);if(!protectedViews.count(k))continue;
  auto ins=op.getArray("ins");if(!ins||ins->size()!=1)continue;auto input=(*ins)[0].getAsObject();if(!input)continue;
  auto source=input->getArray("source");if(!source||source->size()<2)continue;auto tag=(*source)[0].getAsString();auto index=(*source)[1].getAsInteger();if(!tag||*tag!="op"||!index)continue;
  auto predecessor=key(*index,num(*input,"result_index"));if(viewKeys.count(predecessor))protectedViews.insert(predecessor);
 }
 // A private shape-only SSA value is a descriptor, not a materialized
 // source tensor. Published results always retain their own output storage.
 for(auto&v:*ops)if(auto op=v.getAsObject())if(contiguousView(*op)){
  auto k=key(num(*op,"id"),0);if(protectedViews.count(k))continue;
  auto ins=resolve(op->getArray("ins"),argNames,values);if(ins.empty()||ins[0].empty())continue;
  auto name=values.at(k),root=ins[0];if(viewRoots.count(root))root=viewRoots.at(root);
  viewRoots[name]=root;auto&typ=*tensors.getObject(name);
  internalViews.push_back(Obj{{"tensor",name},{"source",root},{"shape",shapeArray(ints(typ,"shape"))},{"dtype",str(typ,"dtype")},{"source_op_index",num(*op,"source_op_index",num(*op,"id"))},{"result_index",0}});
 }
 auto storage=[&](const std::string&n){auto it=viewRoots.find(n);return it==viewRoots.end()?n:it->second;};
 for(size_t oi=0;oi<ops->size();++oi){const auto &op=*(*ops)[oi].getAsObject();auto id=num(op,"id",oi);auto sourceId=num(op,"source_op_index",id);
   auto ins=resolve(op.getArray("ins"),argNames,values),outs=resolve(op.getArray("outs"),argNames,values);
   for(size_t i=0;i<outs.size();++i)if(unusedDestination(op,i))outs[i]="";
   std::vector<std::string>res,captures;auto rs=op.getArray("results");if(rs)for(size_t ri=0;ri<rs->size();++ri)res.push_back(values.at(key(id,ri)));
   if(auto body=op.getObject("scalar_body"))captures=resolve(body->getArray("captures"),argNames,values);
   Obj step{{"task_index",int64_t(oi)},{"source_op_index",sourceId},{"source_op",copy((*ops)[oi])},{"input_tensors",namesArray(ins)},{"output_tensors",namesArray(outs)},{"result_tensors",namesArray(res)},{"capture_tensors",namesArray(captures)}};
   Arr indices;std::string kind=str(op,"kind");bool contraction=kind=="linalg.matmul"||kind=="linalg.batch_matmul"||kind=="linalg.matmul_transpose_b"||str(op,"family")=="contraction";
   bool deviceContraction=contraction&&ins.size()>=2&&!ins[0].empty()&&!ins[1].empty()&&str(*tensors.getObject(ins[0]),"dtype")=="i8"&&str(*tensors.getObject(ins[1]),"dtype")=="i8";
   bool deviceEpilogue=false;
   if(deadInitializers.count(id)||(res.size()==1&&viewRoots.count(res[0]))){step["kind"]="alias";}
   else if(!contraction&&ins.size()==2&&res.size()==1&&!ins[0].empty()&&!ins[1].empty()&&isBiasAdd(op,*tensors.getObject(ins[0]),*tensors.getObject(ins[1]),*tensors.getObject(res[0]))){
     indices.push_back(int64_t(commands.size()));commands.push_back(Obj{{"opcode","BIAS_ADD"},{"operands",Obj{{"src",ins[0]},{"bias",ins[1]},{"dst",res[0]}}},{"attributes",Obj{{"output_dtype","i32"},{"epilogue",Arr{"bias_add"}}}}});step["kind"]="matmul";deviceEpilogue=true;
   }else if(!contraction&&ins.size()==1&&res.size()==1&&!ins[0].empty()&&str(*tensors.getObject(ins[0]),"dtype")=="i32"&&str(*tensors.getObject(res[0]),"dtype")=="i8"&&ints(*tensors.getObject(ins[0]),"shape")==ints(*tensors.getObject(res[0]),"shape")&&identityMaps(op,ints(*tensors.getObject(ins[0]),"shape").size())&&isClampI8(op)){
     // This is a whole-program source task: the public MOVEMENT opcode is an
     // identity operation and cannot describe narrowing/saturation. The private
     // schedule names the actual accumulator-load and signed-i8 readout work.
     step["kind"]="accelerator_readout";step["source_tensor"]=ins[0];step["result_tensor"]=res[0];
     step["attributes"]=Obj{{"acc_scale",1.0},{"epilogue",Arr{}},{"output_dtype","i8"}};deviceEpilogue=true;
   }else if(deviceContraction&&ins.size()>=2&&!ins[0].empty()&&!ins[1].empty()&&res.size()==1){
     const auto&at=*tensors.getObject(ins[0]);const auto&bt=*tensors.getObject(ins[1]);auto ash=ints(at,"shape"),bsh=ints(bt,"shape");
     if(str(at,"dtype")!="i8"||str(bt,"dtype")!="i8")throw std::runtime_error("accelerator contraction requires signed i8 operands");
     Obj attrs{{"output_dtype",str(*tensors.getObject(res[0]),"dtype")},{"epilogue",Arr{}}};
     if(!outs.empty()&&!outs[0].empty())attrs["initial_tensor"]=outs[0];
     else if(auto os=op.getArray("outs")){if(!os->empty())if(auto init=(*os)[0].getAsObject())if(auto val=init->get("const_value"))attrs["initial_value"]=copy(*val);}
     if(ash.size()>2){indices.push_back(int64_t(commands.size()));commands.push_back(Obj{{"opcode","BATCHED_MATMUL"},{"operands",Obj{{"a",ins[0]},{"w",ins[1]},{"dst",res[0]}}},{"attributes",std::move(attrs)}});}
     else if(kind=="linalg.matmul_transpose_b") {indices.push_back(int64_t(commands.size()));commands.push_back(Obj{{"opcode","ATTENTION_QK"},{"operands",Obj{{"q",ins[0]},{"k",ins[1]},{"dst",res[0]}}},{"attributes",std::move(attrs)}});}
     else {std::string acc="acc"+std::to_string(id);indices.push_back(int64_t(commands.size()));commands.push_back(Obj{{"opcode","MATMUL"},{"operands",Obj{{"lhs",ins[0]},{"rhs",ins[1]},{"dst",acc}}},{"attributes",Obj{}}});indices.push_back(int64_t(commands.size()));commands.push_back(Obj{{"opcode","COMMIT"},{"operands",Obj{{"src",acc},{"dst",res[0]}}},{"attributes",std::move(attrs)}});}
     step["kind"]="matmul";
   }else step["kind"]="source";
   bool host=str(step,"kind")=="source";
   std::string region;if(auto prov=op.getObject("prov"))region=str(*prov,"region_id");
   if(host){Obj lane{{"source_op_index",sourceId},{"task_index",int64_t(oi)},{"lane","host"},{"reason",hostReason(op)}};if(!region.empty())lane["region_id"]=region;lanePlacement.push_back(std::move(lane));}
   bool aliasStep=str(step,"kind")=="alias";step["command_indices"]=std::move(indices);schedule.push_back(std::move(step));
   Arr reads,writes,regions;std::set<std::string>seen;for(auto group:{ins,outs,captures})for(auto&n:group)if(!n.empty()&&seen.insert(storage(n)).second)reads.push_back(storage(n));for(auto&n:res)if(!n.empty()&&!viewRoots.count(n))writes.push_back(n);
   if(auto prov=op.getObject("prov")){auto region=str(*prov,"region_id");if(!region.empty())regions.push_back(region);}
   Arr covered; int64_t previousSource=oi?num(*(*ops)[oi-1].getAsObject(),"source_op_index",int64_t(oi-1)):-1; for(int64_t si=previousSource+1;si<=sourceId;++si)covered.push_back(si);
   tasks.push_back(Obj{{"task_index",int64_t(oi)},{"kind",aliasStep?"alias":deviceContraction||deviceEpilogue?"accelerator":"host"},{"source_op_indices",std::move(covered)},{"source_region_ids",std::move(regions)},{"instruction_start",int64_t(oi)},{"instruction_end",int64_t(oi+1)},{"reads",std::move(reads)},{"writes",std::move(writes)}});
 }
 if(!tasks.empty()&&!ops->empty()){
   auto finalSource=num(*ops->back().getAsObject(),"source_op_index",ops->size()-1);
   auto sourceCount=num(inv,"source_op_count",ops->size());
   auto covered=tasks.back().getAsObject()->getArray("source_op_indices");
   for(auto si=finalSource+1;si<sourceCount;++si)covered->push_back(si);
 }
 // A returned entry argument or repeated return owns a distinct ABI destination.
 // Represent its required copy explicitly in the internal schedule.
 for(size_t i=0;i<returns->size();++i){const auto&r=*(*returns)[i].getAsObject();auto src=edge(r,argNames,values);if(src==outNames[i])continue;if(src.empty())throw std::runtime_error("return of initializer or constant is not represented by payload");
   declare(outNames[i],r,"output");Obj rec{{"kind","tensor.cast"},{"ins",Arr{copy((*returns)[i])}},{"outs",Arr{}},{"results",Arr{Obj{{"shape",shapeArray(ints(r,"shape"))},{"dtype",str(r,"dtype")}}}}};
   auto lastTask=int64_t(tasks.empty()?0:tasks.size()-1),lastSource=ops->empty()?0:num(*ops->back().getAsObject(),"source_op_index",ops->size()-1);
   schedule.push_back(Obj{{"kind","source"},{"task_index",lastTask},{"source_op_index",lastSource},{"source_op",std::move(rec)},{"input_tensors",Arr{src}},{"output_tensors",Arr{}},{"result_tensors",Arr{outNames[i]}},{"capture_tensors",Arr{}},{"command_indices",Arr{}}});
   if(!tasks.empty())if(auto task=tasks.back().getAsObject()){task->getArray("writes")->push_back(outNames[i]);auto read=task->getArray("reads");bool found=false;for(auto&v:*read)if(auto s=v.getAsString())if(*s==storage(src))found=true;if(!found)read->push_back(storage(src));}
 }
 // Public commands can only reference declared logical tensors. Commands
 // consuming a private descriptor belong to the compiler's source schedule;
 // keep their exact logical dimensions without fabricating public ABI storage.
 Arr publicCommands;
 for(auto&v:schedule)if(auto step=v.getAsObject())if(auto indices=step->getArray("command_indices")){
  bool privateCommand=false;
  for(auto&i:*indices){auto&cmd=*commands[*i.getAsInteger()].getAsObject();for(auto field:{"operands","attributes"})if(auto attrs=cmd.getObject(field))for(auto&entry:*attrs)if(auto value=entry.second.getAsString())if(viewRoots.count(value->str()))privateCommand=true;}
  Arr revised,privateCommands;
  for(auto&i:*indices){auto&cmd=commands[*i.getAsInteger()];if(privateCommand)privateCommands.push_back(copy(cmd));else{revised.push_back(int64_t(publicCommands.size()));publicCommands.push_back(copy(cmd));}}
  if(privateCommand)(*step)["compiler_commands"]=std::move(privateCommands);
  (*step)["command_indices"]=std::move(revised);
 }
 commands=std::move(publicCommands);
 if(!viewRoots.empty()){
  Arr materializedArgs,materializedValues;
  for(auto&v:abiArgs)if(!viewRoots.count(str(*v.getAsObject(),"tensor")))materializedArgs.push_back(copy(v));
  for(auto&v:sourceValues)if(!viewRoots.count(str(*v.getAsObject(),"tensor")))materializedValues.push_back(copy(v));
  abiArgs=std::move(materializedArgs);sourceValues=std::move(materializedValues);
  for(auto&v:viewRoots)tensors.erase(v.first);
 }
 Obj plan{{"schema","mixed_program_plan_v1"},{"source_sha256",c.sourceHash},{"source_op_count",num(inv,"source_op_count",ops->size())},{"tasks",std::move(tasks)},{"schedule_instruction_count",int64_t(std::max<size_t>(1,schedule.size()))},{"prologue_instruction_range",Arr{0,0}},{"epilogue_instruction_range",Arr{int64_t(schedule.size()),int64_t(schedule.size())}},{"entry_bindings",std::move(entries)},{"source_values",std::move(sourceValues)},{"output_bindings",namesArray(outNames)}};
 c.cb=Obj{{"abi_version","0.1"},{"target","gemmini_rocket_scalar_int8_v0"},{"backend","external_backend"},{"tensors",std::move(tensors)},{"commands",std::move(commands)},{"kernel_abi",Obj{{"kind","whole_program"},{"args",std::move(abiArgs)},{"outputs",std::move(abiOutputs)}}},{"params",Obj{{"compiler_internal_views",std::move(internalViews)},{"compiler_tensor_aliases",std::move(aliases)},{"compiler_schedule",std::move(schedule)},{"global_program_plan",std::move(plan)},{"lane_placement",std::move(lanePlacement)}}}};
}
}
