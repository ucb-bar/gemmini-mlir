#include "compiler.h"
#include <set>
#include <algorithm>
#include <functional>
#include <cmath>
namespace backend {
namespace {
const Object &params(const Object &e) { static const Object empty; auto *p=e.getObject("parameters"); return p?*p:empty; }
std::string textValue(const Value &v) { auto s=v.getAsString(); if(!s) throw std::runtime_error("expected value identifier"); return s->str(); }
std::vector<std::string> operands(const Object&e) { std::vector<std::string> r; if(auto*a=e.getArray("operands")) for(auto&v:*a)r.push_back(textValue(v)); return r; }
Array strings(const std::vector<std::string>&s) { Array a; for(auto &x:s)a.push_back(x);return a; }
Array numbers(const std::vector<int64_t>&s) { Array a;for(auto x:s)a.push_back(x);return a; }
Object tensor(const Object &o,const std::string &role) { return Object{{"shape",Value(Array(arr(o,"shape")))},{"dtype",str(o,"dtype")},{"role",role}}; }
void copyField(Object &dst,llvm::StringRef key,const Object &src,llvm::StringRef source) { if(auto*v=src.get(source))dst[key]=Value(*v); }
void uniquePush(std::vector<std::string>&v,const std::string&s) {if(std::find(v.begin(),v.end(),s)==v.end())v.push_back(s);}
bool isEpilogue(const std::string &op) {return op=="add_bias"||op=="scale_round"||op=="round_shift_right"||op=="relu"||op=="cast"||op=="max_pool";}
// Fused pooling keeps only window-local accelerator state. Padding must be
// neutral after narrowing, and every output window must have a real input tap.
bool devicePoolGeometry(const Object&pool,bool relu) {
 auto&pr=params(pool);auto vectorParam=[&](llvm::StringRef key,std::vector<int64_t> fallback){if(auto*a=pr.getArray(key)){fallback.clear();for(auto&v:*a){auto n=v.getAsInteger();if(!n)return std::vector<int64_t>{};fallback.push_back(*n);}}return fallback;};auto spatial=vectorParam("spatial_shape",{}),window=vectorParam("window",{}),stride=vectorParam("stride",{}),padding=vectorParam("padding",{0,0,0,0});
 bool neutralPad=true;if(auto*v=pr.get("padding_value"))if(v->kind()!=Value::Null){auto pad=v->getAsInteger();neutralPad=pad&&(*pad<=-128||(relu&&*pad<=0));}
 if(!neutralPad||spatial.size()!=2||window.size()!=2||stride.size()!=2||padding.size()!=4)return false;
 for(int axis=0;axis<2;++axis){int64_t h=spatial[axis],w=window[axis],step=stride[axis],before=padding[axis],after=padding[axis+2];if(h<=0||w<=0||step<=0||before<0||after<0||h+before+after<w)return false;int64_t count=(h+before+after-w)/step+1;if(w-1-before<0||(count-1)*step-before>=h)return false;}
 return true;
}
struct Group { std::vector<int> expressions; std::string category; bool commands=true; };
}
Object lowerCommands(const Object &p,const std::string &sha) {
 const auto &ins=arr(p,"inputs"),&es=arr(p,"expressions"),&outs=arr(p,"outputs");
 std::map<std::string,const Object*> defs;
 std::map<std::string,int> producer;
 std::map<std::string,std::vector<int>> users;
 std::map<std::string,std::vector<std::string>> outputNames;
 std::set<std::string> occupied;
 for(auto &v:ins) {auto&e=*v.getAsObject(); auto id=str(e,"id"); if(defs.count(id))throw std::runtime_error("duplicate input identifier");defs[id]=&e; occupied.insert(str(e,"name",id));}
 for(size_t i=0;i<es.size();++i) {auto&e=*es[i].getAsObject();auto id=str(e,"id");if(defs.count(id))throw std::runtime_error("duplicate expression identifier");defs[id]=&e;producer[id]=i;for(auto&x:operands(e))users[x].push_back(i);}
 for(auto &v:outs){auto&o=*v.getAsObject();auto id=str(o,"value");if(!defs.count(id))throw std::runtime_error("undefined output value "+id);outputNames[id].push_back(str(o,"name"));occupied.insert(str(o,"name"));}
 auto expr=[&](int i)->const Object& {return *es[i].getAsObject();};
 auto next=[&](int i)->int {auto id=str(expr(i),"id");auto &u=users[id]; if(outputNames.count(id)||u.size()!=1||u[0]<=i)return -1; auto in=operands(expr(u[0]));return !in.empty()&&in[0]==id?u[0]:-1;};
 // The batched command has no broadcasting or implicit shared weights. Keep
 // its complete rank and batch prefix, rather than reducing it to one matrix.
 for(int i=0;i<(int)es.size();++i){auto&e=expr(i);if(str(e,"operator")!="matmul")continue;auto in=operands(e);if(in.size()!=2)throw std::runtime_error("matmul requires two operands");auto a=shape(*defs.at(in[0])),b=shape(*defs.at(in[1])),d=shape(e);
  bool valid=a.size()>=2&&a.size()==b.size()&&a.size()==d.size();
  if(valid){for(size_t q=0;q<a.size();++q)if(a[q]<=0||b[q]<=0||d[q]<=0)valid=false;for(size_t q=0;q+2<a.size();++q)if(a[q]!=b[q]||a[q]!=d[q])valid=false;size_t r=a.size();if(a[r-1]!=b[r-2]||a[r-2]!=d[r-2]||b[r-1]!=d[r-1])valid=false;}
  if(!valid)throw std::runtime_error("matmul requires matching positive batch prefixes and contraction dimensions");
 }
 // A contraction readout requesting half-up integer requant cannot be replaced
 // by the target's ties-to-even acc_scale stage. Recognize the stage from the
 // source graph, only while every wide result is unobserved and single-use and
 // the chain ends in saturating i8 readout. Independent host shifts remain H.
 int declinedExpression=-1;
 for(int i=0;i<(int)es.size()&&declinedExpression<0;++i){auto&root=expr(i);auto op=str(root,"operator");if(op!="matmul"&&op!="conv2d")continue;auto in=operands(root);if(in.size()!=2||str(root,"dtype")!="i32"||str(*defs.at(in[0]),"dtype")!="i8"||str(*defs.at(in[1]),"dtype")!="i8")continue;
  int j=i,requant=-1;while(true){int k=next(j);if(k<0)break;auto&e=expr(k);auto ko=str(e,"operator");if(ko=="cast"){if(str(e,"dtype")=="i8"&&str(params(e),"conversion","saturate")=="saturate"&&outputNames.count(str(e,"id"))&&requant>=0)declinedExpression=requant;break;}if(str(e,"dtype")!="i32")break;
   if(ko=="round_shift_right"){auto&pr=params(e);if(integer(pr,"shift")>0&&str(pr,"rounding","half_up")=="half_up")requant=k;else if(integer(pr,"shift")>0)break;}
   else if(ko!="add_bias"&&ko!="relu")break;
   j=k;
  }
 }
 std::vector<Group> groups; std::vector<bool> assigned(es.size());
 // The external command vocabulary has a narrowing acc_scale stage. Never use
 // that stage for an observable i32 scale or across arithmetic following it.
 for(int i=0;i<(int)es.size();++i) {
  if(assigned[i])continue;
  Group g;g.expressions.push_back(i);auto op=str(expr(i),"operator");g.category=op;
  int j=i;
  if(op=="matmul"||op=="conv2d"||op=="scaled_add"||op=="scale_round"||op=="relu"||op=="cast"||op=="movement"||op=="identity") {
   while(true){if(op=="matmul"&&shape(*defs.at(operands(expr(i)).at(0))).size()>2)break;int k=next(j);if(k<0)break;auto ko=str(expr(k),"operator");
    bool ready=true;for(auto &dep:operands(expr(k))){auto it=producer.find(dep);if(it!=producer.end()&&it->second>=i&&std::find(g.expressions.begin(),g.expressions.end(),it->second)==g.expressions.end())ready=false;}
    if(!ready)break;
    bool allow=isEpilogue(ko);
    if(op=="scaled_add")allow=ko=="relu"||ko=="cast";
    if(op=="scale_round")allow=ko=="relu"||ko=="max_pool"||ko=="cast";
    if(op=="relu")allow=ko=="max_pool"||ko=="cast";
    if(op=="cast"||op=="movement"||op=="identity")allow=ko=="max_pool"||ko=="cast";
    // Accelerator readout currently has uniform positive scaling and relu.
    // Keep unsupported stages in their own source-owned host tasks.
    if(ko=="max_pool"&&!(op=="matmul"||op=="conv2d"||op=="scale_round"||op=="relu"||(op=="cast"&&str(expr(i),"dtype")=="i32"&&str(*defs.at(operands(expr(i)).at(0)),"dtype")=="i8")))allow=false;
    if(ko=="round_shift_right"||(ko=="add_bias"&&op!="conv2d"))allow=false;
    if(ko=="scale_round"){auto &pr=params(expr(k));auto scale=pr.getNumber("factor");if(pr.getArray("factors")||!scale||!std::isfinite(*scale)||*scale<=0)allow=false;}
    if(!allow)break;
    if(ko=="cast" && str(expr(k),"dtype")!="i8" && str(expr(k),"dtype")!=str(expr(j),"dtype"))break;
    g.expressions.push_back(k);j=k;if(ko=="cast"&&str(expr(k),"dtype")=="i8")break;
   }
   // A scale's clamp is safe only if it reaches i8 through monotone relu /
   // pooling. A later bias or shift would observe the premature saturation.
   int scale=-1; bool legal=true; std::set<std::string> stages;
   for(size_t q=1;q<g.expressions.size();++q){auto qo=str(expr(g.expressions[q]),"operator");if(qo!="cast"&&!stages.insert(qo).second)legal=false;if(qo=="scale_round"){if(scale>=0)legal=false;scale=q;}else if(scale>=0 && qo!="relu"&&qo!="cast"&&qo!="max_pool")legal=false;}
   if(scale>=0 && str(expr(g.expressions.back()),"dtype")!="i8")legal=false;
   // A cast is not a no-op if followed by a bias, requant or another scale.
   bool narrow=false;for(size_t q=1;q<g.expressions.size();++q){auto qo=str(expr(g.expressions[q]),"operator");if(narrow&&qo!="relu"&&qo!="cast"&&qo!="max_pool")legal=false;if(qo=="cast"&&str(expr(g.expressions[q]),"dtype")=="i8")narrow=true;}
   if(narrow&&str(expr(g.expressions.back()),"dtype")!="i8")legal=false;
   if(op=="matmul"||op=="conv2d"){int stage=0;for(size_t q=1;q<g.expressions.size();++q){auto qo=str(expr(g.expressions[q]),"operator");int order=qo=="add_bias"?0:qo=="scale_round"?1:qo=="relu"?2:qo=="max_pool"?3:4;if(order<stage)legal=false;stage=order;}}
   if((op=="matmul"||op=="conv2d")&&g.expressions.size()>1&&str(expr(g.expressions.back()),"dtype")!="i8")legal=false;
   bool pooled=false;for(size_t q=1;q<g.expressions.size();++q){auto qo=str(expr(g.expressions[q]),"operator");if(pooled&&qo!="cast")legal=false;if(qo=="max_pool")pooled=true;}
   if(pooled){if(str(expr(g.expressions.back()),"operator")!="cast"||str(expr(g.expressions.back()),"dtype")!="i8")legal=false;for(size_t q=0;q+1<g.expressions.size();++q)if(str(expr(g.expressions[q]),"dtype")!="i32")legal=false;}
   if(op=="scaled_add"&&g.expressions.size()>1&&str(expr(g.expressions.back()),"dtype")!="i8")legal=false;
   if(op=="scale_round"){auto&pr=params(expr(i));if(pooled&&pr.getArray("factors"))legal=false;bool positive=true;if(auto*a=pr.getArray("factors")){for(auto&v:*a){auto f=v.getAsNumber();if(!f||!std::isfinite(*f)||*f<=0)positive=false;}}else{auto f=pr.getNumber("factor");positive=f&&std::isfinite(*f)&&*f>0;}if(!positive||str(expr(g.expressions.back()),"dtype")!="i8")legal=false;}
   if(op=="cast"&&g.expressions.size()>1){bool pool=false;for(auto x:g.expressions)if(str(expr(x),"operator")=="max_pool")pool=true;if(!pool||str(expr(g.expressions.back()),"dtype")!="i8")legal=false;}
   // Preserve the established standalone bias boundary except for complete,
   // proven CONV2D+bias+pool readouts. COMMIT bias retains a separate task;
   // bias in CONV2D is an explicit source stage.
   bool biased=false,activated=false;const Object*poolStage=nullptr;
   for(auto x:g.expressions){auto&e=expr(x);auto eo=str(e,"operator");if(eo=="relu")activated=true;if(eo=="max_pool")poolStage=&e;if(eo=="add_bias"){biased=true;auto operandsBias=operands(e);if(operandsBias.size()!=2)legal=false;else{auto&b=*defs.at(operandsBias[1]);auto bs=shape(b),rs=shape(expr(i));if(str(b,"dtype")!="i32"||bs.size()!=1||rs.empty()||bs[0]!=rs.back())legal=false;}}}
   if(biased){auto inRoot=operands(expr(i));if(op!="conv2d"||!poolStage||!devicePoolGeometry(*poolStage,activated)||str(expr(i),"dtype")!="i32"||inRoot.size()!=2||str(*defs.at(inRoot[0]),"dtype")!="i8"||str(*defs.at(inRoot[1]),"dtype")!="i8")legal=false;}
   if(!legal)g.expressions.resize(1);
  }
  for(auto x:g.expressions)assigned[x]=true;
  groups.push_back(std::move(g));
 }
 // Assign a physical ABI tensor only to each group boundary. Fused source
 // results stay SSA values and have no invented allocation or pointer argument.
 Object tensors,values,convIm2col,poolPrecast,poolPartial,poolDeviceFused;Array abiArgs,abiOutputs,entryBindings,sourceValues,outputBindings,compilerTemporaries;
 std::map<std::string,std::string> convScratch;
 std::map<int,std::vector<std::string>> poolScratch;
 std::map<std::string,std::string> names;
 auto fresh=[&](std::string n){if(!occupied.count(n)){occupied.insert(n);return n;}int k=0;std::string r;do{r=n+"_"+std::to_string(k++);}while(occupied.count(r));occupied.insert(r);return r;};
 auto addTensor=[&](const std::string&name,const Object&o,const std::string&role){if(tensors.get(name))return;tensors[name]=tensor(o,role);abiArgs.push_back(Object{{"tensor",name},{"access",role=="output"||role=="intermediate"?"write":"read"}});};
 std::map<std::string,std::string> inputRole;
 for(auto&ev:es){auto&e=*ev.getAsObject();auto in=operands(e);auto op=str(e,"operator");if((op=="matmul"||op=="conv2d")&&in.size()>1)inputRole[in[1]]="weight";if(op=="add_bias"&&in.size()>1)inputRole[in[1]]="bias";}
 for(auto &v:ins){auto&e=*v.getAsObject();auto id=str(e,"id"),name=str(e,"name",id);names[id]=name;values[id]=name;addTensor(name,e,inputRole.count(id)?inputRole[id]:"input");entryBindings.push_back(name);}
 // Output declaration order is source return order, independent of scheduling.
 for(auto &v:outs){auto&o=*v.getAsObject();auto id=str(o,"value"),name=str(o,"name");addTensor(name,*defs.at(id),"output");abiOutputs.push_back(name);outputBindings.push_back(name);if(!names.count(id)){names[id]=name;values[id]=name;}}
 for(auto&g:groups){int last=g.expressions.back();auto&e=expr(last);auto id=str(e,"id");if(!names.count(id)){auto name=fresh("tmp_"+id);names[id]=name;values[id]=name;addTensor(name,e,"intermediate");}}
 // Validate convolution geometry before deciding storage. Supported integer
 // contractions gather directly by DMA and do not materialize host im2col.
 // Any remaining host im2col allocation has an explicit source-result owner.
 for(auto &g:groups){int rootIndex=g.expressions.front();auto&e=expr(rootIndex);if(str(e,"operator")!="conv2d")continue;auto in=operands(e);auto is=shape(*defs.at(in.at(0))),ws=shape(*defs.at(in.at(1)));auto&pr=params(e);auto&kernel=arr(pr,"kernel");if(is.size()!=4||kernel.size()<2)throw std::runtime_error("invalid convolution geometry");
  auto arrayParam=[&](llvm::StringRef key,std::vector<int64_t>d){if(auto*a=pr.getArray(key)){d.clear();for(auto&v:*a)d.push_back(*v.getAsInteger());}return d;};
  auto stride=arrayParam("stride",{1,1}),padding=arrayParam("padding",{0,0,0,0}),dilation=arrayParam("dilation",{1,1});int64_t kh=*kernel[0].getAsInteger(),kw=*kernel[1].getAsInteger();
  if(stride.size()!=2||padding.size()!=4||dilation.size()!=2||kh<=0||kw<=0||stride[0]<=0||stride[1]<=0||dilation[0]<=0||dilation[1]<=0)throw std::runtime_error("invalid convolution geometry");
  for(auto x:is)if(x<=0)throw std::runtime_error("invalid convolution input shape");for(auto x:ws)if(x<=0)throw std::runtime_error("invalid convolution weight shape");for(auto x:padding)if(x<0)throw std::runtime_error("invalid convolution padding");
  if(str(pr,"input_axes","NHWC")!="NHWC"||str(pr,"layout","nhwc")!="nhwc")throw std::runtime_error("unsupported convolution layout");
  if(!((ws.size()==2&&ws[0]==kh*kw*is[3])||(ws.size()==4&&ws[0]==kh&&ws[1]==kw&&ws[2]==is[3])))throw std::runtime_error("convolution weight dimensions do not match kernel and input channels");
  int64_t oh=(is[1]+padding[0]+padding[2]-dilation[0]*(kh-1)-1)/stride[0]+1,ow=(is[2]+padding[1]+padding[3]-dilation[1]*(kw-1)-1)/stride[1]+1;auto os=shape(e);
  if(oh<=0||ow<=0||!((os.size()==2&&os[0]==is[0]*oh*ow&&os[1]==ws.back())||(os.size()==4&&os[0]==is[0]&&os[1]==oh&&os[2]==ow&&os[3]==ws.back())))throw std::runtime_error("convolution output dimensions do not match geometry");
  // The integer spatial runtime addresses NHWC taps directly, including
  // striding, dilation and zero-padding; im2col is mathematical notation only.
  if(str(*defs.at(in[0]),"dtype")=="i8"&&str(*defs.at(in[1]),"dtype")=="i8"&&str(e,"dtype")=="i32")continue;
  // Dense 1x1 input already is im2col even for a host contraction.
  if(kernel.size()==2&&kh==1&&kw==1&&ws.size()==2&&stride==std::vector<int64_t>{1,1}&&padding==std::vector<int64_t>{0,0,0,0})continue;
  auto id=str(e,"id"),name=fresh("conv_"+id+"_im2col");Object description{{"shape",Array{is[0]*oh*ow,kh*kw*is[3]}},{"dtype",str(*defs.at(in[0]),"dtype")}};addTensor(name,description,"intermediate");convScratch[id]=name;convIm2col[id]=name;
  auto*a=e.getArray("source_op_indices");int64_t source=a&&!a->empty()?*a->back().getAsInteger():rootIndex;compilerTemporaries.push_back(Object{{"tensor",name},{"source_op_index",source},{"source_result_index",integer(e,"source_result_index")},{"purpose","NHWC zero-padded convolution im2col"}});
 }
 // Quantization is monotone, so an unobserved i32 maximum followed by
 // saturating i8 cast may read out to i8 before pooling. The full prepool
 // tensor and the optional wide-window reduction tile are real ABI scratch.
 for(size_t gi=0;gi<groups.size();++gi){auto&g=groups[gi];int rootIndex=g.expressions.front();auto&root=expr(rootIndex);auto rootOp=str(root,"operator"),rootID=str(root,"id");for(auto index:g.expressions){auto&pool=expr(index);if(str(pool,"operator")!="max_pool")continue;auto inputs=operands(pool);if(inputs.empty())continue;bool fusedNarrow=(rootOp=="matmul"||rootOp=="conv2d"||rootOp=="scale_round"||rootOp=="relu")&&g.expressions.size()>1&&str(expr(g.expressions.back()),"dtype")=="i8";if(fusedNarrow){auto rootInputs=operands(root);if(rootOp=="matmul"||rootOp=="conv2d")fusedNarrow=rootInputs.size()>=2&&str(*defs.at(rootInputs[0]),"dtype")=="i8"&&str(*defs.at(rootInputs[1]),"dtype")=="i8";else fusedNarrow=!rootInputs.empty()&&str(*defs.at(rootInputs[0]),"dtype")=="i32";}bool directI8=rootOp=="max_pool"&&str(*defs.at(inputs[0]),"dtype")=="i8"&&str(pool,"dtype")=="i8";bool movement=rootOp=="cast"&&str(*defs.at(operands(root).at(0)),"dtype")=="i8"&&str(expr(g.expressions.back()),"dtype")=="i8";if(!fusedNarrow&&!directI8&&!movement)continue;
   auto*a=pool.getArray("source_op_indices");int64_t source=a&&!a->empty()?*a->back().getAsInteger():index;
   auto temporary=[&](const std::string&stem,Array shape,const std::string&purpose){auto name=fresh(stem);Object description{{"shape",std::move(shape)},{"dtype","i8"}};addTensor(name,description,"intermediate");compilerTemporaries.push_back(Object{{"tensor",name},{"source_op_index",source},{"source_result_index",integer(pool,"source_result_index")},{"purpose",purpose}});poolScratch[gi].push_back(name);return name;};
   // Fused contraction and i32 readout runtimes reduce each spatial window
   // in accelerator storage, with no full prepool tensor crossing the ABI.
   bool deviceFused=fusedNarrow&&(rootOp=="matmul"||rootOp=="conv2d"||rootOp=="scale_round"||rootOp=="relu")&&str(root,"dtype")=="i32";
   if(deviceFused){bool relu=false;for(auto x:g.expressions)if(str(expr(x),"operator")=="relu")relu=true;deviceFused=devicePoolGeometry(pool,relu);}
   if(deviceFused)poolDeviceFused[rootID]=true;
   else if(fusedNarrow)poolPrecast[rootID]=temporary("pool_"+rootID+"_precast",Array(arr(*defs.at(inputs[0]),"shape")),"Monotone saturating i8 readout before unobserved spatial maximum");
   auto&window=arr(params(pool),"window");bool partial=false;for(auto&v:window)if(*v.getAsInteger()>3)partial=true;
   if(partial&&!deviceFused)poolPartial[str(pool,"id")]=temporary("pool_"+str(pool,"id")+"_partial",Array{16},"Device spatial maximum partial reduction tile");
  }}
 // Source result bindings exist precisely for declared materialized buffers.
 for(auto&g:groups){auto&e=expr(g.expressions.back());auto id=str(e,"id");auto *indices=e.getArray("source_op_indices");int64_t source=indices&&!indices->empty()?*indices->back().getAsInteger():g.expressions.back();
  if(outputNames.count(id)){for(auto&name:outputNames[id])sourceValues.push_back(Object{{"op_index",source},{"result_index",integer(e,"source_result_index")},{"tensor",name}});}else sourceValues.push_back(Object{{"op_index",source},{"result_index",integer(e,"source_result_index")},{"tensor",names.at(id)}});
 }
 auto named=[&](const std::string&id)->std::string{auto it=names.find(id);if(it==names.end())throw std::runtime_error("unmaterialized value crosses a fusion boundary: "+id);return it->second;};
 Array commands,groupMetadata,tasks,lanePlacement;
 auto command=[&](const std::string&opcode,Object operands,Object attributes){commands.push_back(Object{{"opcode",opcode},{"operands",std::move(operands)},{"attributes",std::move(attributes)}});};
 for(size_t gi=0;gi<groups.size();++gi){auto&g=groups[gi];auto&root=expr(g.expressions.front());auto &last=expr(g.expressions.back());auto in=operands(root);auto op=str(root,"operator");auto dst=named(str(last,"id"));size_t commandStart=commands.size();Object attrs{{"output_dtype",str(last,"dtype")}};Array epilogue;std::string bias;
  auto poolAttrs=[&](const Object&e){auto&pr=params(e);copyField(attrs,"pool_in_dims",pr,"spatial_shape");copyField(attrs,"pool_size",pr,"window");copyField(attrs,"pool_stride",pr,"stride");copyField(attrs,"pool_padding",pr,"padding");auto*v=pr.get("padding_value");if(v && v->kind()!=Value::Null)attrs["pool_pad_value"]=Value(*v);};
  for(size_t q=1;q<g.expressions.size();++q){auto&e=expr(g.expressions[q]);auto eo=str(e,"operator");auto&pr=params(e);
   if(eo=="add_bias"){epilogue.push_back("bias_add");bias=named(operands(e).at(1));}
   else if(eo=="scale_round"){epilogue.push_back("acc_scale");copyField(attrs,"acc_scale",pr,"factor");copyField(attrs,"acc_scale_by_column",pr,"factors");}
   else if(eo=="round_shift_right"){epilogue.push_back("requant");copyField(attrs,"requant_shift",pr,"shift");}
   else if(eo=="relu")epilogue.push_back("relu");
   else if(eo=="max_pool"){epilogue.push_back("maxpool");poolAttrs(e);}
  }
  attrs["epilogue"]=std::move(epilogue);
  bool represented=true;
  if(op=="matmul"){
   auto sh=shape(*defs.at(in.at(0)));
   if(sh.size()>2 && g.expressions.size()==1)command("BATCHED_MATMUL",Object{{"a",named(in[0])},{"w",named(in[1])},{"dst",dst}},Object{});
   else {auto handle="acc_"+std::to_string(gi);command("MATMUL",Object{{"lhs",named(in.at(0))},{"rhs",named(in.at(1))},{"dst",handle}},Object{});Object operands{{"src",handle},{"dst",dst}};if(!bias.empty())operands["bias"]=bias;command("COMMIT",std::move(operands),std::move(attrs));}
  }else if(op=="conv2d"){
   auto&pr=params(root);auto ifmShape=shape(*defs.at(in.at(0))),wShape=shape(*defs.at(in.at(1)));auto kernel=arr(pr,"kernel");
   if(kernel.size()==2){kernel.push_back(ifmShape.back());kernel.push_back(wShape.back());}
   attrs["kernel"]=std::move(kernel);attrs["layout"]="nhwc";copyField(attrs,"stride",pr,"stride");copyField(attrs,"padding",pr,"padding");copyField(attrs,"dilation",pr,"dilation");if(!bias.empty())attrs["bias"]=bias;
   command("CONV2D",Object{{"ifm",named(in.at(0))},{"weight",named(in.at(1))},{"dst",dst}},std::move(attrs));
  }else if(op=="scaled_add"){
   auto&pr=params(root);copyField(attrs,"lhs_scale",pr,"lhs_factor");copyField(attrs,"rhs_scale",pr,"rhs_factor");copyField(attrs,"bound_lsb",pr,"allowed_error_lsb");command("RESIDUAL_ADD",Object{{"lhs",named(in.at(0))},{"rhs",named(in.at(1))},{"dst",dst}},std::move(attrs));
  }else if(op=="add_bias" && g.expressions.size()==1){attrs.erase("epilogue");command("BIAS_ADD",Object{{"src",named(in.at(0))},{"bias",named(in.at(1))},{"dst",dst}},std::move(attrs));
  }else if((op=="cast"||op=="movement"||op=="identity") && !in.empty()){
   bool hasPool=false;for(auto x:g.expressions)if(str(expr(x),"operator")=="max_pool")hasPool=true;
   if(hasPool && str(last,"dtype")=="i8" && str(*defs.at(in[0]),"dtype")=="i8"){
    if(!attrs.get("acc_scale")&&!attrs.get("acc_scale_by_column"))attrs["acc_scale"]=1.0;
    attrs["epilogue"]=Array{"acc_scale","maxpool"};command("MAXPOOL2D",Object{{"src",named(in[0])},{"dst",dst}},std::move(attrs));
   }else if(g.expressions.size()==1 && (str(root,"dtype")==str(*defs.at(in[0]),"dtype") || (str(*defs.at(in[0]),"dtype")=="i8"&&str(root,"dtype")=="i32"))){attrs["semantic"]="mvin_mvout";command("MOVEMENT",Object{{"src",named(in[0])},{"dst",dst}},std::move(attrs));}
   else represented=false;
  }else if(op=="max_pool" && in.size()==1 && str(*defs.at(in[0]),"dtype")=="i8" && str(last,"dtype")=="i8"){
   poolAttrs(root);attrs["acc_scale"]=1.0;attrs["epilogue"]=Array{"acc_scale","maxpool"};command("MAXPOOL2D",Object{{"src",named(in[0])},{"dst",dst}},std::move(attrs));
  }else represented=false;
  // Unsupported general host expressions remain explicit source-owned tasks.
  // A whole-program ABI represents them without fabricating accelerator work.
  std::vector<std::string> reads,writes,regionIDs;std::set<std::string> local;
  for(auto x:g.expressions)local.insert(str(expr(x),"id"));
  Array indices,expressionIndices;
  std::set<int64_t> seenIndices;
  for(auto x:g.expressions){auto&e=expr(x);expressionIndices.push_back(x);for(auto&v:operands(e))if(!local.count(v))uniquePush(reads,named(v));auto rid=str(e,"region_id");if(!rid.empty())uniquePush(regionIDs,rid);if(auto*a=e.getArray("source_op_indices")){for(auto&v:*a){auto k=*v.getAsInteger();if(seenIndices.insert(k).second)indices.push_back(k);}}else if(seenIndices.insert(x).second)indices.push_back(x);}
  auto lastID=str(last,"id");if(outputNames.count(lastID))for(auto&name:outputNames[lastID])uniquePush(writes,name);else writes.push_back(dst);
  bool device=represented;
  if(op=="matmul"||op=="conv2d")device=device&&in.size()>=2&&str(*defs.at(in[0]),"dtype")=="i8"&&str(*defs.at(in[1]),"dtype")=="i8";
  else if(op=="add_bias")device=device&&str(root,"dtype")=="i32";
  else if(op=="scaled_add"){
   auto&pr=params(root);auto a=pr.getNumber("lhs_factor"),b=pr.getNumber("rhs_factor");auto bound=integer(pr,"allowed_error_lsb",-1);bool integral=a&&b&&std::floor(*a)==*a&&std::floor(*b)==*b;
   device=a&&b&&in.size()==2&&str(*defs.at(in[0]),"dtype")=="i8"&&str(*defs.at(in[1]),"dtype")=="i8"&&std::isfinite(*a)&&std::isfinite(*b)&&*a>=0&&*b>=0&&*a<=16&&*b<=16&&(integral||(*a<=1&&*b<=1&&bound>=1)||bound>=2);
  }
  else if(op=="scale_round"||op=="relu")device=in.size()==1&&str(*defs.at(in[0]),"dtype")=="i32"&&str(last,"dtype")=="i8"&&g.expressions.size()>1;
  else if(op=="cast" && in.size()==1 && str(*defs.at(in[0]),"dtype")=="i32" && str(root,"dtype")=="i8" && g.expressions.size()==1)device=true;
  if(!device){while(commands.size()>commandStart)commands.pop_back();}
  if(convScratch.count(str(root,"id")))uniquePush(writes,convScratch.at(str(root,"id")));
  if(poolScratch.count(gi))for(auto&name:poolScratch.at(gi))uniquePush(writes,name);
  std::string kind=device?"device":"host";
  Object placement{{"task_index",(int64_t)gi},{"lane",kind},{"source_op_indices",Array(indices)}};
  if(!device)placement["reason"]="No implemented accelerator lowering for normalized "+op+" with these types or parameters";
  lanePlacement.push_back(std::move(placement));
  Object task{{"task_index",(int64_t)gi},{"kind",kind},{"source_op_indices",Array(indices)},{"instruction_start",(int64_t)gi},{"instruction_end",(int64_t)gi+1},{"reads",strings(reads)},{"writes",strings(writes)}};
  if(!regionIDs.empty())task["source_region_ids"]=strings(regionIDs);
  tasks.push_back(std::move(task));
  Object metadata{{"expression_indices",std::move(expressionIndices)},{"command_start",(int64_t)commandStart},{"command_end",(int64_t)commands.size()},{"reads",strings(reads)},{"writes",strings(writes)},{"source_op_indices",std::move(indices)},{"kind",kind},{"operator",op}};
  if(!regionIDs.empty())metadata["source_region_ids"]=strings(regionIDs);
  groupMetadata.push_back(std::move(metadata));
 }
 Object plan{{"schema","mixed_program_plan_v1"},{"source_sha256",sha},{"source_op_count",integer(p,"source_op_count",es.size())},{"tasks",std::move(tasks)},{"schedule_instruction_count",(int64_t)groups.size()},{"prologue_instruction_range",Array{0,0}},{"epilogue_instruction_range",Array{(int64_t)groups.size(),(int64_t)groups.size()}},{"entry_bindings",std::move(entryBindings)},{"source_values",std::move(sourceValues)},{"output_bindings",std::move(outputBindings)},{"compiler_temporaries",std::move(compilerTemporaries)}};
 Object aliases;for(auto&it:outputNames)aliases[it.first]=strings(it.second);
 Object cb{{"abi_version","0.1"},{"target","gemmini"},{"backend","external_backend"},{"tensors",std::move(tensors)},{"commands",std::move(commands)},{"kernel_abi",Object{{"kind","whole_program"},{"args",std::move(abiArgs)},{"outputs",std::move(abiOutputs)}}},{"params",Object{{"global_program_plan",std::move(plan)},{"lane_placement",std::move(lanePlacement)}}},{"_compiler",Object{{"values",std::move(values)},{"groups",std::move(groupMetadata)},{"output_aliases",std::move(aliases)},{"conv_im2col",std::move(convIm2col)},{"pool_precast",std::move(poolPrecast)},{"pool_partial",std::move(poolPartial)},{"pool_device_fused",std::move(poolDeviceFused)}}}};
 Array pointerArgs;for(auto&v:*cb.getObject("kernel_abi")->getArray("args"))pointerArgs.push_back(str(*v.getAsObject(),"tensor"));cb["compiler_pointer_abi"]=Object{{"version",1},{"arguments",std::move(pointerArgs)}};
 if(declinedExpression>=0){cb["commands"]=Array{};cb["declined"]=Object{{"op","requant"},{"reason","Contraction i8 readout requires positive round-half-up integer requant; this target readout applies ties-to-even acc_scale, relu and maxpool, and has no route applying the required requant stage"}};}
 return cb;
}
}
