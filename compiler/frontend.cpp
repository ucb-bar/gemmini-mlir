#include "compiler.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/DenseMap.h"
#include <set>
#include <functional>
#include <regex>
namespace backend { namespace {
std::string dtype(mlir::Type t){if(auto s=mlir::dyn_cast<mlir::ShapedType>(t)) t=s.getElementType();std::string r;llvm::raw_string_ostream os(r);t.print(os);return r;}
Array dims(mlir::Type t){Array a;if(auto s=mlir::dyn_cast<mlir::RankedTensorType>(t)){if(!s.hasStaticShape())throw std::runtime_error("dynamic tensors are unsupported");for(auto d:s.getShape()){if(d<=0)throw std::runtime_error("nonpositive tensor dimension");a.push_back(d);}}return a;}
Value attr(mlir::Attribute a){if(auto i=mlir::dyn_cast<mlir::IntegerAttr>(a))return i.getInt();if(auto f=mlir::dyn_cast<mlir::FloatAttr>(a))return f.getValueAsDouble();if(auto s=mlir::dyn_cast<mlir::StringAttr>(a))return s.str();if(auto d=mlir::dyn_cast<mlir::DenseIntElementsAttr>(a)){Array r;for(auto v:d.getValues<llvm::APInt>())r.push_back(v.getSExtValue());return r;}if(auto d=mlir::dyn_cast<mlir::DenseFPElementsAttr>(a)){Array r;for(auto v:d.getValues<llvm::APFloat>())r.push_back(v.convertToDouble());return r;}if(auto d=mlir::dyn_cast<mlir::DenseI64ArrayAttr>(a)){Array r;for(auto v:d.asArrayRef())r.push_back(v);return r;}if(auto d=mlir::dyn_cast<mlir::ArrayAttr>(a)){Array r;for(auto v:d)r.push_back(attr(v));return r;}throw std::runtime_error("unsupported constant/attribute type");}
Value affine(mlir::AffineExpr e){if(auto d=mlir::dyn_cast<mlir::AffineDimExpr>(e))return int64_t(d.getPosition());if(auto c=mlir::dyn_cast<mlir::AffineConstantExpr>(e))return Object{{"constant",c.getValue()}};auto b=mlir::dyn_cast<mlir::AffineBinaryOpExpr>(e);if(!b)throw std::runtime_error("affine symbols unsupported");std::string op;switch(e.getKind()){case mlir::AffineExprKind::Add:op="add";break;case mlir::AffineExprKind::Mul:op="mul";break;case mlir::AffineExprKind::Mod:op="mod";break;case mlir::AffineExprKind::FloorDiv:op="floordiv";break;case mlir::AffineExprKind::CeilDiv:op="ceildiv";break;default:throw std::runtime_error("unsupported affine expression");}return Object{{"op",op},{"lhs",affine(b.getLHS())},{"rhs",affine(b.getRHS())}};}
std::string printOp(mlir::Operation*op){std::string r;llvm::raw_string_ostream os(r);op->print(os);return r;}
Object parseMLIR(const std::string &source){
 mlir::DialectRegistry registry;registry.insert<mlir::func::FuncDialect,mlir::linalg::LinalgDialect,mlir::tensor::TensorDialect,mlir::arith::ArithDialect,mlir::math::MathDialect>();mlir::MLIRContext context(registry);context.allowUnregisteredDialects();// Older mathematical exporters place the yield attribute dictionary before
 // its operands; LLVM 23 accepts the equivalent dictionary after the operands.
 // This is an assembly spelling migration only, with no operation insertion.
 std::string assembly=std::regex_replace(source,std::regex(R"(linalg\.yield (\{[^\n]*?\}) (%[^\s]+) : ([^\n]+))"),"linalg.yield $2 $1 : $3");
 auto module=mlir::parseSourceString<mlir::ModuleOp>(assembly,&context);if(!module)throw std::runtime_error("MLIR parser rejected source");if(mlir::failed(mlir::verify(*module)))throw std::runtime_error("MLIR verifier rejected source");
 mlir::func::FuncOp fn;for(auto f:module->getOps<mlir::func::FuncOp>())if(!f.isDeclaration()){if(fn)throw std::runtime_error("expected one defined function");fn=f;}if(!fn||!llvm::hasSingleElement(fn.getBody()))throw std::runtime_error("expected one single-block function");
 llvm::DenseMap<mlir::Value,std::string> ids;std::map<std::string,mlir::Type> valueTypes;llvm::DenseMap<mlir::Value,mlir::Attribute> constants;llvm::DenseMap<mlir::Operation*,int64_t> sourceIndices;std::set<int64_t> foldedInitializers;Array inputs,exprs,outputs;auto inNames=fn->getAttrOfType<mlir::ArrayAttr>("merlin.logical_inputs"),outNames=fn->getAttrOfType<mlir::ArrayAttr>("merlin.logical_outputs");
 for(auto a:fn.getArguments()){unsigned i=a.getArgNumber();std::string id="i"+std::to_string(i),name=id;if(inNames&&i<inNames.size())name=mlir::cast<mlir::StringAttr>(inNames[i]).str();ids[a]=id;valueTypes[id]=a.getType();inputs.push_back(Object{{"id",id},{"name",name},{"shape",dims(a.getType())},{"dtype",dtype(a.getType())}});}
 int64_t index=0;for(auto &operation:fn.front()){auto*op=&operation;auto name=op->getName().getStringRef();if(name=="func.return"){unsigned n=0;for(auto v:op->getOperands()){std::string out="output"+std::to_string(n);if(outNames&&n<outNames.size())out=mlir::cast<mlir::StringAttr>(outNames[n]).str();outputs.push_back(Object{{"name",out},{"value",ids.lookup(v)},{"shape",dims(v.getType())},{"dtype",dtype(v.getType())}});++n;}continue;}
 sourceIndices[op]=index;if(op->getNumResults()!=1)throw std::runtime_error("unsupported direct operation result count: "+name.str());auto result=op->getResult(0);std::string id="v"+std::to_string(index);ids[result]=id;valueTypes[id]=result.getType();Object e{{"id",id},{"shape",dims(result.getType())},{"dtype",dtype(result.getType())},{"source_op_indices",Array{index}},{"source_result_index",0}};if(auto r=op->getAttrOfType<mlir::StringAttr>("prov.region_id"))e["region_id"]=r.str();if(auto r=op->getAttrOfType<mlir::StringAttr>("prov.placement"))e["placement"]=r.str();Array operands;for(auto v:op->getOperands()){if(!ids.count(v))throw std::runtime_error("unbound MLIR operand");operands.push_back(ids.lookup(v));}Object params;std::string kind;
 if(name=="arith.constant"){kind="constant";auto value=op->getAttr("value");params["value"]=attr(value);constants[result]=value;}
 else if(name=="tensor.empty"){kind="empty";operands=Array{};}
 else if(name=="tensor.splat"||name=="linalg.fill"){kind="fill";operands=Array{ids.lookup(op->getOperand(0))};if(auto c=constants.lookup(op->getOperand(0)))constants[result]=c;}
 else if(name=="tensor.collapse_shape"||name=="tensor.expand_shape"||name=="tensor.cast"){kind="reshape";operands=Array{ids.lookup(op->getOperand(0))};}
 else if(name=="linalg.transpose"){kind="transpose";operands=Array{ids.lookup(op->getOperand(0))};params["permutation"]=attr(op->getAttr("permutation"));}
 else if(auto lin=mlir::dyn_cast<mlir::linalg::LinalgOp>(op)){
 auto maps=lin.getIndexingMapsArray();bool contraction=lin.getNumDpsInputs()==2&&lin.getNumDpsInits()==1&&lin.getNumReductionLoops()==1&&result.getType().isIntOrIndexOrFloat()==false;bool zero=false;if(op->getNumOperands()>2){auto c=constants.lookup(op->getOperand(2));if(auto i=mlir::dyn_cast_or_null<mlir::IntegerAttr>(c))zero=i.getInt()==0;}
 // A matmul requires the canonical indexing maps and the multiply-add body.
 unsigned rank=maps.empty()?0:maps[0].getNumResults();
 bool mm=contraction&&maps.size()==3&&rank>=2&&maps[0].getNumDims()==rank+1&&maps[1].getNumResults()==rank&&maps[2].getNumResults()==rank;
 if(mm){
  auto d=[&](unsigned p){return mlir::getAffineDimExpr(p,&context);};
  // Canonical row-major [batch..., M, K] x [batch..., K, N].
  // This recognizes both named matmul/batch_matmul and equivalent generics.
  // Prefixes must agree exactly: no implicit broadcast or transpose is fused.
  for(unsigned b=0;b+2<rank;++b)mm=mm&&maps[0].getResult(b)==d(b)&&maps[1].getResult(b)==d(b)&&maps[2].getResult(b)==d(b);
  unsigned m=rank-2,n=rank-1,k=rank;
  mm=mm&&maps[0].getResult(m)==d(m)&&maps[0].getResult(n)==d(k)&&maps[1].getResult(m)==d(k)&&maps[1].getResult(n)==d(n)&&maps[2].getResult(m)==d(m)&&maps[2].getResult(n)==d(n);
  auto iterators=lin.getIteratorTypesArray();for(unsigned axis=0;axis<iterators.size();++axis)mm=mm&&iterators[axis]==(axis==k?mlir::utils::IteratorType::reduction:mlir::utils::IteratorType::parallel);
  bool mul=false,add=false;for(auto&b:op->getRegion(0).front()){mul|=b.getName().getStringRef()=="arith.muli";add|=b.getName().getStringRef()=="arith.addi";}mm=mm&&mul&&add&&dtype(result.getType())=="i32"&&dtype(op->getOperand(0).getType())=="i8"&&dtype(op->getOperand(1).getType())=="i8";
 if(mm){auto &bb=op->getRegion(0).front();auto *sum=bb.getTerminator()->getOperand(0).getDefiningOp();mm=sum&&sum->getName().getStringRef()=="arith.addi";mlir::Value product;if(mm){if(sum->getOperand(0)==bb.getArgument(2))product=sum->getOperand(1);else if(sum->getOperand(1)==bb.getArgument(2))product=sum->getOperand(0);else mm=false;}auto *pr=product?product.getDefiningOp():nullptr;mm=mm&&pr&&pr->getName().getStringRef()=="arith.muli";if(mm){auto *lhs=pr->getOperand(0).getDefiningOp(),*rhs=pr->getOperand(1).getDefiningOp();mm=lhs&&rhs&&lhs->getName().getStringRef()=="arith.extsi"&&rhs->getName().getStringRef()=="arith.extsi"&&((lhs->getOperand(0)==bb.getArgument(0)&&rhs->getOperand(0)==bb.getArgument(1))||(lhs->getOperand(0)==bb.getArgument(1)&&rhs->getOperand(0)==bb.getArgument(0)));}}
}
 if(mm&&zero){kind="matmul";operands=Array{ids.lookup(op->getOperand(0)),ids.lookup(op->getOperand(1))};
 // The zero accumulator is implemented by the contraction itself. Fold only
 // source initializer nodes whose original SSA result is exclusively owned by
 // this initializer chain; shared values and observable returns remain intact.
 // Region-less nodes, or nodes in this contraction's existing region, can be
 // covered by its task without losing source-region provenance.
 std::set<int64_t> owned;
 auto rootRegion=op->getAttrOfType<mlir::StringAttr>("prov.region_id");
 std::function<void(mlir::Value,mlir::Operation*)> collectInitializer;
 collectInitializer=[&](mlir::Value value,mlir::Operation*consumer){
  auto *def=value.getDefiningOp();
  if(!def||def->getBlock()!=op->getBlock()||def->getNumResults()!=1||!value.hasOneUse()||value.use_begin()->getOwner()!=consumer)return;
  auto n=def->getName().getStringRef();
  if(n!="arith.constant"&&n!="tensor.empty"&&n!="tensor.splat"&&n!="linalg.fill")return;
  auto region=def->getAttrOfType<mlir::StringAttr>("prov.region_id");
  if(region&&(!rootRegion||region.getValue()!=rootRegion.getValue()))return;
  auto found=sourceIndices.find(def);if(found==sourceIndices.end()||found->second>=index)return;
  owned.insert(found->second);
  for(auto operand:def->getOperands())collectInitializer(operand,def);
 };
 collectInitializer(op->getOperand(2),op);
 Array coverage;for(auto sourceIndex:owned){coverage.push_back(sourceIndex);foldedInitializers.insert(sourceIndex);}coverage.push_back(index);e["source_op_indices"]=std::move(coverage);
}
 else {
 kind="generic";Array mapJSON,loopShape;std::vector<int64_t> bounds(lin.getNumLoops(),-1);for(auto [operand,map]:llvm::zip(op->getOperands(),maps)){Array a;auto type=mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());for(auto [j,x]:llvm::enumerate(map.getResults())){a.push_back(affine(x));if(auto d=mlir::dyn_cast<mlir::AffineDimExpr>(x)){int64_t b=type.getDimSize(j);auto&p=bounds[d.getPosition()];if(p!=-1&&p!=b)throw std::runtime_error("inconsistent loop bounds");p=b;}}mapJSON.push_back(std::move(a));}for(auto b:bounds){if(b<=0)throw std::runtime_error("cannot infer static generic loop bounds");loopShape.push_back(b);}params["loop_shape"]=std::move(loopShape);params["indexing_maps"]=std::move(mapJSON);params["input_count"]=int64_t(lin.getNumDpsInputs());params["destination_index"]=int64_t(lin.getNumDpsInputs());
 auto&body=op->getRegion(0).front();llvm::DenseMap<mlir::Value,std::string> scalarIds;Array args,instructions;for(auto a:body.getArguments()){std::string s="a"+std::to_string(a.getArgNumber());scalarIds[a]=s;args.push_back(s);}unsigned si=0;std::function<std::string(mlir::Value)> scalarId=[&](mlir::Value v){auto found=scalarIds.find(v);if(found!=scalarIds.end())return found->second;auto c=constants.lookup(v);if(!c||mlir::isa<mlir::ShapedType>(v.getType())){auto external=ids.find(v);if(external==ids.end())throw std::runtime_error("generic captures value outside the source function");std::string externalId=external->second;bool present=false;for(auto &operand:operands)if(operand.getAsString()==externalId)present=true;if(!present)operands.push_back(externalId);scalarIds[v]=externalId;return externalId;}std::string s="s"+std::to_string(si++);scalarIds[v]=s;instructions.push_back(Object{{"id",s},{"op","arith.constant"},{"operands",Array{}},{"dtype",dtype(v.getType())},{"value",attr(c)}});return s;};std::string yield;
 for(auto&b:body){auto bn=b.getName().getStringRef();if(bn=="linalg.yield"){if(b.getNumOperands()!=1)throw std::runtime_error("multiple generic yields unsupported");yield=scalarId(b.getOperand(0));continue;}if(b.getNumResults()!=1||b.getNumRegions())throw std::runtime_error("unsupported scalar region operation");Array ss;for(auto v:b.getOperands())ss.push_back(scalarId(v));std::string s="s"+std::to_string(si++);scalarIds[b.getResult(0)]=s;Object instruction{{"id",s},{"op",bn.str()},{"operands",std::move(ss)},{"dtype",dtype(b.getResult(0).getType())}};if(auto v=b.getAttr("value"))instruction["value"]=attr(v);if(auto index=mlir::dyn_cast<mlir::linalg::IndexOp>(&b))instruction["dimension"]=int64_t(index.getDim());if(auto cmp=mlir::dyn_cast<mlir::arith::CmpIOp>(&b))instruction["predicate"]=mlir::arith::stringifyCmpIPredicate(cmp.getPredicate()).str();else if(auto cmp=mlir::dyn_cast<mlir::arith::CmpFOp>(&b))instruction["predicate"]=mlir::arith::stringifyCmpFPredicate(cmp.getPredicate()).str();instructions.push_back(std::move(instruction));}params["scalar_program"]=Object{{"args",std::move(args)},{"instructions",std::move(instructions)},{"yield",yield}};
 // Recognize accelerator epilogues by their scalar SSA semantics and maps,
 // never by optional provenance labels.
 auto *term=body.getTerminator();auto y=term->getOperand(0);
 auto isArg=[&](mlir::Value v,unsigned i){return i<body.getNumArguments()&&v==body.getArgument(i);};
 auto constantInt=[&](mlir::Value v,int64_t value){auto c=constants.lookup(v);if(!c)if(auto *d=v.getDefiningOp())if(d->getName().getStringRef()=="arith.constant")c=d->getAttr("value");auto a=mlir::dyn_cast_or_null<mlir::IntegerAttr>(c);return a&&a.getInt()==value;};
 auto isOp=[](mlir::Value v,llvm::StringRef n)->mlir::Operation*{auto *d=v.getDefiningOp();return d&&d->getName().getStringRef()==n?d:nullptr;};
 bool identity=maps.size()>=2&&maps.front().isIdentity()&&maps.back().isIdentity()&&lin.getNumReductionLoops()==0;
 if(identity&&lin.getNumDpsInputs()==1){
 if(auto *tr=isOp(y,"arith.trunci")){if(dtype(result.getType())=="i8")if(auto *hi=isOp(tr->getOperand(0),"arith.minsi")){mlir::Value low;if(constantInt(hi->getOperand(1),127))low=hi->getOperand(0);else if(constantInt(hi->getOperand(0),127))low=hi->getOperand(1);if(low)if(auto *lo=isOp(low,"arith.maxsi"))if((isArg(lo->getOperand(0),0)&&constantInt(lo->getOperand(1),-128))||(isArg(lo->getOperand(1),0)&&constantInt(lo->getOperand(0),-128))){kind="cast";operands=Array{ids.lookup(op->getOperand(0))};params["conversion"]="saturate";}}}
 if(auto *r=isOp(y,"arith.maxsi"))if((isArg(r->getOperand(0),0)&&constantInt(r->getOperand(1),0))||(isArg(r->getOperand(1),0)&&constantInt(r->getOperand(0),0))){kind="relu";operands=Array{ids.lookup(op->getOperand(0))};}
 if(auto *r=isOp(y,"arith.xori"))if((isArg(r->getOperand(0),0)&&constantInt(r->getOperand(1),1))||(isArg(r->getOperand(1),0)&&constantInt(r->getOperand(0),1))){kind="xor_low_bit";operands=Array{ids.lookup(op->getOperand(0))};}
 }
 if(identity&&lin.getNumDpsInputs()==2&&maps[1].getNumResults()==1&&maps[1].getResult(0)==mlir::getAffineDimExpr(maps[0].getNumDims()-1,&context))if(auto *sum=isOp(y,"arith.addi"))if((isArg(sum->getOperand(0),0)&&isArg(sum->getOperand(1),1))||(isArg(sum->getOperand(0),1)&&isArg(sum->getOperand(1),0))){kind="add_bias";operands=Array{ids.lookup(op->getOperand(0)),ids.lookup(op->getOperand(1))};}

 }
 }
 else if(name=="tensor.concat"){kind="concat";params["dimension"]=attr(op->getAttr("dim"));}
 else if(name=="quant_ext.quantize_per_tensor"||name=="quant_ext.dequantize_per_tensor"){kind=name=="quant_ext.quantize_per_tensor"?"quantize_per_tensor":"dequantize_per_tensor";for(auto key:{"quant_min","quant_max"})if(auto a=op->getAttr(key))params[key]=attr(a);}
 else if(name=="tensor.extract_slice"||name=="tensor.insert_slice"){
  kind=name=="tensor.extract_slice"?"extract_slice":"insert_slice";
  for(auto key:{"static_offsets","static_sizes","static_strides"})params[key]=attr(op->getAttr(key));
  if(name=="tensor.extract_slice"&&op->getNumOperands()==1){
   auto vector=[&](const Object &p,llvm::StringRef key){std::vector<int64_t> v;for(auto &x:arr(p,key))v.push_back(*x.getAsInteger());return v;};
   auto resultType=mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
   auto originalSource=op->getOperand(0);auto *previousOp=originalSource.getDefiningOp();
   // A private source view can disappear only when its original SSA value
   // has exactly this use. Returns and shared views retain their own writes.
   if(previousOp&&previousOp->getName().getStringRef()=="tensor.extract_slice"&&previousOp->getBlock()==op->getBlock()&&previousOp->getNumOperands()==1&&originalSource.hasOneUse()&&originalSource.use_begin()->getOwner()==op){
    auto previousIndex=sourceIndices.find(previousOp);
    auto previousRegion=previousOp->getAttrOfType<mlir::StringAttr>("prov.region_id");
    auto rootRegion=op->getAttrOfType<mlir::StringAttr>("prov.region_id");
    bool sameRegion=!previousRegion||(rootRegion&&previousRegion.getValue()==rootRegion.getValue());
    auto previousPlacement=previousOp->getAttrOfType<mlir::StringAttr>("prov.placement");
    auto rootPlacement=op->getAttrOfType<mlir::StringAttr>("prov.placement");
    bool samePlacement=!previousPlacement||(rootPlacement&&previousPlacement.getValue()==rootPlacement.getValue());
    if(sameRegion&&samePlacement&&previousIndex!=sourceIndices.end()&&previousIndex->second<index){
     auto &previous=*exprs[previousIndex->second].getAsObject();
     auto &previousParameters=*previous.getObject("parameters");
     auto &previousOperands=arr(previous,"operands");
     if(previousOperands.size()==1){
      auto baseId=previousOperands[0].getAsString()->str();
      auto baseType=mlir::dyn_cast<mlir::RankedTensorType>(valueTypes.at(baseId));
      auto middleType=mlir::dyn_cast<mlir::RankedTensorType>(originalSource.getType());
      auto firstOffsets=vector(previousParameters,"static_offsets"),firstSizes=vector(previousParameters,"static_sizes"),firstStrides=vector(previousParameters,"static_strides");
      auto secondOffsets=vector(params,"static_offsets"),secondSizes=vector(params,"static_sizes"),secondStrides=vector(params,"static_strides");
      bool valid=baseType&&middleType&&resultType&&baseType.hasStaticShape()&&middleType.hasStaticShape()&&resultType.hasStaticShape()&&baseType.getRank()==middleType.getRank()&&baseType.getRank()==resultType.getRank()&&baseType.getElementType()==resultType.getElementType()&&middleType.getElementType()==resultType.getElementType();
      unsigned rank=valid?baseType.getRank():0;
      valid=valid&&firstOffsets.size()==rank&&firstSizes.size()==rank&&firstStrides.size()==rank&&secondOffsets.size()==rank&&secondSizes.size()==rank&&secondStrides.size()==rank;
      auto inBounds=[](int64_t offset,int64_t size,int64_t stride,int64_t extent){int64_t distance,last;return offset>=0&&size>0&&stride>0&&extent>0&&!__builtin_mul_overflow(size-1,stride,&distance)&&!__builtin_add_overflow(offset,distance,&last)&&last<extent;};
      Array composedOffsets,composedStrides;
      for(unsigned axis=0;valid&&axis<rank;++axis){
       int64_t displacement,offset,stride;
       valid=firstSizes[axis]==middleType.getDimSize(axis)&&secondSizes[axis]==resultType.getDimSize(axis)&&inBounds(firstOffsets[axis],firstSizes[axis],firstStrides[axis],baseType.getDimSize(axis))&&inBounds(secondOffsets[axis],secondSizes[axis],secondStrides[axis],middleType.getDimSize(axis));
       valid=valid&&!__builtin_mul_overflow(secondOffsets[axis],firstStrides[axis],&displacement)&&!__builtin_add_overflow(firstOffsets[axis],displacement,&offset)&&!__builtin_mul_overflow(firstStrides[axis],secondStrides[axis],&stride);
       if(valid)valid=inBounds(offset,secondSizes[axis],stride,baseType.getDimSize(axis));
       if(valid){composedOffsets.push_back(offset);composedStrides.push_back(stride);}
      }
      if(valid){
       params["static_offsets"]=std::move(composedOffsets);params["static_strides"]=std::move(composedStrides);operands=Array{baseId};
       std::set<int64_t> coverage;for(auto &v:arr(previous,"source_op_indices"))coverage.insert(*v.getAsInteger());for(auto &v:arr(e,"source_op_indices"))coverage.insert(*v.getAsInteger());Array merged;for(auto sourceIndex:coverage)merged.push_back(sourceIndex);e["source_op_indices"]=std::move(merged);
       foldedInitializers.insert(previousIndex->second);
      }
     }
    }
   }
   // A complete, unit-stride slice has the source's dense linear ordering.
   // Check the composed recipe against its actual base, not the old view.
   auto sourceType=mlir::dyn_cast<mlir::RankedTensorType>(valueTypes.at(operands[0].getAsString()->str()));
   auto offsets=vector(params,"static_offsets"),sizes=vector(params,"static_sizes"),strides=vector(params,"static_strides");
   bool identity=sourceType&&sourceType.hasStaticShape()&&sourceType==result.getType();
   if(identity){
    unsigned rank=sourceType.getRank();identity=offsets.size()==rank&&sizes.size()==rank&&strides.size()==rank;
    for(unsigned axis=0;identity&&axis<rank;++axis)identity=offsets[axis]==0&&strides[axis]==1&&sizes[axis]==sourceType.getDimSize(axis);
   }
   if(identity)kind="reshape";
  }
 }
 else throw std::runtime_error("unsupported mathematical MLIR operation: "+name.str());
 // Named full-overwrite operations use the same empty-seed proof as
 // generic scalar maps. The original MLIR body and iteration maps establish
 // overwrite; an operation name alone never establishes initializer ownership.
 if(name=="linalg.generic"||name=="linalg.transpose"||name=="linalg.broadcast"||name=="linalg.copy"||name=="linalg.fill")if(auto lin=mlir::dyn_cast<mlir::linalg::LinalgOp>(op))if(lin.getNumDpsInits()==1&&op->getNumRegions()==1){
  unsigned destination=lin.getNumDpsInputs();auto seed=op->getOperand(destination);auto *empty=seed.getDefiningOp();
  auto &body=op->getRegion(0).front();auto maps=lin.getIndexingMapsArray();
  bool overwrite=destination<body.getNumArguments()&&body.getArgument(destination).use_empty()&&!maps.empty();
  std::vector<int64_t> bounds(lin.getNumLoops(),-1);
  for(auto [operand,map]:llvm::zip(op->getOperands(),maps)){
   auto tensor=mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
   for(auto [axis,expression]:llvm::enumerate(map.getResults()))if(auto dimension=mlir::dyn_cast<mlir::AffineDimExpr>(expression)){
    unsigned loop=dimension.getPosition();
    if(!tensor||!tensor.hasStaticShape()||loop>=bounds.size()){overwrite=false;continue;}
    int64_t extent=tensor.getDimSize(axis);if(extent<=0||(bounds[loop]!=-1&&bounds[loop]!=extent))overwrite=false;bounds[loop]=extent;
   }
  }
  for(auto extent:bounds)if(extent<=0)overwrite=false;
  auto resultType=mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  auto outputMap=maps.back();std::set<unsigned> axes;
  overwrite=overwrite&&resultType&&outputMap.getNumResults()==unsigned(resultType.getRank());
  for(auto [axis,expression]:llvm::enumerate(outputMap.getResults())){
   auto dimension=mlir::dyn_cast<mlir::AffineDimExpr>(expression);
   if(!dimension||!axes.insert(dimension.getPosition()).second){overwrite=false;break;}
   unsigned loop=dimension.getPosition();
   if(loop>=bounds.size()||bounds[loop]<=0||!resultType||bounds[loop]!=resultType.getDimSize(axis)){overwrite=false;break;}
  }
  auto emptyRegion=empty?empty->getAttrOfType<mlir::StringAttr>("prov.region_id"):mlir::StringAttr();
  auto rootRegion=op->getAttrOfType<mlir::StringAttr>("prov.region_id");
  bool sameRegion=!emptyRegion||(rootRegion&&emptyRegion.getValue()==rootRegion.getValue());
  if(overwrite&&empty&&empty->getName().getStringRef()=="tensor.empty"&&empty->getBlock()==op->getBlock()&&seed.hasOneUse()&&seed.use_begin()->getOwner()==op&&sameRegion){
   auto source=sourceIndices.find(empty);
   if(source!=sourceIndices.end()&&source->second<index){
    params["elided_destination"]=true;
    // Several named/specialized normalizations already omit the unused
    // destination. Generic maps still carry it before their captured values.
    if(destination<operands.size()&&operands[destination].getAsString()==ids.lookup(seed))operands.erase(operands.begin()+destination);
    std::set<int64_t> coverage{source->second};for(auto &v:arr(e,"source_op_indices"))coverage.insert(*v.getAsInteger());Array merged;for(auto sourceIndex:coverage)merged.push_back(sourceIndex);e["source_op_indices"]=std::move(merged);
    foldedInitializers.insert(source->second);
   }
  }
 }
 e["operator"]=kind;e["operands"]=std::move(operands);e["parameters"]=std::move(params);exprs.push_back(std::move(e));++index;
 }
 Array retained;for(auto &expression:exprs){auto &normalized=*expression.getAsObject();auto &coverage=arr(normalized,"source_op_indices");if(foldedInitializers.count(*coverage.back().getAsInteger()))continue;retained.push_back(std::move(expression));}
 return Object{{"schema","numerical_program.v1"},{"inputs",std::move(inputs)},{"expressions",std::move(retained)},{"outputs",std::move(outputs)},{"source_op_count",index},{"source_format","generic_math_mlir_v1"}};
}
}
Object parseProgram(const std::string&source){auto first=source.find_first_not_of(" \t\r\n");if(first==std::string::npos)throw std::runtime_error("empty source program");if(source[first]!='{')return parseMLIR(source);auto v=llvm::json::parse(source);if(!v)throw std::runtime_error("invalid JSON: "+llvm::toString(v.takeError()));auto*obj=v->getAsObject();if(!obj||str(*obj,"schema")!="numerical_program.v1")throw std::runtime_error("expected numerical_program.v1");Object program=clone(*obj);auto*expressions=program.getArray("expressions");if(!expressions)throw std::runtime_error("missing expressions");std::set<std::string> ids,names;for(auto&input:arr(program,"inputs")){auto*o=input.getAsObject();if(!o||str(*o,"id").empty()||!ids.insert(str(*o,"id")).second)throw std::runtime_error("invalid input id");for(auto d:shape(*o))if(d<=0)throw std::runtime_error("invalid input shape");if(str(*o,"dtype").empty())throw std::runtime_error("missing input dtype");}
 int64_t i=0;for(auto&e:*expressions){auto*o=e.getAsObject();if(!o||str(*o,"id").empty()||ids.count(str(*o,"id")))throw std::runtime_error("invalid expression id");for(auto&operand:arr(*o,"operands")){auto s=operand.getAsString();if(!s||!ids.count(s->str()))throw std::runtime_error("expression references unavailable value");}for(auto d:shape(*o))if(d<=0)throw std::runtime_error("invalid expression shape");if(str(*o,"operator").empty()||str(*o,"dtype").empty())throw std::runtime_error("missing expression operator/dtype");ids.insert(str(*o,"id"));(*o)["source_op_indices"]=Array{i++};o->erase("region_id");}
 for(auto&o:arr(program,"outputs")){auto*p=o.getAsObject();if(!p||!ids.count(str(*p,"value"))||str(*p,"name").empty()||!names.insert(str(*p,"name")).second)throw std::runtime_error("invalid output binding");}program["source_op_count"]=i;return program;}
}
