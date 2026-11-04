"""Bounded RHS block DMA using the admitted Gemmini software layout.

The selected raw gemmini.h matmul move-in-B loop copies up to MAX_BLOCK_LEN
adjacent N tiles into (k*J+j)*DIM local rows. gemmini_params.h defines DIM=16,
MAX_BYTES=64 and MAX_BLOCK_LEN=4. This authored address law follows that exact
source layout; independent final-band checking retains all original epochs.
No packet mask supplies a hardware capability or completion guarantee.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw,TileLoadMapping,propose_tile_load_coalescing
from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
from merlin.sched.ir import Const,Scaled,Sum,Var
from . import isa

# Content pins for the selected raw, independently authored target software.
REFERENCE_FILES = {
 'originals/upstream-exo/public-reference/headers/gemmini.h': '06988403de641bbf22fea84e76debd863ab3185498f2b48ceb4b475d4af0914e',
 'originals/upstream-exo/public-reference/headers/gemmini_params.h': 'd6db18f8edbfeb854780286d8c2afb81ae3b9103224522478a9047d5550272e5',
}

def _linear(e):
 if type(e)is Const:return e.value,{}
 if type(e)is Var:return 0,{e.name:1}
 if type(e)is Scaled:
  c,t=_linear(e.expr);return c*e.factor,{k:v*e.factor for k,v in t.items()}
 if type(e)is Sum:
  c,t=0,{}
  for x in e.terms:
   a,b=_linear(x);c+=a
   for k,v in b.items():t[k]=t.get(k,0)+v
  return c,{k:v for k,v in t.items() if v}
 raise ValueError('block DMA requires an affine column offset')

def _quotient(e,divisor):
 c,t=_linear(e)
 if c%divisor or any(v%divisor for v in t.values()):raise ValueError('block DMA band is not tile aligned')
 return Const(c//divisor)+sum((Var(k)*(v//divisor) for k,v in t.items()),Const(0))

@dataclass(frozen=True)
class DenseRhsTransferResult:
 checked: object
 realization: dict
 @property
 def kernel(self):return self.checked.kernel
 def revalidate(self):self.checked.revalidate()
 def materialize_scf(self):return self.checked.materialize_scf()
 def emit_llvm(self,builder,lower_call):
  from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
  self.revalidate();bridge=FnBuilderIndexAdapter(builder)
  result=emit_index_program(self.kernel,bridge,lambda name,args:lower_call(name,{k:bridge.value(v) for k,v in args.items()}))
  self.revalidate();return result


def select_rhs_block_transfers(original,problem,block,resources,storage,adapter,*,consumer=None):
 """Return final checked wide calls, or preserve the exact incumbent.

 Every proposed command is projected from its real fields, expanded into ALL
 primitive bands, and rechecked against source storage/lifetime/epoch contracts.
 The primitive one-band tail is unchanged. Source pitches/configuration and
 block stride remain the same original channel state in the terminal encoder.
 """
 original.revalidate()
 if problem.tile_n!=isa.DIM or block.n<2:
  return original,{'applied':False,'reason':'no adjacent primitive N tiles in a panel'}
 identity=sha256(repr(sorted(REFERENCE_FILES.items())).encode()).hexdigest()
 def project(call):
  if call.instr not in ('gemmini_mvin_lhs','gemmini_mvin_weight'):return None
  op=adapter.project(call)
  if type(op)is not Load:raise ValueError('block DMA source is not a complete decoded load')
  left=op.role=='lhs';pitch=problem.k if left else problem.n
  law=TileLoadAddressLaw(isa.DIM,isa.DIM,64,1,pitch,isa.DIM,'selected-gemmini-software-block-mvin:'+identity)
  return TileLoadMapping(op.role,'spad',op.dram_row*pitch+op.dram_col,op.row,op.rows,op.cols,law)
 def encode(call,columns):
  return replace(call,args=tuple((k,Const(columns) if k=='cols' else v) for k,v in call.args))
 def encode_band(call,band):
  op=adapter.project(call);old=project(call)
  offset=_quotient(band.source_byte+old.source_byte*(-1),1)
  col=op.dram_col+offset
  op=replace(op,block=(op.block[0],op.block[1]+_quotient(offset,isa.DIM)),
   dram_col=col,row=band.local_row,rows=band.rows,cols=band.columns)
  return adapter.encode(op)
 proposal,proposal_report=propose_tile_load_coalescing(original.kernel,
  project=lambda call:project(call) if call.instr=='gemmini_mvin_weight' else None,
  encode=encode,max_tiles=4,coalesce_tail=True)
 if not proposal_report['changes']:return original,dict(applied=False,proposal=proposal_report)
 from merlin.compile.scheduling.dense_ready_readout import DenseReadyReadoutResult,admit_ready_readout_tile_loads
 if consumer is None:
  consumer=admit_ready_readout_tile_loads if type(original) is DenseReadyReadoutResult else admit_panel_tile_loads
 checked=consumer(original,proposal,problem,block,resources,storage,adapter,
  project=project,encode_band=encode_band,timeout_ms=30000)
 receipt=dict(applied=True,source_sha256=original.kernel.digest(),kernel_sha256=checked.kernel.digest(),
  reference_files=REFERENCE_FILES,proposal=proposal_report,projection=checked.realization['tile_load_projection'],
  completion='all original full-device drains and finite issued-band bounds retained')
 result=DenseRhsTransferResult(checked,dict(original.realization,kernel_sha256=checked.kernel.digest(),rhs_block_transfers=receipt))
 result.revalidate()
 return result,receipt
