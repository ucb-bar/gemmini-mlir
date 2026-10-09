"""RV64GC register scheduling for eight independent source FMA chains."""

from dataclasses import dataclass

from merlin.llvmlower.source_fma_batch import SourceFmaBatchContract


@dataclass(frozen=True)
class SourceFmaBatchCapability:
    isa: str
    abi: str
    contract: SourceFmaBatchContract

    def header(self) -> str:
        if self.isa != 'rv64gc' or self.abi != 'lp64d':
            raise ValueError('RV64GC/lp64d source FMA batch capability required')
        if not isinstance(self.contract, SourceFmaBatchContract):
            raise ValueError('typed source FMA batch contract required')
        self.contract.validate()
        # Every output is read/write and early-clobber. It cannot share a
        # register with a still-needed input belonging to another lane.
        instructions = '\\n\\t'.join(
            f'fmadd.s %{lane},%{lane + 8},%{lane},%16'
            for lane in range(8)
        )
        outputs = ','.join(f'"+&f"(product[{lane}])' for lane in range(8))
        inputs = ','.join(f'"f"(fraction[{lane}])' for lane in range(8))
        return ('#ifndef GEMMINI_HOST_SOURCE_FMA_BATCH_H\n'
                '#define GEMMINI_HOST_SOURCE_FMA_BATCH_H\n'
                'static inline void gemmini_host_source_fma_eight(\n'
                ' const float fraction[8],float product[8],float coefficient) {\n'
                f' __asm__("{instructions}"\n'
                f'         : {outputs}\n'
                f'         : {inputs},"f"(coefficient));\n'
                '}\n'
                '#define MERLIN_SOURCE_F32_FMA_EIGHT gemmini_host_source_fma_eight\n'
                '#endif\n')
