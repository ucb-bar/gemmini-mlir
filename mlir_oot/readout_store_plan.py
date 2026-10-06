"""Typed primitive two-output store contract; shared arithmetic proof is Merlin."""
from dataclasses import dataclass
from merlin.llvmlower.enclosed_readout import prove
from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange


@dataclass(frozen=True)
class PairedReadoutPlan:
    source_scales: tuple[float, ...]
    store_scales: tuple[float, float]
    accumulator_min: int
    accumulator_max: int
    relu: bool = False

    def certificate(self):
        if type(self.source_scales) is not tuple or type(self.store_scales) is not tuple:
            raise ValueError('immutable scale tuples required')
        certificate = prove(self.source_scales, self.store_scales,
                            self.accumulator_min, self.accumulator_max, relu=self.relu)
        if not certificate['exact_pair_decoder']:
            raise ValueError('store pair does not determine exact source readout')
        return certificate

    def require_conv_producer(self, shape):
        self.certificate()
        if shape.output_dtype != 'i32' or shape.scale != 1.0 or shape.relu:
            raise ValueError('paired stores require an unscaled unactivated i32 producer')
        domain = IntegerSumProductsRange(9 * shape.cin, -128, 127, -128, 127)
        domain.require_contained(self.accumulator_min, self.accumulator_max)
        return domain
