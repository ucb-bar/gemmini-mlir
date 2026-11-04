"""Optional checked device-horizontal transport for the original pooled source."""
from merlin.compile.scheduling.spatial_horizontal_flat import SpatialHorizontalFlat
from merlin.compile.scheduling.spatial_residue_runs import SpatialResidueRuns


def propose_pool_horizontal_flat(producer):
    """Return a legal independent nonresident layout or preserve the incumbent."""
    if producer.window.kernel[1]*producer.window.input_shape[-1] <= producer.primitive:
        return None
    try:
        return SpatialHorizontalFlat(SpatialResidueRuns(producer,
            max(producer.rhs_row+producer.primitive,producer.lhs_row+producer.row_tile)))
    except ValueError:
        return None
