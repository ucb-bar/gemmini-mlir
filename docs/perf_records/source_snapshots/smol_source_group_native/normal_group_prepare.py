from pathlib import Path
import json
from merlin.llvmlower.ordered_bf16_group_binding import SourceExactGroupPreparation
w=Path(__file__).resolve().parent
source=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle/model.mlir')
prepare=SourceExactGroupPreparation(expected_source_sha256='4814509b8e11a5c819b1f9ae63f01f89f72e9edf9c3d0ec9d5cd0ab6b35de2dc')
selected=prepare(source,w/'normal_group_preparation')
print(json.dumps({'source_groups':prepare.receipt['source_groups'],'source_contractions':prepare.receipt['source_contractions'],'selected_source':str(selected),'numeric_provider_installed':False}),flush=True)
