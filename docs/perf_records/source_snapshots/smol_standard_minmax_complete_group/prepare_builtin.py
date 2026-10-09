"""Reuse frozen original data/link; select explicit separate min/max policy."""
from pathlib import Path
import hashlib,json
work=Path.cwd()/'out/artifacts/probes/smol-minmax-20261006'
original=(work/'prepare.py').read_text()
source=original.replace("arm=work/'guarded';", "arm=work/'builtin';")
source=source.replace("work/'guarded_minmax.h',arm/'guarded_minmax.h'", "work/'builtin_minmax.h',arm/'guarded_minmax.h'")
source=source.replace("str(work/'guarded_minmax.h')", "str(work/'builtin_minmax.h'),str(work/'prepare_builtin.py')")
source=source.replace('Guarded standard min/max experiment; source zero/NaN operands retain original library.', 'Explicit standard min/max experiment; min/max signed-zero and NaN-payload distinctions unobserved.')
source=source.replace("'source_signed_zero_nan_library_paths_preserved':True", "'min_max_signed_zero_nan_payload_unobserved':True")
assert source!=original
(work/'builtin_build_source.py').write_text(source)
exec(compile(source,str(work/'builtin_build_source.py'),'exec'))
