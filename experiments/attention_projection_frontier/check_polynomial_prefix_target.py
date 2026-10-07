"""Independent rational source oracle against prefix enclosures on target."""
from pathlib import Path
B=Path(__file__).resolve().parents[2]
s=(B/'out/artifacts/probes/prepared-polynomial-constants/target_driver.py').read_text()
s=s.replace('prepared-polynomial-constants/','source-polynomial-prefix-table/')
s=s.replace('#include "prepared_polynomial_constants.h"','#include "polynomial_prefix_table.h"')
s=s.replace('merlin_polynomial_constants_admit','merlin_polynomial_prefix_admit').replace('merlin_polynomial_constants_four','merlin_polynomial_prefix_four')
s=s.replace('merlin_interval_bits(interval.lo)!=got||merlin_interval_bits(interval.hi)!=got','merlin_interval_bits(interval.lo)>got||merlin_interval_bits(interval.hi)<got')
s=s.replace('merlin_interval_bits(ys[j].lo)!=lower||merlin_interval_bits(ys[j].hi)!=upper','merlin_interval_bits(ys[j].lo)>lower||merlin_interval_bits(ys[j].hi)<upper')
(B/'out/artifacts/probes/source-polynomial-prefix-table/target_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
