"""Check numeric RV64GC operand fixtures against the installed ISA declarations."""

import argparse
import json
from pathlib import Path

from mlir_oot.cpu_fp_distances import direct_target, encoding_forms, fp_effects


def check(header, output):
    forms = encoding_forms(header.read_text())
    # Independent operand fixtures use GNU's numeric disassembly spelling.
    examples = (
        ("fadd.s f10,f11,f12", 0x00C58553, 4, (10,), (11, 12)),
        ("fdiv.s f10,f11,f12", 0x18C58553, 4, (10,), (11, 12)),
        ("fmadd.s f10,f11,f12,f13", 0x68C58543, 4, (10,), (11, 12, 13)),
        ("fcvt.s.d f10,f11", 0x40158553, 4, (10,), (11,)),
        ("fcvt.w.s x10,f11", 0xC0058553, 4, (), (11,)),
        ("fcvt.s.w f10,x11", 0xD0058553, 4, (10,), ()),
        ("fmv.x.w x10,f11", 0xE0058553, 4, (), (11,)),
        ("fclass.s x10,f11", 0xE0059553, 4, (), (11,)),
        ("fmv.w.x f10,x11", 0xF0058553, 4, (10,), ()),
        ("flw f10,0(x11)", 0x0005A507, 4, (10,), ()),
        ("fsw f10,0(x11)", 0x00A5A027, 4, (), (10,)),
        ("c.fld f8,0(x8)", 0x2000, 2, (8,), ()),
        ("c.fsd f8,0(x8)", 0xA000, 2, (), (8,)),
        ("c.fldsp f8,0(x2)", 0x2402, 2, (8,), ()),
        ("c.fsdsp f8,0(x2)", 0xA022, 2, (), (8,)),
    )
    for name, word, width, defs, uses in examples:
        assert fp_effects(word, width, forms) == (defs, uses, True), name
    assert fp_effects(0x40258553, 4, forms)[2] is False, "half-format conversion is outside RV64GC"
    assert fp_effects(0xFFFFFFFF, 4, forms)[2] is False
    # +8/-8/-4 branch targets, including compressed loop edges.
    for word, width, delta in ((0x00000463, 4, 8), (0xFE000CE3, 4, -8),
                               (0x0080006F, 4, 8), (0xFF9FF06F, 4, -8),
                               (0xBFF5, 2, -4), (0xDC75, 2, -4)):
        assert direct_target(0x1000, word, width) == 0x1000 + delta
    record = {"schema": "cpu_fp_operand_decoder_check_v1", "status": "pass",
              "operand_examples": len(examples), "unsupported_form_refusals": 2,
              "signed_branch_targets": 6, "cycles_or_reordering_claimed": False,
              "fixtures": [{"disassembly": n, "encoding": hex(w), "width": z} for n, w, z, _, _ in examples]}
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("CPU_FP_OPERAND_DECODER_PASS", len(examples) + 8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding-header", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.encoding_header, args.out)
