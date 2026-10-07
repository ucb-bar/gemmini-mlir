"""Independently verify the selected final body against its relocatable object."""
from pathlib import Path
import hashlib
import json
import struct

PACKET = Path('/scratch/agustin/tmp/gemmini-current-source-residue-20261007/docs/perf_records/current_source_residue_2086_whole_qualification.json')
OUT = Path(__file__).resolve().parent / 'source_residue_whole_link_review.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sections(path):
    data = Path(path).read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01'
    assert struct.unpack_from('<H', data, 18)[0] == 243
    table = struct.unpack_from('<Q', data, 40)[0]
    width, count, strings = struct.unpack_from('<HHH', data, 58)
    assert width == 64 and count > 0
    rows = [struct.unpack_from('<IIQQQQIIQQ', data, table+i*width) for i in range(count)]
    names = data[rows[strings][4]:rows[strings][4]+rows[strings][5]]
    return data, rows, [names[r[0]:].split(b'\0', 1)[0].decode() for r in rows]


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits-1)) else value


packet = json.loads(PACKET.read_text())
entry_path = Path(packet['actual_selected_entry_and_relocation_closure']['path'])
assert sha(entry_path) == packet['actual_selected_entry_and_relocation_closure']['sha256']
entry = json.loads(entry_path.read_text())
assert entry['status'] == 'PASS' and entry['selected_symbol_entry_count'] == 1
elf = Path(packet['candidate_elf']['path'])
bundle = json.loads(Path(packet['selected_bundle']['path']).read_text())
objects = [Path(path) for path, digest in packet['pins'].items()
    if path.endswith('/gemmini_exact_requant_43/kernel.o') and digest == entry['selected_object_sha256']]
assert objects
obj = objects[0]
assert sha(obj) == entry['selected_object_sha256']
assert sha(elf) == packet['candidate_elf']['sha256'] == entry['elf_sha256']
obj_data, obj_rows, obj_names = sections(obj)
executable = [i for i, row in enumerate(obj_rows) if row[2] & 4 and row[5]]
assert len(executable) == 1
index = executable[0]
row = obj_rows[index]
source = bytearray(obj_data[row[4]:row[4]+row[5]])
elf_data, elf_rows, elf_names = sections(elf)
start, end = entry['selected_symbol_bounds']['start'], entry['selected_symbol_bounds']['end']
container = [row for row in elf_rows if row[2] & 4 and row[3] <= start < end <= row[3]+row[5]]
assert len(container) == 1 and end-start == len(source)
row = container[0]
target = bytearray(elf_data[row[4]+start-row[3]:row[4]+end-row[3]])
saved_comparison_body = (entry_path.parent/'candidate.text').read_bytes()
relocations, relax = [], []
for row in obj_rows:
    if row[1] != 4 or row[7] != index:
        continue
    assert row[9] == 24
    symbols = obj_rows[row[6]]
    assert symbols[9] == 24
    for location in range(row[4], row[4]+row[5], 24):
        offset, info, addend = struct.unpack_from('<QQq', obj_data, location)
        kind, symbol = info & 0xFFFFFFFF, info >> 32
        if kind == 51:
            relax.append(offset)
            continue
        assert kind in (16, 17, 44), (offset, kind)
        name, flags, visibility, section, value, size = struct.unpack_from('<IBBHQQ', obj_data, symbols[4]+symbol*24)
        assert section == index
        expected_target = start+value+addend
        width = 2 if kind == 44 else 4
        old = int.from_bytes(source[offset:offset+width], 'little')
        final = int.from_bytes(target[offset:offset+width], 'little')
        if kind == 16:
            mask = 0x1FFF07F
            assert final & 0x7F == 0x63
            immediate = ((final >> 31) << 12) | (((final >> 7) & 1) << 11) | (((final >> 25) & 63) << 5) | (((final >> 8) & 15) << 1)
            displacement = signed(immediate, 13)
        elif kind == 17:
            mask = 0xFFF
            assert final & 0x7F == 0x6F
            immediate = ((final >> 31) << 20) | (((final >> 21) & 1023) << 1) | (((final >> 20) & 1) << 11) | (((final >> 12) & 255) << 12)
            displacement = signed(immediate, 21)
        else:
            mask = 0xFFFF ^ ((1 << 12) | (3 << 10) | (3 << 5) | (3 << 3) | (1 << 2))
            assert final & 3 == 1 and final >> 13 in (6, 7)
            immediate = (((final >> 12) & 1) << 8) | (((final >> 10) & 3) << 3) | (((final >> 5) & 3) << 6) | (((final >> 3) & 3) << 1) | (((final >> 2) & 1) << 5)
            displacement = signed(immediate, 9)
        assert old & mask == final & mask, (offset, 'opcode/register changed')
        assert start+offset+displacement == expected_target, (offset, 'target mismatch')
        source[offset:offset+width] = (old & mask).to_bytes(width, 'little')
        target[offset:offset+width] = (final & mask).to_bytes(width, 'little')
        relocations.append(dict(offset=offset, kind=kind, width=width, target=expected_target))
assert source == target
assert bytes(target) == saved_comparison_body
assert len(relocations) == 11 and sorted(relax) == [13876]
assert sorted(r['kind'] for r in relocations) == [16]*9+[17,44]
result = dict(status='PASS', source_object=str(obj), source_object_sha256=sha(obj),
    elf=str(elf), elf_sha256=sha(elf), full_selected_body_bytes=len(source),
    symbol=entry['selected_symbol'], actual_entry_count=1,
    relocation_targets_and_registers=relocations, relax_metadata_offsets=relax,
    all_other_bytes_equal=True, independent_from_agent_boolean=True,
    source_packet=str(PACKET), source_packet_sha256=sha(PACKET), entry_receipt=str(entry_path),
    entry_receipt_sha256=sha(entry_path), script_sha256=sha(__file__))
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(status='PASS', selected_bytes=len(source), exact_relocations=len(relocations))))
