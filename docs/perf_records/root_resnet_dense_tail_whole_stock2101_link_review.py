"""Root actual ELF/symbol/RELA proof for both selected stationary-tail bodies."""
from pathlib import Path
import hashlib,json,struct
PACKET=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/docs/perf_records/current_dense_stationary_tail_2095_whole_qualification.json')
OUT=Path(__file__).resolve().parent/'dense_tail_whole_link_review.json'
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


def lookup_symbol(data,rows,name):
    found=[]
    for row in rows:
        if row[1]!=2:continue
        assert row[9]==24
        strings=rows[row[6]];names=data[strings[4]:strings[4]+strings[5]]
        for offset in range(row[4],row[4]+row[5],24):
            n,flags,visibility,section,value,size=struct.unpack_from('<IBBHQQ',data,offset)
            if names[n:].split(b'\0',1)[0].decode()==name:
                found.append((section,value,size))
    assert len(found)==1,(name,found)
    return found[0]

def verify(selected,elf):
    obj=Path(selected['selected_object'])
    assert sha(obj)==selected['selected_object_sha256']
    obj_data,obj_rows,obj_names=sections(obj)
    elf_data,elf_rows,elf_names=sections(elf)
    index,source_start,source_size=lookup_symbol(obj_data,obj_rows,selected['symbol'])
    elf_section,start,size=lookup_symbol(elf_data,elf_rows,selected['symbol'])
    assert source_start==0 and source_size==size==selected['body_bytes']
    assert start==selected['symbol_start']
    assert selected['actual_entry_count']==1
    row=obj_rows[index]
    assert row[2]&4 and row[5]==size
    source=bytearray(obj_data[row[4]:row[4]+row[5]])
    row=elf_rows[elf_section]
    assert row[2]&4 and row[3]<=start<start+size<=row[3]+row[5]
    target=bytearray(elf_data[row[4]+start-row[3]:row[4]+start-row[3]+size])
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
            name, flags, visibility, section, value, target_symbol_size = struct.unpack_from('<IBBHQQ', obj_data, symbols[4]+symbol*24)
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
    assert source==target and size==len(source)==selected['body_bytes']
    assert len(relocations)==len(selected['relocations'])
    assert relocations==selected['relocations'] and relax==selected['relax_offsets']
    return dict(symbol=selected['symbol'],body_bytes=size,actual_entry_count=1,
        source_object=str(obj),source_object_sha256=sha(obj),
        all_other_bytes_equal=True,relocation_targets_and_registers=relocations,
        relax_metadata_offsets=relax,actual_elf_symbol_bounds=[start,start+size])
packet=json.loads(PACKET.read_text())
assert sha(PACKET)=='765ebdf9db0728b9e18922135579f669a1d85028f6dd295cdef2b1809c82f26f'
assert len(packet['pins'])==2671
for path,digest in packet['pins'].items():assert sha(path)==digest,path
entry_row=packet['actual_selected_entry_and_fullbody_relocation_closure']
entry_path=Path(entry_row['path']);assert sha(entry_path)==entry_row['sha256']
entry=json.loads(entry_path.read_text())
elf=Path(packet['candidate_elf']['path']);assert sha(elf)==packet['candidate_elf']['sha256']==entry['elf_sha256']
assert len(entry['selected'])==19
selected=[verify(row,elf) for row in entry['selected']]
assert {x['symbol'] for x in selected}=={x['kernel'] for x in packet['selected_actual19']}
assert len({x['symbol'] for x in selected})==19
result=dict(schema='root_actual_nineteen_dense_body_elf_symbol_RELA_verification_v1',status='PASS',
    elf=str(elf),elf_sha256=sha(elf),selected=selected,independent_from_agent_boolean=True,
    source_packet=str(PACKET),source_packet_sha256=sha(PACKET),entry_receipt=str(entry_path),entry_receipt_sha256=sha(entry_path),script_sha256=sha(__file__))
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='PASS',selected_bodies=len(selected),body_bytes=[x['body_bytes'] for x in selected],exact_relocations=[len(x['relocation_targets_and_registers']) for x in selected])),flush=True)
