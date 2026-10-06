"""Reproduce consumer trap alone, then link the unchanged Merlin allocator."""
from pathlib import Path
import hashlib
import json
import subprocess
from mlir_oot.no_fsm_audit import audit_elf

root = Path.cwd()
work = root / 'out/artifacts/probes/smol-word-enclosure-20261006'
old = work / 'consumer_capsules_ordered'
core = Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
out = work / 'consumer_capsules_printf'
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
recipe = json.loads((old / 'control/build.json').read_text())
for p, expected in recipe['pins'].items():
    assert sha(p) == expected, p
gcc = recipe['commands'][0][0]
runtime = core / 'merlin/runtime/baremetal/spike'
allocator = out / 'malloc.o'
allocator_command = [gcc, '-O2', '-ffreestanding', '-fno-builtin', '-march=rv64gc',
    '-mabi=lp64d', '-mcmodel=medany', '-DMERLIN_ARENA_BASE_ADDR=0xc0000000ULL',
    '-DMERLIN_ARENA_SIZE_BYTES=0x1000000ULL', '-c', str(runtime / 'merlin_malloc.c'),
    '-o', str(allocator)]
subprocess.run(allocator_command, check=True)
compat = r'''
extern void printstr(const char*);
extern void tohost_exit(uintptr_t) __attribute__((noreturn));
void htif_putc(char c){char s[2]={c,0};printstr(s);}
void htif_puts(const char*s){printstr(s);}
void htif_putd(long v){printf("%ld",v);}
void htif_puthex(unsigned long long v){printf("0x%lx",(unsigned long)v);}
void htif_exit(int code){tohost_exit(code);}
uintptr_t handle_trap(uintptr_t cause,uintptr_t epc,uintptr_t regs[32]){
 uintptr_t value;__asm__ volatile("csrr %0,mtval":"=r"(value));
 printf("CONSUMER_TRAP cause=%lu pc=%lx value=%lx\n",cause,epc,value);
 tohost_exit(1337);
}
'''
source = (old / 'driver.c').read_text()
assert source.count('int main(void){') == 1
source = source.replace('int main(void){', compat + '\nint main(void){')
# Common benchmark printf is the linked HTIF implementation; libc puts is not.
source = source.replace(' puts("WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS");',
 ' printf("%s\\n","WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS");')
source = source.replace(' printf("WORKSPACE_GROUP_CYCLES %lu\\n",elapsed);', '')

for name, provider, add_allocator, diagnostic in [
    ('control', root / 'out/artifacts/probes/smol-minmax-20261006/core_selected/provider.o', True, False),
    ('words', work / 'words_exact_headers/provider.o', True, False),
    ('table10', root / 'out/artifacts/probes/smol-polynomial-table-20261006/table10/provider.o', True, False),
]:
    arm = out / name
    arm.mkdir()
    driver = arm / 'driver.c'
    arm_source = source
    if diagnostic:
        arm_source = arm_source.replace('int main(void){', 'int main(void){\n int s=check_consumer(original_source);printf("ORIGINAL_CONSUMER_DIAGNOSTIC %d\\n",s);return s;\n')
    driver.write_text(arm_source)
    command = [a.replace(str(old / 'driver.c'), str(driver)).replace(str(old / 'driver.o'), str(arm / 'driver.o')) for a in recipe['commands'][0]]
    subprocess.run(command, check=True)
    link = [a.replace(str(old / 'control/model.elf'), str(arm / 'model.elf')).replace(str(old / 'driver.o'), str(arm / 'driver.o')) for a in recipe['link']]
    if provider:
        link = [str(provider) if a == str(root / 'out/artifacts/probes/smol-minmax-20261006/core_selected/provider.o') else a for a in link]
    if add_allocator:
        link.insert(link.index('-lm'), str(allocator))
    subprocess.run(link, check=True)
    audit = audit_elf((arm / 'model.elf').read_bytes())
    assert audit['status'] == 'pass', audit
    pins = dict(recipe['pins'])
    for p in [driver, arm / 'driver.o', allocator, runtime / 'merlin_malloc.c', runtime / 'htif.h', Path(__file__), *(Path(a) for a in link if Path(a).is_file())]:
        pins[str(p.resolve())] = sha(p)
    (arm / 'build.json').write_text(json.dumps({'scope': 'Original compiled consumer with original source/goldens. Unchanged production Merlin bump allocator plus benchmark console ABI bridge outside provider ROI; diagnostic variants run consumer only.', 'compile': command, 'allocator_compile': allocator_command, 'link': link, 'pins': pins, 'nofsm': audit, 'diagnostic': diagnostic, 'original_consumer_unchanged': True}, indent=2) + '\n')
    watch = (old / 'control/watch.py').read_text()
    if diagnostic:
        watch = watch.replace('WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS', 'ORIGINAL_CONSUMER_DIAGNOSTIC 0')
    (arm / 'watch.py').write_text(watch)
    print(name, sha(arm / 'model.elf'), audit['status'], flush=True)
