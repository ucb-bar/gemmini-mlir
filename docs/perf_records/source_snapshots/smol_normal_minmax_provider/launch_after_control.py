"""One conditional strict execution; never restart an observed live process."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
w=Path(__file__).resolve().parent;old=w.parent/'normal_attention_provider';root=w.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(status,**fields):
 (w/'sequential_launch_state.json').write_text(json.dumps({'status':status,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields},indent=2)+'\n')
fd=os.open(w/'sequential_launch.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
write('waiting_for_prior_terminal_pass')
while not (old/'target_validation.json').exists():time.sleep(30)
r=json.loads((old/'target_validation.json').read_text());old_intent=json.loads((old/'target_intent.json').read_text())
assert r['status']=='pass' and r['returncode']==0 and r['final_elf_unchanged'],'Prior target failed: no new execution'
assert sha(old/'build/model.elf')==r['elf_sha256']==old_intent['elf_sha256']
assert sha(old/'target_spike.log')==r['log_sha256'] and r['expected_output_sha256']==old_intent['expected_output_sha256']
pid=json.loads((old/'target_process.json').read_text())['spike_pid']
cmdline=Path(f'/proc/{pid}/cmdline')
if cmdline.exists():assert str(old/'build/model.elf').encode() not in cmdline.read_bytes(),'Prior exact simulator still live'
receipt=root/'docs/perf_records/smol_normal_minmax_provider_build_native.json';new=json.loads(receipt.read_text())
for p,h in new['pins'].items():assert sha(p)==h,(p,'qualification pin changed')
assert new['native']['bitwise_mismatches']==0 and new['source_fallback']['mismatches']==0
assert not (w/'target_process.json').exists() and not (w/'target_intent.json').exists(),'Existing target intent refuses duplicate execution'
with (w/'target_watch.log').open('x')as log:
 p=subprocess.Popen(['bash',str(w/'run_target.sh')],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(w/'target_watcher.json').write_text(json.dumps({'pid':p.pid,'launched_after_control_receipt_sha256':sha(old/'target_validation.json'),'qualification_sha256':sha(receipt),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
write('launched_once',watcher_pid=p.pid,prior_receipt_sha256=sha(old/'target_validation.json'))
