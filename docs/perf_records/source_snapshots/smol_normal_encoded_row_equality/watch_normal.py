from pathlib import Path
import json,time,os,subprocess,hashlib
w=Path(__file__).resolve().parent;pid=json.loads((w/'build_process.json').read_text())['pid'];start=time.time()
while not(w/'build_result.json').exists():
 try:os.kill(pid,0)
 except ProcessLookupError:
  (w/'normal_watch_state.json').write_text(json.dumps({'state':'build_terminated_without_result','elapsed_seconds':time.time()-start},indent=2)+'\n');raise SystemExit(1)
 if time.time()-start>7200:
  (w/'normal_watch_state.json').write_text(json.dumps({'state':'observation_limit_build_still_owned','pid':pid},indent=2)+'\n');raise SystemExit(2)
 time.sleep(10)
(w/'normal_watch_state.json').write_text(json.dumps({'state':'native_and_source_fallback_running','build_result_sha256':hashlib.sha256((w/'build_result.json').read_bytes()).hexdigest()},indent=2)+'\n')
with(w/'native.log').open('w')as log:r=subprocess.run(['bash',str(w/'run_native.sh')],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
(w/'normal_watch_state.json').write_text(json.dumps({'state':'pass'if r.returncode==0 else'failed','returncode':r.returncode,'elapsed_seconds':time.time()-start,'normal_native_validation':str(w/'native/validation.json'),'source_fallback_validation':str(w/'native/source_fallback_validation.json'),'whole_target_execution':'not launched'},indent=2)+'\n')
