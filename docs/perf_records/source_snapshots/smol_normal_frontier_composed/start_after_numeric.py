from pathlib import Path
import json,subprocess,os,time
w=Path(__file__).resolve().parent
v=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed/native/validation.json')
while not v.exists():
 os.kill(2343046,0);time.sleep(10)
r=json.loads(v.read_text());assert r['bitwise_mismatches']==0 and r['allclose'] and r['calls'][:3]==[48,0,0]
env=dict(os.environ,PYTHONPATH='.:/scratch/agustin/tmp/merlin-polynomial-four-cell-20261006/src')
with(w/'prepare.log').open('w')as f:subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(w/'prepare.py')],check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
with(w/'build.log').open('w')as f:p=subprocess.Popen(['bash',str(w/'run.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(w/'build_process.json').write_text(json.dumps({'pid':p.pid})+'\n')
with(w/'normal_watch.log').open('w')as f:q=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(w/'watch_normal.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(w/'watch_process.json').write_text(json.dumps({'pid':q.pid})+'\n')
print('build',p.pid,'watch',q.pid,flush=True)
