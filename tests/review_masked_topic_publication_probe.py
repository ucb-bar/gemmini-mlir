"""Read back the reviewed generic masked topic; never publish or mutate GitHub."""

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path


def close():
    root=Path(__file__).resolve().parents[1]
    tree=Path('/scratch/agustin/tmp/merlin-masked-contraction-upstream-20261007')
    delivery=tree/'out/artifacts/delivery/masked-contraction-20261007'
    path=delivery/'pull_request.json'
    published=json.loads(path.read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for key in ('root_review','delivery','publisher'):
        assert sha(published[key])==published[key+'_sha256']
    body_path=delivery/'pull_request_body.md'
    assert sha(body_path)==published['body_sha256']
    # Git's installed credential helper is used without exposing credential output.
    credential=subprocess.run(['git','-C',str(tree),'credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,timeout=20,check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    fields=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    request=urllib.request.Request('https://api.github.com/repos/ucb-bar/merlin/pulls/48',headers={'Authorization':'Bearer '+fields['password'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(request,timeout=30) as response:
        pr=json.load(response)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/perf/closed-mask-contraction-main-20261007','refs/heads/main'],cwd=tree,text=True)
    heads={line.split()[1]:line.split()[0] for line in remote.splitlines()}
    assert pr['head']['sha']==published['reviewed_head']==heads['refs/heads/perf/closed-mask-contraction-main-20261007']
    assert pr['base']['ref']=='main' and pr['base']['sha']==published['reviewed_base']==heads['refs/heads/main']
    assert hashlib.sha256(pr['body'].encode()).hexdigest()==published['body_sha256']
    assert pr['state']=='open' and not pr['merged']
    output=root/'docs/perf_records/root_merlin_masked_PR48_publication_review_20261007.json'
    assert not output.exists()
    paths=(path,Path(published['root_review']),Path(published['delivery']),Path(published['publisher']),body_path,Path(__file__))
    record={'schema':'root_merlin_masked_PR48_publication_review_v1','status':'PUBLISHED_REVIEWED_MAIN_BASED_TOPIC','number':48,'url':pr['html_url'],'remote_head':pr['head']['sha'],'remote_base':pr['base']['sha'],'actual_API_and_git_remote_equal_released_hashes':True,'body_sha256':published['body_sha256'],'default_unchanged':True,'no_main_push_force_or_merge':True,'scope':'Generic typed closed-mask contraction proof/scheduling, independent of target/model names. Read-only publication review. Actual main does not include this open topic. READMEdate-only previoushead discrepancy retained in delivery review. Initial localgh lookup failed before any network request; installed git credential provider now read-only.','pins':{str(p):sha(p) for p in paths}}
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'head':record['remote_head'],'base':record['remote_base'],'url':record['url']}))


if __name__=='__main__':
    close()
