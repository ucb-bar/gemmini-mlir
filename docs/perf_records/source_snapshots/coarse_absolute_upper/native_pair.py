from pathlib import Path
import json,subprocess,os
w=Path(__file__).resolve().parent
base=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed/validate_native.py').read_text()
for bucket,per_group in [('.',576)]:
 case=w/bucket;s=base.replace('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed',str(case)).replace('encoded rows plus producer spans, exact casts, probability bins and four-cell schedule','absolute-product diagnostic cost partition with explicitly enlarged native-only caller workspace')
 s=s.replace("s=['#include <stdint.h>'","s=['#include <stdlib.h>','#include <string.h>', 'typedef struct{void*allocated;void*aligned;int64_t offset,size,stride;} W1;', 'extern size_t group_provider_workspace_bytes(void);', 'static W1 workspace_view;static unsigned char*workspace_owner;', 'static void*workspace(void){if(!workspace_owner){size_t n=group_provider_workspace_bytes();workspace_owner=malloc(n+128);if(!workspace_owner)abort();memset(workspace_owner,0xa5,n+128);workspace_view=(W1){workspace_owner,(void*)(((uintptr_t)workspace_owner+63)&~(uintptr_t)63),0,(int64_t)n,1};}return &workspace_view;}', 'int finish_workspace(void){size_t n=workspace_view.size;unsigned char*p=workspace_view.aligned;for(unsigned char*q=workspace_owner;q<p;q++)if(*q!=0xa5)return 0;for(unsigned char*q=p+n;q<workspace_owner+n+128;q++)if(*q!=0xa5)return 0;free(workspace_owner);workspace_owner=0;return 1;}', '#include <stdint.h>'")
 # The original list includes stdint after injected declarations; move standard types before them.
 s=s.replace("s=['#include <stdlib.h>'","s=['#include <stdint.h>','#include <stddef.h>','#include <stdlib.h>'")
 s=s.replace('uintptr_t p=((uintptr_t*)a12)[1];','a12=workspace();uintptr_t p=((uintptr_t*)a12)[1];')
 s=s.replace('if product_calls%480==0:',f'if product_calls%{per_group}==0:').replace("product_calls//480",f'product_calls//{per_group}').replace('product_calls==48*480',f'product_calls==48*{per_group}')
 s=s.replace('product_calls==48*576','48*480<=product_calls<=48*576')
 s=s.replace("r={'scope':", "assert lib.finish_workspace()==1\nr={'scope':")
 (case/'validate_native.py').write_text(s)
 with(case/'native.log').open('w')as f:r=subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(case/'validate_native.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='/scratch/agustin/tmp/merlin-coarse-absolute-upper-20261006/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
 assert r.returncode==0,(bucket,r.returncode)
 print(bucket,'PASS',flush=True)
