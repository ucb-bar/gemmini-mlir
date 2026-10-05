import unittest
from mlir_oot.golden_device_profile import emit, parse_profile

class BoundaryProfileTest(unittest.TestCase):
    manifest = {'expected_device_calls':3,'boundaries':[{'id':0},{'id':1}]}
    valid = ('PROFILE_SUM 31 13 18 3 0\nPROFILE_CALL 0 0 10 4\n'
             'PROFILE_CALL 1 1 5 3\nPROFILE_CALL 2 0 2 6\nPROFILE_TAIL 1\n')
    def test_complete_intervals(self):
        self.assertEqual(parse_profile(self.valid,self.manifest)['host_gap_counter'],18)
    def test_missing_boundary_refused(self):
        with self.assertRaisesRegex(ValueError,'cover'):
            parse_profile(self.valid.replace('PROFILE_CALL 1 1','PROFILE_CALL 1 0'),self.manifest)
    def test_nonconserved_intervals_refused(self):
        with self.assertRaisesRegex(ValueError,'conserve'):
            parse_profile(self.valid.replace('PROFILE_TAIL 1','PROFILE_TAIL 2'),self.manifest)
    def test_overflow_refused(self):
        with self.assertRaisesRegex(ValueError,'cover'):
            parse_profile(self.valid.replace('31 13 18 3 0','31 13 18 3 1'),self.manifest)
    def test_unsupported_abi_refused(self):
        with self.assertRaises(ValueError):emit([('unsafe;name',3)])
        with self.assertRaises(ValueError):emit([('valid_name',5)])


def test_leaf_profile_reconstructs_and_checks_original_object_bytes(tmp_path):
    import hashlib,shutil,subprocess
    from pathlib import Path
    import pytest
    from mlir_oot.golden_device_profile import leaf_kernel_profile
    linker=shutil.which('ld.lld');cc=shutil.which('cc')
    if not linker or not cc:pytest.skip('native link tools unavailable')
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'kernel.c').write_text('void kernel(void*a,void*b,void*c){}')
    (bundle/'adapter.c').write_text('extern void kernel(void*,void*,void*); void adapter(void*a,void*b,void*c){kernel(a,b,c);}')
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    comps=[]
    for name in ('kernel','adapter'):
        obj=bundle/(name+'.o');argv=[cc,'-O2','-c',str(bundle/(name+'.c')),'-o',str(obj)];subprocess.run(argv,check=True)
        comps.append({'compiler_argv':argv,'object_sha256':sha(obj)})
    component=bundle/'requant.o';subprocess.run([linker,'-r',str(bundle/'kernel.o'),str(bundle/'adapter.o'),'-o',str(component)],check=True)
    top=tmp_path/'merged.o';argv=[linker,'-r',str(component),'-o',str(top)];subprocess.run(argv,check=True)
    catalog={'compilation':{'linker_argv':argv,'object_sha256':sha(top)},'fused_requantizations':[{'kernel':'kernel','compilation':comps[0],'adapter_compilation':comps[1]}],'kernels':[]}
    work=tmp_path/'profile';work.mkdir()
    leaves,rows,receipts=leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
    assert leaves==[bundle/'kernel.o',bundle/'adapter.o']
    assert rows[0]['symbol']=='kernel' and receipts[0]['sha256']==sha(component)
    (bundle/'kernel.o').write_bytes((bundle/'kernel.o').read_bytes()+b'changed')
    with pytest.raises(ValueError,match='identity mismatch'):leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
