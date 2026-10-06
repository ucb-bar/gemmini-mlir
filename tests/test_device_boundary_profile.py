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


def test_repeated_symbol_counts_are_required_when_manifest_declares_them():
    import pytest
    manifest=dict(BoundaryProfileTest.manifest,expected_symbol_calls={'0':2,'1':1})
    parse_profile(BoundaryProfileTest.valid,manifest)
    # Same total count, all symbols present, conserved intervals, wrong multiplicity.
    wrong=BoundaryProfileTest.valid.replace('PROFILE_CALL 2 0','PROFILE_CALL 2 1')
    with pytest.raises(ValueError,match='per-symbol'):
        parse_profile(wrong,manifest)


def test_segmented_leaf_profile_replaces_only_the_active_boundary(tmp_path):
    import hashlib, json, shutil, subprocess
    from pathlib import Path
    import pytest
    from mlir_oot.golden_device_profile import leaf_kernel_profile
    linker=shutil.which('ld.lld'); cc=shutil.which('cc')
    if not linker or not cc: pytest.skip('native link tools unavailable')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def bundle(name,kernel,adapter):
        directory=tmp_path/name;directory.mkdir();comps=[]
        for stem,body in [('kernel',f'void {kernel}(void*a,void*b,void*c){{}}'),
                          ('adapter',f'extern void {kernel}(void*,void*,void*);void {adapter}(void*a,void*b,void*c){{{kernel}(a,b,c);}}')]:
            source=directory/(stem+'.c');source.write_text(body);obj=directory/(stem+'.o')
            argv=[cc,'-O2','-c',str(source),'-o',str(obj)];subprocess.run(argv,check=True)
            comps.append({'compiler_argv':argv,'object_sha256':sha(obj)})
        obj=directory/(name+'.o');argv=[linker,'-r',str(directory/'kernel.o'),str(directory/'adapter.o'),'-o',str(obj)];subprocess.run(argv,check=True)
        return obj,comps,argv
    old,oldc,oldlink=bundle('requant','original_kernel','original_adapter')
    new,newc,newlink=bundle('segmented_inputs','accepted_kernel','accepted_adapter')
    merged=tmp_path/'merged.o';argv=[linker,'-r',str(old),str(new),'-o',str(merged)];subprocess.run(argv,check=True)
    source=tmp_path/'source.mlir';source.write_text('builtin.module {func.func private @original_adapter() func.func private @accepted_adapter() func.func @forward(){func.call @accepted_adapter() : () -> () func.return}}')
    route={'source_symbol':'original_adapter','accepted_symbol':'accepted_adapter','accepted_kernel':'accepted_kernel',
           'address':{'rows':3,'cols':5},'compilation':newc[0],'adapter_compilation':newc[1]}
    original_route={'symbol':'original_adapter','kernel':'original_kernel','compilation':oldc[0],'adapter_compilation':oldc[1]}
    original_manifest=old.parent/'requant.json'
    original_manifest.write_text(json.dumps({'schema':'gemmini_exact_captured_requant_bundle_v1',
        'object_sha256':sha(old),'routes':[original_route]}))
    original_graph={'linker_argv':oldlink,'object_sha256':sha(old),
                    'requant_manifest_sha256':sha(original_manifest)}
    catalog={'compilation':{'linker_argv':argv,'object_sha256':sha(merged),
             'original':original_graph,
             'segment':{'linker_argv':newlink,'object_sha256':sha(new)},
             'segmented_bindings':{'schema':'gemmini_segmented_input_bindings_v1','routes':[route]}},
             'source_snapshot':str(source),
             'segmented_input_acceptance':{'selected_source_snapshot_sha256':sha(source),
                'source_rewrites':[{'source_symbol':'original_adapter','accepted_symbol':'accepted_adapter','calls':1}]},
             'fused_requantizations':[dict(original_route, source_symbol='original_adapter',
                symbol='accepted_adapter',kernel='accepted_kernel',compilation=newc[0],adapter_compilation=newc[1])],
             'kernels':[]}
    work=tmp_path/'profile';work.mkdir()
    leaves,rows,receipts=leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
    assert leaves==[old.parent/'kernel.o',old.parent/'adapter.o',new.parent/'kernel.o',new.parent/'adapter.o']
    assert [r['symbol']for r in rows]==['accepted_kernel']
    assert rows[0]['source_symbol']=='original_adapter' and len(receipts)==2
    assert receipts[0]['original_manifest_sha256']==sha(original_manifest)
    parse_profile('PROFILE_SUM 10 4 6 1 0\nPROFILE_CALL 0 0 5 4\nPROFILE_TAIL 1\n',
                  {'boundaries':[{'id':0}],'expected_device_calls':1,'expected_symbol_calls':{'0':1}})
    saved=original_manifest.read_bytes()
    original_manifest.write_bytes(saved+b'\n')
    with pytest.raises(ValueError,match='manifest identity'):
        leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
    original_manifest.write_bytes(saved)
    active=catalog['fused_requantizations'][0]
    active['kernel']='original_kernel'
    with pytest.raises(ValueError,match='replaced primitive'):
        leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
    active['kernel']='accepted_kernel'
    rewrites=catalog['segmented_input_acceptance']['source_rewrites']
    rewrites[0]['calls']=2
    with pytest.raises(ValueError,match='source call coverage'):
        leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
    rewrites[0]['calls']=1
    source.write_text(source.read_text().replace('accepted_adapter() :','original_adapter() :'))
    with pytest.raises(ValueError,match='source identity'):
        leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
    catalog['segmented_input_acceptance']['selected_source_snapshot_sha256']=sha(source)
    with pytest.raises(ValueError,match='source call coverage'):
        leaf_kernel_profile(catalog,tmp_path,work,Path(linker).parent)
