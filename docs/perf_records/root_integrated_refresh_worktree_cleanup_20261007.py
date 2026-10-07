"""Archive and remove five reviewed, integrated, inactive delivery worktrees."""
from pathlib import Path
import hashlib
import json
import os
import pwd
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'out/artifacts/audits/integrated-refresh-cleanup-20261007/archive'
OUT.mkdir(exist_ok=False)
MAIN = 'eb15a85ce550ec63c27530d56af66893fe629766'
PLAN = [('merlin-existing-pr40-refresh-20261007', 'd50e092f3ecfc3af574caf56cc45eba40da9d74d', ['merlin/tests/ir/test_static_llvm_cfg.py', 'src/merlin/llvmlower/static_llvm_cfg.py']), ('merlin-existing-pr46-refresh-20261007', 'fa7c40f067c16520d6bc78bbd7d00070ab01e1ac', ['merlin/tests/dse/test_execution_boundaries.py', 'src/merlin/perf/execution_boundaries.py']), ('merlin-existing-pr47-refresh-20261007', '6e511428581373f950d2ac445decf182c610b355', ['merlin/tests/ir/test_bounded_rne_lanes.py', 'merlin/tests/ir/test_bounded_rne_maps.py', 'merlin/tests/ir/test_bounded_rne_packet_llvm.py', 'src/merlin/llvmlower/bounded_rne_lanes.py', 'src/merlin/llvmlower/bounded_rne_maps.py', 'src/merlin/llvmlower/bounded_rne_packet_llvm.py']), ('merlin-existing-pr50-refresh-20261007', 'e96d7068d8612353f656fd7c622ad661e022ee5a', ['merlin/tests/ir/test_quantized_affine_domain.py', 'merlin/tests/ir/test_typed_integer_domains.py', 'src/merlin/llvmlower/quantized_affine_domain.py', 'src/merlin/llvmlower/typed_integer_domains.py']), ('merlin-pr41-refresh-20261007', '347e3d71359fa2a932e1253b3e12bb0072fc5e38', ['merlin/tests/dse/test_address_locality.py', 'src/merlin/perf/address_locality.py']), ('merlin-pr43-refresh-20261007', '4f1387483be16dc029ee515c1a1595c6fb452243', ['merlin/tests/dse/test_depgraph.py', 'src/merlin/perf/depgraph.py']), ('merlin-pr44-refresh-20261007', '27676c182ff5eb55ad14b219e0e67f262dfe461e', ['merlin/tests/ir/test_immutable_llvm_base.py', 'src/merlin/llvmlower/immutable_llvm_base.py']), ('merlin-pr51-refresh-20261007', '85c1ba7ee64761dc244bb0db6bc59b3b008fafd9', ['merlin/tests/ir/test_observation_boundary.py', 'src/merlin/llvmlower/observation_boundary.py'])]

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def live_uses(path):
    findings = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            for link in [entry/'cwd', entry/'exe', *list((entry/'fd').iterdir())]:
                try:
                    target = os.readlink(link)
                except OSError:
                    continue
                if target == str(path) or target.startswith(str(path)+'/'):
                    findings.append(dict(pid=int(entry.name), reference=str(link), target=target))
        except (OSError, PermissionError):
            continue
    return findings

def inventory(path):
    regular, links = {}, {}
    for folder, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = [x for x in dirs if x != '.git']
        for name in dirs:
            child = Path(folder)/name
            owner = pwd.getpwuid(child.lstat().st_uid).pw_name
            assert 'jack' not in name.lower() and 'jack' not in owner.lower(), ('Forbidden directory; refuse before descent',child)

        for name in names + dirs:
            if name == '.git':
                continue
            item = Path(folder)/name
            relative = str(item.relative_to(path))
            if item.is_symlink():
                links[relative] = os.readlink(item)
            elif item.is_file():
                regular[relative] = dict(bytes=item.stat().st_size, sha256=sha(item))
    return regular, links

assert git('rev-parse', 'HEAD') == git('ls-remote', 'origin', 'refs/heads/main').split()[0] == MAIN
assert not git('status', '--porcelain')
manifest = dict(schema='integrated_delivery_worktree_archive_successors_v1', published_main=MAIN,
    remote_verified=True, scope='Complete owned source, ignored/untracked outputs and links are retained except Git administrative pointers. Old pin paths map to exact archived members. Shared Git objects, branches and primary stores remain.',
    prior_cleanup='/scratch/agustin/tmp/merlin-approved-topics-main-20261007/out/artifacts/audits/approved-topics-20261007/cleanup.json',
    agent_dependency_confirmation='All three current agents reference_parity/tiny_performance/firesim_recovery explicitly confirmed no live source/build dependencies on listed PR review worktrees. Active original source/target/runtime/qualification owners retained.',
    rows=[])
result = dict(schema='integrated_delivery_worktree_cleanup_v1', published_main=MAIN, removed=[], retained=[], archive_bytes=0, logical_source_bytes_removed=0)
for name, short_head, modules in PLAN:
    path = Path('/scratch/agustin/tmp')/name
    assert path.is_dir() and path != ROOT and (path/'.git').is_file()
    assert path.stat().st_uid == os.getuid()
    head = git('rev-parse', 'HEAD', cwd=path)
    assert head.startswith(short_head) and not git('status', '--porcelain', '--untracked-files=no', cwd=path)
    uses = live_uses(path)
    if uses:
        result['retained'].append(dict(path=str(path), reason='Live process references', uses=uses))
        continue
    # These production topics are on main; two now include explicitly reviewed
    # follow-on changes (prepared source policy and the corrected argv slot).
    diff = git('diff', head, MAIN, '--', *modules)
    assert not diff, (path, 'Reviewed topic code/tests differ from published main')
    diff_path = OUT/(name+'.superseding_main.patch')
    diff_path.write_text(diff+'\n')
    regular, links = inventory(path)
    archive = OUT/(name+'.tar.gz')
    refused = []
    def admit(member):
        if '.git' in Path(member.name).parts:
            return None
        if member.isdir():
            try:
                owner = pwd.getpwuid(member.uid).pw_name
            except KeyError:
                owner = ''
            if 'jack' in member.name.lower() or 'jack' in owner.lower():
                refused.append(member.name)
                return None
        return member
    with tarfile.open(archive, 'w:gz', compresslevel=6, dereference=False) as stream:
        stream.add(path, arcname=name, filter=admit)
    assert not refused, ('Forbidden directories; refusing removal', refused)
    archived_regular, archived_links = {}, {}
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            relative = str(Path(*Path(member.name).parts[1:]))
            if member.isfile():
                data = stream.extractfile(member).read()
                archived_regular[relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            elif member.issym():
                archived_links[relative] = member.linkname
    assert archived_regular == regular and archived_links == links
    files_manifest = OUT/(name+'.files.json')
    save(files_manifest, dict(original_root=str(path), files=regular, links=links))
    row = dict(path=str(path), head=head, archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        logical_source_bytes=sum(x['bytes'] for x in regular.values()), recovery_branch=git('symbolic-ref','--short','HEAD',cwd=path), files=len(regular), links=len(links), file_manifest=str(files_manifest), file_manifest_sha256=sha(files_manifest),
        successor_rule=f'{path}/RELATIVE -> {archive} member {name}/RELATIVE', published_main=MAIN,
        reviewed_production_modules=modules, superseding_main_diff=str(diff_path), superseding_main_diff_sha256=sha(diff_path))
    manifest['rows'].append(row)
    save(OUT/'worktree_archive_successors.json', manifest)
    assert not live_uses(path) and inventory(path) == (regular, links)
    assert git('rev-parse', 'HEAD', cwd=path) == head
    common = git('rev-parse', '--path-format=absolute', '--git-common-dir', cwd=path)
    argv = ['git', '--git-dir='+common, 'worktree', 'remove', '--force', str(path)]
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and not path.exists(), p.stderr
    result['removed'].append(dict(path=str(path), head=head, archive=str(archive), archive_sha256=row['archive_sha256'], files=row['files'], removal_argv=argv))
    result['archive_bytes'] += row['archive_bytes']
    result['logical_source_bytes_removed'] += row['logical_source_bytes']
    result['logical_bytes_reclaimed_net'] = result['logical_source_bytes_removed']-result['archive_bytes']
    result['disk_claim'] = 'Logical regular-file bytes minus retained compressed archives; not a df measurement.'
    save(OUT/'cleanup.json', result)
    print(json.dumps(dict(removed=name, logical_bytes_reclaimed_net=result['logical_bytes_reclaimed_net'])), flush=True)
save(OUT/'cleanup.json', result)
