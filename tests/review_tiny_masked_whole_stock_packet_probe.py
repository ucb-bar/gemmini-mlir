"""Close the normal masked whole composition and release one stock observation."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf
from review_rms4_fast_stock_packet_probe import digest


def review(args):
    if args.output.exists():
        raise ValueError("review requires fresh output")
    q=json.loads(args.packet.read_text())
    pins=q["pins"]|q["review_copy_pins"]
    if len(pins)!=474:
        raise ValueError("whole archive coverage differs")
    for path,expected in pins.items():
        if digest(Path(path))!=expected:
            raise ValueError("frozen artifact changed: "+path)
    candidate=Path(q["candidate"]["ELF"])
    control=Path(q["control"]["ELF"])
    reproduced=Path(q["control"]["reproduced"])
    if digest(control)!=q["control"]["ELF_SHA"] or digest(reproduced)!=digest(control) or digest(candidate)!=q["candidate"]["ELF_SHA"]:
        raise ValueError("candidate/control executable binding differs")
    audit=audit_elf(candidate.read_bytes())
    if audit["status"]!="pass" or audit["forbidden"] or audit["unknown"]:
        raise ValueError("final zeroFSM executable audit failed")
    adapter_path=Path(q["candidate"]["standard_adapter"])
    adapter=json.loads(adapter_path.read_text())
    build_path=Path(adapter["controlled_link_path"])
    build=json.loads(build_path.read_text())
    old_build_path=control.parent.parent/"build.json"
    old_build=json.loads(old_build_path.read_text())
    for path,expected in build["candidate_objects"].items():
        if digest(Path(path))!=expected:
            raise ValueError("changed linked object: "+path)
        if Path(path).name!="model.o" and old_build["candidate_objects"].get(path)!=expected:
            raise ValueError("unexpected nonmodel object change")
    if len(build["candidate_objects"])!=len(old_build["candidate_objects"]):
        raise ValueError("linked leaf coverage changed")
    old_model=next(Path(p) for p in old_build["candidate_objects"] if Path(p).name=="model.o")
    reproduced_model=reproduced.parent/"target/model.o"
    if digest(reproduced_model)!=digest(old_model):
        raise ValueError("normal control object does not reproduce champion")
    def options(argv):
        return ["MODEL_OBJECT" if Path(value).name=="model.o" else "OUTPUT" if i==len(argv)-1 else value for i,value in enumerate(argv)]
    if options(build["candidate_link_argv"])!=options(old_build["candidate_link_argv"]):
        raise ValueError("other controlled link options/leaves changed")
    if build["normal_source_masked_contractions"]!=22 or not build["all155devicebindings"] or build["table_bytes"]!=524288:
        raise ValueError("normal source or continuation resource coverage changed")
    reference=np.load(Path(adapter["reference_path"]))
    gold=np.load(Path(adapter["torch_golden_path"]))
    if reference.dtype!=np.float32 or reference.shape!=(1,8,32000) or reference.shape!=gold.shape:
        raise ValueError("original whole shape/type differs")
    raw_sha=hashlib.sha256(reference.astype('<f4',copy=False).tobytes()).hexdigest()
    if raw_sha!="ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3" or not np.allclose(reference,gold,atol=.03125,rtol=.02):
        raise ValueError("original compiled/Torch whole gate failed")
    strict_path=Path(adapter["source_qualification_path"])
    strict=json.loads(strict_path.read_text())
    console_path=Path(adapter["spike_console_path"])
    console=console_path.read_text()
    if strict["exit_code"]!=0 or strict["elf_sha256"]!=digest(candidate) or strict["nofsm_audit"]!=audit or digest(console_path)!=adapter["spike_console_sha256"]:
        raise ValueError("strict terminal/ELF/stdout binding differs")
    if re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([a-f0-9]{64})$',console,re.M)!=[("256000","1024000",raw_sha)] or re.findall(r'^METRIC memref_rank_mismatch (\d+)$',console,re.M)!=["0"] or len(re.findall(r'^DONE$',console,re.M))!=1:
        raise ValueError("whole original digest/DONE/rank contract failed")
    cycles=re.findall(r'^METRIC cycles (\d+)$',console,re.M)
    if cycles!=["130504904"]:
        raise ValueError("strict retirement proxy differs")
    result={"schema":"root_tiny_masked_whole_stock_release_v1","status":"RELEASED_ONE_NORMAL_WHOLE_STOCK_OBSERVATION","pins_reclosed":len(pins),"whole_original_outputs":256000,"raw_original_sha256":raw_sha,"Torch_gate":{"atol":.03125,"rtol":.02,"passed":True},"control_job":2056,"control_cycles":412672134,"control_elf_sha256":digest(control),"control_object_and_final_elf_reproduced":True,"unchanged_nonmodel_linked_objects":len(build["candidate_objects"])-1,"normal_typed_source_mask_count":22,"device_boundaries":155,"candidate":str(candidate),"candidate_elf_sha256":digest(candidate),"standard_adapter":str(adapter_path),"final_nofsm":audit,"strict_retired_instructions":130504904,"strict_is_not_hardware_cycles":True,"hardware_alias":"alveo_u250_firesim_gemmini_rocket_stock","hardware_config":"FireSimGemminiRocketConfig","allowed_runs":1,"requires":["Exact staged ELF/stock bitstream/runworkload identity.","Original full256000 digest, single DONE, rank mismatch0, standard reference adapter and originalTorch gate.","Real completeforward metric; one observation compared to actual2056, no section-to-whole attribution."],"whole_prediction":"UNKNOWN","default_promotion":False,"scope":"Normal typed mask observation plus unchanged source-exact continuation. Explicit documented fusionguard0 in BOTH historical-control arms only; main default unchanged. All original effects/accuracy gates retained. Capsule allocator differs from whole runtime and capsule16.38% is not a whole forecast.","pins":{str(p):digest(p) for p in (args.packet,adapter_path,build_path,old_build_path,strict_path,console_path,Path(__file__))}}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","pins_reclosed","whole_original_outputs","control_object_and_final_elf_reproduced","strict_retired_instructions")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    review(parser.parse_args())
