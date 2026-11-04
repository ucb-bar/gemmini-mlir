"""Consume live Merlin graph contracts before whole-source device discovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def prepare_host_layout(source: str, contract, workdir: str | Path):
    """Apply the existing checked transform without serializing permissions.

    The caller supplies the original typed contract or live software policy.
    Saved reports are observations only and are never accepted as contracts.
    """
    from merlin.llvmlower.graph_optimizations import GraphOptimizationContract, optimize_host_graph
    from merlin.llvmlower.host_graph_policy import HostGraphPolicy

    if type(contract) not in (GraphOptimizationContract, HostGraphPolicy):
        raise TypeError('host layout requires an original typed Merlin graph contract or live policy')
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    if type(contract) is HostGraphPolicy:
        contract.revalidate()
        contract.retain(work)
    else:
        contract.validate(source)
    selected, report = optimize_host_graph(source, contract)
    source_path, output_path = work / 'input.mlir', work / 'prepared.mlir'
    source_path.write_text(source)
    output_path.write_text(selected)
    receipt = dict(schema='submission.host_graph_preparation.v1',
        binding_stage='before_device_region_discovery',
        source=dict(path=str(source_path.resolve()), sha256=hashlib.sha256(source.encode()).hexdigest()),
        output=dict(path=str(output_path.resolve()), sha256=hashlib.sha256(selected.encode()).hexdigest()),
        changed=source != selected, graph_optimization=report,
        scope='checked source transformation; no runtime or performance qualification')
    (work / 'preparation.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    return selected, receipt


def revalidate_host_layout(source: str, contract):
    """Recheck the live source binding or software policy after compilation."""
    from merlin.llvmlower.graph_optimizations import GraphOptimizationContract
    from merlin.llvmlower.host_graph_policy import HostGraphPolicy
    if type(contract) is GraphOptimizationContract:
        contract.validate(source)
    elif type(contract) is HostGraphPolicy:
        contract.revalidate()
    else:
        raise TypeError('host layout requires the original live typed contract')
