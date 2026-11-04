"""Explicit writer installation under complete source consumer observations.

A supplied numerical theorem must prove all retained integer/scale observations.
This binder validates live source identities and permissions before mutation; it
neither proves that theorem nor admits a target or inferred performance policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .closed_group_writer import ClosedGroupWriterContract, _pin, install_closed_group_writers
from .ordered_bf16_group_binding import _context_digest, verify_group_module_coverage
from .quantized_consumer_frontier import (
    QuantizedConsumerFrontier,
    find_quantized_consumer_frontiers,
    quantized_consumer_semantic_sha256,
    validate_quantized_consumer_frontier,
)


@dataclass(frozen=True)
class ConsumerObservationWitness:
    frontier: object
    consumer_semantic_sha256: str
    observed_types: tuple[str, ...]
    numerical_witness_sha256: str
    effect_witness_sha256: str
    fenv_policy: str


@dataclass(frozen=True)
class ConsumerObservationContract:
    """Explicit theorem identity for one complete source consumer grammar."""

    consumer_semantic_sha256: str
    observed_types: tuple[str, ...]
    numerical_witness_sha256: str
    effect_witness_sha256: str
    fenv_policy: str


class ConsumerObservedGroupPreparation:
    """Opt-in normal prepared-source callback with supplied complete proofs.

    Actual typed producer and consumer semantics select only supported paths.
    Unsupported live consumers retain source execution. Multiple applicable
    proofs for one producer refuse; this API never chooses between numerics.
    Provider arithmetic, source fallback and target qualification remain explicit
    external obligations. There is no default route or performance prediction.
    """

    def __init__(
        self, provider_contracts, consumer_contracts, *, source_preparation=None, reuse_private_workspaces=False
    ):
        from .ordered_bf16_group_binding import SourceExactGroupPreparation

        self.provider_contracts = tuple(provider_contracts)
        self.consumer_contracts = tuple(consumer_contracts)
        self.source_preparation = source_preparation or SourceExactGroupPreparation()
        if type(reuse_private_workspaces) is not bool:
            raise ValueError("private workspace storage reuse permission must be explicit boolean")
        self.reuse_private_workspaces = reuse_private_workspaces
        self.receipt = None
        producer_keys = set()
        for contract in self.provider_contracts:
            if (
                not isinstance(contract, ClosedGroupWriterContract)
                or contract.numerical_policy != "exact_consumer_observations"
                or not _pin(contract.source_semantic_sha256)
            ):
                raise ValueError("explicit complete producer observation contract required")
            if contract.source_semantic_sha256 in producer_keys:
                raise ValueError("ambiguous complete producer contracts")
            producer_keys.add(contract.source_semantic_sha256)
        consumer_keys = set()
        for contract in self.consumer_contracts:
            if (
                not isinstance(contract, ConsumerObservationContract)
                or any(
                    not _pin(value)
                    for value in (
                        contract.consumer_semantic_sha256,
                        contract.numerical_witness_sha256,
                        contract.effect_witness_sha256,
                    )
                )
                or not contract.observed_types
                or contract.fenv_policy != "rne_returned_values"
            ):
                raise ValueError("explicit complete consumer observation contract required")
            if contract.consumer_semantic_sha256 in consumer_keys:
                raise ValueError("ambiguous complete consumer contracts")
            consumer_keys.add(contract.consumer_semantic_sha256)

    def install(self, module, source_receipt):
        """Validate live coverage, discover consumers and install supported proofs."""
        from xdsl.dialects import func

        from .closed_group_writer import source_function_semantic_sha256

        verify_group_module_coverage(module, source_receipt)
        records = {record["symbol"]: record for record in source_receipt["records"]}
        functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
        providers = {contract.source_semantic_sha256: contract for contract in self.provider_contracts}
        candidates = {
            symbol: providers[semantic]
            for symbol in records
            if (semantic := source_function_semantic_sha256(functions[symbol])) in providers
        }
        calls = [op for op in module.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data in records]
        frontiers = find_quantized_consumer_frontiers(module, source_values=tuple(call.results[0] for call in calls))
        proofs = {contract.consumer_semantic_sha256: contract for contract in self.consumer_contracts}
        selected, witnesses = {}, []
        for frontier in frontiers:
            if not frontier.source_uses_closed:
                continue
            semantic = quantized_consumer_semantic_sha256(frontier)
            proof = proofs.get(semantic)
            if proof is None or proof.observed_types != tuple(
                str(value.type) for value in frontier.observation_outputs
            ):
                continue
            matched = {}
            for value in frontier.source_values:
                symbol = value.owner.callee.root_reference.data
                contract = candidates.get(symbol)
                if (
                    contract is not None
                    and contract.numerical_witness_sha256 == proof.numerical_witness_sha256
                    and contract.fenv_policy == proof.fenv_policy
                ):
                    if symbol in selected:
                        raise ValueError("ambiguous live consumer coverage for source producer")
                    matched[symbol] = contract
            if matched:
                selected.update(matched)
                witnesses.append(
                    ConsumerObservationWitness(
                        frontier,
                        semantic,
                        proof.observed_types,
                        proof.numerical_witness_sha256,
                        proof.effect_witness_sha256,
                        proof.fenv_policy,
                    )
                )
        if self.reuse_private_workspaces:
            from .private_workspace_pool import validate_source_workspace_pool

            validate_source_workspace_pool(module, selected)
        result = install_consumer_observed_group_writers(module, source_receipt, selected, consumer_witnesses=witnesses)
        if self.reuse_private_workspaces:
            from .private_workspace_pool import pool_private_writer_workspaces

            result["private_workspace_pool"] = pool_private_writer_workspaces(module, result["fresh_writer_report"])
        return dict(
            result,
            discovered_integer_frontiers=len(frontiers),
            compatible_source_producers=len(candidates),
            source_cpu_groups=len(records) - len(selected),
        )

    def __call__(self, source_path, workdir):
        import hashlib
        import json
        from pathlib import Path

        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.xdsl_dialects._common import text

        workdir = Path(workdir)
        exact_path = self.source_preparation(source_path, workdir / "source_control")
        source_receipt = self.source_preparation.receipt
        module = parse_mlir_text(Path(exact_path).read_text())
        installed = self.install(module, source_receipt)
        module.verify()
        workdir.mkdir(parents=True, exist_ok=True)
        selected = workdir / "consumer_observed_groups.mlir"
        selected.write_text(text(module))
        self.receipt = dict(
            installed,
            selected_path=str(selected.resolve()),
            selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
            source_control_receipt=source_receipt,
        )
        (workdir / "consumer_observed_groups.json").write_text(json.dumps(self.receipt, indent=2) + "\n")
        return selected


def install_consumer_observed_group_writers(module, source_receipt, contracts, *, consumer_witnesses):
    """Check every selected producer's current consumer before writer binding.

    Producers may return observationally equivalent BF16 values only with an
    explicitly supplied complete consumer numerical/effect witness. The original
    typed consumer and all escaping observations remain in the caller. Every
    selected actual source call must be covered exactly once, and source/view
    residual escapes refuse. Unselected source groups remain ordinary source.
    """
    from xdsl.dialects import func
    from xdsl.ir import OpResult

    verify_group_module_coverage(module, source_receipt)
    records = {record["symbol"]: record for record in source_receipt["records"]}
    selected_calls = {
        operation: operation.callee.root_reference.data
        for operation in module.walk()
        if isinstance(operation, func.CallOp) and operation.callee.root_reference.data in contracts
    }
    if set(contracts) - records.keys():
        raise ValueError("observed writer is not a retained source group")
    if any(
        not isinstance(contract, ClosedGroupWriterContract)
        or contract.numerical_policy != "exact_consumer_observations"
        for contract in contracts.values()
    ):
        raise ValueError("explicit exact consumer observation policy required")
    all_operations = set(module.walk())
    covered = set()
    consumers = []
    for supplied in consumer_witnesses:
        if not isinstance(supplied, ConsumerObservationWitness):
            raise ValueError("explicit live consumer witness required")
        frontier = supplied.frontier
        if not isinstance(frontier, QuantizedConsumerFrontier):
            raise ValueError("explicit live typed consumer frontier required")
        validate_quantized_consumer_frontier(frontier)
        if (
            not frontier.source_uses_closed
            or any(operation not in all_operations for operation in frontier.operations)
            or any(
                not isinstance(value, OpResult) or value.owner not in all_operations for value in frontier.source_values
            )
        ):
            raise ValueError("consumer has a foreign source or unquantized escape")
        if (
            not _pin(supplied.consumer_semantic_sha256)
            or supplied.consumer_semantic_sha256 != quantized_consumer_semantic_sha256(frontier)
            or supplied.observed_types != tuple(str(value.type) for value in frontier.observation_outputs)
            or not _pin(supplied.numerical_witness_sha256)
            or not _pin(supplied.effect_witness_sha256)
            or supplied.fenv_policy != "rne_returned_values"
        ):
            raise ValueError("complete consumer numeric, effect and observation witness required")
        producers = []
        for value in frontier.source_values:
            call = value.owner
            if not isinstance(call, func.CallOp) or call.callee.root_reference.data not in records:
                raise ValueError("consumer source is not an actual retained group call")
            if call not in selected_calls:
                continue
            symbol = selected_calls[call]
            if call in covered:
                raise ValueError("selected source call has duplicate consumer coverage")
            contract = contracts[symbol]
            if (
                contract.numerical_witness_sha256 != supplied.numerical_witness_sha256
                or contract.fenv_policy != supplied.fenv_policy
            ):
                raise ValueError("writer proof does not cover this consumer obligation")
            covered.add(call)
            producers.append(symbol)
        if not producers:
            raise ValueError("consumer witness covers no selected source call")
        consumers.append(
            dict(
                consumer_semantic_sha256=supplied.consumer_semantic_sha256,
                consumer_context_sha256=_context_digest(frontier.integer_output.owner),
                selected_source_symbols=producers,
                observed_types=list(supplied.observed_types),
                numerical_witness_sha256=supplied.numerical_witness_sha256,
                effect_witness_sha256=supplied.effect_witness_sha256,
                fenv_policy=supplied.fenv_policy,
                unquantized_source_escapes=0,
            )
        )
    if covered != set(selected_calls):
        raise ValueError("selected source call lacks complete consumer coverage")
    # All consumer source checks precede the existing source/ABI writer validation
    # and any body replacement. The existing writer binder checks full writes,
    # input preservation, borrowed lifetime, source fallback and implementation.
    result = install_closed_group_writers(module, source_receipt, contracts)
    return dict(
        result,
        consumer_obligations=consumers,
        consumer_observation_calls=len(covered),
        numerical_policy="exact_consumer_observations",
        scope="Explicit supplied complete consumer proof; original quantization remains. No numeric theorem, target coverage or timing admission inferred.",
    )
