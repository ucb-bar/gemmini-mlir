"""Check actual borrowed descriptors using Merlin's existing typed lexer.

This is an emitted ABI witness. Source element semantics and fresh allocation
lifetime are established by the separately pinned typed acceptance contracts.
It does not infer alias safety, physical DRAM traffic or cycle savings.
"""

import argparse
import hashlib
import json
from math import prod
from pathlib import Path

from merlin.llvmlower.late_quant_rne import _tokens


def calls(tokens, name):
    result = []
    for index, token in enumerate(tokens):
        if token.text != "@" + name or index < 2:
            continue
        if [v.text for v in tokens[index - 2 : index]] != ["call", "void"]:
            continue
        if tokens[index + 1].text != "(":
            raise ValueError("expected emitted call argument list")
        arguments, current = [], []
        cursor = index + 2
        while tokens[cursor].text != ")":
            word = tokens[cursor].text
            if word == ",":
                arguments.append(current)
                current = []
            else:
                if word in "()[]{}<>":
                    raise ValueError("nested call operands require another ABI witness")
                current.append(word)
            cursor += 1
        arguments.append(current)
        if any(len(arg) != 2 or arg[0] not in ("ptr", "i64") for arg in arguments):
            raise ValueError("unsupported emitted ranked descriptor operands")
        result.append({"arguments": arguments, "source_offset": token.start})
    return result


def pin(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--host-llvm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    tokens = _tokens(args.host_llvm.read_text())
    routes = {route["symbol"]: route for route in catalog["fused_requantizations"]}
    records = []
    for source in catalog["segmented_input_acceptance"]["source_rewrites"]:
        route = routes[source["accepted_symbol"]]
        shape = route["segmented_input"]["owner_shape"]
        rank, width = len(shape), 3 + 2 * len(shape)
        consumers = calls(tokens, source["accepted_symbol"])
        producers = calls(tokens, source["owner_producers"][0])
        if source["calls"] != 1 or len(consumers) != 1 or len(producers) != 1:
            raise ValueError(
                "this emitted witness requires one unique producer/consumer call"
            )
        incoming = consumers[0]["arguments"][:width]
        outgoing = producers[0]["arguments"][-width:]
        if incoming != outgoing:
            raise ValueError(
                "accepted input is not the actual producer result descriptor"
            )
        if (
            source["address"]["dtype"] != "i8"
            or source["address"]["element_bytes"] != 1
        ):
            raise ValueError(
                "actual borrowed allocation must preserve physical i8 storage"
            )
        dense = [prod(shape[i + 1 :]) for i in range(rank)]
        constants = [0, *shape, *dense]
        if incoming[2:] != [["i64", str(value)] for value in constants]:
            raise ValueError("actual input descriptor differs from dense typed owner")
        if calls(tokens, source["source_symbol"]):
            raise ValueError("original materialized consumer still executes")
        records.append(
            {
                "accepted_symbol": source["accepted_symbol"],
                "owner_producer": source["owner_producers"][0],
                "actual_descriptor": incoming,
                "producer_call_offset": producers[0]["source_offset"],
                "consumer_call_offset": consumers[0]["source_offset"],
                "same_actual_producer_result_descriptor": True,
                "materialized_original_consumer_calls": 0,
                "avoided_logical_temporary_bytes": source["address"]["rows"]
                * source["address"]["cols"],
            }
        )
    result = {
        "schema": "segmented_input_emitted_host_descriptor_closure_v1",
        "pins": {
            "catalog": pin(args.catalog),
            "host_llvm": pin(args.host_llvm),
            "driver": pin(Path(__file__)),
        },
        "routes": records,
        "scope": "Actual consumer descriptor exactly equals actual producer output descriptor; original materialized consumer no longer executes. Typed source/lifetime proofs remain separately bound.",
        "physical_dram_traffic": "UNKNOWN",
        "cycles": "UNKNOWN",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("EMITTED_BORROWED_DESCRIPTOR_CLOSED", len(records))


if __name__ == "__main__":
    main()
