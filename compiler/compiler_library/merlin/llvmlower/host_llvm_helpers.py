"""Explicit immutable helper IR linkage inside normal host code generation.

Upstream LLVM verifies and links the supplied modules, then performs only its
always-inline pass. Required scalar helpers must have matching declared types,
actual original references and no remaining address/call references afterwards.
An annotation or object membership never proves inlining. Semantic equivalence,
numeric effects and helper source/ABI qualification remain caller obligations.
Empty selection preserves the original file and performs no compiler work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.proc import run_checked

from .compilation_recipe import CompilationRecipe
from .late_quant_rne import _functions, _identity, _tokens


@dataclass(frozen=True)
class HelperIR:
    path: Path
    sha256: str

    def validate(self) -> Path:
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("complete lowercase helper digest required")
        path = Path(self.path).resolve(strict=True)
        if not path.is_file() or sha256_file(path) != self.sha256:
            raise ValueError("helper IR file is absent, unsupported or changed")
        return path


def _scalar_type(token):
    return token in {"void", "ptr", "half", "bfloat", "float", "double"} or (
        token.startswith("i") and token[1:].isdecimal() and int(token[1:]) > 0
    )


def _headers(source):
    """Required helper ABI subset; LLVM verifies the complete module grammar."""
    tokens = _tokens(source)
    headers, positions = {}, set()
    for index, token in enumerate(tokens):
        if token.text not in {"declare", "define"}:
            continue
        end = index + 1
        while end < len(tokens) and not tokens[end].text.startswith("@"):
            end += 1
        if end + 1 >= len(tokens) or tokens[end + 1].text != "(":
            raise ValueError("unresolved LLVM function header")
        name = _identity(tokens[end].text)
        positions.add(end)
        preamble = [item.text for item in tokens[index + 1 : end]]
        return_type = preamble[-1] if preamble else ""
        default_cc = not any(item == "cc" or (item.endswith("cc") and item != "ccc") for item in preamble)
        cursor, depth, arguments, segment = end + 2, 1, [], []
        while cursor < len(tokens) and depth:
            value = tokens[cursor].text
            if value == ")" and depth == 1:
                if segment:
                    arguments.append(segment)
                depth = 0
                break
            if value == "," and depth == 1:
                arguments.append(segment)
                segment = []
            else:
                segment.append(value)
                depth += (value == "(") - (value == ")")
            cursor += 1
        if depth:
            raise ValueError("unresolved LLVM function arguments")
        unsupported_abi = {
            "addrspace",
            "byval",
            "byref",
            "sret",
            "inalloca",
            "preallocated",
            "swiftself",
            "swifterror",
            "nest",
        }
        extension_attributes = {"signext", "zeroext", "inreg"}
        scalar = (
            _scalar_type(return_type)
            and not unsupported_abi.intersection(preamble)
            and all(
                part and _scalar_type(part[0]) and part[0] != "void" and not unsupported_abi.intersection(part)
                for part in arguments
            )
        )
        signature = (
            (
                (return_type, tuple(sorted(extension_attributes.intersection(preamble)))),
                tuple((part[0], tuple(sorted(extension_attributes.intersection(part)))) for part in arguments),
            )
            if scalar and default_cc
            else None
        )
        if name in headers:
            raise ValueError("duplicate LLVM function header: " + name)
        headers[name] = {
            "signature": signature,
            "defined": token.text == "define",
            "public": token.text == "define" and not {"private", "internal"}.intersection(preamble),
        }
    return tokens, headers, positions


def _remaining_references(source, names):
    tokens, _headers_by_name, positions = _headers(source)
    return [
        {"symbol": _identity(token.text), "offset": token.start}
        for index, token in enumerate(tokens)
        if index not in positions and token.text.startswith("@") and _identity(token.text) in names
    ]


def _module_context(source):
    tokens = _tokens(source)
    context = {}
    for index, token in enumerate(tokens[:-3]):
        if token.text != "target" or tokens[index + 1].text not in {"triple", "datalayout"}:
            continue
        field = tokens[index + 1].text
        value = tokens[index + 3].text
        if tokens[index + 2].text != "=" or not value.startswith('"') or field in context:
            raise ValueError("unresolved or duplicate explicit LLVM module context")
        context[field] = _identity(value)
    return context


def link_and_inline(source: Path, workdir: Path, *, helpers=(), required_inlined_symbols=(), timeout=180):
    """Retain completed helper linkage and exact call-removal evidence.

    Helper files are already compiled by their caller, with its actual host
    flags and target layout. This utility selects neither flags nor helper
    semantics. Matching scalar/pointer signatures and extension attributes use
    the default calling convention; unsupported pointer ABIs, varargs, aggregates
    and nondefault address spaces refuse. All shared declarations are checked;
    opaque-pointer linker type adaptation does not establish ABI equivalence.
    LLVM rejects incompatible module linkage; original public functions survive.
    """
    from .toolchain import llvm_link, llvm_opt

    source, workdir = Path(source), Path(workdir)
    helpers, required = tuple(helpers), tuple(required_inlined_symbols)
    if not source.is_file():
        raise ValueError("actual lowered host IR file required")
    if not helpers:
        if required:
            raise ValueError("required helpers have no selected IR definitions")
        return source
    if any(not isinstance(helper, HelperIR) for helper in helpers):
        raise ValueError("immutable typed helper bindings required")
    if (
        not required
        or any(not isinstance(name, str) or not name for name in required)
        or len(set(required)) != len(required)
    ):
        raise ValueError("distinct required helper symbols must be explicit")
    paths = tuple(helper.validate() for helper in helpers)
    if len(set(paths)) != len(paths) or source.resolve() in paths:
        raise ValueError("helper inputs must be distinct from each other and the host module")
    original = source.read_bytes()
    _, source_headers, _ = _headers(original.decode())
    contexts = [_module_context(original.decode())]
    definitions = {}
    shared_headers = dict(source_headers)
    for path in paths:
        _, headers, _ = _headers(path.read_text())
        contexts.append(_module_context(path.read_text()))
        for name, row in headers.items():
            if name in shared_headers:
                previous = shared_headers[name]
                if previous["signature"] is None or previous["signature"] != row["signature"]:
                    raise ValueError(
                        "shared function types/calling convention/extension ABI differ or are unsupported: " + name
                    )
            else:
                shared_headers[name] = row
            if row["defined"]:
                if source_headers.get(name, {}).get("defined"):
                    raise ValueError("selected helper would replace an original function definition: " + name)
                if name in definitions:
                    raise ValueError("multiple selected helper definitions: " + name)
                definitions[name] = row
    if any(len({context[field] for context in contexts if field in context}) > 1 for field in ("triple", "datalayout")):
        raise ValueError("selected modules have conflicting explicit target layout/triple")
    for name in required:
        declared, defined = source_headers.get(name), definitions.get(name)
        if not declared or declared["defined"] or not defined or declared["signature"] is None:
            raise ValueError("required helper lacks a supported original declaration/selected definition: " + name)
        if declared["signature"] != defined["signature"]:
            raise ValueError("required helper source and definition types/calling convention differ: " + name)
        if not any(
            token.text.startswith("@") and _identity(token.text) == name
            for body in _functions(_tokens(original.decode()))
            for token in body
        ):
            raise ValueError("required helper has no actual source caller: " + name)
    if workdir.exists() and (not workdir.is_dir() or any(workdir.iterdir())):
        raise ValueError("helper linkage needs a fresh owned workdir")
    workdir.mkdir(parents=True, exist_ok=True)
    recipe = CompilationRecipe(workdir, producer=Path(__file__))
    recipe.bind_preparation("host_source", source)
    for index, path in enumerate(paths):
        recipe.bind_preparation("helper_" + str(index), path)

    def run(argv, inputs, output):
        recipe.run(argv, runner=lambda args: run_checked(args, timeout=timeout), inputs=inputs, output=output)
        for helper in helpers:
            helper.validate()
        if source.read_bytes() != original:
            raise ValueError("helper compilation modified the original host IR")

    verified = workdir / "host.bc"
    run([llvm_opt(), "-passes=verify", source, "-o", verified], [source], verified)
    modules = [verified]
    for index, path in enumerate(paths):
        verified_helper = workdir / ("helper_" + str(index) + ".bc")
        run([llvm_opt(), "-passes=verify", path, "-o", verified_helper], [path], verified_helper)
        modules.append(verified_helper)
    linked = workdir / "linked.bc"
    run([llvm_link(), *modules, "-o", linked], modules, linked)
    selected = workdir / "model.ll"
    run([llvm_opt(), "-S", "-passes=always-inline,verify", linked, "-o", selected], [linked], selected)
    remaining = _remaining_references(selected.read_text(), required)
    if remaining:
        raise ValueError("required helper still has a call/address reference after actual inlining: " + str(remaining))
    _, selected_headers, _ = _headers(selected.read_text())
    public = {name for name, row in source_headers.items() if row["public"]}
    if not public <= {name for name, row in selected_headers.items() if row["defined"]}:
        raise ValueError("helper linking removed an original public function")
    recipe.record.update(
        required_inlined_symbols=list(required),
        remaining_helper_references=remaining,
        original_public_functions=sorted(public),
        explicit_module_contexts=contexts,
        scope=(
            "explicit immutable LLVM helper linkage and actual inlining; "
            "numeric/ABI effects and later object/link closure caller-owned"
        ),
    )
    recipe.completed_product(selected, kind="llvm_ir")
    return selected


def merlin_host_llvm_transform(*, helpers=(), required_inlined_symbols=(), timeout=180):
    """Use existing normal host transform callback without changing its defaults."""
    helpers, required = tuple(helpers), tuple(required_inlined_symbols)

    def transform(source, workdir):
        return link_and_inline(source, workdir, helpers=helpers, required_inlined_symbols=required, timeout=timeout)

    return transform
