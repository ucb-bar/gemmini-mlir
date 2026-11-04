"""Bytes a body of results is attributed to, which no longer exist and cannot be reproduced.

A separate module from :mod:`merlin.common.provenance` because it answers a different question. That
one asks "is this checkout the declared revision" and expects an answer either way. This one is for the
case where there is no answer to be had and there never will be again, and the design follows from
refusing to let that case borrow either of the ordinary answers.

THE MEASURED CASE. A pinned simulator-compiler binary was rebuilt in place, and by the time anyone
noticed, 244 artifacts recorded the old digest as the tool that produced them. Three things can be done
with that, and two are worse than the loss:

* repoint the declaration at the bytes now on disk -- every one of those artifacts keeps asserting a
  digest nothing on the host has, while the registry says the pin is fine. The attribution is then not
  broken, it is silently WRONG, which is the failure the pin registry exists to prevent;
* delete the declaration -- the citations dangle with no account of what they point at;
* record the loss: what was lost, when, why it cannot be rebuilt, and what DOES still verify.

A loss record CLAIMS its digest: the pin and artifact loaders are cross-checked against it, so nothing in
the registry can re-declare those bytes as its own. And :func:`lost_for_digest` makes a citation
resolvable -- the artifacts citing lost bytes never re-hash them, so they do not fail and never will, and
asking is the only way anyone finds out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "LOST_REQUIRED_FIELDS",
    "UNRECOVERABLE",
    "LostArtifact",
    "claimed_digests",
    "load_lost",
    "lost_for_digest",
    "parse_lost",
]

#: The state of a declared artifact whose bytes no longer exist anywhere and cannot be rebuilt. Not
#: spelled as a missing digest or an empty string: "gone, and here is the account" is a different fact
#: from "not yet recorded" (an empty ``Artifact.digest``) and from "present but wrong" (a digest mismatch).
UNRECOVERABLE = "UNRECOVERABLE"

#: Fields a loss record must state. Each is a question a reader of a dangling citation asks.
LOST_REQUIRED_FIELDS: tuple[str, ...] = ("digest", "what", "lost_on", "why_unrecoverable", "still_verifies")


@dataclass(frozen=True)
class LostArtifact:
    """One set of bytes recorded as gone, with the account of how.

    ``superseded_by`` records what occupies the path NOW. It is bookkeeping about the path, never a
    substitute for the lost bytes, and nothing reads it as one.
    """

    name: str
    digest: str
    what: str
    lost_on: str
    why_unrecoverable: str
    still_verifies: tuple[str, ...] = ()
    path: str = ""
    root_env: str = ""
    bytes_declared: int | None = None
    discovered_on: str = ""
    cause: str = ""
    superseded_by: str = ""
    cited_by: str = ""
    do_not: tuple[str, ...] = ()
    #: What now stands as the witness for the results that cited the lost bytes, and whether it is a
    #: WEAKER attestation than the one it replaces. A decision someone made, so it is a field a reader
    #: can find and disagree with, not a remark in ``notes``.
    attestation_decision: str = ""
    notes: str = ""
    state: str = UNRECOVERABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.name,
            "state": self.state,
            "digest": self.digest,
            "path": self.path,
            "what": self.what,
            "lost_on": self.lost_on,
            "discovered_on": self.discovered_on,
            "cause": self.cause,
            "why_unrecoverable": self.why_unrecoverable,
            "superseded_by": self.superseded_by,
            "cited_by": self.cited_by,
            "still_verifies": list(self.still_verifies),
            "do_not": list(self.do_not),
            "attestation_decision": self.attestation_decision,
            "bytes_declared": self.bytes_declared,
        }


#: Parsed at most once per (file, mtime, size), the same key as ``provenance.load_pins``: the loss
#: records are read on every pin and artifact load (the cross-check), and a grade performs thousands.
_LOST_MEMO: dict[tuple[str, int, int], dict[str, LostArtifact]] = {}


def load_lost(path: str | Path | None = None) -> dict[str, LostArtifact]:
    """Every declared UNRECOVERABLE artifact. An empty mapping when the registry declares none."""
    import yaml

    from .provenance import PinsError, pins_path

    p = Path(path) if path is not None else pins_path()
    if not p.is_file():
        raise PinsError(f"no pin registry at {p}")
    try:
        st = p.stat()
        memo_key: tuple[str, int, int] | None = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:  # unstattable: parse it, and do not remember what we cannot key
        memo_key = None
    if memo_key is not None and memo_key in _LOST_MEMO:
        return dict(_LOST_MEMO[memo_key])
    out = parse_lost(yaml.safe_load(p.read_text(encoding="utf-8")) or {}, p)
    if memo_key is not None:
        _LOST_MEMO[memo_key] = dict(out)
    return out


def _digest(src: Path, name: str, field_name: str, raw: Any) -> str:
    from .provenance import PinsError

    if not isinstance(raw, str):
        raise PinsError(
            f"{src}: lost artifact {name!r} {field_name} must be a quoted sha256 string; YAML read "
            f"{type(raw).__name__}, which loses leading zeros"
        )
    digest = raw.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PinsError(
            f"{src}: lost artifact {name!r} {field_name} {raw!r} is not a 64-character sha256; the digest "
            "IS the thing being recorded, so a partial one records nothing"
        )
    return digest


def parse_lost(document: Any, src: Path) -> dict[str, LostArtifact]:
    """The loss records out of an ALREADY-PARSED registry document.

    Split from :func:`load_lost` so the pin and artifact loaders cross-check against the document THEY
    parsed: re-reading would double the parses of a file loaded thousands of times per grade, and would
    let a loader and its own cross-check see two different versions of the file.
    """
    from .provenance import PinsError

    if not isinstance(document, dict):
        raise PinsError(f"{src}: loss registry must be a mapping")
    entries = document.get("lost_artifacts", {})
    if not isinstance(entries, dict):
        raise PinsError(f"{src}: 'lost_artifacts' must be a mapping of name -> declaration")
    out: dict[str, LostArtifact] = {}
    seen: dict[str, str] = {}
    for name, body in entries.items():
        if not isinstance(name, str) or not name.strip():
            raise PinsError(f"{src}: lost artifact name must be a nonempty string")
        if not isinstance(body, dict):
            raise PinsError(f"{src}: lost artifact {name!r} must be a mapping")
        state = str(body.get("state") or UNRECOVERABLE)
        if state != UNRECOVERABLE:
            raise PinsError(
                f"{src}: lost artifact {name!r} declares state {state!r}; this section records only "
                f"{UNRECOVERABLE!r}. Something recoverable belongs in `artifacts:` with its digest."
            )
        missing = [f for f in LOST_REQUIRED_FIELDS if not body.get(f)]
        if missing:
            raise PinsError(
                f"{src}: lost artifact {name!r} does not state {missing}; a loss record that does not say "
                "what was lost, when, why it cannot be rebuilt and what still verifies leaves a reader of "
                "a dangling citation exactly where they started"
            )
        for field in ("what", "lost_on", "why_unrecoverable"):
            if not isinstance(body[field], str) or not body[field].strip():
                raise PinsError(f"{src}: lost artifact {name!r} {field} must be a nonempty string")
        for field in ("still_verifies", "do_not"):
            values = body.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise PinsError(f"{src}: lost artifact {name!r} {field} must be a list of nonempty strings")
        digest = _digest(src, str(name), "digest", body["digest"])
        if digest in seen:
            raise PinsError(
                f"{src}: lost artifacts {seen[digest]!r} and {name!r} both claim digest {digest[:12]}; "
                "one set of bytes is lost once"
            )
        seen[digest] = str(name)
        superseded = (
            _digest(src, str(name), "superseded_by", body["superseded_by"]) if body.get("superseded_by") else ""
        )
        if superseded == digest:
            raise PinsError(
                f"{src}: lost artifact {name!r} declares superseded_by equal to its own lost digest, which "
                "would say the bytes replaced themselves"
            )
        declared = body.get("bytes_declared")
        if declared is not None and (isinstance(declared, bool) or not isinstance(declared, int) or declared < 0):
            raise PinsError(f"{src}: lost artifact {name!r} bytes_declared must be a nonnegative integer")
        out[str(name)] = LostArtifact(
            name=str(name),
            digest=digest,
            what=str(body["what"]),
            lost_on=str(body["lost_on"]),
            why_unrecoverable=str(body["why_unrecoverable"]),
            still_verifies=tuple(str(s) for s in body["still_verifies"]),
            path=str(body.get("path") or ""),
            root_env=str(body.get("root_env") or ""),
            bytes_declared=int(declared) if declared is not None else None,
            discovered_on=str(body.get("discovered_on") or ""),
            cause=str(body.get("cause") or ""),
            superseded_by=superseded,
            cited_by=str(body.get("cited_by") or ""),
            do_not=tuple(str(d) for d in (body.get("do_not") or ())),
            attestation_decision=str(body.get("attestation_decision") or ""),
            notes=str(body.get("notes") or ""),
        )
    return out


def claimed_digests(path: str | Path | None = None, *, document: Any = None) -> dict[str, LostArtifact]:
    """``{digest: record}`` for every set of bytes a loss record claims. Used by the loaders' checks.

    ``document`` is an already-parsed registry: pass it when the caller has one, so the cross-check
    reads the same bytes the caller did and the file is parsed once.
    """
    if document is not None:
        from .provenance import pins_path

        records = parse_lost(document, Path(path) if path is not None else pins_path())
    else:
        records = load_lost(path)
    return {rec.digest: rec for rec in records.values()}


def lost_for_digest(digest: str, *, path: str | Path | None = None) -> LostArtifact | None:
    """The loss record claiming ``digest``, or None.

    This turns a dangling citation into an answered one: an artifact recording lost bytes does not
    re-hash them at load, so it never fails, and asking is the only way a reader learns they are gone.
    """
    want = str(digest or "").strip().lower()
    if not want:
        return None
    return claimed_digests(path).get(want)
