"""Deterministic pointer-based context handoff contracts.

The mechanism models a small startup pointer that references user-controlled
content. It verifies identity, supersession, and content integrity before a
continuity-dependent execution may consume that content.

It performs no network access and is not an OpenAI product integration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

POINTER_SCHEMA = "glaciereq.context-pointer.v1"
HANDOFF_SCHEMA = "glaciereq.context-handoff-receipt.v1"
HANDOFF_EVIDENCE_STATE = "LOCAL_POINTER_HANDOFF_CONTRACT_NO_OPENAI_RUNTIME_INTEGRATION"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContextHandoffError(ValueError):
    """Raised when required continuity state cannot be resolved exactly."""


def _bytes(content: str | bytes) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    raise TypeError("context content must be str or bytes")


def _digest_bytes(content: str | bytes) -> str:
    return hashlib.sha256(_bytes(content)).hexdigest()


def _digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextPointer:
    pointer_id: str
    target: str
    authority: str
    content_sha256: str
    version: int = 1
    required: bool = True
    supersedes: tuple[str, ...] = ()

    def validate(self) -> None:
        for name, value in (
            ("pointer_id", self.pointer_id),
            ("target", self.target),
            ("authority", self.authority),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContextHandoffError(f"{name} must be non-empty text")
        if not _SHA256_RE.fullmatch(self.content_sha256):
            raise ContextHandoffError("content_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ContextHandoffError("version must be a positive integer")
        if not isinstance(self.required, bool):
            raise ContextHandoffError("required must be boolean")
        if len(self.supersedes) != len(set(self.supersedes)):
            raise ContextHandoffError("supersedes entries must be unique")
        if self.pointer_id in self.supersedes:
            raise ContextHandoffError("a pointer cannot supersede itself")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": POINTER_SCHEMA,
            "pointer_id": self.pointer_id,
            "target": self.target,
            "authority": self.authority,
            "content_sha256": self.content_sha256,
            "version": self.version,
            "required": self.required,
            "supersedes": list(self.supersedes),
        }


def build_pointer(
    pointer_id: str,
    target: str,
    content: str | bytes,
    *,
    authority: str = "user",
    version: int = 1,
    required: bool = True,
    supersedes: tuple[str, ...] = (),
) -> ContextPointer:
    """Create a pointer bound to exact user-controlled content bytes."""

    pointer = ContextPointer(
        pointer_id=pointer_id,
        target=target,
        authority=authority,
        content_sha256=_digest_bytes(content),
        version=version,
        required=required,
        supersedes=supersedes,
    )
    pointer.validate()
    return pointer


def _active_pointers(pointers: list[ContextPointer]) -> list[ContextPointer]:
    seen: set[str] = set()
    for pointer in pointers:
        pointer.validate()
        if pointer.pointer_id in seen:
            raise ContextHandoffError("pointer_id values must be unique")
        seen.add(pointer.pointer_id)

    superseded = {item for pointer in pointers for item in pointer.supersedes}
    return [pointer for pointer in pointers if pointer.pointer_id not in superseded]


def resolve_handoff(
    pointers: list[ContextPointer],
    resolver: Mapping[str, str | bytes],
) -> dict[str, object]:
    """Resolve active pointers and fail closed when required state is absent or changed."""

    active = _active_pointers(pointers)
    resolved: list[dict[str, object]] = []
    skipped_optional: list[str] = []

    for pointer in sorted(active, key=lambda item: (item.authority, item.pointer_id)):
        if pointer.target not in resolver:
            if pointer.required:
                raise ContextHandoffError(
                    f"required context pointer {pointer.pointer_id!r} could not be resolved"
                )
            skipped_optional.append(pointer.pointer_id)
            continue

        content = resolver[pointer.target]
        observed = _digest_bytes(content)
        if observed != pointer.content_sha256:
            raise ContextHandoffError(
                f"context digest mismatch for pointer {pointer.pointer_id!r}"
            )
        resolved.append(
            {
                "pointer_id": pointer.pointer_id,
                "target": pointer.target,
                "authority": pointer.authority,
                "version": pointer.version,
                "content_sha256": observed,
                "content_bytes": len(_bytes(content)),
            }
        )

    body: dict[str, object] = {
        "schema": HANDOFF_SCHEMA,
        "evidence_state": HANDOFF_EVIDENCE_STATE,
        "resolved": resolved,
        "skipped_optional": sorted(skipped_optional),
        "active_pointer_count": len(active),
        "resolved_pointer_count": len(resolved),
        "runtime_integration": False,
        "network_access": False,
    }
    body["receipt_sha256"] = _digest_json(body)
    return body
