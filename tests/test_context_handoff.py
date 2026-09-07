from __future__ import annotations

import unittest

from context_handoff import (
    HANDOFF_EVIDENCE_STATE,
    ContextHandoffError,
    build_pointer,
    resolve_handoff,
)


class ContextHandoffTests(unittest.TestCase):
    def test_required_user_pointer_resolves_with_deterministic_receipt(self) -> None:
        content = "LOOK FIRST. THEN TALK.\nsource_priority=user"
        pointer = build_pointer(
            "user-continuity",
            "user://continuity/bootstrap",
            content,
            authority="user",
        )
        first = resolve_handoff([pointer], {pointer.target: content})
        second = resolve_handoff([pointer], {pointer.target: content})
        self.assertEqual(first, second)
        self.assertEqual(first["resolved_pointer_count"], 1)
        self.assertEqual(first["resolved"][0]["pointer_id"], "user-continuity")
        self.assertEqual(first["evidence_state"], HANDOFF_EVIDENCE_STATE)
        self.assertFalse(first["runtime_integration"])
        self.assertEqual(len(first["receipt_sha256"]), 64)

    def test_required_pointer_missing_fails_closed(self) -> None:
        pointer = build_pointer("required", "user://required", "content")
        with self.assertRaises(ContextHandoffError, msg="required state must not be skipped"):
            resolve_handoff([pointer], {})

    def test_digest_mismatch_fails_closed(self) -> None:
        pointer = build_pointer("config", "user://config", "version-one")
        with self.assertRaises(ContextHandoffError):
            resolve_handoff([pointer], {"user://config": "silently-changed"})

    def test_new_pointer_supersedes_old_pointer(self) -> None:
        old = build_pointer("config-v1", "user://config/v1", "old", version=1)
        new = build_pointer(
            "config-v2",
            "user://config/v2",
            "new",
            version=2,
            supersedes=("config-v1",),
        )
        receipt = resolve_handoff(
            [old, new],
            {
                "user://config/v1": "old",
                "user://config/v2": "new",
            },
        )
        self.assertEqual(receipt["active_pointer_count"], 1)
        self.assertEqual(
            [row["pointer_id"] for row in receipt["resolved"]],
            ["config-v2"],
        )

    def test_optional_missing_pointer_is_explicitly_recorded(self) -> None:
        pointer = build_pointer(
            "optional",
            "user://optional",
            "not-present",
            required=False,
        )
        receipt = resolve_handoff([pointer], {})
        self.assertEqual(receipt["resolved_pointer_count"], 0)
        self.assertEqual(receipt["skipped_optional"], ["optional"])

    def test_duplicate_pointer_ids_fail_closed(self) -> None:
        first = build_pointer("same", "user://a", "a")
        second = build_pointer("same", "user://b", "b")
        with self.assertRaises(ContextHandoffError):
            resolve_handoff([first, second], {"user://a": "a", "user://b": "b"})


if __name__ == "__main__":
    unittest.main()
