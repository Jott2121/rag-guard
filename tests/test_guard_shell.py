"""Shell-guard decisions: rebasing pulls, rewrite flags, blank commands."""
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / ".cursor" / "hooks" / "guard-shell.sh"
HOOKS = ROOT / ".cursor" / "hooks.json"


def run(payload: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(GUARD)],
        input=payload.encode(),
        capture_output=True,
        cwd=ROOT,
    )


def decision(payload: str) -> dict:
    result = run(payload)
    if result.returncode != 0:
        raise AssertionError(
            f"expected JSON decision, exit {result.returncode}: {result.stderr!r}"
        )
    return json.loads(result.stdout)


class GuardShellTests(unittest.TestCase):
    def test_matcher_sends_git_pull_to_the_guard(self):
        hooks = json.loads(HOOKS.read_text())
        matcher = hooks["hooks"]["beforeShellExecution"][0]["matcher"]
        self.assertIn("git pull", matcher.split("|"))

    def test_rewrite_flags_are_denied_away_from_the_subcommand(self):
        denied = [
            "git push origin --force",
            "git push origin -f",
            "git push -fu origin",
            "git push --force",
            "git push --force-with-lease origin",
            "git reset HEAD --hard",
            "git reset --hard HEAD",
        ]
        for command in denied:
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "deny", command)
        asked = decision(json.dumps({"command": "git push -u origin main"}))
        self.assertEqual(asked["permission"], "ask")
        for command in ("git reset HEAD", "git reset --soft HEAD"):
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "allow", command)

    def test_plain_pull_is_allowed_and_rebase_pull_is_denied(self):
        allowed = decision(json.dumps({"command": "git pull origin main"}))
        self.assertEqual(allowed["permission"], "allow")
        denied = decision(json.dumps({"command": "git pull --rebase origin main"}))
        self.assertEqual(denied["permission"], "deny")

    def test_missing_blank_and_non_string_commands_are_denied(self):
        payloads = [
            "{}",
            '{"command": ""}',
            '{"command": "   "}',
            '{"command": null}',
            '{"command": 1}',
            '{"command": ["git", "status"]}',
        ]
        for payload in payloads:
            got = decision(payload)
            self.assertEqual(got["permission"], "deny", payload)

    def test_invalid_json_exits_nonzero(self):
        result = run("{")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()
