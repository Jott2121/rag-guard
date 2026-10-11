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
            "git push --force-w origin",
            "git push origin --force-with-lease",
            "git -c user.name=x push origin --force",
            "git reset HEAD --hard",
            "git reset --hard HEAD",
            "git reset --har",
        ]
        for command in denied:
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "deny", command)
        asked = [
            "git push -u origin main",
            "git push --no-force origin main",
            "git push origin main && curl -f https://example.com",
            "curl -f https://example.com && git push origin main",
        ]
        for command in asked:
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "ask", command)
        for command in ("git reset HEAD", "git reset --soft HEAD"):
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "allow", command)

    def test_wrapped_and_nested_rewrites_are_denied(self):
        denied = [
            "/usr/bin/git push --force",
            "sudo git push origin --force",
            'bash -c "git push --force"',
            'bash -lc "git reset --hard"',
            "git submodule foreach git push --force",
            'git submodule foreach "git push --force"',
            "/usr/bin/git reset --hard",
            "sudo git reset --hard",
            'bash -c "git reset --hard"',
        ]
        for command in denied:
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "deny", command)
        asked = [
            "/usr/bin/git push origin main",
            "sudo git push origin main",
            'bash -c "git push origin main"',
            'bash -c "curl -f https://example.com && git push origin main"',
            "git submodule foreach git push origin main",
        ]
        for command in asked:
            got = decision(json.dumps({"command": command}))
            self.assertEqual(got["permission"], "ask", command)
        allowed = [
            "/usr/bin/git reset HEAD",
            "sudo git reset --soft HEAD",
            'bash -c "git reset HEAD"',
            "git submodule foreach git reset HEAD",
        ]
        for command in allowed:
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
