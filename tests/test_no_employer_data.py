"""Hash matching, whole-tree coverage, and fail-closed behavior of the employer-data scan.

Planted terms are synthetic and live only in temp git repos built per test.
Marker samples are concatenated so this file is not itself a hit.
"""
import importlib.util
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "bin" / "check_no_employer_data.py"
_SPEC = importlib.util.spec_from_file_location("check_no_employer_data", _PATH)
scan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan)

PLANTED = "zorblax quantics"


def _tok(*parts: str) -> str:
    return "".join(parts)


class TempRepo:
    def __init__(self, files: dict[str, str], hash_lines: list[str] | None):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        if hash_lines is not None:
            files = {**files, str(scan.HASH_FILE): "\n".join(hash_lines) + "\n"}
        for rel, body in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)

    def run(self, env: dict[str, str] | None = None) -> tuple[int, str]:
        err = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(err):
            code = scan.main(self.root, env or {})
        return code, err.getvalue()

    def close(self) -> None:
        self._tmp.cleanup()


class HashScanTests(unittest.TestCase):
    def repo(self, files, hash_lines=("# header", scan.sha256(PLANTED))):
        r = TempRepo(files, None if hash_lines is None else list(hash_lines))
        self.addCleanup(r.close)
        return r

    def test_planted_term_is_caught_in_any_folder_without_echoing_it(self):
        for rel in ["results/run.json", "eval/fixture.txt", ".cursor/rules/x.mdc", "AGENTS.md", ".github/x.yml"]:
            r = self.repo({"ok.txt": "nothing here\n", rel: f"line one\nfrom {PLANTED.upper()} ltd\n"})
            code, err = r.run()
            self.assertEqual(code, 1, rel)
            self.assertIn(f"{rel}:2: blocked term", err)
            self.assertNotIn("zorblax", err.lower())

    def test_single_word_hyphen_and_glued_forms_match(self):
        hashes = {scan.sha256("zorblax"), scan.sha256(PLANTED)}
        for line in ["Zorblax", "zorblax-quantics", "ZorblaxQuantics", "zorblax_quantics"]:
            self.assertTrue(scan.term_hits("f", line, hashes), line)
        for line in ["zorblaxy", "xzorblax", "zorbla"]:
            self.assertEqual(scan.term_hits("f", line, {scan.sha256("zorblax")}), [], line)

    def test_clean_tree_passes(self):
        code, err = self.repo({"README.md": "hello\n## About\nbio\n", "results/a.json": "{}\n"}).run()
        self.assertEqual((code, err), (0, ""))

    def test_blocked_terms_env_adds_terms(self):
        r = self.repo({"notes.md": "Visit Frobnitz Works today\n"})
        self.assertEqual(r.run()[0], 0)
        code, err = r.run({"BLOCKED_TERMS": " other , FROBNITZ   works "})
        self.assertEqual(code, 1)
        self.assertIn("notes.md:1: blocked term", err)

    def test_blocked_terms_env_rejects_three_words(self):
        with self.assertRaises(SystemExit):
            scan.env_hashes({"BLOCKED_TERMS": "one two three"})

    def test_missing_hash_file_fails_closed(self):
        with self.assertRaises(SystemExit):
            self.repo({"a.txt": "x\n"}, hash_lines=None).run()

    def test_empty_or_comment_only_hash_file_fails_closed(self):
        for lines in ([], ["", "# only a comment"]):
            with self.assertRaises(SystemExit):
                self.repo({"a.txt": "x\n"}, hash_lines=lines).run()

    def test_malformed_hash_line_fails_closed(self):
        with self.assertRaises(SystemExit):
            self.repo({"a.txt": "x\n"}, hash_lines=["not-a-hash"]).run()

    def test_committed_hash_file_loads(self):
        self.assertGreaterEqual(len(scan.load_hashes(scan.ROOT / scan.HASH_FILE)), 1)

    def test_files_mode_for_hook(self):
        r = self.repo({"bad.txt": f"{PLANTED}\n", "good.txt": "fine\n"})
        with redirect_stderr(StringIO()):
            self.assertEqual(scan.scan_paths([str(r.root / "bad.txt")], r.root, {}), 1)
            self.assertEqual(scan.scan_paths([str(r.root / "good.txt")], r.root, {}), 0)


class MarkerTests(unittest.TestCase):
    def test_short_tokens_are_whole_words_only(self):
        clean = ["militarily", "hereditarily", "cuisine", "sanitarily"]
        for word in clean:
            self.assertEqual(scan.line_hits(word), [], word)

        hits = [
            _tok("CU", "I"),
            _tok("IT", "AR"),
            _tok("EXPORT", " CONTROLLED"),
            _tok("PROPRIET", "ARY"),
        ]
        for word in hits:
            self.assertEqual(scan.line_hits(word), [1], word)

    def test_author_bio_is_not_scanned(self):
        text = "shipping takes 3 days\n## About\nBuilt by someone. " + _tok("IT", "AR") + " note in the bio.\n"
        body = scan.readme_scannable(text)
        self.assertEqual(scan.line_hits(body), [])
        self.assertEqual(scan.line_hits(text), [3])

    def test_readme_without_about_fails_closed(self):
        with self.assertRaises(SystemExit):
            scan.readme_scannable("no bio heading here\n")

    def test_git_failure_fails_closed(self):
        with self.assertRaises(SystemExit):
            scan.git_stdout(["git", "ls-files", "--not-a-real-flag"])

    def test_this_repo_is_clean(self):
        self.assertEqual(scan.main(), 0)


if __name__ == "__main__":
    unittest.main()
