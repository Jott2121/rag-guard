"""Word-boundary and fail-closed behavior of the employer-data scan.

Sample tokens are concatenated so this file is not itself a hit.
"""
import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "bin" / "check_no_employer_data.py"
_SPEC = importlib.util.spec_from_file_location("check_no_employer_data", _PATH)
scan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan)


def _tok(*parts: str) -> str:
    return "".join(parts)


class MarkerTests(unittest.TestCase):
    def test_short_tokens_are_whole_words_only(self):
        clean = ["militarily", "hereditarily", "cuisine", "xlmcox", "sanitarily"]
        for word in clean:
            self.assertEqual(scan.line_hits(word), [], word)

        hits = [
            _tok("lm", "co"),
            _tok("CU", "I"),
            _tok("IT", "AR"),
            _tok("lock", "heed"),
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
