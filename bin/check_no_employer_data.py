"""Fail if tracked files contain blocked employer terms, export-control markers, or build junk.

Employer terms are never stored in plain text. `.github/blocked-terms.sha256`
holds one SHA-256 per line of a lowercased term (words joined by one space).
Every tracked file is tokenized into lowercase alphanumeric words (once as-is,
once split at camelCase boundaries), and each word, each adjacent 2-word
phrase, and each adjacent pair glued together is hashed and looked up. Terms longer than two words cannot match. The optional
`BLOCKED_TERMS` env var adds comma-separated plain-text terms (a CI secret).
This scan covers the whole tree with no exclusions. Hits are reported by path,
line, and hash prefix only, so CI logs never echo the term.

Export-control markers are generic, so they stay as a regex. The README author
bio (from "## About" through EOF) and policy files that name the markers on
purpose (`.cursor/`, `AGENTS.md`, `.github/`) are skipped for markers only.

A missing, empty, or malformed hash file, a git failure, or an unsupported
`BLOCKED_TERMS` entry exits non-zero. The check does not pass when it cannot run.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH_FILE = Path(".github") / "blocked-terms.sha256"
WORD = re.compile(r"[a-z0-9]+")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
HEX64 = re.compile(r"[0-9a-f]{64}")
# Pieces, so a scan of this file is not a hit.
_MARKER_SRC = "|".join([
    r"\b" + "CU" + "I" + r"\b",
    r"\b" + "IT" + "AR" + r"\b",
    "EXPORT" + " CONTROLLED",
    r"\b" + "PROPRIET" + "ARY" + r"\b",
])
MARKER = re.compile(_MARKER_SRC, re.IGNORECASE)
JUNK = re.compile(r"(?:^|/)__pycache__/|\.pyc$|^\.coverage$")


def sha256(term: str) -> str:
    return hashlib.sha256(term.encode("utf-8")).hexdigest()


def normalize(term: str) -> str:
    return " ".join(WORD.findall(term.lower()))


def load_hashes(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"blocked-terms hash file unreadable: {path}: {exc}") from exc
    hashes: set[str] = set()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not HEX64.fullmatch(line):
            raise SystemExit(f"{path}:{lineno}: not a lowercase hex SHA-256")
        hashes.add(line)
    if not hashes:
        raise SystemExit(f"blocked-terms hash file has no hashes: {path}")
    return hashes


def env_hashes(env: Mapping[str, str]) -> set[str]:
    hashes: set[str] = set()
    for raw in env.get("BLOCKED_TERMS", "").split(","):
        term = normalize(raw)
        if not term:
            continue
        if term.count(" ") > 1:
            raise SystemExit("BLOCKED_TERMS entries must be one or two words")
        hashes.add(sha256(term))
    return hashes


def candidates(line: str) -> Iterator[str]:
    for source in (line, CAMEL.sub(" ", line)):
        words = WORD.findall(source.lower())
        yield from words
        for a, b in zip(words, words[1:]):
            yield f"{a} {b}"
            yield a + b


def term_hits(path: str, text: str, hashes: set[str]) -> list[str]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for digest in {sha256(c) for c in candidates(line)} & hashes:
            hits.append(f"{path}:{lineno}: blocked term (sha256 {digest[:12]})")
    return hits


def line_hits(text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines(), 1) if MARKER.search(line)]


def readme_scannable(text: str) -> str:
    body, sep, _bio = text.partition("\n## About\n")
    if not sep:
        raise SystemExit("README.md has no ## About section to treat as the author bio")
    return body


def marker_skipped(path: str) -> bool:
    return path == "AGENTS.md" or path.startswith(".cursor/") or path.startswith(".github/")


def git_stdout(args: list[str], root: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"{' '.join(args)} failed ({result.returncode}): {detail}")
    return result.stdout


def tracked_files(root: Path = ROOT) -> list[str]:
    raw = git_stdout(["git", "ls-files", "-z"], root)
    return [path for path in raw.split("\0") if path]


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def marker_lines(path: str, text: str) -> list[str]:
    if marker_skipped(path) or "\0" in text:
        return []
    if path == "README.md":
        text = readme_scannable(text)
    return [f"{path}:{n}: export-control marker" for n in line_hits(text)]


def blocked_hashes(root: Path, env: Mapping[str, str]) -> set[str]:
    return load_hashes(root / HASH_FILE) | env_hashes(env)


def scan_paths(paths: Iterable[str], root: Path, env: Mapping[str, str]) -> int:
    hashes = blocked_hashes(root, env)
    hits: list[str] = []
    for path in paths:
        hits.extend(term_hits(path, read_text(Path(path)), hashes))
    if hits:
        print("\n".join(hits), file=sys.stderr)
        return 1
    return 0


def main(root: Path = ROOT, env: Mapping[str, str] = os.environ) -> int:
    hashes = blocked_hashes(root, env)
    files = tracked_files(root)
    junk = [path for path in files if JUNK.search(path)]
    if junk:
        print("committed build junk:", file=sys.stderr)
        print("\n".join(junk), file=sys.stderr)
        return 1
    hits: list[str] = []
    for path in files:
        text = read_text(root / path)
        hits.extend(term_hits(path, text, hashes))
        hits.extend(marker_lines(path, text))
    if hits:
        print("employer term / export-control marker:", file=sys.stderr)
        print("\n".join(hits), file=sys.stderr)
        return 1
    print(f"no blocked terms ({len(hashes)} hashes) or export-control markers in {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--files"]:
        sys.exit(scan_paths(sys.argv[2:], ROOT, os.environ))
    sys.exit(main())
