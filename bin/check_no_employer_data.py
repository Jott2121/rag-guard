"""Fail if tracked files contain employer / export-control markers or build junk.

Short tokens use word boundaries so ordinary words are not hits. The README
author bio (from "## About" through EOF) is the one biographical blurb that
is not scanned. Policy files that name the markers on purpose are skipped:
`.cursor/`, `AGENTS.md`, and `.github/`.

The marker expression is assembled from pieces so this file does not itself
contain the tokens it searches for.

A git or read failure exits non-zero. The check does not pass when the
scanner itself errors.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pieces, so a scan of this file is not a hit.
_MARKER_SRC = "|".join([
    "lock" + "heed",
    r"\b" + "lm" + "co" + r"\b",
    r"\b" + "CU" + "I" + r"\b",
    r"\b" + "IT" + "AR" + r"\b",
    "EXPORT" + " CONTROLLED",
    r"\b" + "PROPRIET" + "ARY" + r"\b",
])
MARKER = re.compile(_MARKER_SRC, re.IGNORECASE)
JUNK = re.compile(r"(?:^|/)__pycache__/|\.pyc$|^\.coverage$")


def line_hits(text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines(), 1) if MARKER.search(line)]


def readme_scannable(text: str) -> str:
    body, sep, _bio = text.partition("\n## About\n")
    if not sep:
        raise SystemExit("README.md has no ## About section to treat as the author bio")
    return body


def skipped(path: str) -> bool:
    return path == "AGENTS.md" or path.startswith(".cursor/") or path.startswith(".github/")


def git_stdout(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"{' '.join(args)} failed ({result.returncode}): {detail}")
    return result.stdout


def tracked_files() -> list[str]:
    raw = git_stdout(["git", "ls-files", "-z"])
    return [path for path in raw.split("\0") if path]


def scan_file(path: str) -> list[str]:
    data = (ROOT / path).read_bytes()
    if b"\0" in data:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    if path == "README.md":
        text = readme_scannable(text)
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if MARKER.search(line):
            hits.append(f"{path}:{lineno}:{line}")
    return hits


def main() -> int:
    files = tracked_files()
    junk = [path for path in files if JUNK.search(path)]
    if junk:
        print("committed build junk:", file=sys.stderr)
        print("\n".join(junk), file=sys.stderr)
        return 1
    hits: list[str] = []
    for path in files:
        if skipped(path):
            continue
        hits.extend(scan_file(path))
    if hits:
        print("employer/export-control marker:", file=sys.stderr)
        print("\n".join(hits), file=sys.stderr)
        return 1
    print(f"no employer/export-control markers in {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
