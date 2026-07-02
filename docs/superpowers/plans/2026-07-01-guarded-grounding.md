# Guarded Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This plan was adversarially reviewed against the real rag-guard + Bow code before hand-off.** Where a task touches existing Bow/rag-guard code, a **CONFIRM** step tells you to open the real file and match names/signatures/line numbers before writing.

**Goal:** Make rag-guard fire automatically on every conversation — grounding answers in Jeff's notes, escalating to the web when notes fall short, and corroborating across ≥2 independent, authority-weighted sources — delivered through a Claude Code hook (advisory) and a Bow wrap (teeth).

**Architecture:** One shared core (persistent, fingerprinted TF-IDF index over memory + wikis) plus a web-verification engine, consumed by two thin adapters. A three-tier "truth ladder" runs on both surfaces: local notes → web search → cross-source corroboration, every answer stamped by confidence and cited.

**Tech Stack:** Python 3.11+ (stdlib only for the `rag_guard` core; `pytest` for tests). Web/model calls live in adapter layers via the `claude -p` subprocess seam (Bow) and the main agent's own web tools (CLI). No new runtime dependencies in `rag_guard`.

## Global Constraints

- `rag_guard` core has **zero runtime dependencies** (stdlib only). Model/web calls live in adapter/verifier layers, never imported by the core retrieval/guard modules. (spec §3)
- Python **3.11+**; tests pass on 3.11–3.13.
- **Coverage floor = 90%.** CI enforces `pytest --cov=rag_guard --cov-fail-under=90` (`.github/workflows/ci.yml`). Stay green, key-free, network-free (`FakeProvider` + mocked web/model).
- Doc dict shape: `{"id": str, "text": str, "source": str, "weight": float}`. Retriever hit shape: `{"id","text","score"}`. `RagGuard.answer` return shape unchanged.
- Never hardcode a model in Bow — all selection via `bow/routing.py` (use the module's named constants, e.g. `FABLE`, `SONNET`).
- CC hook is **retrieval-only, fail-open**, registered timeout 15s (target <1s); never breaks the user's prompt.
- Bow guard is **fail-toward-labeled**: never drop an answer; degrade to a distinct stamp (`guard unavailable` / `web check failed`) vs. a genuine `unverified` verdict.
- Corpus is markdown-text-only; exclude audio/PDF/images/gpx/code/`.git`/`.venv`/`node_modules`/caches.

---

## File Structure

**rag-guard repo (`~/rag-guard`):**
- `rag_guard/corpus.py` — *new* — walk sources, chunk markdown, emit `{id,text,source,weight}` (memory chunks weighted higher).
- `rag_guard/retriever.py` — *modify* — precomputed per-doc norms, `from_index()`, `index_state()`, weight-aware ranking.
- `rag_guard/index.py` — *new* — save/load, corpus fingerprint (incl. chunk params), `get_index()`.
- `rag_guard/config.py` — *new* — roots, cache path, `MIN_SCORE`, `MAX_WEB_SOURCES`.
- `rag_guard/service.py` — *new* — warm singleton `query()`.
- `rag_guard/cli.py` — *new* — stdin/argv → cached retrieval → JSON stdout.
- `rag_guard/stamps.py` — *new* — confidence-ladder stamps incl. error stamps.
- `rag_guard/webverify.py` — *new* — Tier 2/3: injectable search, independence, authority-aware corroboration → `Verdict`.
- `rag_guard/reindex.py` — *new* — nightly rebuild entry.
- `bin/hook_userpromptsubmit.py`, `bin/__init__.py` — *new* — CC adapter.
- `pyproject.toml` — *modify* — `[project.scripts]`.
- `ops/com.jeffotterson.ragguard-reindex.plist` — *new*.
- `tests/test_corpus.py`, `test_index.py`, `test_config.py`, `test_service.py`, `test_cli.py`, `test_stamps.py`, `test_webverify.py`, `test_reindex.py`, `test_hook.py`, `test_eval_grounding.py` — *new*; `tests/test_retriever.py` — *append*.
- `eval/__init__.py`, `eval/grounding_cases.py`, `eval/run_grounding.py` — *new*.
- `README.md` — *modify* — landscape/prior-art + case study.

**bow repo (`~/bow`) — tests live in the top-level `tests/` dir:**
- `bow/ragguard_wrap.py` — *new* — `GuardedBrain` (truth ladder, session-key aware, skip predicate).
- `bow/ragguard_judge.py` — *new* — cross-family Tier-1 groundedness judge (role `rag_guard`).
- `bow/ragguard_webverifier.py` — *new* — tool-enabled web verifier (role `rag_guard_web`).
- `bow/daemon.py` — *modify* — wrap the constructed `Brain` in `main()`.
- `bow/routing.py` — *modify* — add `rag_guard` + `rag_guard_web` to `DEFAULT_ROLES`.
- `tests/test_ragguard_wrap.py`, `tests/test_ragguard_judge.py`, `tests/test_ragguard_webverifier.py` — *new*; `tests/test_routing.py` — *append*.

---

## Phase 0 — Core: corpus + index + query API

### Task 1: Corpus chunker with source weighting

**Files:** Create `rag_guard/corpus.py`; Test `tests/test_corpus.py`

**Interfaces:**
- Produces: `chunk_text(text, *, chunk_chars=800, overlap=100) -> list[str]`; `build_corpus(roots, *, chunk_chars=800, overlap=100, exclude_suffixes=DEFAULT_EXCLUDE) -> list[dict]` emitting `{"id","text","source","weight"}`. Memory-source chunks get `weight=1.15`; all others `weight=1.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py
from rag_guard.corpus import chunk_text, build_corpus

def test_chunk_respects_size_and_overlap():
    text = "abcdefghij" * 20
    chunks = chunk_text(text, chunk_chars=80, overlap=20)
    assert all(len(c) <= 80 for c in chunks)
    assert len(chunks) >= 3
    assert chunks[0][-20:] == chunks[1][:20]

def test_chunk_short_text_single_chunk():
    assert chunk_text("hello world", chunk_chars=800, overlap=100) == ["hello world"]

def test_build_corpus_reads_markdown_and_skips_excluded(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes 3 days")
    (tmp_path / "b.aiff").write_bytes(b"\x00\x01")
    (tmp_path / "notes.txt").write_text("ignore me")
    docs = build_corpus([str(tmp_path)])
    ids = [d["id"] for d in docs]
    assert any("a.md#0" in i for i in ids)
    assert all(".aiff" not in i and "notes.txt" not in i for i in ids)
    assert docs[0]["text"] == "shipping takes 3 days"
    assert docs[0]["weight"] == 1.0

def test_memory_source_gets_higher_weight(tmp_path):
    mem = tmp_path / "memory"; mem.mkdir()
    (mem / "fact.md").write_text("Jeff prefers plain text")
    docs = build_corpus([str(mem)])
    assert docs[0]["weight"] == 1.15
```

- [ ] **Step 2: Run test to verify it fails** — `PYTHONPATH=. python3 -m pytest tests/test_corpus.py -v` → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/corpus.py
"""Build a retrievable corpus of {id,text,source,weight} chunks from markdown files."""
from __future__ import annotations

import os

DEFAULT_EXCLUDE = (".aiff", ".m4a", ".mp3", ".wav", ".pdf", ".png", ".jpg",
                   ".jpeg", ".gif", ".gpx", ".canvas", ".zip")
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
MEMORY_WEIGHT = 1.15


def chunk_text(text: str, *, chunk_chars: int = 800, overlap: int = 100) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []
    step = max(1, chunk_chars - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_chars]
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
    return chunks


def _weight_for(path: str) -> float:
    return MEMORY_WEIGHT if (os.sep + "memory" + os.sep) in (path + os.sep) or \
        os.path.basename(os.path.dirname(path)) == "memory" else 1.0


def build_corpus(roots: list[str], *, chunk_chars: int = 800, overlap: int = 100,
                 exclude_suffixes: tuple[str, ...] = DEFAULT_EXCLUDE) -> list[dict]:
    docs: list[dict] = []
    for root in roots:
        if os.path.isfile(root):
            _add_file(root, root, docs, chunk_chars, overlap, exclude_suffixes)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in sorted(filenames):
                if name.endswith(".md"):
                    _add_file(os.path.join(dirpath, name), root, docs,
                              chunk_chars, overlap, exclude_suffixes)
    return docs


def _add_file(path, source, docs, chunk_chars, overlap, exclude_suffixes):
    if path.endswith(exclude_suffixes):
        return
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return
    base = source if os.path.isdir(source) else os.path.dirname(source)
    rel = os.path.relpath(path, base)
    weight = _weight_for(path)
    for i, chunk in enumerate(chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)):
        docs.append({"id": f"{rel}#{i}", "text": chunk, "source": source, "weight": weight})
```

- [ ] **Step 4: Run test to verify it passes** — `PYTHONPATH=. python3 -m pytest tests/test_corpus.py -v` → PASS (4)

- [ ] **Step 5: Commit** — `git add rag_guard/corpus.py tests/test_corpus.py && git commit -m "feat(corpus): markdown chunker with memory-source weighting"`

### Task 2: Retriever — norms, from_index, weight-aware ranking

**Files:** Modify `rag_guard/retriever.py`; Test `tests/test_retriever.py` (append)

- [ ] **Step 0: CONFIRM real code**

Open `rag_guard/retriever.py`. Confirm these exist and match: `self.idf`, `self._vecs`, `self._vectorize(toks)`, module fn `_toks(text)`, `Retriever._cosine(a,b)`, and that `__init__` builds a **smoothed** idf (`math.log((1+n)/(1+d))+1.0`, positive for a 1-doc corpus). If names differ, adapt the code below to the real names.

**Interfaces:**
- Produces: `self._norms: list[float]`; `Retriever.from_index(docs, idf, vecs, norms) -> Retriever` (no re-tokenize); `Retriever.index_state() -> {docs, idf, vecs, norms}`. `retrieve()` multiplies each doc's cosine by `doc.get("weight", 1.0)` before ranking, and **deletes the now-unused `_cosine` method** (folded inline) to keep coverage clean.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py  (append)
from rag_guard.retriever import Retriever

def test_from_index_bypasses_rebuild_and_matches():
    docs = [{"id": "a", "text": "shipping takes three days"},
            {"id": "b", "text": "returns within thirty days"}]
    built = Retriever(docs)
    loaded = Retriever.from_index(**built.index_state())
    assert loaded.retrieve("shipping") == built.retrieve("shipping")

def test_norms_precomputed_positive():
    r = Retriever([{"id": "a", "text": "hello world"}, {"id": "b", "text": "other text"}])
    assert r._norms[0] > 0

def test_weight_breaks_ties_toward_memory():
    docs = [{"id": "wiki", "text": "shipping policy details", "weight": 1.0},
            {"id": "mem", "text": "shipping policy details", "weight": 1.15}]
    top = Retriever(docs).retrieve("shipping policy details", k=1)[0]
    assert top["id"] == "mem"
```

- [ ] **Step 2: Run test to verify it fails** — `PYTHONPATH=. python3 -m pytest tests/test_retriever.py -k "from_index or norms or weight" -v` → FAIL (`AttributeError: ... from_index`)

- [ ] **Step 3: Write minimal implementation**

In `__init__`, after `self._vecs = [...]`:
```python
        self._norms = [self._norm(v) for v in self._vecs]
```
Add methods, and **replace** `retrieve` (delete the old `_cosine` static method entirely):
```python
    @staticmethod
    def _norm(vec):
        import math
        return math.sqrt(sum(v * v for v in vec.values()))

    @classmethod
    def from_index(cls, docs, idf, vecs, norms):
        self = cls.__new__(cls)
        self.docs, self.idf, self._vecs, self._norms = docs, idf, vecs, norms
        return self

    def index_state(self):
        return {"docs": self.docs, "idf": self.idf, "vecs": self._vecs, "norms": self._norms}

    def retrieve(self, query, k=3):
        import math
        qv = self._vectorize(_toks(query))
        qn = math.sqrt(sum(v * v for v in qv.values()))
        scored = []
        for i, d in enumerate(self.docs):
            dn = self._norms[i]
            if qn and dn:
                common = set(qv) & set(self._vecs[i])
                score = sum(qv[t] * self._vecs[i][t] for t in common) / (qn * dn)
            else:
                score = 0.0
            score *= d.get("weight", 1.0)
            scored.append({"id": d["id"], "text": d["text"], "score": round(score, 6)})
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[:k]
```

- [ ] **Step 4: Run test to verify it passes** — `PYTHONPATH=. python3 -m pytest tests/test_retriever.py -v` → PASS

- [ ] **Step 5: Commit** — `git add rag_guard/retriever.py tests/test_retriever.py && git commit -m "feat(retriever): norms, from_index, weight-aware ranking"`

### Task 3: Persistent index + fingerprint (incl. chunk params)

**Files:** Create `rag_guard/index.py`; Test `tests/test_index.py`

**Interfaces:**
- Produces: `fingerprint(roots, *, chunk_chars=800, overlap=100) -> str`; `save_index(retriever, fp, path)`; `load_index(path) -> (Retriever, str) | None`; `get_index(cache_path, roots, *, chunk_chars=800, overlap=100) -> Retriever`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
import time
from rag_guard.index import fingerprint, get_index

def _seed(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes three days")
    (tmp_path / "b.md").write_text("returns within thirty days")

def test_get_index_builds_then_caches(tmp_path):
    _seed(tmp_path); cache = tmp_path / "idx.json"
    r1 = get_index(str(cache), [str(tmp_path)])
    assert cache.exists() and r1.retrieve("shipping")[0]["score"] > 0
    assert get_index(str(cache), [str(tmp_path)]).retrieve("shipping") == r1.retrieve("shipping")

def test_fingerprint_changes_on_file_and_chunkparams(tmp_path):
    _seed(tmp_path)
    fp1 = fingerprint([str(tmp_path)])
    time.sleep(0.01); (tmp_path / "a.md").write_text("longer different content here")
    assert fingerprint([str(tmp_path)]) != fp1
    assert fingerprint([str(tmp_path)], chunk_chars=400) != fingerprint([str(tmp_path)])

def test_get_index_rebuilds_on_change(tmp_path):
    _seed(tmp_path); cache = tmp_path / "idx.json"
    get_index(str(cache), [str(tmp_path)])
    (tmp_path / "a.md").write_text("bravo replacement content indeed")
    assert get_index(str(cache), [str(tmp_path)]).retrieve("bravo")[0]["score"] > 0
```

- [ ] **Step 2: Run test to verify it fails** — `PYTHONPATH=. python3 -m pytest tests/test_index.py -v` → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/index.py
"""Persist the TF-IDF index; rebuild only when the corpus fingerprint changes."""
from __future__ import annotations

import hashlib
import json
import os

from rag_guard.corpus import build_corpus
from rag_guard.retriever import Retriever

_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def _walk_md(root):
    if os.path.isfile(root):
        if root.endswith(".md"):
            yield root
        return
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        for name in sorted(fn):
            if name.endswith(".md"):
                yield os.path.join(dp, name)


def fingerprint(roots, *, chunk_chars=800, overlap=100) -> str:
    h = hashlib.sha256()
    h.update(f"{chunk_chars}:{overlap}:".encode())
    for root in sorted(roots):
        for path in _walk_md(root):
            try:
                st = os.stat(path)
            except OSError:
                continue
            h.update(f"{path}:{int(st.st_mtime)}:{st.st_size};".encode())
    return h.hexdigest()


def save_index(retriever, fp, path):
    payload = {"fingerprint": fp, **retriever.index_state()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def load_index(path):
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    fp = payload.pop("fingerprint", None)
    try:
        return Retriever.from_index(payload["docs"], payload["idf"],
                                    payload["vecs"], payload["norms"]), fp
    except KeyError:
        return None


def get_index(cache_path, roots, *, chunk_chars=800, overlap=100) -> Retriever:
    current = fingerprint(roots, chunk_chars=chunk_chars, overlap=overlap)
    cached = load_index(cache_path)
    if cached is not None and cached[1] == current:
        return cached[0]
    retriever = Retriever(build_corpus(roots, chunk_chars=chunk_chars, overlap=overlap))
    save_index(retriever, current, cache_path)
    return retriever
```

- [ ] **Step 4: Run test to verify it passes** — `PYTHONPATH=. python3 -m pytest tests/test_index.py -v` → PASS (3). Add a test that feeds a corrupt cache file and asserts `load_index` returns `None` (covers the except branch).

- [ ] **Step 5: Commit** — `git add rag_guard/index.py tests/test_index.py && git commit -m "feat(index): fingerprinted persistence with rebuild-on-change"`

### Task 4: Config

**Files:** Create `rag_guard/config.py`; Test `tests/test_config.py`

**Interfaces:** `default_roots() -> list[str]`; `cache_path() -> str`; `MIN_SCORE = 0.05`; `MAX_WEB_SOURCES = 6`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from rag_guard import config

def test_cache_path_and_constants():
    assert config.cache_path().endswith("index.json")
    assert config.MIN_SCORE > 0 and config.MAX_WEB_SOURCES >= 2

def test_default_roots_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_GUARD_ROOTS", str(tmp_path))
    (tmp_path / "x.md").write_text("hi")
    assert str(tmp_path) in config.default_roots()
```

- [ ] **Step 2: Run test to verify it fails** — `PYTHONPATH=. python3 -m pytest tests/test_config.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/config.py
"""Corpus roots, cache path, and thresholds."""
from __future__ import annotations

import glob
import os

MIN_SCORE = 0.05
MAX_WEB_SOURCES = 6


def cache_path() -> str:
    base = os.path.expanduser("~/.cache/rag-guard")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "index.json")


def default_roots() -> list[str]:
    override = os.environ.get("RAG_GUARD_ROOTS")
    if override:
        candidates = override.split(":")
    else:
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, ".claude/projects/-Users-jeffreyotterson/memory"),
            *glob.glob(os.path.join(home, "Documents/*-Wiki")),
            os.path.join(home, ".claude/CLAUDE.md"),
        ]
    return [c for c in candidates if os.path.exists(c)]
```

- [ ] **Step 4: Run test to verify it passes** — PASS (2)
- [ ] **Step 5: Commit** — `git add rag_guard/config.py tests/test_config.py && git commit -m "feat(config): roots, cache, thresholds"`

### Task 5: Warm-singleton service

**Files:** Create `rag_guard/service.py`; Test `tests/test_service.py`

**Interfaces:** `query(text, k=5, *, roots=None, cache=None) -> list[dict]`; `reset()`. NOTE: `query()` recomputes `fingerprint` per call (a stat-walk); acceptable for the corpus size but the hot path relies on the OS stat cache. If profiling shows it's too slow, add a short TTL before re-fingerprinting.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_service.py
from rag_guard import service

def test_query_returns_hits(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes three days")
    (tmp_path / "b.md").write_text("returns within thirty days")
    service.reset()
    hits = service.query("shipping", roots=[str(tmp_path)], cache=str(tmp_path / "i.json"))
    assert hits and hits[0]["score"] > 0

def test_reset_clears_singleton(tmp_path):
    (tmp_path / "a.md").write_text("alpha"); (tmp_path / "b.md").write_text("beta")
    service.reset()
    service.query("alpha", roots=[str(tmp_path)], cache=str(tmp_path / "i.json"))
    assert service._SINGLETON is not None
    service.reset()
    assert service._SINGLETON is None
```

- [ ] **Step 2: Run test to verify it fails** — FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/service.py
"""Warm, importable retrieval singleton for in-process callers (Bow)."""
from __future__ import annotations

from rag_guard import config
from rag_guard.index import fingerprint, get_index

_SINGLETON = None
_FP = None


def reset() -> None:
    global _SINGLETON, _FP
    _SINGLETON, _FP = None, None


def query(text, k=5, *, roots=None, cache=None) -> list[dict]:
    global _SINGLETON, _FP
    roots = roots or config.default_roots()
    cache = cache or config.cache_path()
    fp = fingerprint(roots)
    if _SINGLETON is None or _FP != fp:
        _SINGLETON = get_index(cache, roots)
        _FP = fp
    return _SINGLETON.retrieve(text, k)
```

- [ ] **Step 4: Run test to verify it passes** — PASS (2)
- [ ] **Step 5: Commit** — `git add rag_guard/service.py tests/test_service.py && git commit -m "feat(service): warm retrieval singleton"`

### Task 6: CLI query entry (stdin → JSON)

**Files:** Create `rag_guard/cli.py`; Modify `pyproject.toml`; Test `tests/test_cli.py`

**Interfaces:** `run(argv, stdin_text) -> {query, hits, support, grounded}`; console script `rag-guard-query`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from rag_guard import cli

def _seed(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes three days")
    (tmp_path / "b.md").write_text("returns within thirty days")

def test_run_reports_support(tmp_path):
    _seed(tmp_path)
    out = cli.run(["--roots", str(tmp_path), "--cache", str(tmp_path / "i.json")],
                  "how long is shipping")
    assert out["query"] == "how long is shipping" and out["hits"]
    assert out["support"] > 0 and out["grounded"] is True

def test_run_low_support_not_grounded(tmp_path):
    _seed(tmp_path)
    out = cli.run(["--roots", str(tmp_path), "--cache", str(tmp_path / "i.json")],
                  "zzzznonsense termxyz")
    assert out["grounded"] is False
```

- [ ] **Step 2: Run test to verify it fails** — FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/cli.py
"""stdin/argv -> cached retrieval -> JSON stdout. Retrieval-only; no model call."""
from __future__ import annotations

import argparse
import json
import sys

from rag_guard import config, service


def run(argv, stdin_text) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("query", nargs="*")
    args = ap.parse_args(argv)
    q = " ".join(args.query) or (stdin_text or "").strip()
    roots = args.roots.split(":") if args.roots else None
    hits = service.query(q, args.k, roots=roots, cache=args.cache)
    support = max((h["score"] for h in hits), default=0.0)
    return {"query": q, "hits": hits, "support": round(support, 6),
            "grounded": support >= config.MIN_SCORE}


def main():
    stdin_text = "" if sys.stdin.isatty() else sys.stdin.read()
    print(json.dumps(run(sys.argv[1:], stdin_text)))


if __name__ == "__main__":
    main()
```

In `pyproject.toml` add:
```toml
[project.scripts]
rag-guard-query = "rag_guard.cli:main"
```

- [ ] **Step 4: Run test to verify it passes** — PASS (2)
- [ ] **Step 5: Commit** — `git add rag_guard/cli.py pyproject.toml tests/test_cli.py && git commit -m "feat(cli): stdin->JSON retrieval entry"`

---

## Phase 1 — Freshness backstop

### Task 7: Nightly reindex + launchd

**Files:** Create `rag_guard/reindex.py`, `ops/com.jeffotterson.ragguard-reindex.plist`; Test `tests/test_reindex.py`

**Interfaces:** `reindex() -> int` (deletes cache, rebuilds, returns doc count).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reindex.py
from rag_guard import reindex

def test_reindex_builds_cache(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("hello world"); (tmp_path / "b.md").write_text("more text")
    cache = tmp_path / "i.json"
    monkeypatch.setattr("rag_guard.config.default_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr("rag_guard.config.cache_path", lambda: str(cache))
    assert reindex.reindex() >= 1 and cache.exists()
```

- [ ] **Step 2: Run test to verify it fails** — FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/reindex.py
"""Force a full index rebuild — used by the nightly launchd backstop."""
from __future__ import annotations

import os

from rag_guard import config
from rag_guard.index import get_index


def reindex() -> int:
    cache = config.cache_path()
    try:
        os.remove(cache)
    except OSError:
        pass
    return len(get_index(cache, config.default_roots()).docs)


if __name__ == "__main__":
    print(reindex())
```

Create `ops/com.jeffotterson.ragguard-reindex.plist` (StartCalendarInterval Hour 3), `ProgramArguments` = `/usr/bin/python3 -m rag_guard.reindex`, `WorkingDirectory` + `PYTHONPATH` = `/Users/jeffreyotterson/rag-guard`. Install (human): `cp ops/*.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.jeffotterson.ragguard-reindex.plist`.

- [ ] **Step 4: Run test to verify it passes** — PASS
- [ ] **Step 5: Commit** — `git add rag_guard/reindex.py ops/ tests/test_reindex.py && git commit -m "feat(reindex): nightly rebuild backstop"`

---

## Phase 2 — Web-verify engine (Tier 2/3)

### Task 8: Confidence stamps (incl. error stamps)

**Files:** Create `rag_guard/stamps.py`; Test `tests/test_stamps.py`

**Interfaces:** constants `GROUNDED, WEB_VERIFIED, SINGLE_SOURCE, CONFLICT, UNVERIFIED, GENERAL_ONLY, GUARD_UNAVAILABLE, WEB_CHECK_FAILED`; `stamp_answer(answer, status, sources) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stamps.py
from rag_guard import stamps

def test_grounded_has_banner():
    assert "GROUNDED" in stamps.stamp_answer("ok", stamps.GROUNDED, ["m/x.md#0"])

def test_unverified_and_error_banners():
    assert "UNVERIFIED" in stamps.stamp_answer("m", stamps.UNVERIFIED, [])
    assert "guard" in stamps.stamp_answer("m", stamps.GUARD_UNAVAILABLE, []).lower()
    assert "web" in stamps.stamp_answer("m", stamps.WEB_CHECK_FAILED, []).lower()

def test_web_verified_lists_sources():
    out = stamps.stamp_answer("x", stamps.WEB_VERIFIED, ["https://a.gov", "https://b.org"])
    assert "a.gov" in out and "b.org" in out
```

- [ ] **Step 2: Run test to verify it fails** — FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/stamps.py
"""Confidence-ladder stamps appended to a delivered answer."""
from __future__ import annotations

GROUNDED = "grounded"
WEB_VERIFIED = "web_verified"
SINGLE_SOURCE = "single_source"
CONFLICT = "conflict"
UNVERIFIED = "unverified"
GENERAL_ONLY = "general_only"
GUARD_UNAVAILABLE = "guard_unavailable"
WEB_CHECK_FAILED = "web_check_failed"

_BANNER = {
    GROUNDED: "✔ GROUNDED — from your notes",
    WEB_VERIFIED: "✔ WEB-VERIFIED — corroborated across independent sources",
    SINGLE_SOURCE: "⚠ SINGLE SOURCE — found on the web, one source only",
    CONFLICT: "⚠ SOURCES CONFLICT — sources disagree; see both",
    UNVERIFIED: "⚠ UNVERIFIED — I couldn't back this in your notes or the web",
    GENERAL_ONLY: "⚠ No relevant notes found — answering from general knowledge only",
    GUARD_UNAVAILABLE: "⚠ guard unavailable — grounding check failed; answer unverified",
    WEB_CHECK_FAILED: "⚠ web check failed — could not verify online; answer unverified",
}


def stamp_answer(answer, status, sources) -> str:
    lines = [answer.rstrip(), "", f"[{_BANNER.get(status, _BANNER[UNVERIFIED])}]"]
    if sources:
        lines.append("sources: " + ", ".join(sources))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes** — PASS (3)
- [ ] **Step 5: Commit** — `git add rag_guard/stamps.py tests/test_stamps.py && git commit -m "feat(stamps): confidence + error banners"`

### Task 9: Web-verify engine (authority-aware corroboration)

**Files:** Create `rag_guard/webverify.py`; Test `tests/test_webverify.py`

**SCOPE NOTE (honest):** full syndication/echo-chamber detection needs source *content*, which the injected `search_fn` doesn't provide in v1. v1 independence = distinct registered publishers; syndication detection is a documented **known limitation / future work** (see README + spec §11). Authority weighting IS enforced now: social-only agreement cannot reach WEB-VERIFIED.

**Interfaces:**
- Produces: `publisher_of(url)->str`; `authority_tier(url)->int` (3 official, 2 established, 1 social); `independent(sources)->list`; `verify_claim(query, candidate, search_fn, *, min_sources=2, max_sources=None) -> Verdict` where `Verdict={status, sources, confidence, conflict, contradicts_local}`. A claim reaches `WEB_VERIFIED` only if ≥`min_sources` independent supporters **and at least one supporter is tier ≥ 2**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webverify.py
from rag_guard import webverify, stamps

def test_publisher_and_authority():
    assert webverify.publisher_of("https://www.irs.gov/pub/x") == "irs.gov"
    assert webverify.authority_tier("https://irs.gov/x") == 3
    assert webverify.authority_tier("https://twitter.com/u/1") == 1

def test_independent_dedupes_by_publisher():
    srcs = [{"url": "https://a.gov/1"}, {"url": "https://a.gov/2"}, {"url": "https://b.org/1"}]
    assert len(webverify.independent(srcs)) == 2

def test_two_independent_nonsocial_is_web_verified():
    def sf(q): return [{"url": "https://a.gov/x", "supports": True},
                       {"url": "https://b.org/y", "supports": True}]
    v = webverify.verify_claim("q", "a", sf, min_sources=2)
    assert v["status"] == stamps.WEB_VERIFIED and len(v["sources"]) == 2

def test_social_only_is_not_web_verified():
    def sf(q): return [{"url": "https://twitter.com/a/1", "supports": True},
                       {"url": "https://reddit.com/b/2", "supports": True}]
    v = webverify.verify_claim("q", "a", sf, min_sources=2)
    assert v["status"] != stamps.WEB_VERIFIED  # social leads need a higher tier

def test_single_supporting_is_single_source():
    v = webverify.verify_claim("q", "a", lambda q: [{"url": "https://a.gov/x", "supports": True}])
    assert v["status"] == stamps.SINGLE_SOURCE

def test_disagreement_is_conflict():
    def sf(q): return [{"url": "https://a.gov/x", "supports": True},
                       {"url": "https://b.org/y", "supports": False}]
    assert webverify.verify_claim("q", "a", sf)["conflict"] is True

def test_no_results_is_unverified():
    assert webverify.verify_claim("q", "a", lambda q: [])["status"] == stamps.UNVERIFIED

def test_max_sources_caps_fetches():
    big = [{"url": f"https://s{i}.org/x", "supports": True} for i in range(20)]
    v = webverify.verify_claim("q", "a", lambda q: big, max_sources=3)
    assert len(v["sources"]) <= 3
```

- [ ] **Step 2: Run test to verify it fails** — FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# rag_guard/webverify.py
"""Tier 2/3: search the web and corroborate a claim across independent, authority-
weighted sources. Web access is INJECTED via search_fn (returns [{"url","supports"}]),
so this module is network-free and unit-testable. Syndication detection is future work;
v1 independence = distinct publishers, with social-only agreement blocked from verified."""
from __future__ import annotations

from urllib.parse import urlparse

from rag_guard import stamps

_OFFICIAL = (".gov", ".mil", ".gov.uk")
_SOCIAL = {"twitter.com", "x.com", "reddit.com", "facebook.com", "t.me",
           "instagram.com", "tiktok.com"}


def publisher_of(url):
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def authority_tier(url):
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(_OFFICIAL):
        return 3
    return 1 if publisher_of(url) in _SOCIAL else 2


def independent(sources):
    seen, out = set(), []
    for s in sources:
        pub = publisher_of(s["url"])
        if pub not in seen:
            seen.add(pub)
            out.append(s)
    return out


def verify_claim(query, candidate, search_fn, *, min_sources=2, max_sources=None):
    raw = search_fn(query) or []
    for s in raw:
        s["publisher"] = publisher_of(s["url"])
        s["authority_tier"] = authority_tier(s["url"])
    uniq = independent(raw)
    if max_sources:
        uniq = uniq[:max_sources]
    supporting = [s for s in uniq if s.get("supports")]
    refuting = [s for s in uniq if not s.get("supports")]
    has_credible = any(s["authority_tier"] >= 2 for s in supporting)

    if supporting and refuting:
        status, conflict = stamps.CONFLICT, True
    elif len(supporting) >= min_sources and has_credible:
        status, conflict = stamps.WEB_VERIFIED, False
    elif supporting:
        status, conflict = stamps.SINGLE_SOURCE, False
    else:
        status, conflict = stamps.UNVERIFIED, False

    confidence = min(1.0, len(supporting) / float(min_sources)) if supporting else 0.0
    return {"status": status, "sources": uniq, "confidence": round(confidence, 3),
            "conflict": conflict, "contradicts_local": False}
```

- [ ] **Step 4: Run test to verify it passes** — PASS (8)
- [ ] **Step 5: Commit** — `git add rag_guard/webverify.py tests/test_webverify.py && git commit -m "feat(webverify): authority-aware corroboration + source cap"`

---

## Phase 3 — Claude Code hook adapter

### Task 10: UserPromptSubmit hook (protocol injection)

**Files:** Create `bin/__init__.py`, `bin/hook_userpromptsubmit.py`; Test `tests/test_hook.py`

**Interfaces:** `build_output(prompt, hits, support) -> dict`; `main()`. Retrieval-only; silent when `support < MIN_SCORE`; fail-open. NOTE: `bin/` is outside `--cov=rag_guard`, so hook coverage doesn't count toward the gate — keep hook logic thin and test it for correctness only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook.py
from bin import hook_userpromptsubmit as hook

def test_output_contract_when_grounded():
    hits = [{"id": "memory/pref.md#0", "text": "Jeff prefers plain text", "score": 0.4}]
    out = hook.build_output("what format?", hits, 0.4)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "Jeff prefers plain text" in ctx and "corroborate" in ctx.lower()

def test_silent_when_low_support():
    out = hook.build_output("obscure", [], 0.0)
    assert out["hookSpecificOutput"]["additionalContext"] == ""
```

- [ ] **Step 2: Run test to verify it fails** — FAIL (create empty `bin/__init__.py` so `bin` imports as a package)

- [ ] **Step 3: Write minimal implementation**

```python
# bin/hook_userpromptsubmit.py
"""Claude Code UserPromptSubmit hook: ground the prompt in Jeff's notes.
Retrieval-only, fail-open. Injects passages + the truth-ladder protocol. Never blocks."""
from __future__ import annotations

import json
import sys

from rag_guard import config, service

_PROTOCOL = (
    "GROUNDING PROTOCOL: Prefer the notes above. If they don't cover the question and it "
    "may be newer than your training cutoff, search the web, corroborate the claim across "
    ">=2 independent sources (prefer primary/official over social), cite them, and flag "
    "conflicts or anything contradicting Jeff's notes. State the answer's confidence: "
    "grounded / web-verified / single-source / unverified."
)


def build_output(prompt, hits, support):
    if support < config.MIN_SCORE or not hits:
        ctx = ""
    else:
        passages = "\n".join(f"- ({h['id']}) {h['text']}" for h in hits)
        ctx = f"Relevant notes from Jeff's knowledge base:\n{passages}\n\n{_PROTOCOL}"
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ctx}}


def main():
    try:
        payload = json.load(sys.stdin)
        hits = service.query(payload.get("prompt", ""), 5)
        support = max((h["score"] for h in hits), default=0.0)
        print(json.dumps(build_output(payload.get("prompt", ""), hits, support)))
    except Exception:
        pass  # fail-open
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes** — PASS (2)

- [ ] **Step 5: Register (human) + commit**

Add to `~/.claude/settings.json` under `hooks.UserPromptSubmit` a command hook: `PYTHONPATH=/Users/jeffreyotterson/rag-guard python3 /Users/jeffreyotterson/rag-guard/bin/hook_userpromptsubmit.py`, `"timeout": 15`. Verify: `echo '{"prompt":"what text format does Jeff prefer"}' | PYTHONPATH=. python3 bin/hook_userpromptsubmit.py`.
`git add bin/ tests/test_hook.py && git commit -m "feat(hook): UserPromptSubmit grounding + protocol"`

---

## Phase 4 — Bow wrap adapter

> **CONFIRM before Phase 4:** open `bow/brain.py`, `bow/daemon.py`, `bow/routing.py`, `bow/claudejson.py`. Verified facts this phase relies on (re-check line numbers): `Brain.ask(self, message, key="default")` (brain.py ~245); daemon calls `self.brain.ask(message, key)` (daemon.py ~641) and `self.brain.ask(text)` (~715); `routing.DEFAULT_ROLES` uses named constants (`FABLE`, `SONNET`) and `routing.resolve(role)` raises on unknown; `claudejson.result_dict(stdout)` may return `None`; the module-level claude-binary resolver is `bow.brain._resolve_claude_bin()`. Bow tests live in `~/bow/tests/`.

### Task 11: Routing roles

**Files:** Modify `bow/routing.py` (`DEFAULT_ROLES` only — NOT `OPENAI_ROLES`, which is asserted claude-free); Test `tests/test_routing.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_routing.py  (append)
from bow import routing

def test_rag_guard_roles_resolve():
    assert routing.resolve("rag_guard")["model"].startswith("claude-")
    assert routing.resolve("rag_guard_web")["model"].startswith("claude-")
```

- [ ] **Step 2: Run test to verify it fails** — `cd ~/bow && python3 -m pytest tests/test_routing.py -k rag_guard -v` → FAIL

- [ ] **Step 3: Write minimal implementation** — in `DEFAULT_ROLES`, using the module constants:

```python
    "rag_guard": {"model": FABLE, "effort": "medium"},       # cross-family judge (toolless)
    "rag_guard_web": {"model": SONNET, "effort": "medium"},  # tool-enabled web verifier
```

- [ ] **Step 4: Run test to verify it passes** — PASS
- [ ] **Step 5: Commit** — `cd ~/bow && git add bow/routing.py tests/test_routing.py && git commit -m "feat(routing): rag_guard + rag_guard_web roles"`

### Task 12: Cross-family groundedness judge (role `rag_guard`)

**Files:** Create `bow/ragguard_judge.py`; Test `tests/test_ragguard_judge.py`

**Interfaces:** `judge_grounded(answer, contexts, *, runner=None) -> bool | None` — fail-CLOSED: a cross-family `claude -p` (role `rag_guard`, toolless) returns `GROUNDED`/`UNSUPPORTED`; parse to bool. `runner(prompt)->str` injectable. On any error return `None` (caller → `GUARD_UNAVAILABLE`). Lexical `rag_guard.guard.groundedness` is used by the caller as a cheap pre-filter; THIS is the decision.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ragguard_judge.py
from bow.ragguard_judge import judge_grounded

def test_grounded_verdict_true():
    assert judge_grounded("shipping is 3 days", ["shipping takes 3 days"],
                          runner=lambda p: "GROUNDED") is True

def test_unsupported_verdict_false():
    assert judge_grounded("the ceo is bob", ["shipping info"],
                          runner=lambda p: "UNSUPPORTED") is False

def test_runner_error_returns_none():
    def bad(p): raise RuntimeError("boom")
    assert judge_grounded("x", ["y"], runner=bad) is None
```

- [ ] **Step 2: Run test to verify it fails** — FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# bow/ragguard_judge.py
"""Cross-family Tier-1 groundedness judge (routing role 'rag_guard'), fail-closed.
Modeled on loopeval.py (independent refute-first), NOT the fail-open judge.py."""
from __future__ import annotations

_PROMPT = (
    "You are a strict groundedness judge. Is the ANSWER fully supported by the CONTEXT "
    "below? Reply with exactly one word: GROUNDED or UNSUPPORTED.\n\n"
    "CONTEXT:\n{ctx}\n\nANSWER:\n{ans}"
)


def _default_runner(prompt):
    import subprocess
    from bow import routing
    from bow.brain import _resolve_claude_bin
    from bow.claudejson import result_dict
    model = routing.resolve("rag_guard")["model"]
    out = subprocess.run([_resolve_claude_bin(), "-p", prompt, "--model", model,
                          "--output-format", "json"],
                         capture_output=True, text=True, timeout=60)
    rd = result_dict(out.stdout)
    return (rd.get("result", "") if rd else "")


def judge_grounded(answer, contexts, *, runner=None):
    runner = runner or _default_runner
    try:
        verdict = runner(_PROMPT.format(ctx="\n".join(contexts), ans=answer)).strip().upper()
        return verdict.startswith("GROUNDED")
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes** — PASS (3)
- [ ] **Step 5: Commit** — `cd ~/bow && git add bow/ragguard_judge.py tests/test_ragguard_judge.py && git commit -m "feat(bow): cross-family groundedness judge"`

### Task 13: GuardedBrain (truth ladder, session-key aware, skip predicate)

**Files:** Create `bow/ragguard_wrap.py`; Test `tests/test_ragguard_wrap.py`

- [ ] **Step 0: CONFIRM** `rag_guard.guard.groundedness(answer, contexts, threshold) -> {"grounded","support"}` and `rag_guard.guard.redact_pii(text) -> str` (open `rag_guard/guard.py`).

**Interfaces:** `GuardedBrain(inner, *, retrieve_fn, judge_fn, verify_fn, ground_threshold=0.5, max_retries=2)` exposing `ask(text, key="default") -> str`. `judge_fn(answer, contexts)->bool|None` (Task 12), `verify_fn(query, answer)->Verdict|None` (Task 15). Slash-command / empty inputs pass through unstamped via `_should_guard`. Corrective loop **reformulates** by appending retrieved context to the retry message.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ragguard_wrap.py
from bow.ragguard_wrap import GuardedBrain
from rag_guard import stamps

class FakeInner:
    def __init__(self, replies): self.replies = list(replies); self.calls = []
    def ask(self, text, key="default"):
        self.calls.append((text, key))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]

def g_hits(_): return [{"id": "memory/x.md#0", "text": "shipping takes three days", "score": 0.6}]
def no_hits(_): return []

def test_passes_key_through():
    inner = FakeInner(["shipping takes three days"])
    gb = GuardedBrain(inner, retrieve_fn=g_hits, judge_fn=lambda a, c: True,
                      verify_fn=lambda q, a: None)
    gb.ask("how long is shipping", "chat42")
    assert inner.calls[0][1] == "chat42"

def test_grounded_stamped_no_retry():
    inner = FakeInner(["shipping takes three days"])
    gb = GuardedBrain(inner, retrieve_fn=g_hits, judge_fn=lambda a, c: True,
                      verify_fn=lambda q, a: None)
    out = gb.ask("how long is shipping")
    assert "GROUNDED" in out and len(inner.calls) == 1

def test_ungrounded_then_web_verified_after_retries():
    inner = FakeInner(["g1", "g2", "g3"])
    verdict = {"status": stamps.WEB_VERIFIED,
               "sources": [{"url": "https://a.gov/x"}, {"url": "https://b.org/y"}]}
    gb = GuardedBrain(inner, retrieve_fn=no_hits, judge_fn=lambda a, c: False,
                      verify_fn=lambda q, a: verdict, max_retries=2)
    out = gb.ask("what changed last week")
    assert "WEB-VERIFIED" in out and len(inner.calls) == 3

def test_unverifiable_best_effort_never_empty():
    inner = FakeInner(["my best guess"])
    gb = GuardedBrain(inner, retrieve_fn=no_hits, judge_fn=lambda a, c: False,
                      verify_fn=lambda q, a: {"status": stamps.UNVERIFIED, "sources": []},
                      max_retries=0)
    out = gb.ask("obscure")
    assert "my best guess" in out and "general knowledge" in out.lower()

def test_guard_unavailable_when_judge_errors():
    inner = FakeInner(["ans"])
    gb = GuardedBrain(inner, retrieve_fn=g_hits, judge_fn=lambda a, c: None,
                      verify_fn=lambda q, a: None, max_retries=0)
    assert "guard unavailable" in gb.ask("q").lower()

def test_slash_command_passthrough_unstamped():
    inner = FakeInner(["/help output"])
    gb = GuardedBrain(inner, retrieve_fn=g_hits, judge_fn=lambda a, c: True,
                      verify_fn=lambda q, a: None)
    out = gb.ask("/help")
    assert out == "/help output"

def test_pii_redacted():
    inner = FakeInner(["email a@b.com"])
    gb = GuardedBrain(inner, retrieve_fn=g_hits, judge_fn=lambda a, c: True,
                      verify_fn=lambda q, a: None)
    assert "a@b.com" not in gb.ask("contact")
```

- [ ] **Step 2: Run test to verify it fails** — FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# bow/ragguard_wrap.py
"""Wrap Brain.ask() with the truth ladder: local grounding (cross-family judge) ->
corrective loop -> web verification -> confidence-stamped, PII-redacted delivery.
Session key is preserved. Non-factual turns (slash-commands, empty) pass through."""
from __future__ import annotations

from rag_guard import stamps
from rag_guard.guard import groundedness, redact_pii


def _should_guard(text):
    t = (text or "").strip()
    return bool(t) and not t.startswith("/")


class GuardedBrain:
    def __init__(self, inner, *, retrieve_fn, judge_fn, verify_fn,
                 ground_threshold=0.5, max_retries=2):
        self._inner = inner
        self._retrieve = retrieve_fn
        self._judge = judge_fn
        self._verify = verify_fn
        self._threshold = ground_threshold
        self._max_retries = max_retries

    def ask(self, text, key="default"):
        if not _should_guard(text):
            return self._inner.ask(text, key)

        answer = self._inner.ask(text, key)
        hits = self._retrieve(text)
        contexts = [h["text"] for h in hits]

        attempts = 0
        judged_error = False
        while True:
            grounded = None
            if contexts:
                # cheap lexical pre-filter, then authoritative cross-family judge
                if groundedness(answer, contexts, self._threshold).get("grounded"):
                    grounded = self._judge(answer, contexts)
                else:
                    grounded = self._judge(answer, contexts)
            if grounded is True:
                return self._deliver(answer, stamps.GROUNDED, [h["id"] for h in hits])
            if grounded is None and contexts:
                judged_error = True
            if attempts >= self._max_retries:
                break
            attempts += 1
            retry_msg = text if not contexts else f"{text}\n\nUse these notes:\n" + \
                "\n".join(contexts)
            answer = self._inner.ask(retry_msg, key)
            hits = self._retrieve(text)
            contexts = [h["text"] for h in hits]

        if judged_error:
            return self._deliver(answer, stamps.GUARD_UNAVAILABLE, [])

        verdict = self._verify(text, answer)
        if verdict is None:
            status = stamps.GENERAL_ONLY if not contexts else stamps.WEB_CHECK_FAILED
            srcs = []
        else:
            status = verdict["status"]
            srcs = [s["url"] for s in verdict.get("sources", [])]
        return self._deliver(answer, status, srcs)

    def _deliver(self, answer, status, sources):
        return stamps.stamp_answer(redact_pii(answer), status, sources)
```

- [ ] **Step 4: Run test to verify it passes** — `cd ~/bow && python3 -m pytest tests/test_ragguard_wrap.py -v` → PASS (7)
- [ ] **Step 5: Commit** — `cd ~/bow && git add bow/ragguard_wrap.py tests/test_ragguard_wrap.py && git commit -m "feat(bow): GuardedBrain truth ladder"`

### Task 14: Web verifier (tool-enabled) + wire real fns + daemon injection

**Files:** Create `bow/ragguard_webverifier.py`; Modify `bow/ragguard_wrap.py` (add `guarded(inner)`), `bow/daemon.py`; Test `tests/test_ragguard_webverifier.py`, `tests/test_ragguard_wrap.py` (append)

**Interfaces:** `run_web_verifier(query, answer, *, runner=None) -> Verdict`; `guarded(inner) -> GuardedBrain` with production fns.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ragguard_webverifier.py
import json
from bow.ragguard_webverifier import run_web_verifier
from rag_guard import stamps

def test_parses_results_into_verdict():
    def r(p): return json.dumps([{"url": "https://a.gov/x", "supports": True},
                                 {"url": "https://b.org/y", "supports": True}])
    assert run_web_verifier("q", "a", runner=r)["status"] == stamps.WEB_VERIFIED

def test_runner_error_is_unverified():
    def bad(p): raise RuntimeError("no net")
    assert run_web_verifier("q", "a", runner=bad)["status"] == stamps.UNVERIFIED
```

```python
# tests/test_ragguard_wrap.py  (append)
from bow.ragguard_wrap import guarded

def test_guarded_factory():
    class Inner:
        def ask(self, t, key="default"): return "hi"
    assert hasattr(guarded(Inner()), "ask")
```

- [ ] **Step 2: Run test to verify it fails** — FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# bow/ragguard_webverifier.py
"""Tool-enabled web verifier (routing role 'rag_guard_web'). Asks a claude -p sub-agent
WITH web tools to return JSON [{"url","supports"}] for the claim, then corroborates via
rag_guard.webverify. Bounded by config.MAX_WEB_SOURCES."""
from __future__ import annotations

import json

from rag_guard import config, stamps
from rag_guard.webverify import verify_claim

_PROMPT = (
    "Search the web to check this claim. Return ONLY a JSON array of "
    '{{"url": <source url>, "supports": <true if the source supports the claim, false if '
    "it contradicts it>}}, at most {n} items, distinct publishers, prefer primary/official "
    "sources.\n\nClaim (answer to \"{q}\"):\n{a}"
)


def _default_runner(prompt):
    import subprocess
    from bow import routing
    from bow.brain import _resolve_claude_bin
    from bow.claudejson import result_dict
    model = routing.resolve("rag_guard_web")["model"]
    # Web tools MUST be enabled (do NOT use LEAN_TOOLLESS flags here).
    out = subprocess.run(
        [_resolve_claude_bin(), "-p", prompt, "--model", model,
         "--output-format", "json", "--allowedTools", "WebSearch,WebFetch"],
        capture_output=True, text=True, timeout=120)
    rd = result_dict(out.stdout)
    return (rd.get("result", "") if rd else "")


def run_web_verifier(query, answer, *, runner=None):
    runner = runner or _default_runner
    try:
        prompt = _PROMPT.format(n=config.MAX_WEB_SOURCES, q=query, a=answer)
        results = json.loads(runner(prompt))
        return verify_claim(query, answer, lambda q: results,
                            min_sources=2, max_sources=config.MAX_WEB_SOURCES)
    except Exception:
        return {"status": stamps.UNVERIFIED, "sources": [], "confidence": 0.0,
                "conflict": False, "contradicts_local": False}
```

Append to `bow/ragguard_wrap.py`:
```python
def guarded(inner):
    from rag_guard.service import query as _q
    from bow.ragguard_judge import judge_grounded
    from bow.ragguard_webverifier import run_web_verifier

    def retrieve_fn(text):
        try:
            return _q(text, 5)
        except Exception:
            return []

    return GuardedBrain(inner, retrieve_fn=retrieve_fn,
                        judge_fn=judge_grounded, verify_fn=run_web_verifier)
```

In `bow/daemon.py::main()`, right after the `Brain(...)` is constructed and BEFORE it's passed to `Daemon(...)` (CONFIRM the real variable name):
```python
    from bow.ragguard_wrap import guarded
    brain = guarded(brain)
```

- [ ] **Step 4: Run test to verify it passes** — `cd ~/bow && python3 -m pytest tests/test_ragguard_webverifier.py tests/test_ragguard_wrap.py -v` → PASS
- [ ] **Step 5: Commit** — `cd ~/bow && git add bow/ragguard_webverifier.py bow/ragguard_wrap.py bow/daemon.py tests/ && git commit -m "feat(bow): tool-enabled web verifier + daemon injection"`

---

## Phase 5 — Eval, coverage, docs

### Task 15: Real-corpus grounding eval (four expectation classes)

**Files:** Create `eval/__init__.py`, `eval/grounding_cases.py`, `eval/run_grounding.py`; Test `tests/test_eval_grounding.py`

**Interfaces:** `CASES` (labeled, keys `query` + one of `expect_grounded`/`expect_refusal`); `run(query_fn, verify_fn) -> dict` scoring `grounded_rate`, `refusal_rate`, and web-tier outcome counts. Uses `service.query` (real) + a **mocked** `verify_fn`. Does NOT use `evaluate.evaluate` (that scores `gold`/`expect_refusal` off a `RagGuard`, incompatible with these labels).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_grounding.py
from eval.grounding_cases import CASES
from eval.run_grounding import run
from rag_guard import stamps

def test_cases_shape():
    assert len(CASES) >= 10
    for c in CASES:
        assert "query" in c and ("expect_grounded" in c or "expect_refusal" in c)

def test_run_scores_metrics():
    def qf(t): return [{"id": "m#0", "text": t, "score": 0.9}] if "grounded" in t else []
    def vf(q, a): return {"status": stamps.WEB_VERIFIED, "sources": []}
    cases = [{"query": "grounded thing", "expect_grounded": True},
             {"query": "unknown thing", "expect_refusal": True}]
    res = run(qf, vf, cases=cases)
    assert 0.0 <= res["grounded_rate"] <= 1.0 and res["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails** — FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# eval/grounding_cases.py
"""Labeled cases over Jeff's real corpus. expect_grounded => notes cover it; expect_refusal
=> notes don't (CLI advises / Bow escalates to web). Tune after running the live index."""
CASES = [
    {"query": "what text format does Jeff prefer for deliverables", "expect_grounded": True},
    {"query": "what is Jeff's mortgage rate", "expect_grounded": True},
    {"query": "what is Bow", "expect_grounded": True},
    {"query": "what is rag-guard", "expect_grounded": True},
    {"query": "what vehicle is Lizzo", "expect_grounded": True},
    {"query": "Jeff's target MBA graduation", "expect_grounded": True},
    {"query": "what did Jeff hunt in the Pintlers", "expect_grounded": True},
    {"query": "capital of France", "expect_refusal": True},
    {"query": "current price of bitcoin right now", "expect_refusal": True},
    {"query": "who won the game last night", "expect_refusal": True},
]
```

```python
# eval/run_grounding.py
"""Run CASES through retrieval (+ mocked web verify) and report metrics."""
from __future__ import annotations

from eval.grounding_cases import CASES
from rag_guard import config


def run(query_fn, verify_fn, *, cases=None) -> dict:
    cases = cases if cases is not None else CASES
    grounded_hits = refuse_hits = grounded_ok = refuse_ok = 0
    web_outcomes = {}
    for c in cases:
        hits = query_fn(c["query"])
        support = max((h["score"] for h in hits), default=0.0)
        is_grounded = support >= config.MIN_SCORE
        if c.get("expect_grounded"):
            grounded_hits += 1; grounded_ok += int(is_grounded)
        if c.get("expect_refusal"):
            refuse_hits += 1; refuse_ok += int(not is_grounded)
            v = verify_fn(c["query"], "candidate")
            web_outcomes[v["status"]] = web_outcomes.get(v["status"], 0) + 1
    return {"n": len(cases),
            "grounded_rate": round(grounded_ok / grounded_hits, 3) if grounded_hits else None,
            "refusal_rate": round(refuse_ok / refuse_hits, 3) if refuse_hits else None,
            "web_outcomes": web_outcomes}
```

- [ ] **Step 4: Run test to verify it passes** — PASS (2)
- [ ] **Step 5: Commit** — `git add eval/ tests/test_eval_grounding.py && git commit -m "test(eval): grounding eval over real corpus"`

### Task 16: Coverage gate + full suite

- [ ] **Step 1:** `cd ~/rag-guard && PYTHONPATH=. python3 -m pytest -q --cov=rag_guard --cov-report=term-missing --cov-fail-under=90` → all pass, coverage ≥90%. If below, add in-scope tests (e.g. `index.load_index` corrupt-file branch, `webverify` conflict+cap branches).
- [ ] **Step 2:** `cd ~/bow && python3 -m pytest tests/test_ragguard_wrap.py tests/test_ragguard_judge.py tests/test_ragguard_webverifier.py tests/test_routing.py -q` → all pass.
- [ ] **Step 3:** `git add -A && git commit -m "test: close coverage gaps to 90% floor"`

### Task 17: README landscape + case study

- [ ] **Step 1:** Add "Where this sits in the landscape" — honestly name prior art: Corrective RAG (CRAG), Self-RAG, Guardrails AI / NeMo groundedness rails, RAGAS/TruLens faithfulness eval. State the differentiator: zero-dependency stdlib core, packaged authority-weighted cross-source corroboration tier, and a documented production integration (Bow + Claude Code hook). Note syndication detection as future work.
- [ ] **Step 2:** Add "Case study: running it on my own ops" — the three-tier truth ladder, both surfaces, confidence ladder with an example stamped answer.
- [ ] **Step 3:** `git add README.md && git commit -m "docs: landscape/prior-art + production case study"`

---

## Self-Review (author, post-QC)

- **QC findings applied:** Bow `ask(message, key)` arity + session-key threading fixed (T13); orphaned `rag_guard` role now consumed by a real cross-family judge (T12, new); authority-weighting enforced in the verdict (T9); coverage floor corrected to 90% + real CI command (T16); bow tests moved to `tests/` (all Phase-4 tasks); routing uses named constants, DEFAULT_ROLES only (T11); web verifier enables web tools + resolves the real binary + guards `None` from `result_dict` (T14); memory-source weighting implemented + tie-break test (T1/T2); fingerprint includes chunk params (T3); distinct `guard unavailable` / `web check failed` stamps (T8/T13); config source cap (T4/T9/T14); eval rewritten to a real runner with four outcome classes (T15); ≥2-doc fixtures avoid the 1-doc TF-IDF zero-score trap; `_cosine` deletion keeps coverage clean (T2).
- **Downscoped honestly:** syndication/echo-chamber detection deferred (needs source content) — documented in T9, README (T17), spec §11; `contradicts_local` remains a `Verdict` field but stays `False` in v1 (surfacing it is future work) — noted here so it isn't mistaken for complete.
- **CONFIRM steps** added everywhere the plan touches existing rag-guard/Bow code (T2, Phase-4 header, T13).
```
