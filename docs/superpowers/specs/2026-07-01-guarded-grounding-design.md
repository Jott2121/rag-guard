# Guarded Grounding — wiring rag-guard into ops

**Date:** 2026-07-01
**Status:** Design approved; implementation pending (Sonnet 5 builds, Opus judges)
**Owner:** Jeff Otterson

## 1. Problem & goal

`rag-guard` today is an inert, well-tested library — a "tool in a drawer." Its guards
(refuse-when-unsupported, groundedness, PII redaction) never fire because nothing calls
`RagGuard.answer()`, and it has no real corpus, no persistence, and no query entry point.

**Goal:** make rag-guard fire *automatically* on every conversation Jeff has with an AI, grounded
first in Jeff's own knowledge base (auto-memory + Obsidian wikis) and, when that can't support the
answer, **escalating to the web and verifying the answer across independent sources** before
delivering it — every answer stamped with how confident to be, and cited.

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Surfaces | **Both**: Claude Code CLI hook (advisory) **and** Bow wrap (teeth) |
| Corpus | **Memory + wikis** (auto-memory weighted highest) + global `CLAUDE.md` |
| CLI no-support behavior | **Advise, don't block** |
| Every CLI session | **Yes** — registered in global `~/.claude/settings.json` |
| Freshness | **Auto** — fingerprint check on use + nightly launchd rebuild backstop |
| Bow local behavior | **Corrective loop**, cap **2 retries** |
| **Web escalation** | **Auto** when local support is low |
| **Corroboration bar** | **2 independent sources** must agree → "web-verified" |
| **Fallback** | If unverifiable anywhere → **labeled best-effort** (⚠ UNVERIFIED), never empty |
| **Web tier scope** | **In v1** |
| Delivery | Sonnet 5 implements **test-first**; Opus judges adversarially before "done" |

## 3. Non-goals (v1)

- Guarding Bow's non-interactive call sites (routines, research digest/deepdive, reply-drafter,
  skill-miner, frontier grade/audio). Interactive chat only. Documented, not silently skipped.
- Semantic/embedding retrieval for the *local* corpus. v1 is stdlib TF-IDF; embeddings swap in later
  behind the same `retrieve()` seam with no adapter changes.
- Post-answer PII redaction on the Claude Code side (a prompt hook runs *before* the answer). PII
  redaction applies on the Bow side, where we control the finished answer.
- An inverted index / ANN. Corpus is ~11 MB; the O(N) scan is fine at this scale.

## 4. Architecture: one core, two adapters, a three-tier truth ladder

```
                 Jeff's notes (memory + wikis + CLAUDE.md)
                                 │
                      ┌──────────▼───────────┐
                      │   rag_guard CORE     │  corpus → persisted, fingerprinted index
                      │  retrieve() + guards │  → retrieve + groundedness + PII
                      └──────┬───────┬───────┘
                             │       │
           ┌─────────────────┘       └──────────────────┐
           ▼                                             ▼
     CC HOOK adapter (advisory)              BOW WRAP adapter (teeth)
     injects grounding + a web-              runs the truth ladder itself
     verification PROTOCOL; the
     main agent (Opus) executes it

        THE TRUTH LADDER (both surfaces, same logic):
        ┌─────────────────────────────────────────────────────────┐
        │ Tier 1  LOCAL   retrieve from notes → groundedness check  │
        │           │ grounded → ✔ GROUNDED (cite notes)            │
        │           ▼ not grounded (support low)                    │
        │ Tier 2  WEB     search (firecrawl / fetch MCP / WebSearch)│
        │           ▼ candidate answer found                        │
        │ Tier 3  CORROBORATE  ≥2 independent sources agree?        │
        │           ├ yes → ✔ WEB-VERIFIED (cite sources)           │
        │           ├ 1 only → ⚠ SINGLE SOURCE (cite)               │
        │           ├ disagree → ⚠ SOURCES CONFLICT (show both)     │
        │           └ none → ⚠ UNVERIFIED (best-effort, never empty)│
        └─────────────────────────────────────────────────────────┘
```

**Confidence ladder (the stamp on every answer):**

| Stamp | Meaning |
|---|---|
| ✔ GROUNDED | backed by Jeff's notes |
| ✔ WEB-VERIFIED | ≥2 independent sources agree — cited |
| ⚠ SINGLE SOURCE | found on the web, one source only — cited, lower confidence |
| ⚠ SOURCES CONFLICT | sources disagree — both shown |
| ⚠ UNVERIFIED | couldn't back it anywhere — best-effort, flagged |

**Source-authority weighting** (used to pick/weight corroborating sources): primary/official
(government, company announcements, official docs) > established news > social (an X post is a
*lead to verify*, not proof). Also flag when a web answer **contradicts** Jeff's own notes.

## 5. Component design

### 5.1 Core — corpus builder (`rag_guard/corpus.py`, new)

- `build_corpus(sources, *, chunk_chars=800, overlap=100, exclude=...) -> list[dict]`
- Walks configured sources, **markdown text only**, chunks with overlap, emits stable
  `{"id": f"{relpath}#{chunk_idx}", "text": chunk, "source": <tag>}`.
- **Sources (priority order):** (1) `~/.claude/projects/-Users-jeffreyotterson/memory/`;
  (2) Obsidian wikis `~/Documents/*-Wiki`; (3) `~/.claude/CLAUDE.md`.
- **Exclusions:** NotebookLM audio (`.m4a/.aiff`), PDFs, `.gpx`, images, `node_modules`, `.git`,
  `.venv`, caches, code/data dirs.
- Memory-source chunks carry a priority weight (tie-break toward purpose-built facts).

### 5.2 Core — persistent index (`rag_guard/index.py`, new)

Today `Retriever.__init__` eagerly rebuilds the whole index (`retriever.py:35-43`); no persistence.
Add:
- `Retriever.save(path)` / `Retriever.load(path)` bypassing the eager rebuild — serialize `docs`,
  `idf`, `_vecs`, and **precomputed per-doc L2 norms**.
- `get_index(cache_path, sources) -> Retriever`: load cache; compare a **corpus fingerprint** (hash
  of file list + mtimes + chunk params); rebuild (incrementally where practical) if stale; else reuse.

### 5.3 Core — query API (`rag_guard/service.py` + `rag_guard/cli.py`, new)

- `service.query(text, k=5)`: module-level **warm singleton** retrieval for Bow. (No separate `get_guard()` — the Bow adapter composes the guards directly, so a warm `RagGuard` singleton isn't needed.)
- `cli.py main()`: stdin/argv query → load **cached** index → **retrieval only** → JSON stdout. Wire
  as `[project.scripts]` in `pyproject.toml`. This is the hook's entry point.

### 5.4 Freshness

- Per-query cheap fingerprint check (mtimes + count) → rebuild if changed (corpus is tiny, sub-second).
  A saved note is searchable on the next message. **Backstop:** nightly launchd rebuild. Bow's warm
  singleton re-checks the fingerprint and reloads on change.

### 5.5 Web verification tier (`rag_guard/webverify.py`, new)

The Tier-2/Tier-3 engine, shared in spirit by both surfaces.

- `verify_claim(query, candidate_answer, *, min_sources=2) -> Verdict` where
  `Verdict = {status, sources:[{url, publisher, authority_tier, supports:bool}], confidence, conflict:bool, contradicts_local:bool}`
  and `status ∈ {web_verified, single_source, conflict, unverified}`.
- **Search:** firecrawl MCP (`firecrawl_search`, the configured provider) primary; fetch MCP /
  built-in WebSearch/WebFetch as fallbacks.
- **Independence rule (v1):** sources count as independent only if different registered
  domains / publishers. Full **syndication/echo-chamber detection is future work** — it needs
  source *content* the injected search seam doesn't return in v1; documented as a known
  limitation, not silently assumed.
- **Contradiction flag:** `Verdict.contradicts_local` is defined in the shape but stays `False`
  in v1; computing/surfacing a web-vs-notes contradiction is future work.
- **Authority weighting:** primary/official > established news > social. An X post is a lead; it must
  be corroborated by a higher tier to count.
- **Bounded effort:** cap searches/fetches (config) so latency/cost stay sane; only invoked on the
  fallback path.

### 5.6 CC hook adapter (`bin/hook_userpromptsubmit.py`, new)

- Registered under `UserPromptSubmit` in `~/.claude/settings.json` → fires in **every** session.
- Reads hook JSON from stdin; retrieves top-k from the cached index (**no model call**); computes
  support score. Emits `{"hookSpecificOutput": {"hookEventName":"UserPromptSubmit","additionalContext":<text>}}`.
- `additionalContext` = retrieved passages + support summary + **the truth-ladder protocol**:
  *"If your notes don't cover this and it may be newer than your training cutoff, search the web,
  corroborate the claim across ≥2 independent sources (prefer primary/official), cite them, and flag
  conflicts or anything that contradicts Jeff's notes. Stamp the answer's confidence."*
  The **main agent (Opus) executes** the web steps with its own web tools — the hook only injects the
  protocol + the local context.
- **Silent when nothing relevant** (top score < threshold → inject little/nothing).
- **Hard constraints:** 30 s cap (target < 1 s); retrieval-only; **fail-open** (never break the prompt).

### 5.7 Bow wrap adapter (`bow/ragguard_wrap.py`, new)

- **Integration point:** wrap `Brain.ask()` — the single chokepoint every interactive answer flows
  through (Telegram/voice/PTT/Ghost-group); its return value *is* the delivered message
  (`bow/brain.py`, result ~line 295 → return ~line 300). Teeth; cannot be bypassed.
- **Injection:** subclass/decorator injected at `bow/daemon.py::main()` where `Brain(...)` is built
  (~line 801) and passed to `Daemon(...)` (~line 806). One wrapper covers `_ask` and
  `_process_voice_text`.
- **Flow (the truth ladder with teeth):**
  1. Brain answers.
  2. **Tier 1:** groundedness vs. local corpus (separate **cross-family** call, new routing role
     `rag_guard`, fail-**closed** `loopeval.py` pattern — not fail-open `judge.py`). Grounded →
     ✔ GROUNDED, deliver.
  3. Not grounded → corrective loop: reformulate, re-retrieve, re-ask brain, up to **2 retries**.
  4. Still not grounded locally → **Tier 2/3:** `webverify.verify_claim(...)` (routing role
     `rag_guard_web`, tool-enabled). Stamp per the confidence ladder (✔ WEB-VERIFIED / ⚠ SINGLE
     SOURCE / ⚠ SOURCES CONFLICT).
  5. If unverifiable anywhere → deliver best answer stamped **⚠ UNVERIFIED**; if retrieval + web both
     found nothing, stamp escalates to "answering from general knowledge only." **Never empty.**
  6. PII redaction on the final delivered answer; citations appended.
- Skips `_rotation_flush`/self-heal (no user-facing answer), deterministic fast-paths
  (scenes/lights/TV), and slash-commands (no factual claims).

## 6. Interfaces & contracts

- **Hook stdout:** `{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"..."}}`, exit 0.
- **Provider seam (unchanged):** any object with `complete(prompt)->str`. Bow guard + web-verifier
  reuse the `claude -p` subprocess pattern via `routing.resolve('rag_guard' | 'rag_guard_web')`.
- **Retriever hit:** `{"id","text","score"}`. **`RagGuard.answer`:** `{answer,refused,grounded,support,sources,trace}`.
- **Verdict (new):** `{status, sources:[{url,publisher,authority_tier,supports}], confidence, conflict, contradicts_local}`.

## 7. Error handling / fail modes

- **CC hook: fail-open.** Any error → inject nothing, exit 0. Never block the prompt.
- **Bow guard: fail-toward-labeled.** Guard/verifier model or web error → deliver the brain's answer
  stamped `⚠ guard unavailable` / `⚠ web check failed`, never drop the answer.
- **Web unreliable:** corroboration prefers primary/authoritative sources and detects syndication to
  resist echo-chamber agreement; conflicts are surfaced, not hidden.
- **Index cache corrupt/missing:** rebuild; if that fails, degrade gracefully (hook silent; Bow
  delivers unstamped with a logged warning).

## 8. Testing strategy (test-first)

- **Unit:** corpus builder (chunking, id stability, exclusions); index save/load + fingerprint
  invalidation + norms; hook output contract (valid JSON, silent-when-empty, fail-open); Bow wrap
  (grounded pass-through, loop retries, web escalation, each confidence stamp, PII, never-empty);
  webverify (independence rule, authority weighting, ≥2 corroboration, conflict detection) with
  **mocked web** for determinism.
- **Eval:** extend the harness with a labeled set over the **real** corpus (grounded / web-verified /
  conflict / unverified expectations). Keep the ~99% coverage bar. `FakeProvider` + mock web keep CI
  key-free and network-free.

## 9. Delivery model

1. Opus writes the implementation plan (test-first, phased).
2. Sonnet 5 implements phase by phase.
3. Deterministic gate: tests + coverage pass.
4. **Opus judges** adversarially (correctness, matches Bow's real code, web-tier edge cases,
   testability) → green-light or fix list. Fix loop until clean.

## 10. Phasing (web tier in v1)

- **Phase 0 — Core:** corpus + persistent fingerprinted index + query API.
- **Phase 1 — Freshness:** fingerprint auto-rebuild + nightly launchd backstop.
- **Phase 2 — Web verify engine:** `webverify.py` (search + independence + corroboration + authority)
  with mocked-web tests.
- **Phase 3 — CC hook:** adapter + truth-ladder protocol injection + register in global settings.
- **Phase 4 — Bow wrap:** local corrective loop + web tier + confidence stamps + routing roles.
- **Phase 5 — Tests/eval/QC:** labeled eval on real corpus; Opus judge gate.

Each phase boundary is a viable stop.

## 11. Risks

- **Latency/cost of the web tier** (mitigated: only on fallback, bounded searches/fetches, cap 2 retries).
- **Web ≠ truth / echo chambers** (mitigated in v1: authority weighting so social-only can't verify, and conflicts surfaced not hidden; full syndication detection is future work).
- **Defining "independent"** (v1: distinct registered domains/publishers; content-level syndication detection deferred).
- **Softened local guarantee:** Bow always answers (labeled), not hard-refuse — intended for a
  personal chief-of-staff.
- **Hook latency** (mitigated: cached index, fingerprint-only hot path, no model call).
- **TF-IDF lexical misses** (accepted for v1; embeddings later behind the seam).
- **Hook noise in unrelated sessions** (mitigated: silent below threshold).
- **v1 Bow teeth cover interactive chat only** (documented).
