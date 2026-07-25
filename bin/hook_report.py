"""Did the injected passages get used? Joins hook-fire records against Claude Code session
transcripts and reports it, so gate changes stop being guesses.

Run:  PYTHONPATH=. python3 bin/hook_report.py [--transcripts DIR] [--log PATH]

METHOD. Each fire record carries the session_id and the absolute paths of the notes it
injected. Claude Code writes one transcript per session (<session_id>.jsonl) with a
timestamp on every entry. So for each fire we ask: after this moment, in this session, did
any tool call touch one of the files we injected?

HOW TO READ THE RESULT -- this is a proxy, and it cuts both ways:
  * touched   -> the notes were topically right. The injection did NOT save the read,
                 but it was relevant.
  * untouched -> EITHER the injection was noise, OR it was good enough that the agent
                 never needed the file. One number cannot separate those.
So `touched` is best read as a LOWER BOUND on relevance, not a usage rate. The signal that
actually decides push-vs-pull is the untouched rate being extreme: near 0% means the
injection is doing nothing the agent could not do itself; very high means it is on-topic
but not saving work. Report both and say which.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter

from rag_guard import config, hooklog

DEFAULT_TRANSCRIPTS = os.path.expanduser("~/.claude/projects/-Users-jeffreyotterson")


def _iso_to_epoch(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _strings(value):
    """Every string anywhere in a decoded tool input."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def session_file_touches(transcript_path):
    """[(epoch, tool_input_text)] for every tool call the session made.

    Works on the DECODED entry rather than the raw line: tool schemas vary (Read/Edit pass
    file_path, Bash embeds the path inside a command string where the quotes are
    backslash-escaped), and matching raw text misses the escaped cases. Restricted to
    tool_use blocks on purpose -- a path merely mentioned in prose is not a touch."""
    touches = []
    try:
        with open(transcript_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                when = _iso_to_epoch(entry.get("timestamp"))
                if when is None:
                    continue
                content = (entry.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        blob = " ".join(_strings(part.get("input") or {}))
                        if blob:
                            touches.append((when, blob))
    except OSError:
        return []
    return touches


def analyze(events, transcripts_dir):
    fires = [e for e in events if e.get("fired")]
    cache = {}
    rows = []
    for fire in fires:
        sid = fire.get("session_id")
        if not sid:
            continue
        if sid not in cache:
            cache[sid] = session_file_touches(os.path.join(transcripts_dir, f"{sid}.jsonl"))
        injected = {c["path"] for c in fire.get("chunks", []) if c.get("path")}
        if not injected:
            continue
        after = fire.get("ts", 0)
        # Substring containment against the decoded tool input: exact, and unlike a path
        # regex it survives paths containing spaces.
        used = {p for p in injected
                for when, blob in cache[sid] if when >= after and p in blob}
        rows.append({"fire": fire, "injected": injected, "used": used})
    return rows


def report(events, transcripts_dir, out=sys.stdout):
    total = len(events)
    if not total:
        print("No hook-fire records yet. Is RAG_GUARD_HOOK_LOG=1 set in the hook command?",
              file=out)
        return 1

    reasons = Counter(e.get("reason", "?") for e in events)
    fires = [e for e in events if e.get("fired")]
    interactive = [e for e in events if e.get("reason") != "not_interactive"]

    print(f"hook-fire log: {total} decisions", file=out)
    print(f"  suppressed as non-interactive : {reasons.get('not_interactive', 0)}", file=out)
    if interactive:
        print(f"  interactive decisions         : {len(interactive)}", file=out)
        print(f"  ...of which FIRED             : {len(fires)} "
              f"({len(fires)/len(interactive):.1%})", file=out)
    for reason, n in reasons.most_common():
        if reason not in ("fired", "not_interactive"):
            print(f"  silent [{reason}]: {n}", file=out)

    if fires:
        chars = sorted(sum(c.get("chars", 0) for c in f.get("chunks", [])) for f in fires)
        ms = sorted(f.get("elapsed_ms", 0) for f in fires)
        print(f"\n  median chars injected per fire: {chars[len(chars)//2]:,}", file=out)
        print(f"  median retrieval latency      : {ms[len(ms)//2]} ms", file=out)

    rows = analyze(events, transcripts_dir)
    print(f"\nUSAGE JOIN (fires whose session transcript was found): {len(rows)}", file=out)
    if not rows:
        print("  Not enough joined data yet. Let it run, then re-check.", file=out)
        return 0

    touched = [r for r in rows if r["used"]]
    print(f"  injected note later touched in-session : {len(touched)}/{len(rows)} "
          f"({len(touched)/len(rows):.1%})", file=out)
    print(f"  never touched                          : {len(rows)-len(touched)}/{len(rows)} "
          f"({1-len(touched)/len(rows):.1%})", file=out)
    print("\n  Read this as a LOWER BOUND on relevance, not a usage rate: an untouched fire\n"
          "  is either noise OR an injection good enough that no read was needed.", file=out)

    hot = Counter()
    for r in rows:
        for p in r["used"]:
            hot[os.path.basename(p)] += 1
    if hot:
        print("\n  most-used injected notes:", file=out)
        for name, n in hot.most_common(8):
            print(f"    {n:>4}  {name}", file=out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default=None, help="hook-fire jsonl (default: config path)")
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    args = ap.parse_args(argv)
    events = hooklog.read_events(args.log or config.hook_log_path())
    return report(events, args.transcripts)


if __name__ == "__main__":
    sys.exit(main())
