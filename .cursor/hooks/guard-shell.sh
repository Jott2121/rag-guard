#!/usr/bin/env bash
# beforeShellExecution guard. Reads {"command": "...", "cwd": "..."} on stdin.
# Denies irreversible / publishing commands; asks on network fetches. Fails closed (hooks.json failClosed: true).
set -euo pipefail
input="$(cat)"
# Exit 2: well-formed payload, but command is missing, blank, or not a string.
# Any other non-zero status (invalid JSON, non-object) stays non-zero so failClosed denies.
set +e
cmd="$(printf '%s' "$input" | python3 -c '
import json, shlex, sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(1)
if not isinstance(data, dict):
    sys.exit(1)
cmd = data.get("command", None)
if not isinstance(cmd, str) or not cmd.strip():
    sys.exit(2)

# Git accepts any unambiguous prefix of a long option, so --force-w is
# --force-with-lease. Judge each simple command on its own so curl -f or
# rm -rf beside a plain git push is not a force-push.
FORCE = ("force", "force-with-lease", "force-if-includes")

def segments(text):
    parts, buf, quote, i = [], [], None, 0
    while i < len(text):
        c = text[i]
        if quote:
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'\''", "\""):
            quote = c
            buf.append(c)
            i += 1
            continue
        if c in "\n;|":
            parts.append("".join(buf)); buf = []; i += 1; continue
        if text.startswith("&&", i) or text.startswith("||", i):
            parts.append("".join(buf)); buf = []; i += 2; continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return [p for p in parts if p.strip()]

def tokens_of(text):
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()

def is_git_bin(token):
    name = token.replace("\\", "/").rsplit("/", 1)[-1]
    return name == "git" or name.lower() == "git.exe"

def invocation_at(tokens, i):
    """Subcommand and args after a git binary at tokens[i]."""
    i += 1
    while i < len(tokens):
        t = tokens[i]
        if t in ("-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace"):
            i += 2
            continue
        if t.startswith(("--git-dir=", "--work-tree=", "--exec-path=", "--namespace=")):
            i += 1
            continue
        if t in ("-p", "--paginate", "--no-pager", "--bare", "--no-replace-objects"):
            i += 1
            continue
        if t.startswith("-"):
            return None, []
        return t, tokens[i + 1:]
    return None, []

def long_name(token):
    if not token.startswith("--") or token.startswith("--no-"):
        return None
    return token[2:].split("=", 1)[0]

def is_force(token):
    if token.startswith("-") and not token.startswith("--"):
        return "f" in token[1:]
    name = long_name(token)
    return bool(name) and any(opt.startswith(name) for opt in FORCE)

def is_hard(token):
    name = long_name(token)
    return bool(name) and "hard".startswith(name)

def walk(text, depth=0):
    if depth > 3:
        return
    for segment in segments(text):
        tokens = tokens_of(segment)
        i = 0
        while i < len(tokens):
            if is_git_bin(tokens[i]):
                sub, args = invocation_at(tokens, i)
                if sub == "push" and any(is_force(t) for t in args):
                    sys.exit(3)
                if sub == "reset" and any(is_hard(t) for t in args):
                    sys.exit(4)
            elif any(ch.isspace() for ch in tokens[i]):
                walk(tokens[i], depth + 1)
            i += 1

walk(cmd)

sys.stdout.write(cmd)
')"
parse_status=$?
set -e
# json.dumps so a command with quotes, backslashes, or newlines stays valid JSON.
json_msg() {
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
perm, user, agent = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {"permission": perm}
if user:
    payload["user_message"] = user
if agent:
    payload["agent_message"] = agent
sys.stdout.write(json.dumps(payload) + "\n")
PY
}
deny() { json_msg deny "$1" "$1 Open a PR instead; a human will do this step."; exit 0; }
ask()  { json_msg ask "$1" ""; exit 0; }
if [ "$parse_status" -eq 2 ]; then
  deny "Shell guard blocked a missing command."
fi
if [ "$parse_status" -eq 3 ]; then
  deny "Force-push blocked by repo policy."
fi
if [ "$parse_status" -eq 4 ]; then
  deny "History rewrite blocked by repo policy."
fi
if [ "$parse_status" -ne 0 ]; then
  exit "$parse_status"
fi
case "$cmd" in
  *"rebase"*)                                        deny "History rewrite blocked by repo policy." ;;
  *"git tag"*|*"twine upload"*|*"uv publish"*|*"gh release"*) deny "Releases/publishing are human-only." ;;
  *"rm -rf /"*|*"rm -rf ~"*|*'rm -rf $HOME'*)         deny "Destructive delete blocked." ;;
  *"curl"*"| sh"*|*"curl"*"| bash"*|*"wget"*"| sh"*)   deny "Pipe-to-shell blocked." ;;
  *"git push"*)                                       ask  "Agent wants to push a branch. Allow?" ;;
  *"curl"*|*"wget"*)                                  ask  "Agent wants network access: $cmd" ;;
esac
json_msg allow "" ""
