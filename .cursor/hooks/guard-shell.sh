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
# rm -rf beside a plain git push is not a force-push. Path-qualified git,
# wrappers, shell -c strings, and a later nested git count too.
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

def basename(token):
    return token.rsplit("/", 1)[-1]

def is_git(token):
    if token == "git":
        return True
    if basename(token) != "git":
        return False
    return token.startswith(("/", "./", "../", "~/"))

def is_shell(token):
    return basename(token) in ("sh", "bash", "dash", "zsh", "ksh", "ash")

def git_invocations(tokens):
    found = []
    i = 0
    n = len(tokens)
    while i < n:
        if not is_git(tokens[i]):
            i += 1
            continue
        j = i + 1
        sub = None
        while j < n:
            t = tokens[j]
            if t in ("-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace"):
                j += 2
                continue
            if t.startswith(("--git-dir=", "--work-tree=", "--exec-path=", "--namespace=")):
                j += 1
                continue
            if t in ("-p", "--paginate", "--no-pager", "--bare", "--no-replace-objects"):
                j += 1
                continue
            if not t.startswith("-"):
                sub = t
            break
        if sub is not None:
            found.append((sub, tokens[j + 1:]))
        i += 1
    return found

def shell_scripts(tokens):
    scripts = []
    n = len(tokens)
    for i, token in enumerate(tokens):
        if not is_shell(token):
            continue
        j = i + 1
        while j < n:
            t = tokens[j]
            if t == "--":
                break
            if t.startswith("--"):
                j += 1
                continue
            if t.startswith("-") and "c" in t[1:]:
                if j + 1 < n:
                    scripts.append(tokens[j + 1])
                break
            if not t.startswith("-"):
                break
            j += 1
    return scripts

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

def rewrite_status(command):
    for segment in segments(command):
        tokens = tokens_of(segment)
        for sub, args in git_invocations(tokens):
            if sub == "push" and any(is_force(t) for t in args):
                return 3
            if sub == "reset" and any(is_hard(t) for t in args):
                return 4
            if sub == "submodule" and "foreach" in args:
                for arg in args[args.index("foreach") + 1:]:
                    if arg.startswith("-"):
                        continue
                    status = rewrite_status(arg)
                    if status:
                        return status
        for script in shell_scripts(tokens):
            status = rewrite_status(script)
            if status:
                return status
    return 0

status = rewrite_status(cmd)
if status:
    sys.exit(status)

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
