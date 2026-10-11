#!/usr/bin/env bash
# beforeShellExecution guard. Reads {"command": "...", "cwd": "..."} on stdin.
# Denies irreversible / publishing commands; asks on network fetches. Fails closed (hooks.json failClosed: true).
set -euo pipefail
input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("command",""))')"
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
case "$cmd" in
  *"push --force"*|*"push -f"*|*"--force-with-lease"*) deny "Force-push blocked by repo policy." ;;
  *"reset --hard"*|*"rebase"*)                        deny "History rewrite blocked by repo policy." ;;
  *"git tag"*|*"twine upload"*|*"uv publish"*|*"gh release"*) deny "Releases/publishing are human-only." ;;
  *"rm -rf /"*|*"rm -rf ~"*|*'rm -rf $HOME'*)         deny "Destructive delete blocked." ;;
  *"curl"*"| sh"*|*"curl"*"| bash"*|*"wget"*"| sh"*)   deny "Pipe-to-shell blocked." ;;
  *"git push"*)                                       ask  "Agent wants to push a branch. Allow?" ;;
  *"curl"*|*"wget"*)                                  ask  "Agent wants network access: $cmd" ;;
esac
json_msg allow "" ""
