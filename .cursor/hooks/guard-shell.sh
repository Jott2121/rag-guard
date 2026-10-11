#!/usr/bin/env bash
# beforeShellExecution guard. Reads {"command": "...", "cwd": "..."} on stdin.
# Denies irreversible / publishing commands; asks on network fetches. Fails closed (hooks.json failClosed: true).
set -euo pipefail
input="$(cat)"
# Exit 2: well-formed payload, but command is missing, blank, or not a string.
# Any other non-zero status (invalid JSON, non-object) stays non-zero so failClosed denies.
set +e
cmd="$(printf '%s' "$input" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(1)
if not isinstance(data, dict):
    sys.exit(1)
cmd = data.get("command", None)
if not isinstance(cmd, str) or not cmd.strip():
    sys.exit(2)
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
if [ "$parse_status" -ne 0 ]; then
  exit "$parse_status"
fi
# Flags may sit anywhere, not only beside the subcommand.
# Denied: `git push origin --force`, `git push origin -f`, `git push -fu`,
# `git reset HEAD --hard`. A plain `git push` still asks below.
set -f
push=0
reset=0
force=0
hard=0
for token in $cmd; do
  token="${token#\'}"
  token="${token%\'}"
  token="${token#\"}"
  token="${token%\"}"
  case "$token" in
    push) push=1 ;;
    reset) reset=1 ;;
    --force|--force=*|--force-with-lease|--force-with-lease=*) force=1 ;;
    --hard|--hard=*) hard=1 ;;
    -*)
      case "$token" in
        --*) ;;
        *)
          case "${token#-}" in
            *f*) force=1 ;;
          esac
          ;;
      esac
      ;;
  esac
done
set +f
if [ "$push" -eq 1 ] && [ "$force" -eq 1 ]; then
  deny "Force-push blocked by repo policy."
fi
if [ "$reset" -eq 1 ] && [ "$hard" -eq 1 ]; then
  deny "History rewrite blocked by repo policy."
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
