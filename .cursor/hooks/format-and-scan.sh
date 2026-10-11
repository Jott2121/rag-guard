#!/usr/bin/env bash
# afterFileEdit (notification-only): format the edited Python file and warn on employer-data / secret patterns.
# It can't block the edit; CI's no-employer-data job is the real gate.
# Employer terms are checked by hash (bin/check_no_employer_data.py --files); a scanner error also warns.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
f="$(python3 -c 'import sys,json; print(json.load(sys.stdin).get("file_path",""))')"
[ -f "$f" ] || exit 0
case "$f" in *.py) command -v ruff >/dev/null && ruff format -q "$f" && ruff check -q --fix "$f" ;; esac
warn() {
  mkdir -p "$root/.cursor/hook-logs"
  echo "$(date -u +%FT%TZ) WARN $1 in $f" >> "$root/.cursor/hook-logs/warnings.log"
}
python3 "$root/bin/check_no_employer_data.py" --files "$f" >/dev/null 2>&1 || warn "blocked employer term (or scanner error)"
if grep -EIiq '\bCUI\b|\bITAR\b|EXPORT CONTROLLED|BEGIN (RSA|OPENSSH) PRIVATE KEY|xai-[A-Za-z0-9]{20,}' "$f"; then
  warn "export-control/secret pattern"
fi
exit 0
