#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# This script is never invoked by tests. Publishing needs the user's authenticated CLI.
command -v gh >/dev/null || { echo "GitHub CLI (gh) required on your authenticated machine." >&2; exit 2; }
repo="${1:-RyoSpiralArchitect/self-fold}"
visibility="${2:-private}"
[[ "$visibility" == private || "$visibility" == public ]] || exit 2
[[ -d .git ]] || { echo "Restore the included Git bundle first; see README." >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Commit or review local changes before publication." >&2; exit 2; }
if git remote get-url origin >/dev/null 2>&1; then
  echo "origin already exists; inspect it instead of replacing it." >&2; exit 2
fi
gh auth status
gh repo create "$repo" "--$visibility" --source=. --remote=origin --push   --description "Semantic self-description recurrence across text and latent states; protocol, probes and reproducible research harness"
