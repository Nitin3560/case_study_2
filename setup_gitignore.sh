#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

touch .gitignore

patterns=(
"__pycache__/"
"*.pyc"
"outputs/"
"*.jsonl"
"ray_tmp/"
)

for p in "${patterns[@]}"; do
  grep -Fxq "$p" .gitignore || echo "$p" >> .gitignore
done

git rm -r --cached --ignore-unmatch __pycache__ 2>/dev/null || true
git rm -r --cached --ignore-unmatch outputs 2>/dev/null || true
git rm -r --cached --ignore-unmatch ray_tmp 2>/dev/null || true
git rm -r --cached --ignore-unmatch envs/__pycache__ evaluation/__pycache__ 2>/dev/null || true
git rm --cached --ignore-unmatch **/*.pyc 2>/dev/null || true
git rm --cached --ignore-unmatch **/*.jsonl 2>/dev/null || true