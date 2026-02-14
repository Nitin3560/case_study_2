#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$(dirname "$ROOT"):${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

ENV_CONFIG="${ENV_CONFIG:-$ROOT/configs/env_easy.yaml}"
SEEDS="${SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/camera_ready}"
mkdir -p "$OUT_DIR"

# If CHECKPOINT is unset and SKIP_TRAIN=0, run RLlib training first.
SKIP_TRAIN="${SKIP_TRAIN:-1}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ "$SKIP_TRAIN" == "0" ]]; then
  python "$ROOT/training/train_rllib_ppo_ctde.py" \
    --env-config "$ENV_CONFIG" \
    --stop-iters "${STOP_ITERS:-120}" \
    --num-workers "${NUM_WORKERS:-0}"
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: CHECKPOINT is required (export CHECKPOINT=/abs/path/to/checkpoint_000000)."
  exit 1
fi

echo "Writing environment metadata..."
{
  echo "date_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$(python --version 2>&1)"
  echo "ray=$(python - <<'PY'
import importlib
try:
    m=importlib.import_module("ray")
    print(m.__version__)
except Exception:
    print("missing")
PY
)"
  echo "torch=$(python - <<'PY'
import importlib
try:
    m=importlib.import_module("torch")
    print(m.__version__)
except Exception:
    print("missing")
PY
)"
  echo "env_config=$ENV_CONFIG"
  echo "checkpoint=$CHECKPOINT"
  echo "seeds=$SEEDS"
} > "$OUT_DIR/env_info.txt"

echo "Running RLlib vs static summary..."
python "$ROOT/evaluation/eval_case2_summary.py" \
  --env-config "$ENV_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --seeds "$SEEDS" \
  --out "$OUT_DIR/case2_summary.json"

echo "Running baseline summary (static/fixed_relay/greedy)..."
python "$ROOT/evaluation/eval_baselines_summary.py" \
  --env-config "$ENV_CONFIG" \
  --seeds "$SEEDS" \
  --methods "static,fixed_relay,greedy" \
  --out "$OUT_DIR/baselines_summary.json"

echo "Generating camera-ready report markdown..."
python - <<'PY'
import json, os
root=os.environ["OUT_DIR"] if "OUT_DIR" in os.environ else ""
if not root:
    raise SystemExit("OUT_DIR env var missing")
case=json.load(open(os.path.join(root,"case2_summary.json")))
base=json.load(open(os.path.join(root,"baselines_summary.json")))
env_info=open(os.path.join(root,"env_info.txt")).read().strip()

def fmt(ms):
    return f"{ms[0]:.2f} ± {ms[1]:.2f}"

lines=[]
lines.append("# Camera-Ready Evidence Report")
lines.append("")
lines.append("## Run Metadata")
lines.append("```text")
lines.append(env_info)
lines.append("```")
lines.append("")
lines.append("## Primary Comparison (RLlib PPO-CTDE vs Static)")
lines.append("| Metric | RLlib PPO-CTDE | Static |")
lines.append("|---|---:|---:|")
lines.append(f"| Delivered completion (%) | {fmt(case['rllib']['task_completion_pct'])} | {fmt(case['static']['task_completion_pct'])} |")
lines.append(f"| Throughput (bps) | {fmt(case['rllib']['throughput_bps'])} | {fmt(case['static']['throughput_bps'])} |")
lines.append(f"| Mean AoI (steps) | {fmt(case['rllib']['mean_aoi_steps'])} | {fmt(case['static']['mean_aoi_steps'])} |")
lines.append("")
lines.append("## Baseline Sweep")
lines.append("| Method | Delivered completion (%) | Throughput (bps) | Mean AoI (steps) |")
lines.append("|---|---:|---:|---:|")
for m in base["methods"]:
    s=base["summary"][m]
    lines.append(f"| {m} | {fmt(s['task_completion_pct'])} | {fmt(s['throughput_bps'])} | {fmt(s['mean_aoi_steps'])} |")
lines.append("")
lines.append("## Notes")
lines.append("- Completion is `task_delivery_pct` (delivered tasks only).")
lines.append("- RLlib eval uses deterministic inference (`explore=False`).")
lines.append("- Use this report with raw JSON artifacts in the same folder.")

out=os.path.join(root,"camera_ready_report.md")
with open(out,"w",encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(out)
PY

echo "Done. Outputs:"
echo "  $OUT_DIR/case2_summary.json"
echo "  $OUT_DIR/baselines_summary.json"
echo "  $OUT_DIR/camera_ready_report.md"
