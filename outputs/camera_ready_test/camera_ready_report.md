# Camera-Ready Evidence Report

## Run Metadata
```text
date_utc=2026-02-13T10:55:52Z
git_commit=f6f024c8190a6092f13d31d8bb7f2debfd820de6
python=Python 3.12.12
ray=2.53.0
torch=2.10.0
env_config=/Users/nitin/Desktop/case_study_2/configs/env_easy.yaml
checkpoint=/Users/nitin/Desktop/case_study_2/outputs/checkpoints/rllib_ppo_ctde/PPO_2026-02-13_02-19-47/PPO_ctde_env_c3054_00000_0_2026-02-13_02-19-48/checkpoint_000000
seeds=1,2
```

## Primary Comparison (RLlib PPO-CTDE vs Static)
| Metric | RLlib PPO-CTDE | Static |
|---|---:|---:|
| Delivered completion (%) | 40.00 ± 14.14 | 0.00 ± 0.00 |
| Throughput (bps) | 1184.00 ± 603.40 | 806.67 ± 273.41 |
| Mean AoI (steps) | 176.25 ± 5.30 | 215.21 ± 2.06 |

## Baseline Sweep
| Method | Delivered completion (%) | Throughput (bps) | Mean AoI (steps) |
|---|---:|---:|---:|
| static | 0.00 ± 0.00 | 806.67 ± 273.41 | 215.21 ± 2.06 |
| fixed_relay | 65.00 ± 7.07 | 9284.00 ± 107.48 | 0.00 ± 0.00 |
| greedy | 50.00 ± 42.43 | 11380.00 ± 1978.01 | 165.00 ± 106.07 |

## Notes
- Completion is `task_delivery_pct` (delivered tasks only).
- RLlib eval uses deterministic inference (`explore=False`).
- Use this report with raw JSON artifacts in the same folder.
