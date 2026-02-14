# Camera-Ready Evidence Report

## Run Metadata
```text
date_utc=2026-02-13T11:29:59Z
git_commit=f6f024c8190a6092f13d31d8bb7f2debfd820de6
python=Python 3.12.12
ray=2.53.0
torch=2.10.0
env_config=/Users/nitin/Desktop/case_study_2/configs/env_easy.yaml
checkpoint=/Users/nitin/Desktop/case_study_2/outputs/checkpoints/rllib_ppo_ctde/PPO_2026-02-13_05-09-45/PPO_ctde_env_820d7_00000_0_2026-02-13_05-09-47/checkpoint_000000
seeds=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
```

## Primary Comparison (RLlib PPO-CTDE vs Static)
| Metric | RLlib PPO-CTDE | Static |
|---|---:|---:|
| Delivered completion (%) | 59.50 ± 29.64 | 0.00 ± 0.00 |
| Throughput (bps) | 13088.93 ± 2097.34 | 674.67 ± 500.08 |
| Mean AoI (steps) | 198.58 ± 45.19 | 212.12 ± 9.92 |

## Baseline Sweep
| Method | Delivered completion (%) | Throughput (bps) | Mean AoI (steps) |
|---|---:|---:|---:|
| static | 0.00 ± 0.00 | 674.67 ± 500.08 | 212.12 ± 9.92 |
| fixed_relay | 62.00 ± 23.75 | 8616.53 ± 1149.66 | 45.12 ± 87.56 |
| greedy | 80.50 ± 20.64 | 12303.20 ± 1518.05 | 81.50 ± 80.49 |

## Notes
- Completion is `task_delivery_pct` (delivered tasks only).
- RLlib eval uses deterministic inference (`explore=False`).
- Use this report with raw JSON artifacts in the same folder.
