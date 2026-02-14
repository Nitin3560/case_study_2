# Camera-Ready Statistical Pack

- env_config: `/Users/nitin/Desktop/case_study_2/configs/env_easy.yaml`
- checkpoint: `/Users/nitin/Desktop/case_study_2/outputs/checkpoints/rllib_ppo_ctde/PPO_2026-02-13_05-43-10/PPO_ctde_env_2d0bb_00000_0_2026-02-13_05-43-12/checkpoint_000000`
- seeds: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20`

| Method | Completion (%) | Throughput (bps) | Mean AoI (steps) |
|---|---:|---:|---:|
| rllib | 74.50 +- 21.14 | 11848.67 +- 1871.30 | 164.65 +- 78.03 |
| static | 0.00 +- 0.00 | 674.67 +- 500.08 | 212.12 +- 9.92 |
| fixed_relay | 62.00 +- 23.75 | 8616.53 +- 1149.66 | 45.12 +- 87.56 |
| greedy | 80.50 +- 20.64 | 12303.20 +- 1518.05 | 81.50 +- 80.49 |

## Wilcoxon p-values (one-sided)
- Alternative hypotheses:
  - completion/throughput: RLlib > baseline
  - AoI: RLlib < baseline

- rllib_vs_static: {"completion_greater": 3.4248771319323876e-05, "throughput_greater": 9.5367431640625e-07, "aoi_less": 0.03366534966068669}
- rllib_vs_fixed_relay: {"completion_greater": 0.05202167958985231, "throughput_greater": 1.9073486328125e-06, "aoi_less": 0.9995066746586725}
- rllib_vs_greedy: {"completion_greater": 0.8326131180137362, "throughput_greater": 0.7847833633422852, "aoi_less": 0.9934851958025942}

Raw JSON: `/Users/nitin/Desktop/case_study_2/outputs/camera_ready_pack_300/camera_ready_pack.json`
