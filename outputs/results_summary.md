# Case Study 2 Results (20 seeds, latest runs)

Seeds: 1-20

| Method | N | Completion % (mean+-std) | Throughput bps (mean+-std) | Mean AoI (mean+-std) |
|---|---:|---:|---:|---:|
| static | 20 | 60.5 +- 15.3 | 1158.9 +- 1294.6 | 134.9 +- 29.7 |
| fixed_relay | 20 | 45.0 +- 19.4 | 2219.2 +- 2198.3 | 187.1 +- 29.6 |
| greedy | 20 | 64.0 +- 22.9 | 3286.1 +- 2235.9 | 166.5 +- 73.0 |
| mappo_full | 20 | 14.5 +- 12.4 | 393.9 +- 698.0 | 209.1 +- 14.8 |
| mappo_task_only | 20 | 5.5 +- 8.6 | 118.5 +- 303.7 | 207.7 +- 4.5 |
| mappo_routing_only | 20 | 3.0 +- 4.6 | 146.5 +- 289.0 | 207.5 +- 4.3 |

Throughput gain vs static (mappo_full): -66.0%