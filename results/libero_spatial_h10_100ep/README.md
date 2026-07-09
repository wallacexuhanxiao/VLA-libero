# LIBERO-Spatial h10 100-Episode Results

Protocol: 10 LIBERO-Spatial tasks x 10 episodes, `policy.n_action_steps=10`, 20k-step checkpoints. Success rates use the first 10 episodes per task for every model.

## Overall

| Model | Success | Rate | Delta vs Baseline |
|---|---:|---:|---:|
| Baseline SmolVLA action expert | 44/100 | 44.0% | +0.0 pts |
| Ver1 Residual MoE Adapter | 43/100 | 43.0% | -1.0 pts |
| Ver2 Action Expert FFN-MoE | 39/100 | 39.0% | -5.0 pts |
| Routed Multi-MLP Adapter | 51/100 | 51.0% | +7.0 pts |
| Ver4 PAMAE-style Expert Mixture | 36/100 | 36.0% | -8.0 pts |

Best model: **Routed Multi-MLP Adapter** (`51/100`, `+7.0` points over baseline).

## Per-Task Success Counts

| Model | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline SmolVLA action expert | 5/10 | 4/10 | 4/10 | 3/10 | 2/10 | 2/10 | 5/10 | 6/10 | 6/10 | 7/10 | 44.0% |
| Ver1 Residual MoE Adapter | 5/10 | 8/10 | 4/10 | 2/10 | 5/10 | 0/10 | 5/10 | 6/10 | 3/10 | 5/10 | 43.0% |
| Ver2 Action Expert FFN-MoE | 5/10 | 4/10 | 4/10 | 2/10 | 1/10 | 2/10 | 7/10 | 6/10 | 4/10 | 4/10 | 39.0% |
| Routed Multi-MLP Adapter | 8/10 | 6/10 | 5/10 | 6/10 | 4/10 | 6/10 | 3/10 | 2/10 | 7/10 | 4/10 | 51.0% |
| Ver4 PAMAE-style Expert Mixture | 5/10 | 2/10 | 1/10 | 2/10 | 3/10 | 3/10 | 4/10 | 6/10 | 4/10 | 6/10 | 36.0% |

Raw `eval_info.json` files and shard logs are included under `raw_eval_info/` and `eval_logs/`.
