# LIBERO-Spatial SmolVLA Action-Modeling Report

## Objective

This experiment evaluates whether lightweight action-modeling modules improve closed-loop LIBERO-Spatial manipulation performance over a vanilla SmolVLA action expert.

The final project version uses **Routed Multi-MLP Adapter** as the main method. It keeps the original SmolVLA action expert Transformer and inserts a residual routed adapter over action-token hidden states before the action expert. The router uses action-token context, action position, and flow timestep to combine multiple MLP branches.

## Setup

- Dataset: `HuggingFaceVLA/libero`
- Suite: `libero_spatial`
- Checkpoint step: `20,000`
- Rollout tasks: LIBERO-Spatial task ids `0..9`
- Episodes: `10` episodes per task, `100` total episodes per policy
- Closed-loop horizon: `policy.n_action_steps=10`
- Metric: closed-loop success rate

All models below are compared under the same `h10` rollout protocol. For comparability, each task uses the first 10 recorded episodes. This matters for the Ver2 run, whose raw shard outputs included extra episodes for some tasks.

Machine-readable artifacts:

```text
results/libero_spatial_h10_100ep/summary.csv
results/libero_spatial_h10_100ep/per_task.csv
results/libero_spatial_h10_100ep/summary.json
results/libero_spatial_h10_100ep/raw_eval_info/
results/libero_spatial_h10_100ep/eval_logs/
```

## Model Variants

| Model | Description |
|---|---|
| Baseline SmolVLA action expert | Vanilla SmolVLA fine-tuning on LIBERO-Spatial. |
| Ver1 Residual MoE Adapter | Residual routed multi-MLP adapter before the action expert. |
| Ver2 Action Expert FFN-MoE | Replaces action expert Transformer FFN/MLP blocks with dense MoE FFNs. |
| Routed Multi-MLP Adapter | Main method. Residual routed adapter with action position and flow timestep routing context. |
| Ver4 PAMAE-style Expert Mixture | Sparse phase-aware mixture of multiple action expert Transformer copies. |

## Overall Results

| Model | Success | Rate | Delta vs Baseline |
|---|---:|---:|---:|
| Baseline SmolVLA action expert | `44/100` | `44.0%` | `+0.0 pts` |
| Ver1 Residual MoE Adapter | `43/100` | `43.0%` | `-1.0 pts` |
| Ver2 Action Expert FFN-MoE | `39/100` | `39.0%` | `-5.0 pts` |
| **Routed Multi-MLP Adapter** | **`51/100`** | **`51.0%`** | **`+7.0 pts`** |
| Ver4 PAMAE-style Expert Mixture | `36/100` | `36.0%` | `-8.0 pts` |

The best-performing variant is **Routed Multi-MLP Adapter**, which improves the 20k-step baseline by `+7.0` absolute success-rate points.

## Per-Task Results

| Model | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline SmolVLA action expert | `5/10` | `4/10` | `4/10` | `3/10` | `2/10` | `2/10` | `5/10` | `6/10` | `6/10` | `7/10` | `44.0%` |
| Ver1 Residual MoE Adapter | `5/10` | `8/10` | `4/10` | `2/10` | `5/10` | `0/10` | `5/10` | `6/10` | `3/10` | `5/10` | `43.0%` |
| Ver2 Action Expert FFN-MoE | `5/10` | `4/10` | `4/10` | `2/10` | `1/10` | `2/10` | `7/10` | `6/10` | `4/10` | `4/10` | `39.0%` |
| **Routed Multi-MLP Adapter** | **`8/10`** | **`6/10`** | **`5/10`** | **`6/10`** | **`4/10`** | **`6/10`** | **`3/10`** | **`2/10`** | **`7/10`** | **`4/10`** | **`51.0%`** |
| Ver4 PAMAE-style Expert Mixture | `5/10` | `2/10` | `1/10` | `2/10` | `3/10` | `3/10` | `4/10` | `6/10` | `4/10` | `6/10` | `36.0%` |

## Interpretation

The vanilla baseline already learns coarse reaching and partial object interaction, but it remains brittle across grasp/place phases. The Routed Multi-MLP Adapter gives the best aggregate closed-loop improvement and is especially stronger on tasks `0`, `1`, `2`, `3`, `4`, `5`, and `8` compared with the vanilla baseline.

The more invasive variants did not help in this run. Ver2 changes the action expert FFNs directly and underperformed the baseline. Ver4 is closer to a sparse expert-mixture design, but at 20k steps it was less stable and reached only `36%`. This makes the Routed Multi-MLP Adapter the best practical method for the current project: it is lightweight, easy to patch into SmolVLA, and gives the strongest measured closed-loop success rate.

## Reproducibility Notes

Rollout evaluation is CPU-heavy because LIBERO/MuJoCo simulation dominates wall-clock time. The final 100-episode evaluations were run as four shards to avoid excessive CPU oversubscription while keeping the same per-task episode count.

Relevant code:

```text
src/vla_libero_moe/phase_aware_moe_adapter.py
scripts/08_train_phase_moe_spatial.sh
docs/MOE_VER2_VER3_PATCH.md
```

The implementation file still uses the historical `phase_aware_moe_adapter.py` name for compatibility, but the method name used in the project and report is **Routed Multi-MLP Adapter**.
