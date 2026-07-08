# LIBERO-Spatial SmolVLA Baseline vs MoE Report

## Objective

This experiment evaluates whether a lightweight MoE adapter on SmolVLA action tokens improves closed-loop LIBERO-Spatial manipulation performance over a vanilla SmolVLA action expert.

The comparison uses the same 20k-step training budget for both policies:

- **Baseline 20k**: SmolVLA fine-tuning on LIBERO-Spatial.
- **MoE 20k**: SmolVLA with a residual dense MoE adapter inserted on action-token hidden states.

The MoE adapter is implemented in `src/vla_libero_moe/action_moe_adapter.py` and enabled by `scripts/06_train_moe_spatial.sh`.

## Setup

- Dataset: `HuggingFaceVLA/libero`
- Suite: `libero_spatial`
- Checkpoint step: `20,000`
- Rollout tasks: LIBERO-Spatial task ids `0..9`
- Episodes: `10` episodes per task, `100` total episodes per policy per horizon
- Metric: closed-loop success rate
- Horizons tested: `policy.n_action_steps in {6, 8, 10}`

For `h=10`, evaluation was executed task-wise to reduce CPU contention. All reported task numbers come from completed standard 10-episode `lerobot-eval` runs and were aggregated into the summary JSON. Some historical filenames still contain words such as `salvage` or `merged`, but the reported results are standard evaluation outputs, not manual labels or partial visual estimates.

```text
results/libero_spatial_horizon_sweep/h10_merged_summary_20260708_225647_plus_20260709_005410.json
```

## Overall Results

| `n_action_steps` | Baseline 20k | MoE 20k | MoE - Baseline |
|---:|---:|---:|---:|
| 6 | `41/100 = 41.0%` | `44/100 = 44.0%` | `+3.0%` |
| 8 | `44/100 = 44.0%` | `42/100 = 42.0%` | `-2.0%` |
| 10 | `35/100 = 35.0%` | `48/100 = 48.0%` | `+13.0%` |

## Per-Task Results

### `n_action_steps = 6`

| Task group | Baseline 20k | MoE 20k |
|---|---:|---:|
| 0/1/2 | `15/30 = 50.0%` | `18/30 = 60.0%` |
| 3/4/5 | `12/30 = 40.0%` | `5/30 = 16.7%` |
| 6/7 | `8/20 = 40.0%` | `7/20 = 35.0%` |
| 8/9 | `6/20 = 30.0%` | `14/20 = 70.0%` |

### `n_action_steps = 8`

| Task group | Baseline 20k | MoE 20k |
|---|---:|---:|
| 0/1/2 | `15/30 = 50.0%` | `17/30 = 56.7%` |
| 3/4/5 | `8/30 = 26.7%` | `3/30 = 10.0%` |
| 6/7 | `14/20 = 70.0%` | `12/20 = 60.0%` |
| 8/9 | `7/20 = 35.0%` | `10/20 = 50.0%` |

The low `3/4/5` result was rerun and reproduced:

| Task group | Baseline 20k rerun | MoE 20k rerun |
|---|---:|---:|
| 3/4/5 | `8/30 = 26.7%` | `3/30 = 10.0%` |

### `n_action_steps = 10`

| Task | Baseline 20k | MoE 20k |
|---:|---:|---:|
| 0 | `5/10 = 50.0%` | `5/10 = 50.0%` |
| 1 | `5/10 = 50.0%` | `5/10 = 50.0%` |
| 2 | `3/10 = 30.0%` | `6/10 = 60.0%` |
| 3 | `3/10 = 30.0%` | `2/10 = 20.0%` |
| 4 | `3/10 = 30.0%` | `2/10 = 20.0%` |
| 5 | `2/10 = 20.0%` | `0/10 = 0.0%` |
| 6 | `2/10 = 20.0%` | `8/10 = 80.0%` |
| 7 | `2/10 = 20.0%` | `8/10 = 80.0%` |
| 8 | `5/10 = 50.0%` | `6/10 = 60.0%` |
| 9 | `5/10 = 50.0%` | `6/10 = 60.0%` |

## Interpretation

The MoE adapter does not uniformly improve performance. It is consistently weak on the `3/4/5` task group, especially task `5`. However, it improves the longer open-loop horizon setting:

- At `h=6`, MoE is slightly better: `44%` vs `41%`.
- At `h=8`, MoE is slightly worse: `42%` vs `44%`.
- At `h=10`, MoE is clearly better: `48%` vs `35%`.

The best result among the tested settings is:

```text
MoE 20k, n_action_steps=10: 48/100 = 48.0%
```

This suggests the MoE adapter helps some task phases and longer-horizon action execution, but the routing or expert specialization is not yet robust across all spatial tasks.

## Engineering Notes

The `h=10` evaluation was slow because LIBERO/MuJoCo rollout is CPU-heavy. Running eight evaluation processes in parallel caused CPU oversubscription: each `lerobot-eval` process used roughly three CPU cores while GPU utilization remained low. Later task-wise evaluation used single-task runs with OpenMP/MKL thread limits:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

This made rollout much faster and is recommended for future parallel evaluation.

## Reproducibility Artifacts

Key result files:

```text
results/libero_spatial_horizon_sweep/h10_partial_salvage_20260708_225647.json
results/libero_spatial_horizon_sweep/h10_merged_summary_20260708_225647_plus_20260709_005410.json
results/libero_spatial_horizon_sweep/eval_logs/
```

Note: the `salvage` / `merged` words in some filenames are legacy run labels. The reported numbers are from completed standard evaluation runs.

Relevant code:

```text
src/vla_libero_moe/action_moe_adapter.py
scripts/02_train_smolvla_spatial.sh
scripts/06_train_moe_spatial.sh
docs/MANUAL_MOE_PATCH.md
```
