# SOP: Official LeRobot LIBERO + SmolVLA Baseline

## Stage 0 — Create environment

```bash
cd /root

git clone https://github.com/wallacexuhanxiao/VLA-libero.git
cd VLA-libero

conda create -n lerobot python=3.10 -y
conda activate lerobot

bash scripts/00_install_lerobot.sh
source configs/env.sh
```

Expected:

```text
lerobot import: OK
MUJOCO_GL=egl
```

On a headless AutoDL/cloud server, keep `MUJOCO_GL=egl`.

---

## Stage 1 — Official environment sanity check

```bash
bash scripts/01_eval_pi05_libero_spatial.sh
```

Purpose:

```text
Verify MuJoCo, LIBERO, camera/state/action protocol, and official rollout.
Do not train anything before this passes.
```

Pass condition:

```text
The command finishes.
It creates eval_logs/pi05_spatial_task0.
The policy shows non-random closed-loop behavior.
```

If this fails, debug installation and environment first. Do not start SmolVLA training yet.

---

## Stage 2 — Fine-tune official SmolVLA on LIBERO-Spatial

```bash
bash scripts/02_train_smolvla_spatial.sh
```

Default training settings:

```text
policy.type: smolvla
dataset: HuggingFaceVLA/libero
env: libero_spatial
steps: 20000
batch size: 4
eval batch size: 1
eval episodes: 1
env eval frequency: 1000
```

For a single RTX 5090:

```text
Start with batch_size=4.
If OOM, change it to batch_size=2.
Keep eval.batch_size=1.
```

Do not modify the model architecture at this stage.

---

## Stage 3 — Find checkpoint

```bash
bash scripts/04_find_checkpoints.sh
```

Look for the final or best checkpoint folder. The default eval script assumes:

```text
outputs/smolvla_libero_spatial/checkpoints/last/pretrained_model
```

If the actual path is different, pass it through the `CHECKPOINT` environment variable.

---

## Stage 4 — Evaluate one task

```bash
CHECKPOINT=/path/to/checkpoint \
TASK_IDS='[0]' \
EPISODES=10 \
bash scripts/03_eval_smolvla_spatial.sh
```

This is the first real closed-loop test.

Main metric:

```text
success_rate
```

Also record video behavior if available:

```text
Does the arm reach the correct object?
Does the gripper open and close correctly?
Does the robot fail at reaching, grasping, lifting, or placement?
```

---

## Stage 5 — Evaluate all LIBERO-Spatial tasks

```bash
CHECKPOINT=/path/to/checkpoint \
TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' \
EPISODES=10 \
bash scripts/03_eval_smolvla_spatial.sh
```

This gives the official-style spatial benchmark result.

---

## Stage 6 — Only after baseline works: Flow Matching research

Do not replace the action head until official SmolVLA baseline works.

Research plan:

```text
1. Find SmolVLA policy implementation inside third_party/lerobot/src/lerobot/policies/.
2. Identify action expert / action head.
3. Add a new policy type or config flag for Flow Matching.
4. Keep dataset, processors, normalization, env, and eval unchanged.
5. Compare official SmolVLA vs Flow action head under exactly the same LIBERO eval.
```

Metrics to compare:

```text
success_rate
action smoothness
failure mode distribution
rollout stability
```

---

## Experiment report format

For every run, record:

```text
Date:
Repo commit:
LeRobot commit:
Command:
GPU:
Steps:
Batch size:
Task ids:
Episodes:
Success rate:
Failure mode:
Notes:
```
