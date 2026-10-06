# DexCraft

DexCraft is a simulated benchmark of dexterous tool use on an xArm6 + LEAP hand, built on
[ManiSkill3](https://github.com/haosulab/ManiSkill). In each task the robot picks up an articulated tool, moves
it to a goal pose (green ghost), and actuates it. This repository contains

1. the six DexCraft tasks, keyboard teleoperation, and a [MimicGen](https://github.com/NVlabs/mimicgen) pipeline
   that turns a handful of teleoperated demos into large datasets;
2. a script that turns [Dexonomy](https://github.com/JYChen18/DexLearn) grasps into hand-object trajectories
   (floating Shadow hand) for pre-training.

| Task             | Tool              | Success: tool at goal pose (≤ 4 cm, ≤ 30°) and ... |
|------------------|-------------------|-----------------------------------------------------|
| `SprayBottle-v1` | spray bottle      | trigger pulled                                      |
| `Lighter-v1`     | lighter           | ignition lever pressed                              |
| `Dispenser-v1`   | soap dispenser    | pump pressed                                        |
| `Pen-v1`         | click pen         | button clicked (pen starts in a stand)              |
| `Pliers-v1`      | pliers            | handles closed                                      |
| `Stapler-v1`     | stapler           | stapler pressed shut                                |

## Installation

```sh
git clone <this repo> && cd DexCraft
pixi install
pixi run install-pytorch3d
pixi run install-robomimic   # robomimic version required by MimicGen
pixi shell
```

## DexCraft tasks

```python
import gymnasium as gym
import dexcraft.envs  # registers the tasks

env = gym.make("SprayBottle-v1", obs_mode="pointcloud", control_mode="pd_ee_target_delta_pose")
```

With `obs_mode="pointcloud"`, observations are flattened to `state` (robot proprioception + TCP pose),
`pointcloud` (1024 × [xyz, is_object, is_hand, 0], arm and table removed), `hand_pcd` (16 hand keypoints),
`hand_pcd_dense` (136 points sampled on the hand) and `rgb` (two 128×128 cameras).

Run a task with random actions (add `--gui` to watch it):
```sh
python scripts/run_env.py --env-id SprayBottle-v1 --obs-mode pointcloud --steps 10
```

**Control modes.** The action is a 6-D end-effector delta followed by 16 LEAP hand joint deltas.
- `pd_ee_target_delta_pose`: raw end-effector delta in m / rad (clipped to ±0.1). Used for MimicGen
  generation, the generated datasets, and policies.
- `pd_ee_target_delta_pose_normalized`: same controller with the arm action normalized to [-1, 1]. Used for
  keyboard teleoperation, i.e. by the source demos.

## Generating demos with MimicGen

The source demos used in the paper (5 per task) and the generated datasets are on Hugging Face:
```sh
huggingface-cli download <HF_REPO> --repo-type dataset --local-dir .
```
This gives `demos/<task>/teleop/` (raw teleoperation), `demos/<task>/source/` (processed, steps 1-5
below already done, so you can skip directly to step 6) and `generated/<task>/` (step 6 already done).
Steps 1-5 show how to make source demos yourself.

The commands use `SprayBottle-v1`; the MimicGen config names are `spray_bottle`, `lighter`, `dispenser`,
`pen`, `pliers` and `stapler`.

1. **Teleoperate.** Opens the GUI and records to `demos/SprayBottle-v1/teleop/trajectory_0.h5`. The arm is
   controlled with `i/k` (x), `j/l` (y), `u/o` (z), `1/2` `3/4` `5/6` (roll, pitch, yaw); the task-specific
   finger keys are printed at start-up (see `FINGER_KEYMAPS` in `dexcraft/teleop.py`). Press `q` to finish.
   ```sh
   python scripts/teleop.py --env-id SprayBottle-v1 --demo-id 0
   ```

2. **Clean up** each demo: drop idle steps and set the goal to the final tool pose.
   ```sh
   python scripts/mimicgen/truncate_demos.py --env-id SprayBottle-v1 --demo-id 0
   python scripts/replay_demo.py --traj-path demos/SprayBottle-v1/source/trajectory_0.h5   # check + video
   ```

3. **Merge** all demos of a task:
   ```sh
   python scripts/mimicgen/merge_demos.py --env-id SprayBottle-v1
   ```

4. **Convert** to robomimic's format (writes `merged_trajectory_mimicgen.hdf5`):
   ```sh
   python scripts/mimicgen/convert_to_robomimic.py --dataset demos/SprayBottle-v1/source/merged_trajectory.h5
   ```

5. **Add MimicGen annotations** (eef / object poses, subtask signals):
   ```sh
   python scripts/mimicgen/prepare_src_dataset.py --dataset demos/SprayBottle-v1/source/merged_trajectory_mimicgen.hdf5
   ```

6. **Generate** demos (settings in `configs/mimicgen/<task>.json`; writes `outputs/mimicgen/spray_bottle/demo.hdf5`).
   `--num_demos` sets the number of successful demos to generate:
   ```sh
   python scripts/mimicgen/generate_dataset.py --config configs/mimicgen/spray_bottle.json --num_demos 100 --auto-remove-exp
   ```

7. **Render observations** (point clouds, hand keypoints and per-subtask goal hand keypoints `goal_gripper_pcd`).
   The generated datasets used in the paper (500 training + 200 validation demos per task) are in the same
   Hugging Face dataset under `generated/<task>/demo.hdf5` and `demo_val.hdf5`; they store states and actions,
   and this step renders their observations.
   ```sh
   python scripts/mimicgen/dataset_states_to_obs.py --input outputs/mimicgen/spray_bottle/demo.hdf5 \
       --output outputs/mimicgen/spray_bottle/demo_obs.hdf5 --num_workers 8
   # or, for the released dataset
   python scripts/mimicgen/dataset_states_to_obs.py --input generated/SprayBottle-v1/demo.hdf5 \
       --output generated/SprayBottle-v1/demo_obs.hdf5 --num_workers 8
   ```

## Dexonomy trajectories for pre-training

Download the Dexonomy grasps and objects following [DexLearn](https://github.com/JYChen18/DexLearn), so that
`<DexLearn>/assets/grasp/Dexonomy_GRASP_shadow/succ_collect/<grasp type>/` and `<DexLearn>/assets/object/` exist.

Visualize one grasp as a trajectory (approach, pregrasp → grasp → squeeze, lift):
```sh
python scripts/dexonomy/replay_grasp.py --dexonomy-root <DexLearn> --grasp-type 1_Large_Diameter --grasp-id 0 --gui
```

Collect trajectories (motion planning to the pregrasp pose, then closing the hand; failed grasps are discarded).
Each trajectory is saved as `<output>/demo_<i>/{actions,state,pointcloud,hand_pcd,goal_gripper_pcd}.npy`:
```sh
python scripts/dexonomy/collect_dataset.py --dexonomy-root <DexLearn> --grasp-type 1_Large_Diameter \
    --output data/dexonomy/1_Large_Diameter --n 100 --num-workers 16
```

## Repository structure

```
dexcraft/
  agents/      xArm6 + LEAP hand (xarm6_leap), floating Shadow hand (floating_shadow)
  envs/        DexCraft tasks (base.py holds the shared logic) and ShadowGrasp-v1 (Dexonomy)
  mimicgen/    MimicGen env interface, configs and robomimic env wrapper
  teleop.py    keyboard teleoperation
  dexonomy.py  loading Dexonomy grasps
scripts/       teleop, replay, MimicGen pipeline, Dexonomy collection
configs/       MimicGen generation configs
assets/        robot and PartNet-Mobility object models
```

## Citation

```bibtex
TODO
```
