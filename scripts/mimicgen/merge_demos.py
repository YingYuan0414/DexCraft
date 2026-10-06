"""Step 3 of the MimicGen pipeline: merge truncated demos into a single file.

    python scripts/mimicgen/merge_demos.py --env-id SprayBottle-v1
"""
import argparse
import re
from glob import glob
from pathlib import Path

from mani_skill.trajectory.merge_trajectory import merge_trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--demo-dir", type=str, default=None, help="default: demos/<env-id>/source")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir or f"demos/{args.env_id}/source")
    traj_paths = sorted(glob(str(demo_dir / "trajectory_*.h5")), key=lambda p: int(re.findall(r"\d+", Path(p).stem)[-1]))
    assert traj_paths, f"no trajectory_*.h5 in {demo_dir}"
    print("Merging", traj_paths)
    merge_trajectories(str(demo_dir / "merged_trajectory.h5"), traj_paths)
