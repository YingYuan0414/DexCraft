from pathlib import Path

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import _axis_angle_rotation, matrix_to_quaternion

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


def quaternion_distance(q1, q2):
    """Compute angle (in radians) between two unit quaternions."""
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    dot_product = torch.sum(q1 * q2, dim=-1).abs()
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    return 2 * torch.acos(dot_product)


def generate_random_quat(
    n: int,
    device=None,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    bounds=((0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)),
):
    """Sample quaternions from extrinsic XYZ euler angles drawn uniformly within `bounds`.

    A locked axis is set to 0 (and its bound is ignored).
    """
    xyz_angles = torch.zeros((n, 3), device=device)
    for i, (lock, (low, high)) in enumerate(zip((lock_x, lock_y, lock_z), bounds)):
        if not lock:
            xyz_angles[:, i] = torch.rand(n, device=device) * (high - low) + low

    # extrinsic rotation (euler_angles_to_matrix is intrinsic)
    matrices = [_axis_angle_rotation(c, e) for c, e in zip("XYZ", torch.unbind(xyz_angles, -1))]
    rot_mats = torch.matmul(torch.matmul(matrices[2], matrices[1]), matrices[0])
    return matrix_to_quaternion(rot_mats)
