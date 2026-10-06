"""Keyboard teleoperation for the xArm6 + LEAP hand (control mode `pd_ee_target_delta_pose_normalized`).

Arm (end-effector delta, same for all tasks):
    i / k : +x / -x        u / o : +z / -z        1 / 2 : +roll  / -roll     5 / 6 : +yaw / -yaw
    j / l : +y / -y                               3 / 4 : +pitch / -pitch
Fingers: task-specific "close / open" key pairs, see FINGER_KEYMAPS (printed at start-up).
    q : quit (and save the recording)
"""
import numpy as np

TELEOP_CONTROL_MODE = "pd_ee_target_delta_pose_normalized"
EE_TRANS_STEP = 0.1  # normalized, i.e. 0.01 m per step
EE_ROT_STEP = 0.1  # normalized, i.e. 0.01 rad per step
HAND_STEP = 0.1  # normalized, i.e. 0.01 rad per step

ARM_KEYS = {
    "i": (0, 1), "k": (0, -1), "j": (1, 1), "l": (1, -1), "u": (2, 1), "o": (2, -1),
    "1": (3, 1), "2": (3, -1), "3": (4, 1), "4": (4, -1), "5": (5, 1), "6": (5, -1),
}

# Hand action indices follow XArm6Leap.hand_joint_names:
#   leap_[1, 5, 9, 12, 0, 4, 8, 13, 2, 6, 10, 14, 3, 7, 11, 15]
# Each entry maps a (close, open) key pair to per-joint weights; holding the key moves those joints by
# weight * HAND_STEP per step. These are the mappings used to record the DexCraft source demos.
_ALL_BUT_INDEX = [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
_INDEX_TIP = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
_THUMB_BASE = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _one_hot(i, w=1.0):
    v = [0.0] * 16
    v[i] = w
    return v


FINGER_KEYMAPS = {
    "SprayBottle-v1": {
        ("y", "h"): _ALL_BUT_INDEX,  # grip the bottle with middle, ring and thumb
        ("t", "g"): _INDEX_TIP,  # index finger pulls the trigger
        ("r", "f"): [1] * 16,  # whole hand
    },
    "Lighter-v1": {
        ("y", "h"): [0.5, 1, 1, 0, 0, 0, 0, 0, 0.5, 1, 1, 0, 0.5, 1, 1, 0],
        ("t", "g"): _one_hot(15, 2),
        ("7", "8"): _one_hot(11, 2),
        ("r", "f"): _one_hot(7, 2),
    },
    "Dispenser-v1": {
        ("y", "h"): [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        ("t", "g"): _THUMB_BASE,
        ("7", "8"): _one_hot(4),
        ("r", "f"): _INDEX_TIP,
    },
    "Pen-v1": {
        ("y", "h"): [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        ("t", "g"): _one_hot(15, 2),
        ("7", "8"): _one_hot(11, 2),
        ("r", "f"): _one_hot(7, 2),
    },
    "Pliers-v1": {
        ("y", "h"): [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0.5, 1, 1, 1, 0.5],
        ("t", "g"): _THUMB_BASE,
    },
    "Stapler-v1": {
        ("y", "h"): [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ("t", "g"): _THUMB_BASE,
    },
}


def get_pressed_keys(viewer, keys):
    assert viewer is not None, "Keyboard teleoperation needs the GUI, use --render-mode human"
    return {k for k in keys if viewer.window.key_down(k)}


def all_keys(env_id):
    finger_keys = [k for pair in FINGER_KEYMAPS[env_id] for k in pair]
    return list(ARM_KEYS) + finger_keys + ["q"]


def get_teleop_action(pressed_keys, env_id):
    """Map pressed keys to a 22-D action (6 arm + 16 hand)."""
    ee_action = np.zeros(6)
    for key, (axis, sign) in ARM_KEYS.items():
        if key in pressed_keys:
            if axis < 3:
                ee_action[axis] = sign * EE_TRANS_STEP
            else:  # only one rotation axis at a time
                ee_action[3:6] = 0
                ee_action[axis] = sign * EE_ROT_STEP

    hand_action = np.zeros(16)
    for (close_key, open_key), weights in FINGER_KEYMAPS[env_id].items():
        if close_key in pressed_keys:
            hand_action = HAND_STEP * np.array(weights, dtype=float)
        if open_key in pressed_keys:
            hand_action = -HAND_STEP * np.array(weights, dtype=float)
    return np.concatenate([ee_action, hand_action])


def keymap_help(env_id):
    lines = [
        "Arm:  i/k: +x/-x   j/l: +y/-y   u/o: +z/-z   1/2: roll   3/4: pitch   5/6: yaw",
        "Fingers (close/open):",
    ]
    for (close_key, open_key), weights in FINGER_KEYMAPS[env_id].items():
        joints = [i for i, w in enumerate(weights) if w != 0]
        lines.append(f"  {close_key}/{open_key}: hand joints {joints}")
    lines.append("q: quit")
    return "\n".join(lines)
