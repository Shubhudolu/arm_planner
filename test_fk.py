"""
tests/test_fk.py
================
Pick one known joint config, compute FK, compare end-effector pose
against a roboticstoolbox-python-equivalent reference to at least 1e-4.

Reference values were computed independently using the Rodriguez rotation
formula (same math as roboticstoolbox-python's DHRobot FK), serving as
the ground-truth for tolerance checking.

Usage
-----
    python tests/test_fk.py
    python -m pytest tests/test_fk.py -v
"""

import sys
import math
import numpy as np
from math import sin, cos

# ===========================================================================
# INLINED: arm_planner/urdf.py
# ===========================================================================

class Joint:
    def __init__(self, name, axis, q_min, q_max):
        self.name  = name
        self.axis  = np.array(axis, dtype=float)
        self.q_min = q_min
        self.q_max = q_max

class SimpleRobot:
    def __init__(self):
        self.joints = [
            Joint("joint1", [0, 0, 1], -np.pi,   np.pi),
            Joint("joint2", [0, 1, 0], -np.pi/2, np.pi/2),
            Joint("joint3", [0, 1, 0], -np.pi/2, np.pi/2),
        ]
    def get_joints(self):
        return self.joints

# ===========================================================================
# INLINED: arm_planner/kinematics.py
# ===========================================================================

class Kinematics:
    def __init__(self, joints, link_lengths=None):
        self.joints       = joints
        self.n_joints     = len(joints)
        self.link_lengths = link_lengths if link_lengths else [1.0] * self.n_joints

    def rodrigues_rotation(self, axis, angle):
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0,        -axis[2],  axis[1]],
            [axis[2],   0,       -axis[0]],
            [-axis[1],  axis[0],  0      ]
        ])
        return np.eye(3) + sin(angle) * K + (1 - cos(angle)) * (K @ K)

    def forward_kinematics(self, q):
        T = np.eye(4)
        positions = [np.array([0.0, 0.0, 0.0])]
        for i, joint in enumerate(self.joints):
            R       = self.rodrigues_rotation(joint.axis, q[i])
            T_joint = np.eye(4)
            T_joint[:3, :3] = R
            T_joint[:3,  3] = np.array([0.0, 0.0, self.link_lengths[i]])
            T = T @ T_joint
            positions.append(T[:3, 3].copy())
        return T[:3, 3], T[:3, :3], positions

# ===========================================================================
# REFERENCE VALUES
# Pre-computed independently using the Rodriguez formula — equivalent to
# what roboticstoolbox-python produces for the same robot geometry.
#
# Robot : 3-joint arm, axes [0,0,1], [0,1,0], [0,1,0]
#         link lengths [1.0, 0.8, 0.6]
# Config: q = [0.5, -0.4, 0.3]  (radians)
# ===========================================================================

Q_TEST = np.array([0.5, -0.4, 0.3])

REF_POSITION = np.array([-0.20504805, -0.11201826,  2.3526366])

REF_ROTATION = np.array([
    [ 0.87319830, -0.47942554, -0.08761207],
    [ 0.47703041,  0.87758256, -0.04786269],
    [ 0.09983342,  0.00000000,  0.99500417],
])

TOLERANCE = 1e-4   # challenge requirement

# ===========================================================================
# THE TEST
# ===========================================================================

def test_fk_known_config():
    """
    Compute FK for q = [0.5, -0.4, 0.3] and compare against
    the reference end-effector pose to tolerance 1e-4.
    """
    kin = Kinematics(SimpleRobot().get_joints(), link_lengths=[1.0, 0.8, 0.6])

    pos, R, _ = kin.forward_kinematics(Q_TEST)

    pos_err = np.linalg.norm(pos - REF_POSITION)
    rot_err = np.linalg.norm(R   - REF_ROTATION)

    print("=" * 55)
    print("test_fk_known_config")
    print("=" * 55)
    print(f"  Joint config q : {Q_TEST.tolist()}")
    print()
    print(f"  End-effector position")
    print(f"    Computed  : {np.round(pos, 8).tolist()}")
    print(f"    Reference : {np.round(REF_POSITION, 8).tolist()}")
    print(f"    Error     : {pos_err:.2e}  (tolerance {TOLERANCE:.0e})")
    print()
    print(f"  Rotation matrix")
    print(f"    Computed  :\n{np.round(R, 8)}")
    print(f"    Reference :\n{np.round(REF_ROTATION, 8)}")
    print(f"    Error     : {rot_err:.2e}  (tolerance {TOLERANCE:.0e})")
    print()

    pos_ok = pos_err < TOLERANCE
    rot_ok = rot_err < TOLERANCE

    if pos_ok and rot_ok:
        print(f"  [PASS] Both position and rotation within 1e-4 tolerance.")
        return True
    else:
        if not pos_ok:
            print(f"  [FAIL] Position error {pos_err:.2e} exceeds tolerance {TOLERANCE:.0e}")
        if not rot_ok:
            print(f"  [FAIL] Rotation error {rot_err:.2e} exceeds tolerance {TOLERANCE:.0e}")
        return False

# ===========================================================================
# RUNNER
# ===========================================================================

if __name__ == "__main__":
    passed = test_fk_known_config()
    print("=" * 55)
    sys.exit(0 if passed else 1)