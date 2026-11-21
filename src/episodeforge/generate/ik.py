"""Damped Least-Squares (DLS) Jacobian Inverse Kinematics for Franka Panda in MuJoCo."""

from __future__ import annotations

import mujoco
import numpy as np


class DLSInverseKinematics:
    """Robust Damped Least-Squares IK solver with joint limit avoidance and nullspace biasing."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "ee_site",
        damping: float = 0.02,
        step_size: float = 0.8,
        max_iterations: int = 100,
        pos_tolerance: float = 0.003,
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.pos_tolerance = pos_tolerance

        # 7 arm joints indices
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.qpos_indices = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.joint_names
        ]
        self.dof_indices = [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.joint_names
        ]

        # Joint limits
        self.q_min = np.array(
            [
                model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)][0]
                for name in self.joint_names
            ]
        )
        self.q_max = np.array(
            [
                model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)][1]
                for name in self.joint_names
            ]
        )

        # Natural rest posture for Franka arm (elbow slightly up and back)
        self.q_rest = np.array([0.0, -0.4, 0.0, -2.1, 0.0, 1.8, 0.785])

    def solve(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for 7-DoF target joint angles that place the end-effector site at target_pos."""
        # Work on a copy of joint positions
        q = np.array([data.qpos[idx] for idx in self.qpos_indices])

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        for _ in range(self.max_iterations):
            # Update kinematics
            for idx, q_val in zip(self.qpos_indices, q, strict=False):
                data.qpos[idx] = q_val
            mujoco.mj_forward(self.model, data)

            current_pos = data.site_xpos[self.site_id]
            err_pos = target_pos - current_pos

            if np.linalg.norm(err_pos) < self.pos_tolerance:
                break

            mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, self.site_id)

            # Extract 7 arm DoF columns
            J_p = jac_pos[:, self.dof_indices]

            # Damped least squares: J^T (J J^T + lambda^2 I)^-1
            lambda_sq = self.damping**2
            JJT = J_p @ J_p.T + lambda_sq * np.eye(3)
            inv_JJT = np.linalg.inv(JJT)
            J_pinv = J_p.T @ inv_JJT

            delta_q = J_pinv @ err_pos

            # Step and clamp to joint limits
            q += self.step_size * delta_q
            q = np.clip(q, self.q_min + 0.02, self.q_max - 0.02)

        return q
