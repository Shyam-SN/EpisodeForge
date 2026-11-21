"""Task definitions, expert motion primitive state machines, and language generation."""

from __future__ import annotations

import enum

import numpy as np

from episodeforge.generate.scene import SceneRandomization


class TaskType(enum.StrEnum):
    PICK_PLACE = "pick_place"
    PUSH = "push"
    STACK = "stack"


LANGUAGE_TEMPLATES = {
    TaskType.PICK_PLACE: [
        "put the red cube in the bowl",
        "pick up the red cube and place it into the green container",
        "move the red block into the bowl",
        "grasp the red cube and transfer it to the target container",
        "place the red block in the green bowl on the table",
    ],
    TaskType.PUSH: [
        "push the block to the green target",
        "slide the red cube into the green target zone",
        "nudge the block forward to the target pad",
        "push the red block onto the green goal mark",
        "shove the cube across the table to the target",
    ],
    TaskType.STACK: [
        "stack the red cube on the yellow cube",
        "place the red block on top of the yellow block",
        "build a two-block tower by stacking the red cube on the yellow base",
        "lift the red block and set it directly on the yellow cube",
        "stack the red cube onto the yellow target cube",
    ],
}


class TaskStateMachine:
    """Motion-primitive state machine generating end-effector waypoints, gripper commands, and hold state."""

    def __init__(
        self,
        task: TaskType,
        rand: SceneRandomization,
        seed: int,
        imperfection_type: str | None = None,
    ):
        self.task = task
        self.rand = rand
        self.rng = np.random.RandomState(seed)
        self.imperfection_type = imperfection_type

        # State machine phases
        self.phase = "APPROACH"
        self.step_in_phase = 0
        self.total_steps = 0
        self.completed = False
        self.held = False

        # Randomize phase durations slightly so trajectories are not identical
        speed_var = int(self.rng.randint(-2, 3))
        self.approach_steps = max(10, 14 + speed_var)
        self.lower_steps = max(8, 12 + speed_var)
        self.squeeze_steps = max(8, 10 + speed_var)
        self.lift_steps = max(14, 18 + speed_var)
        self.move_steps = max(20, 26 + speed_var)
        self.place_steps = max(10, 14 + speed_var)
        self.retreat_steps = max(10, 12 + speed_var)

        # Positions
        self.cube_pos = np.array(rand.obj1_pos, dtype=np.float64)
        self.target_pos = np.array(rand.target_pos, dtype=np.float64)

    def sample_instruction(self) -> str:
        templates = LANGUAGE_TEMPLATES[self.task]
        return str(self.rng.choice(templates))

    def step_waypoint(self, current_cube_pos: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
        """Compute target EE position (x, y, z), gripper command, whether object should be held, and done flag.

        Returns:
            (target_ee_pos, gripper_command, held_flag, done)
        """
        self.step_in_phase += 1
        self.total_steps += 1

        gripper_cmd = 0.0  # Open by default
        target_ee = np.copy(self.cube_pos)
        self.held = False

        if self.task == TaskType.PICK_PLACE:
            if self.phase == "APPROACH":
                target_ee[2] = 0.60
                gripper_cmd = 0.0
                if self.step_in_phase >= self.approach_steps:
                    self.phase = "LOWER"
                    self.step_in_phase = 0

            elif self.phase == "LOWER":
                alpha = min(1.0, self.step_in_phase / float(self.lower_steps))
                target_ee[2] = (1.0 - alpha) * 0.60 + alpha * 0.435
                gripper_cmd = 0.0
                if self.step_in_phase >= self.lower_steps:
                    self.phase = "SQUEEZE"
                    self.step_in_phase = 0

            elif self.phase == "SQUEEZE":
                target_ee[2] = 0.435
                gripper_cmd = 0.012
                # Imperfection: grasp slip triggers opening early
                if self.imperfection_type == "failed_grasp" and self.step_in_phase > 4:
                    self.held = False
                    gripper_cmd = 0.0
                else:
                    self.held = True

                if self.step_in_phase >= self.squeeze_steps:
                    self.phase = "LIFT"
                    self.step_in_phase = 0

            elif self.phase == "LIFT":
                alpha = min(1.0, self.step_in_phase / float(self.lift_steps))
                target_ee[2] = (1.0 - alpha) * 0.435 + alpha * 0.59
                gripper_cmd = 0.012
                if self.imperfection_type == "failed_grasp":
                    self.held = False
                    gripper_cmd = 0.0
                else:
                    self.held = True

                if self.step_in_phase >= self.lift_steps:
                    self.phase = "MOVE"
                    self.step_in_phase = 0

            elif self.phase == "MOVE":
                alpha = min(1.0, self.step_in_phase / float(self.move_steps))
                lift_p = np.array([self.cube_pos[0], self.cube_pos[1], 0.59])
                bowl_p = np.array([self.target_pos[0], self.target_pos[1], 0.59])
                target_ee = (1.0 - alpha) * lift_p + alpha * bowl_p
                gripper_cmd = 0.012
                if self.imperfection_type == "failed_grasp":
                    self.held = False
                    gripper_cmd = 0.0
                else:
                    self.held = True

                if self.step_in_phase >= self.move_steps:
                    self.phase = "PLACE"
                    self.step_in_phase = 0

            elif self.phase == "PLACE":
                alpha = min(1.0, self.step_in_phase / float(self.place_steps))
                bowl_high = np.array([self.target_pos[0], self.target_pos[1], 0.59])
                bowl_low = np.array([self.target_pos[0], self.target_pos[1], 0.460])
                target_ee = (1.0 - alpha) * bowl_high + alpha * bowl_low
                gripper_cmd = 0.012
                self.held = True
                if self.step_in_phase >= self.place_steps:
                    self.phase = "RELEASE"
                    self.step_in_phase = 0

            elif self.phase == "RELEASE":
                target_ee = np.array([self.target_pos[0], self.target_pos[1], 0.460])
                gripper_cmd = 0.0  # Open gripper
                self.held = False
                if self.step_in_phase >= 10:
                    self.phase = "RETREAT"
                    self.step_in_phase = 0

            elif self.phase == "RETREAT":
                alpha = min(1.0, self.step_in_phase / float(self.retreat_steps))
                target_ee = np.array(
                    [self.target_pos[0], self.target_pos[1], (1.0 - alpha) * 0.460 + alpha * 0.62]
                )
                gripper_cmd = 0.0
                self.held = False
                if self.step_in_phase >= self.retreat_steps:
                    self.completed = True

        elif self.task == TaskType.PUSH:
            dir_vec = self.target_pos[:2] - self.cube_pos[:2]
            dir_vec /= np.linalg.norm(dir_vec) + 1e-6
            p_behind_high = np.array(
                [self.cube_pos[0] - 0.07 * dir_vec[0], self.cube_pos[1] - 0.07 * dir_vec[1], 0.55]
            )
            p_behind_low = np.array(
                [self.cube_pos[0] - 0.06 * dir_vec[0], self.cube_pos[1] - 0.06 * dir_vec[1], 0.428]
            )
            p_push = np.array(
                [
                    self.target_pos[0] + 0.03 * dir_vec[0],
                    self.target_pos[1] + 0.03 * dir_vec[1],
                    0.428,
                ]
            )

            if self.phase == "APPROACH":
                target_ee = p_behind_high
                gripper_cmd = 0.015
                if self.step_in_phase >= self.approach_steps:
                    self.phase = "LOWER"
                    self.step_in_phase = 0

            elif self.phase == "LOWER":
                alpha = min(1.0, self.step_in_phase / float(self.lower_steps))
                target_ee = (1.0 - alpha) * p_behind_high + alpha * p_behind_low
                gripper_cmd = 0.015
                if self.step_in_phase >= self.lower_steps:
                    self.phase = "PUSH"
                    self.step_in_phase = 0

            elif self.phase == "PUSH":
                alpha = min(1.0, self.step_in_phase / float(self.move_steps))
                target_ee = (1.0 - alpha) * p_behind_low + alpha * p_push
                gripper_cmd = 0.015
                if self.step_in_phase >= self.move_steps:
                    self.phase = "RETREAT"
                    self.step_in_phase = 0

            elif self.phase == "RETREAT":
                alpha = min(1.0, self.step_in_phase / float(self.retreat_steps))
                target_ee = np.array([p_push[0], p_push[1], (1.0 - alpha) * 0.428 + alpha * 0.58])
                gripper_cmd = 0.015
                if self.step_in_phase >= self.retreat_steps:
                    self.completed = True

        else:  # STACK
            base_cube = np.array(self.target_pos)
            if self.phase == "APPROACH":
                target_ee[2] = 0.60
                gripper_cmd = 0.0
                if self.step_in_phase >= self.approach_steps:
                    self.phase = "LOWER"
                    self.step_in_phase = 0

            elif self.phase == "LOWER":
                alpha = min(1.0, self.step_in_phase / float(self.lower_steps))
                target_ee[2] = (1.0 - alpha) * 0.60 + alpha * 0.435
                gripper_cmd = 0.0
                if self.step_in_phase >= self.lower_steps:
                    self.phase = "SQUEEZE"
                    self.step_in_phase = 0

            elif self.phase == "SQUEEZE":
                target_ee[2] = 0.435
                gripper_cmd = 0.012
                if self.imperfection_type == "failed_grasp" and self.step_in_phase > 4:
                    self.held = False
                    gripper_cmd = 0.0
                else:
                    self.held = True

                if self.step_in_phase >= self.squeeze_steps:
                    self.phase = "LIFT"
                    self.step_in_phase = 0

            elif self.phase == "LIFT":
                alpha = min(1.0, self.step_in_phase / float(self.lift_steps))
                target_ee[2] = (1.0 - alpha) * 0.435 + alpha * 0.58
                gripper_cmd = 0.012
                if self.imperfection_type == "failed_grasp":
                    self.held = False
                    gripper_cmd = 0.0
                else:
                    self.held = True

                if self.step_in_phase >= self.lift_steps:
                    self.phase = "MOVE"
                    self.step_in_phase = 0

            elif self.phase == "MOVE":
                alpha = min(1.0, self.step_in_phase / float(self.move_steps))
                lift_p = np.array([self.cube_pos[0], self.cube_pos[1], 0.58])
                stack_high = np.array([base_cube[0], base_cube[1], 0.58])
                target_ee = (1.0 - alpha) * lift_p + alpha * stack_high
                gripper_cmd = 0.012
                if self.imperfection_type == "failed_grasp":
                    self.held = False
                    gripper_cmd = 0.0
                else:
                    self.held = True

                if self.step_in_phase >= self.move_steps:
                    self.phase = "PLACE"
                    self.step_in_phase = 0

            elif self.phase == "PLACE":
                alpha = min(1.0, self.step_in_phase / float(self.place_steps))
                stack_high = np.array([base_cube[0], base_cube[1], 0.58])
                stack_place = np.array([base_cube[0], base_cube[1], 0.458])
                target_ee = (1.0 - alpha) * stack_high + alpha * stack_place
                gripper_cmd = 0.012
                self.held = True
                if self.step_in_phase >= self.place_steps:
                    self.phase = "RELEASE"
                    self.step_in_phase = 0

            elif self.phase == "RELEASE":
                target_ee = np.array([base_cube[0], base_cube[1], 0.458])
                gripper_cmd = 0.0
                self.held = False
                if self.step_in_phase >= 10:
                    self.phase = "RETREAT"
                    self.step_in_phase = 0

            elif self.phase == "RETREAT":
                alpha = min(1.0, self.step_in_phase / float(self.retreat_steps))
                target_ee = np.array(
                    [base_cube[0], base_cube[1], (1.0 - alpha) * 0.458 + alpha * 0.60]
                )
                gripper_cmd = 0.0
                self.held = False
                if self.step_in_phase >= self.retreat_steps:
                    self.completed = True

        # Injected timeout imperfection
        if self.imperfection_type == "timeout" and self.total_steps > 35:
            self.completed = True

        return target_ee, gripper_cmd, self.held, self.completed
