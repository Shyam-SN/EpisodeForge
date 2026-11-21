"""MuJoCo scene builder for tabletop manipulation with Franka Emika Panda.

Generates self-contained MJCF XML strings with randomized materials,
lighting, object placements, distractors, compliant contact dynamics,
and multi-camera viewpoints.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class SceneRandomization:
    table_color: list[float]
    light_pos: list[float]
    light_diffuse: list[float]
    obj1_pos: list[float]
    obj1_quat: list[float]
    obj1_color: list[float]
    obj2_pos: list[float]
    obj2_quat: list[float]
    obj2_color: list[float]
    target_pos: list[float]
    distractor_pos: list[float] | None
    friction: float
    mass_scale: float
    front_cam_pos: list[float]
    side_cam_pos: list[float]


def sample_randomization(
    seed: int,
    task: str = "pick_place",
) -> SceneRandomization:
    rng = np.random.RandomState(seed)

    # Table color variation
    table_gray = float(rng.uniform(0.75, 0.90))
    table_color = [table_gray, table_gray, table_gray, 1.0]

    # Lighting variation
    light_x = float(rng.uniform(-0.3, 0.3))
    light_y = float(rng.uniform(-0.3, 0.3))
    light_z = float(rng.uniform(2.2, 2.8))
    light_intensity = float(rng.uniform(0.7, 1.3))
    light_diffuse = [0.6 * light_intensity, 0.6 * light_intensity, 0.6 * light_intensity]

    # Primary object (cube) placement: in front of robot on table (x in [0.46, 0.54], y in [-0.10, 0.10])
    obj1_x = float(rng.uniform(0.46, 0.54))
    obj1_y = float(rng.uniform(-0.10, 0.10))
    obj1_z = 0.420  # on table (table height 0.4)
    angle = float(rng.uniform(-0.3, 0.3))
    obj1_quat = [float(np.cos(angle / 2)), 0.0, 0.0, float(np.sin(angle / 2))]

    # Task-specific secondary object / target
    if task == "pick_place":
        # Target bowl / container at side
        target_x = float(rng.uniform(0.42, 0.52))
        target_y = float(rng.choice([-0.22, 0.22]))
        target_z = 0.415
        obj2_pos = [target_x, target_y, target_z]
        obj2_quat = [1.0, 0.0, 0.0, 0.0]
        obj2_color = [0.1, 0.7, 0.2, 1.0]  # Green bowl
    elif task == "push":
        # Target spot on table
        target_x = float(obj1_x + rng.uniform(0.12, 0.16))
        target_y = float(obj1_y + rng.uniform(-0.06, 0.06))
        target_z = 0.405
        obj2_pos = [target_x, target_y, target_z]
        obj2_quat = [1.0, 0.0, 0.0, 0.0]
        obj2_color = [0.2, 0.8, 0.2, 0.5]  # Semi-transparent target zone
    else:  # stack
        # Base cube on table
        base_x = float(rng.uniform(0.44, 0.50))
        base_y = float(rng.choice([-0.16, 0.16]))
        base_z = 0.420
        obj2_pos = [base_x, base_y, base_z]
        obj2_quat = [1.0, 0.0, 0.0, 0.0]
        obj2_color = [0.9, 0.8, 0.1, 1.0]  # Yellow base cube

    obj1_color = [0.85, 0.15, 0.15, 1.0]  # Red manipulated cube

    # Optional distractor object
    has_distractor = rng.rand() > 0.3
    if has_distractor:
        distractor_pos = [float(rng.uniform(0.60, 0.66)), float(rng.uniform(-0.20, 0.20)), 0.420]
    else:
        distractor_pos = None

    friction = float(rng.uniform(0.8, 1.2))
    mass_scale = float(rng.uniform(0.9, 1.1))

    # Camera jitter (1-2 cm)
    front_cam_pos = [
        0.75 + float(rng.uniform(-0.02, 0.02)),
        float(rng.uniform(-0.02, 0.02)),
        0.65 + float(rng.uniform(-0.02, 0.02)),
    ]
    side_cam_pos = [
        0.35 + float(rng.uniform(-0.02, 0.02)),
        0.65 + float(rng.uniform(-0.02, 0.02)),
        0.60 + float(rng.uniform(-0.02, 0.02)),
    ]

    return SceneRandomization(
        table_color=table_color,
        light_pos=[light_x, light_y, light_z],
        light_diffuse=light_diffuse,
        obj1_pos=[obj1_x, obj1_y, obj1_z],
        obj1_quat=obj1_quat,
        obj1_color=obj1_color,
        obj2_pos=obj2_pos,
        obj2_quat=obj2_quat,
        obj2_color=obj2_color,
        target_pos=[obj2_pos[0], obj2_pos[1], obj2_pos[2]],
        distractor_pos=distractor_pos,
        friction=friction,
        mass_scale=mass_scale,
        front_cam_pos=front_cam_pos,
        side_cam_pos=side_cam_pos,
    )


def generate_tabletop_mjcf(rand: SceneRandomization, task: str = "pick_place") -> str:
    """Build a complete MuJoCo XML string with Franka arm, tabletop, and sensors."""
    distractor_xml = ""
    if rand.distractor_pos is not None:
        distractor_xml = f"""
        <body name="distractor" pos="{rand.distractor_pos[0]} {rand.distractor_pos[1]} {rand.distractor_pos[2]}">
          <freejoint name="distractor_joint"/>
          <geom name="distractor_geom" type="cylinder" size="0.025 0.03" rgba="0.6 0.2 0.8 1" mass="{0.05 * rand.mass_scale}"/>
        </body>
        """

    target_obj_xml = ""
    if task == "pick_place":
        target_obj_xml = f"""
        <body name="target_bowl" pos="{rand.obj2_pos[0]} {rand.obj2_pos[1]} {rand.obj2_pos[2]}">
          <geom name="bowl_base" type="cylinder" size="0.065 0.012" rgba="{rand.obj2_color[0]} {rand.obj2_color[1]} {rand.obj2_color[2]} 1"/>
          <geom name="bowl_rim" type="cylinder" size="0.07 0.025" rgba="{rand.obj2_color[0]} {rand.obj2_color[1]} {rand.obj2_color[2]} 0.6"/>
          <site name="bowl_target" pos="0 0 0.02" size="0.01"/>
        </body>
        """
    elif task == "push":
        target_obj_xml = f"""
        <body name="target_zone" pos="{rand.obj2_pos[0]} {rand.obj2_pos[1]} {rand.obj2_pos[2]}">
          <geom name="target_pad" type="cylinder" size="0.055 0.002" rgba="{rand.obj2_color[0]} {rand.obj2_color[1]} {rand.obj2_color[2]} 0.5" contype="0" conaffinity="0"/>
          <site name="push_target" pos="0 0 0.01" size="0.01"/>
        </body>
        """
    else:  # stack
        target_obj_xml = f"""
        <body name="base_cube" pos="{rand.obj2_pos[0]} {rand.obj2_pos[1]} {rand.obj2_pos[2]}">
          <geom name="base_cube_geom" type="box" size="0.022 0.022 0.020" rgba="{rand.obj2_color[0]} {rand.obj2_color[1]} {rand.obj2_color[2]} 1"/>
          <site name="stack_target" pos="0 0 0.04" size="0.01"/>
        </body>
        """

    mjcf = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="franka_tabletop">
  <compiler angle="radian" coordinate="local" meshdir="" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

  <default>
    <geom solref="0.02 1" solimp="0.9 0.95 0.001" friction="{rand.friction * 2.5} 0.05 0.001"/>
  </default>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.9 0.9 0.9" rgb2="0.75 0.75 0.75"/>
    <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light directional="true" pos="{rand.light_pos[0]} {rand.light_pos[1]} {rand.light_pos[2]}" dir="0 0 -1" diffuse="{rand.light_diffuse[0]} {rand.light_diffuse[1]} {rand.light_diffuse[2]}"/>
    <geom name="ground" type="plane" size="2 2 0.1" material="grid_mat"/>

    <!-- Table -->
    <body name="table" pos="0.5 0 0.2">
      <geom name="tabletop" type="box" size="0.45 0.5 0.2" rgba="{rand.table_color[0]} {rand.table_color[1]} {rand.table_color[2]} 1"/>
    </body>

    <!-- Fixed Multi-Cameras -->
    <camera name="front" pos="{rand.front_cam_pos[0]} {rand.front_cam_pos[1]} {rand.front_cam_pos[2]}" fovy="48" mode="targetbody" target="table"/>
    <camera name="side" pos="{rand.side_cam_pos[0]} {rand.side_cam_pos[1]} {rand.side_cam_pos[2]}" fovy="52" mode="targetbody" target="table"/>

    <!-- Primary Manipulated Object -->
    <body name="red_cube" pos="{rand.obj1_pos[0]} {rand.obj1_pos[1]} {rand.obj1_pos[2]}" quat="{rand.obj1_quat[0]} {rand.obj1_quat[1]} {rand.obj1_quat[2]} {rand.obj1_quat[3]}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.018 0.018 0.018" rgba="{rand.obj1_color[0]} {rand.obj1_color[1]} {rand.obj1_color[2]} 1" mass="{0.012 * rand.mass_scale}"/>
      <site name="cube_site" pos="0 0 0" size="0.005"/>
    </body>

    {target_obj_xml}
    {distractor_xml}

    <!-- Franka Emika Panda Robot Arm (Kinematic & Actuated 7-DoF + Gripper) -->
    <body name="panda_base" pos="0 0 0.4">
      <geom name="link0" type="cylinder" size="0.07 0.05" rgba="0.9 0.9 0.9 1"/>

      <body name="panda_link1" pos="0 0 0.14">
        <joint name="panda_joint1" axis="0 0 1" range="-2.8973 2.8973" damping="5.0"/>
        <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.19" size="0.06" rgba="0.9 0.9 0.9 1"/>

        <body name="panda_link2" pos="0 0 0.19">
          <joint name="panda_joint2" axis="0 1 0" range="-1.7628 1.7628" damping="5.0"/>
          <geom name="link2" type="capsule" fromto="0 0 0 0 0 0.12" size="0.06" rgba="0.9 0.9 0.9 1"/>

          <body name="panda_link3" pos="0 0 0.12">
            <joint name="panda_joint3" axis="0 0 1" range="-2.8973 2.8973" damping="5.0"/>
            <geom name="link3" type="capsule" fromto="0 0 0 0 0.08 0.19" size="0.05" rgba="0.9 0.9 0.9 1"/>

            <body name="panda_link4" pos="0 0.08 0.19">
              <joint name="panda_joint4" axis="0 -1 0" range="-3.0718 -0.0698" damping="5.0"/>
              <geom name="link4" type="capsule" fromto="0 0 0 0 0 0.12" size="0.05" rgba="0.9 0.9 0.9 1"/>

              <body name="panda_link5" pos="0 0 0.12">
                <joint name="panda_joint5" axis="0 0 1" range="-2.8973 2.8973" damping="3.0"/>
                <geom name="link5" type="capsule" fromto="0 0 0 0 0 0.18" size="0.045" rgba="0.9 0.9 0.9 1"/>

                <body name="panda_link6" pos="0 0 0.18">
                  <joint name="panda_joint6" axis="0 1 0" range="-0.0175 3.7525" damping="3.0"/>
                  <geom name="link6" type="cylinder" size="0.045 0.05" rgba="0.9 0.9 0.9 1"/>

                  <body name="panda_link7" pos="0 0 0.08">
                    <joint name="panda_joint7" axis="0 0 1" range="-2.8973 2.8973" damping="2.0"/>
                    <geom name="link7" type="cylinder" size="0.04 0.02" rgba="0.8 0.8 0.8 1"/>

                    <!-- Hand / End-Effector -->
                    <body name="panda_hand" pos="0 0 0.04">
                      <geom name="hand" type="box" size="0.035 0.04 0.015" pos="0 0 0.02" rgba="0.2 0.2 0.2 1"/>
                      <site name="ee_site" pos="0 0 0.065" size="0.008" rgba="1 0 0 1"/>
                      <camera name="wrist" pos="0 0.05 0.04" fovy="62" mode="fixed" euler="-0.7 0 0"/>

                      <!-- Left Finger (slides inward in -Y) -->
                      <body name="left_finger" pos="0 0.026 0.06">
                        <joint name="panda_finger_joint1" type="slide" axis="0 -1 0" range="-0.005 0.025" damping="10.0"/>
                        <geom name="finger1" type="box" size="0.012 0.005 0.020" rgba="0.3 0.3 0.3 1"/>
                      </body>

                      <!-- Right Finger (slides inward in +Y) -->
                      <body name="right_finger" pos="0 -0.026 0.06">
                        <joint name="panda_finger_joint2" type="slide" axis="0 1 0" range="-0.005 0.025" damping="10.0"/>
                        <geom name="finger2" type="box" size="0.012 0.005 0.020" rgba="0.3 0.3 0.3 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act_j1" joint="panda_joint1" kp="450" kv="40"/>
    <position name="act_j2" joint="panda_joint2" kp="450" kv="40"/>
    <position name="act_j3" joint="panda_joint3" kp="400" kv="35"/>
    <position name="act_j4" joint="panda_joint4" kp="400" kv="35"/>
    <position name="act_j5" joint="panda_joint5" kp="300" kv="25"/>
    <position name="act_j6" joint="panda_joint6" kp="300" kv="25"/>
    <position name="act_j7" joint="panda_joint7" kp="200" kv="15"/>
    <position name="act_finger1" joint="panda_finger_joint1" kp="200" kv="20"/>
    <position name="act_finger2" joint="panda_finger_joint2" kp="200" kv="20"/>
  </actuator>
</mujoco>
"""
    return mjcf
