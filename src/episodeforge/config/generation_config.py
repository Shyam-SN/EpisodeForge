"""Configuration for synthetic demonstration generation in MuJoCo."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    name: str
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fov: float = 45.0
    pos: list[float]
    quat: list[float]  # [w, x, y, z]


class RandomizationConfig(BaseModel):
    table_texture_jitter: bool = True
    light_intensity_range: tuple[float, float] = (0.7, 1.3)
    object_color_jitter: bool = True
    object_pos_noise_m: float = 0.03
    object_rot_noise_rad: float = 0.35
    distractor_count_range: tuple[int, int] = (0, 2)
    friction_range: tuple[float, float] = (0.7, 1.3)
    mass_range: tuple[float, float] = (0.8, 1.2)


class ImperfectionRateConfig(BaseModel):
    failed_grasp_rate: float = Field(default=0.08, ge=0.0, le=0.5)
    timeout_rate: float = Field(default=0.03, ge=0.0, le=0.2)
    nan_glitch_rate: float = Field(default=0.02, ge=0.0, le=0.2)
    duplicate_episode_rate: float = Field(default=0.02, ge=0.0, le=0.2)
    idle_padding_rate: float = Field(default=0.05, ge=0.0, le=0.3)


class GenerationConfig(BaseModel):
    robot_model: str = "franka_emika_panda"
    tasks: list[str] = Field(default_factory=lambda: ["pick_place", "push", "stack"])
    total_episodes: int = 1000
    control_rate_hz: float = 15.0
    physics_rate_hz: float = 500.0
    cameras: list[CameraConfig] = Field(default_factory=list)
    randomization: RandomizationConfig = Field(default_factory=RandomizationConfig)
    imperfections: ImperfectionRateConfig = Field(default_factory=ImperfectionRateConfig)
    seed: int = 0
    workers: int = 4
    save_format: Literal["png_master", "ffv1"] = "png_master"
