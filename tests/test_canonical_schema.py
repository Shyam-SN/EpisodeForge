"""Unit tests for CanonicalEpisode Pydantic model and explicit conventions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from episodeforge.canonical.schema import (
    CameraConvention,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraStream,
    CanonicalEpisode,
    ConventionsMetadata,
    ProvenanceMetadata,
    QAVerdict,
    QuaternionOrder,
)


def test_canonical_episode_creation():
    cam_intrinsics = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    cam_extrinsics = CameraExtrinsics(
        translation=[0.5, 0.0, 0.6],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        reference_frame="robot_base",
    )
    cam_stream = CameraStream(
        name="front",
        width=640,
        height=480,
        fps_nominal=30.0,
        frame_count=90,
        convention=CameraConvention.OPENCV,
        intrinsics=cam_intrinsics,
        extrinsics=cam_extrinsics,
        timestamps_ns=[i * 33333333 for i in range(90)],
    )

    episode = CanonicalEpisode(
        episode_id="a1b2c3d4e5f67890",
        source="generated_mujoco",
        source_ref="gen_sim_0001",
        task="pick_place",
        instruction="put the red cube in the bowl",
        success=True,
        seed=42,
        conventions=ConventionsMetadata(
            spatial_units="meters",
            angular_units="radians",
            quaternion_order=QuaternionOrder.WXYZ,
        ),
        cameras={"front": cam_stream},
        upstream_qa_verdict=QAVerdict.PASS,
        provenance=ProvenanceMetadata(
            input_content_hash="deadbeef12345678",
            generator_version="0.1.0",
            created_at_utc="2026-11-14T00:00:00Z",
        ),
    )

    assert episode.episode_id == "a1b2c3d4e5f67890"
    assert episode.conventions.quaternion_order == QuaternionOrder.WXYZ
    assert episode.cameras["front"].width == 640
    assert episode.upstream_qa_verdict == QAVerdict.PASS


def test_canonical_episode_validation_failure():
    with pytest.raises(ValidationError):
        # Invalid short episode_id
        CanonicalEpisode(
            episode_id="short",
            source="generated_mujoco",
            source_ref="ref",
            task="pick_place",
            instruction="test",
            success=True,
            provenance=ProvenanceMetadata(
                input_content_hash="hash",
                created_at_utc="now",
            ),
        )
