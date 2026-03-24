from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def rotation_matrix_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    rotation = np.column_stack([normalize(x_axis), normalize(y_axis), normalize(z_axis)])
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("Rotation matrix must be right-handed.")
    return rotation


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = np.array(rotation, dtype=np.float64)
    trace = float(np.trace(rotation))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    return normalize(quat)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(np.array(quat, dtype=np.float64))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.array(self.position, dtype=np.float64))
        object.__setattr__(self, "rotation", np.array(self.rotation, dtype=np.float64))

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        return matrix_to_quat_wxyz(self.rotation)

    def transformed(self, local_offset: np.ndarray) -> np.ndarray:
        return self.position + self.rotation @ np.array(local_offset, dtype=np.float64)


def invert_pose(pose: Pose) -> Pose:
    rotation_t = pose.rotation.T
    position = -(rotation_t @ pose.position)
    return Pose(position=position, rotation=rotation_t)


def compose_pose(parent: Pose, child: Pose) -> Pose:
    return Pose(
        position=parent.position + parent.rotation @ child.position,
        rotation=parent.rotation @ child.rotation,
    )
