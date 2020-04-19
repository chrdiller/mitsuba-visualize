from typing import List

import numpy as np
from scipy.special import comb
from pyquaternion import Quaternion

from math_util.util import find_closest_orthogonal_matrix


def avoid_jumps(cur: Quaternion, prev: Quaternion):
    """
    Avoid jumps between quaternions that happen when a flipped quaternion follows a non-flipped one
    :param cur: Current quaternion
    :param prev: Previous quaternion
    :return: A new current quaternion which is either the original one or the sign-flipped version thereof
    """
    if (prev - cur).norm < (prev + cur).norm:
        return cur
    else:
        return -cur


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    :param i: Step
    :param n: Total number of steps
    :param t: Variable
    :return: Bernstein polynomial at t with parameters i and n
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(control_poses: List[np.ndarray], num_steps=1000):
    """
    Given a set of control points, return the bezier curve defined by the control points.
    See http://processingjs.nihongoresources.com/bezierinfo/
    :param control_poses: List of poses (4x4 numpy arrays (R|t))
    :param num_steps: Number of camera poses to interpolate between first and last keypoint, considering key
    points in between (but not necessarily passing through them)
    :return: List with num_steps interpolated camera poses (4x4 numpy arrays (R|t))
    """
    interpolated_camera_poses = []

    control_quaternions = [Quaternion(matrix=find_closest_orthogonal_matrix(pose[:3, :3])) for pose in control_poses]
    control_rotations = [control_quaternions[0]]
    for quat_idx in range(1, len(control_quaternions)):
        control_rotations.append(avoid_jumps(control_quaternions[quat_idx], control_rotations[-1]))
    control_rotations = np.stack([q.elements for q in control_rotations])
    control_points = np.stack([pose[:3, 3] for pose in control_poses])

    t = np.linspace(0.0, 1.0, num_steps)

    polynomial_array = np.array([bernstein_poly(i, len(control_poses)-1, t) for i in range(0, len(control_poses))])

    interpolated_rotations = np.stack([np.dot(control_rotations[:, i], polynomial_array) for i in range(4)], axis=1)[::-1]
    interpolated_points = np.stack([np.dot(control_points[:, i], polynomial_array) for i in range(3)], axis=1)[::-1]

    for rotation, point in zip(interpolated_rotations, interpolated_points):
        ext = np.eye(4)
        ext[:3, :3] = Quaternion(rotation).rotation_matrix
        ext[:3, 3] = point
        interpolated_camera_poses.append(ext)

    return interpolated_camera_poses


if __name__ == "__main__":
    raise NotImplementedError("Cannot call this math_util script directly")
