from typing import List

import numpy as np
from pyquaternion import Quaternion

from math_util.util import find_closest_orthogonal_matrix


def catmull_rom_spline(camera_poses: List[np.ndarray], num_points: int) -> List[np.ndarray]:
    """
    Interpolate points on a Catmull-Rom spline
    Reading: Edwin Catmull, Raphael Rom, A CLASS OF LOCAL INTERPOLATING SPLINES,
                Computer Aided Geometric Design, Academic Press, 1974, Pages 317-326
    :param camera_poses: List of 4 camera poses. Points are interpolated on the curve segment between pose 1 and 2
    :param num_points: Number of points to interpolate between pose 1 and 2
    :return: A list of num_points point on the curve segment between pose 1 and 2
    """
    # Formulas from https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
    assert len(camera_poses) == 4, "Need exactly 4 camera poses for interpolation"
    interpolated_cameras = []

    # rotations
    q1 = Quaternion(matrix=find_closest_orthogonal_matrix(camera_poses[1][:3, :3]))
    q2 = Quaternion(matrix=find_closest_orthogonal_matrix(camera_poses[2][:3, :3]))

    # translations
    translations = [pose[:3, 3] for pose in camera_poses]

    if np.linalg.norm(translations[1] - translations[2]) < 0.001:
        for i in range(num_points):
            ext = np.eye(4)
            ext[:3, :3] = Quaternion.slerp(q1, q2, float(i) / num_points).rotation_matrix
            ext[:3, 3] = translations[1]
            interpolated_cameras.append(ext)
    else:
        eps = 0.001
        alpha = 0.5  # Default for centripetal catmull rom splines

        # Calculate knot parameters
        t0 = 0.0
        t1 = np.linalg.norm(translations[1] - translations[0]) ** alpha + t0
        t2 = np.linalg.norm(translations[2] - translations[1]) ** alpha + t1
        t3 = np.linalg.norm(translations[3] - translations[2]) ** alpha + t2

        # Calculate points on curve segment C
        new_points = []
        for t in np.arange(t1, t2, (t2 - t1) / float(num_points)):
            A1 = (t1 - t) / max(eps, t1 - t0) * translations[0] + (t - t0) / max(eps, t1 - t0) * translations[1]
            A2 = (t2 - t) / max(eps, t2 - t1) * translations[1] + (t - t1) / max(eps, t2 - t1) * translations[2]
            A3 = (t3 - t) / max(eps, t3 - t2) * translations[2] + (t - t2) / max(eps, t3 - t2) * translations[3]

            B1 = (t2 - t) / max(eps, t2 - t0) * A1 + (t - t0) / max(eps, t2 - t0) * A2
            B2 = (t3 - t) / max(eps, t3 - t1) * A2 + (t - t1) / max(eps, t3 - t1) * A3

            C = (t2 - t) / max(eps, t2 - t1) * B1 + (t - t1) / max(eps, t2 - t1) * B2
            new_points.append(C)

        # For each point, also slerp the rotation matrix
        for idx, point in enumerate(new_points):
            ext = np.eye(4)
            ext[:3, :3] = Quaternion.slerp(q1, q2, float(idx) / len(new_points)).rotation_matrix
            ext[:3, 3] = point
            interpolated_cameras.append(ext)

    return interpolated_cameras


if __name__ == "__main__":
    raise NotImplementedError("Cannot call this math_util script directly")
