import sys
import argparse
from pathlib import Path
from typing import List
import math

import numpy as np
import xml.etree.ElementTree as etree

sys.path.append('')
import set_python_path
from mitsuba.core import Transform, Matrix4x4, Vector3, PluginManager, normalize, AnimatedTransform, Point3
from mitsuba.render import Sensor

import util
from math_util.catmull_rom_spline import catmull_rom_spline
from math_util.bezier_curve import bezier_curve


def convert_meshlab2mitsuba_camera(camera_def: dict) -> Transform:
    """
    Takes a meshlab camera dict (usually loaded from a meshlab xml file) and turns it into a valid mitsuba Transform
    which can then be applied to a sensor
    :param camera_def: The camera def dict, containing at least keys RotationMatrix and TranslationVector
    :return: A mitsuba transform constructed from the given values
    """
    # Meshlab camera matrix is transposed
    matrix = Matrix4x4([
        float(elem) for elem in camera_def['RotationMatrix'].split(' ')[:16]
    ]).transpose()

    # Add translation vector
    translation = [-float(elem) for elem in camera_def['TranslationVector'].split(' ')]
    for i in range(3):
        matrix[i, 3] = translation[i]

    # Make Mitsuba transform and flip rotation signs (y is not flipped, otherwise resulting image will be flipped vertically)
    transform = Transform(matrix)
    transform *= transform.scale(Vector3(-1, 1, -1))

    return transform


def read_meshlab_sensor_transforms(file_path: Path) -> List[Transform]:
    """
    Reads meshlab camera properties from an xml file. To obtain those, simply press ctrl/cmd+c in meshlab, and paste into a text editor.
    If you paste multiple cameras into one file, make sure to have a valid document structure with one root node, e.g.
    <!DOCTYPE ViewState>
    <project>
     <VCGCamera ... />
     <ViewSettings ... />
    </project>

    :param file_path: Path to the xml file to read
    :return: A list of Mitsuba transforms
    """
    assert file_path.suffix == '.xml'
    root_node = etree.parse(file_path).getroot()

    transforms = [convert_meshlab2mitsuba_camera(elem.attrib) for elem in root_node if elem.tag == 'VCGCamera']

    return transforms


def create_transform_on_bbsphere(bbox, radius_multiplier: float, positioning_vector: Vector3, tilt=None) -> Transform:
    """
    Create a camera with origin on the bounding sphere around an object and looking towards the center of that sphere
    Camera center is calculated as: bbsphere_center + radius_multiplier * bbsphere_radius * normalized_positioning_vector
    :param bbox: The bounding box of the object from which to calculate the bounding sphere
    :param radius_multiplier: The value to multiply the bounding sphere's radius with
    (= distance of the camera origin to the center of the bounding sphere)
    :param positioning_vector: The vector pointing towards the camera center
    :param tilt: Transform to apply to camera origin. Usually a small rotation to tilt the camera perspective a little
    :return: A transform on the bbsphere
    """
    bsphere = bbox.getBSphere()
    camera_target = bsphere.center

    camera_offset = radius_multiplier * bsphere.radius * normalize(positioning_vector)
    camera_origin = camera_target + camera_offset

    camera_transform = Transform.lookAt(tilt * camera_origin if tilt is not None else camera_origin, camera_target, Vector3(0., 0., 1.))

    return camera_transform


def create_trajectory_on_bbsphere(bbox, initial_positioning_vector: Vector3, rotation_around: Vector3, num_cameras: int,
                                  radius_multiplier: float, tilt: Transform = None) -> List[Transform]:
    """
    Creates an interpolated trajectory on the bounding sphere of a scene/object
    :param bbox: The bounding box of the object/scene to render
    :param initial_positioning_vector: Controls where to position the camera initially
    :param rotation_around: Specify axis to rotate camera around
    :param num_cameras: Number of cameras to generate
    :param radius_multiplier: Cameras will be placed at distance multiplier * bbsphere radius
    :param tilt: Additional transformation meant to tilt the camera a bit
    :return: List with generated transforms
    """
    transforms = []

    step_angle = 360. / float(num_cameras)
    for camera_idx in range(num_cameras):
        transform = create_transform_on_bbsphere(bbox, radius_multiplier, Transform.rotate(rotation_around, step_angle * camera_idx) * tilt * initial_positioning_vector)
        transforms.append(transform)

    return transforms


def create_interpolated_trajectory(cameras: List[Transform], method: str,
                                   num_cameras_per_m: int = 75, num_cameras_per_rad: int = 225,
                                   num_total_cameras: int = 1000) -> List[Transform]:
    """
    Creates an interpolated camera trajectory using supplied key camera transforms
    :param cameras: Camera transformations to use as control points for interpolation
    :param method: How to interpolate: *bezier* (used for both camera centers and rotations):
    Smooth trajectory, but does not necessarily pass through the control points (except the first and last) or
    *catmullrom* (used for camera centers, using quaternion slerp for rotations):
    Passes through all control points, but needs more tuning to prevent weird behaviour
    :param num_cameras_per_m: catmullrom parameter: Camera centers per meter
    :param num_cameras_per_rad: catmullrom parameter: Camera rotations per radiant
    :param num_total_cameras: bezier parameter: Number of interpolated cameras between first and last key camera pose
    :return: A list of interpolated transforms
    """
    all_interpolated = []

    camera_pose_matrices = [util.convert_transform2numpy(camera) for camera in cameras]
    if method == 'bezier':
        interpolated_cameras = bezier_curve(camera_pose_matrices, num_steps=num_total_cameras)
        for elem in interpolated_cameras:
            position = Point3(*elem[:3, 3].tolist())
            look = Vector3(*elem[:3, :3].dot(np.array([0, 0, 1])).tolist())
            up = Vector3(*(elem[:3, :3].dot(np.array([0, 1, 0]))).tolist())
            all_interpolated.append(Transform.lookAt(position, position + look, up))

    elif method == 'catmullrom':
        assert len(cameras) >= 4
        camera_groups = [camera_pose_matrices[idx: idx + 4] for idx in range(len(cameras) - 3)]
        for camera_group in camera_groups:
            key_positions = (camera_group[1][:3, 3], camera_group[2][:3, 3])
            key_looks = (camera_group[1][:3, :3] * np.array([0, 0, 1]), camera_group[2][:3, :3] * np.array([0, 0, 1]))
            dist = np.linalg.norm(key_positions[1] - key_positions[0])
            angle = math.acos(np.clip(np.sum(key_looks[1] @ key_looks[0]), -1., 1.))

            num_t = int(np.round(dist / 100. * num_cameras_per_m))
            num_r = int(np.round(angle * num_cameras_per_rad))
            num = max(40, max(num_t, num_r))

            interpolated_cameras = catmull_rom_spline(camera_group, num)

            for elem in interpolated_cameras:
                position = Point3(*elem[:3, 3].tolist())
                look = Vector3(*elem[:3, :3].dot(np.array([0, 0, 1])).tolist())
                up = Vector3(*(elem[:3, :3].dot(np.array([0, 1, 0]))).tolist())
                all_interpolated.append(Transform.lookAt(position, position + look, up))

    else:
        raise NotImplementedError(f'The method you chose ({method}) is not implemented')

    return all_interpolated


def create_animated_sensors(trajectory: List[Transform], shutter_time: float = 1., width: int = 1920, height: int = 1440,
                            fov: float = 45., num_samples: int = 256) -> List[Sensor]:
    """
    Create an animated sensor (Applies motion blur to the rendered image)
    :param trajectory: The trajectory containing all Transforms
    :param shutter_time: The shutter time to be used to set the Mitsuba setShutterOpenTime
    :param width: Width of the generated image
    :param height: Height of the generated image
    :param fov: The sensor field of view
    :param num_samples: Number of samples per pixel (controls the noise in the resulting image)
    :return: A list of Mitsuba animated sensors
    """
    animated_sensors = []
    for transform_idx in range(len(trajectory)):
        atrafo = AnimatedTransform()
        atrafo.appendTransform(0, trajectory[transform_idx])
        atrafo.appendTransform(1, trajectory[min(transform_idx + 1, len(trajectory) - 1)])
        atrafo.sortAndSimplify()
        sensor = create_sensor_from_transform(atrafo, width, height, fov, num_samples)
        sensor.setShutterOpen(0)
        sensor.setShutterOpenTime(shutter_time)
        animated_sensors.append(sensor)

    return animated_sensors


def create_sensor_from_transform(transform, width=1920, height=1440, fov=45., num_samples=256) -> Sensor:
    """
    Create a Mitsuba sensor from a camera transform
    :param transform: The transform (camera to world) that this sensor uses
    :param width: The width of the resulting image
    :param height: The height of the resulting image
    :param fov: The field of view in degrees (default 45, meshlab uses 60)
    :param num_samples: Number of samples per pixel
    :return: A Mitsuba sensor
    """
    sensor = PluginManager.getInstance().create({
        'type': 'perspective',
        'film': {
            'type': 'ldrfilm',
            'width': width,
            'height': height,
            'pixelFormat': 'rgba',
            'exposure': 1.0,
            'banner': False
        },
        'sampler': {
            'type': 'ldsampler',
            'sampleCount': num_samples
        },
        'toWorld': transform,
        'fov': fov,
    })

    return sensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Load and parse cameras from different formats for mitsuba. Called directly, it will print the parsed mitsuba camera")
    parser.add_argument('camera_filename', type=str, help="The file containing camera definitions")
    parser.add_argument('--type', default='meshlab', choices=['meshlab'], help="Which type the camera definitions are")
    parser.add_argument('--interpolate', action='store_true', help="Whether to interpolate cameras between the given one (need at least 4 in this case)")

    args = parser.parse_args()

    camera_filepath = Path(args.camera_filename)
    assert camera_filepath.is_file(), "Camera file has to exist"

    if args.type == 'meshlab':
        transforms = read_meshlab_sensor_transforms(camera_filepath)
    else:
        raise NotImplementedError

    print(f"Read {len(transforms)} camera transformations from type {args.type}:")
    print(transforms)

    if args.interpolate:
        interpolated_cameras = create_interpolated_trajectory(transforms, method='bezier')

        points = np.array([elem[3, :3] for elem in interpolated_cameras])
        from io_util import ply

        ply.write_ply('/home/christian/tmp/cameras.ply', points=points, as_text=True)

        print(f"Interpolated {len(interpolated_cameras)} camera transformations:")
        print(interpolated_cameras)
