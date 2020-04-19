import sys
import argparse
from pathlib import Path
from typing import List, Tuple
from itertools import cycle

import numpy as np
from tqdm import tqdm

sys.path.append('')
import set_python_path
from mitsuba.core import PluginManager, Point3, Spectrum, Vector3, Transform, AABB, EWarn
from mitsuba.render import RenderJob

from io_util import ply
import util
import cameras


def create_spheres(pointcloud: np.ndarray, spectrum: Spectrum, sphere_radius: float = 1.) -> List:
    """
    Create little spheres at the pointcloud's points' locations.
    :param pointcloud: 3D pointcloud, as Nx3 ndarray
    :param spectrum: The color spectrum to use with the spheres' diffuse bsdf
    :param sphere_radius: The radius to use for each sphere
    :return: A list of mitsuba shapes, to be added to a Scene
    """
    spheres = []
    for row in pointcloud:
        sphere = PluginManager.getInstance().create({
            'type': 'sphere',
            'center': Point3(float(row[0]), float(row[1]), float(row[2])),
            'radius': sphere_radius,
            'bsdf': {
                'type': 'diffuse',
                'diffuseReflectance': spectrum
            },
        })
        spheres.append(sphere)

    return spheres


def get_pointcloud_bbox(pointcloud: np.ndarray) -> AABB:
    """
    Create a mitsuba bounding box by using min and max on the input ndarray
    :param pointcloud: The pointcloud (Nx3) to calculate the bounding box from
    :return: The bounding box around the given pointcloud
    """
    min_values = np.min(pointcloud, axis=0)
    max_values = np.max(pointcloud, axis=0)
    min_corner = Point3(*[float(elem) for elem in min_values])
    max_corner = Point3(*[float(elem) for elem in max_values])
    return AABB(min_corner, max_corner)


def load_from_ply(filepath: Path) -> np.ndarray:
    """
    Load points from a ply file
    :param filepath: The path of the file to read from
    :return: An ndarray (Nx3) containing all read points
    """
    points = ply.read_ply(str(filepath))['points'].to_numpy()
    return points


def render_pointclouds(pointcloud_paths: List[Tuple[Path, Path]], sensor, sphere_radius=1., num_workers=8) -> None:
    """
    Render pointclouds by pkacing little Mitsuba spheres at all point locations
    :param pointcloud_paths: Path tuples (input_mesh, output_path) for all pointcloud files
    :param sensor: The sensor containing the transform to render with
    :param sphere_radius: Radius of individual point spheres
    :param num_workers: Number of concurrent render workers
    :return:
    """
    queue = util.prepare_queue(num_workers)

    pointcloud_paths = list(pointcloud_paths)
    with tqdm(total=len(pointcloud_paths), bar_format='Total {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {desc}', dynamic_ncols=True) as t:
        util.redirect_logger(tqdm.write, EWarn, t)
        for pointcloud_path in pointcloud_paths:
            input_path, output_path = pointcloud_path
            t.write(f'Rendering {input_path.stem}')
            # Make Pointcloud Spheres
            spheres = create_spheres(load_from_ply(input_path), util.get_predefined_spectrum('orange'), sphere_radius)
            # Make Scene
            scene = util.construct_simple_scene(spheres, sensor)
            scene.setDestinationFile(str(output_path / f'{input_path.stem}.png'))
            # Make Result
            job = RenderJob(f'Render-{input_path.stem}', scene, queue)
            job.start()

            queue.waitLeft(0)
            queue.join()
            t.update()


def main(args):
    input_filepaths = [Path(elem) for elem in args.input]
    output_path = Path(args.output)

    assert all([elem.is_file() for elem in input_filepaths])
    assert output_path.is_dir()

    bbox = get_pointcloud_bbox(load_from_ply(input_filepaths[0]))
    sensor_transform = cameras.create_transform_on_bbsphere(bbox,
                                                            radius_multiplier=3., positioning_vector=Vector3(0, -1, 1),
                                                            tilt=Transform.rotate(util.axis_unit_vector('x'), -25.))
    sensor = cameras.create_sensor_from_transform(sensor_transform, args.width, args.height,
                                                  fov=45., num_samples=args.samples)
    render_pointclouds(zip(input_filepaths, cycle([output_path])), sensor, args.radius, args.workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Render pointcloud with mitsuba by placing little spheres at the points' positions")
    parser.add_argument('input', nargs='+', help="Path(s) to the ply file(s) containing a pointcloud(s) to render")
    parser.add_argument('-o', '--output', required=True, help='Path to write renderings to')

    # Pointcloud parameters
    parser.add_argument('--radius', default=1., type=float, help='Radius of a single point')

    # Sensor parameters
    parser.add_argument('--width', default=1280, type=int, help='Width of the resulting image')
    parser.add_argument('--height', default=960, type=int, help='Height of the resulting image')
    parser.add_argument('--samples', default=128, type=int, help='Number of integrator samples per pixel')

    # General render parameters
    parser.add_argument('--workers', required=False, default=8, type=int, help="How many concurrent workers to use")

    main(parser.parse_args())
