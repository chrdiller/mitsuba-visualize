import sys
import argparse
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

sys.path.append('')
import set_python_path
from mitsuba.core import *
from mitsuba.render import RenderJob

import util
import cameras


def prepare_ply_mesh(filepath: Path, spectrum, transformation=None):
    """
    Prepare a ply mesh: Load from given filepath, apply given color spectrum and (optionally) transformation.
    Uses a two-sided bsdf (so no black faces) and recomputed the normals before returning
    :param filepath: Path to a ply file
    :param spectrum: The color spectrum to use for the diffuse bsdf used for the mesh
    :param transformation: Optional transformation to apply to the mesh (toWorld transform)
    :return: A mitsuba mesh object
    """
    assert filepath.suffix == '.ply', f"{filepath} does not seem to be a ply file"
    mesh = PluginManager.getInstance().create({
        'type': 'ply',
        'filename': str(filepath),
        'bsdf': {
            'type': 'twosided',
            'bsdf': {
                'type': 'diffuse',
                'diffuseReflectance': spectrum
            }
        },
        'toWorld': transformation if transformation is not None else Transform()
    })
    mesh.computeNormals(True)

    return mesh


def render_multiple_perspectives(mesh_path: Tuple[Path, Path], sensors: list, num_workers=8) -> None:
    """
    Render one mesh from multiple camera perspectives
    :param mesh_path: Path tuples (input_filepath, output_dirpath) of the mesh to render
    :param sensors: The Mitsuba sensor definitions to render the mesh with
    :param num_workers: Number of CPU cores to use for rendering
    :return:
    """
    queue = util.prepare_queue(num_workers)

    input_path, output_path = mesh_path
    # Make Mesh
    mesh = prepare_ply_mesh(input_path, util.get_predefined_spectrum('light_blue'))

    with tqdm(total=len(sensors), bar_format='Total {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {desc}', dynamic_ncols=True) as t:
        util.redirect_logger(tqdm.write, EWarn, t)
        for idx, sensor in enumerate(sensors):
            t.write(f"Rendering with sensor {idx}")
            # Make Scene
            scene = util.construct_simple_scene([mesh], sensor)
            scene.setDestinationFile(str(output_path / f'{input_path.stem}-{idx}.png'))
            # Make Result
            job = RenderJob(f'Render-{input_path.stem}-{idx}', scene, queue)
            job.start()

            queue.waitLeft(0)
            queue.join()
            t.update()


def render_multiple_meshes(mesh_paths: List[Tuple[Path, Path]],
                           radius_multiplier=3., positioning_vector=Vector3(0, 1, 1), tilt=Transform.rotate(util.axis_unit_vector('x'), 20.),
                           width=1920, height=1440, num_samples=256, num_workers=8) -> None:
    """
    Render multiple meshes with the camera always in the same relative position (based on the mesh).
    :param mesh_paths: Path tuples (input_filepath, output_dirpath) of the meshes to render
    :param radius_multiplier: Parameter passed to cameras.create_transform_on_bbsphere
    :param positioning_vector: Parameter passed to cameras.create_transform_on_bbsphere
    :param tilt: Parameter passed to cameras.create_transform_on_bbsphere
    :param width: Parameter passed to cameras.create_sensor_from_transform
    :param height: Parameter passed to cameras.create_sensor_from_transform
    :param num_samples: Parameter passed to cameras.create_sensor_from_transform
    :param num_workers: Number of CPU cores to use for rendering
    :return:
    """
    queue = util.prepare_queue(num_workers)

    mesh_paths = list(mesh_paths)
    with tqdm(total=len(mesh_paths), bar_format='Total {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {desc}', dynamic_ncols=True) as t:
        util.redirect_logger(tqdm.write, EWarn, t)
        for mesh_path in mesh_paths:
            input_path, output_path = mesh_path
            t.write(f'Rendering {input_path.stem}')
            # Make Mesh
            mesh = prepare_ply_mesh(input_path, util.get_predefined_spectrum('light_blue'))
            # Make sensor
            sensor_transform = cameras.create_transform_on_bbsphere(mesh.getAABB(), radius_multiplier, positioning_vector, tilt)
            sensor = cameras.create_sensor_from_transform(sensor_transform, width, height, fov=45., num_samples=num_samples)
            # Make Scene
            scene = util.construct_simple_scene([mesh], sensor)
            scene.setDestinationFile(str(output_path / f'{input_path.stem}.png'))
            # Make Result
            job = RenderJob(f'Render-{input_path.stem}', scene, queue)
            job.start()

            queue.waitLeft(0)
            queue.join()
            t.update()


def main(args):
    input_paths = [Path(path_str) for path_str in args.input_paths]
    output_path = Path(args.output)

    assert all([path.exists() for path in input_paths])
    assert output_path.parent.is_dir()

    if not output_path.is_dir():
        output_path.mkdir(parents=False)

    if args.scenes_list is None and args.scene is None:
        mesh_paths = util.generate_mesh_paths(input_paths, output_path)
    elif args.scenes_list is not None and args.scene is None:
        mesh_paths = util.generate_mesh_paths(input_paths, output_path, util.read_filelist(Path(args.scenes_list)))
    else:  # args.scene is not None:
        mesh_paths = util.generate_mesh_paths(input_paths, output_path, [args.scene])

    if args.cameras is None:
        render_multiple_meshes(mesh_paths,
                               radius_multiplier=3., positioning_vector=Vector3(0, 1, 1),
                               tilt=Transform.rotate(util.axis_unit_vector('x'), 20.),
                               width=args.width, height=args.height, num_samples=args.samples,
                               num_workers=args.workers)
    else:
        sensor_transforms = cameras.read_meshlab_sensor_transforms(Path(args.cameras))
        sensors = [cameras.create_sensor_from_transform(transform, args.width, args.height, fov=45., num_samples=args.samples)
                   for transform in sensor_transforms]
        render_multiple_perspectives(next(mesh_paths), sensors, num_workers=args.workers)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Render directory')
    parser.add_argument('input_paths', nargs='*', help='Path(s) to directory containing all files to render')
    parser.add_argument('-o', '--output', required=True, help='Path to write renderings to')

    # Scene render parameters
    parser.add_argument('--scenes_list', required=False, help='Path to file containing filenames to render in base path')
    parser.add_argument('--scene', required=False, help='One scene. Overrides scenes_list')
    parser.add_argument('--cameras', required=False, help='XML file containing meshlab cameras')

    # Sensor parameters
    parser.add_argument('--width', default=1280, type=int, help='Width of the resulting image')
    parser.add_argument('--height', default=960, type=int, help='Height of the resulting image')
    parser.add_argument('--samples', default=128, type=int, help='Number of integrator samples per pixel')

    # General render parameters
    parser.add_argument('--workers', required=False, default=8, type=int, help="How many concurrent workers to use")

    main(parser.parse_args())
