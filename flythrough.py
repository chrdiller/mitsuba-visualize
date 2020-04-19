import sys
import argparse
from pathlib import Path
from typing import Tuple, List
import shutil

from tqdm import tqdm
import moviepy.editor as mpy

sys.path.append('')
import set_python_path
from mitsuba.core import *
from mitsuba.render import RenderJob, RenderQueue

import mesh
import cameras
import util
import pointcloud


def render_trajectory_frames(mesh_path: Tuple[Path, Path], trajectory: List[Transform], queue: RenderQueue,
                             shutter_time: float, width: int = 1920, height: int = 1440, fov: float = 60.,
                             num_samples: int = 32) -> None:
    """
    Render a camera trajectory through a mesh loaded from mesh_path[0] at the Transformations given in trajectory
    :param mesh_path: Path tuple (input_filepath, output_dirpath) of the mesh to render
    :param trajectory: List of Mitsuba Transformations, corresponding to cameras in the trajectory
    :param shutter_time: The camera shutter time. Controls the amount of motion blur between every pair of consecutive frames
    :param width: Parameter passed to cameras.create_sensor_from_transform
    :param height: Parameter passed to cameras.create_sensor_from_transform
    :param fov: The field of view
    :param num_samples: Parameter passed to cameras.create_sensor_from_transform
    :param queue: The Mitsuba render queue to use for all the frames
    :return:
    """
    input_path, output_path = mesh_path

    # Make Mesh
    mesh_obj = mesh.prepare_ply_mesh(input_path, util.get_predefined_spectrum('light_blue'))

    # Create sensors with animation transforms
    sensors = cameras.create_animated_sensors(trajectory, shutter_time, width, height, fov, num_samples)

    scene = util.construct_simple_scene([mesh_obj], sensors[0])

    with tqdm(total=len(sensors), bar_format='Total {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {desc}', dynamic_ncols=True) as t:
        util.redirect_logger(tqdm.write, EError, t)
        t.write(input_path.stem)
        for idx, sensor in enumerate(sensors):
            # Make Scene
            scene.setSensor(sensor)
            scene.setDestinationFile(str(output_path / f'{input_path.stem}-{idx}.png'))
            # Make Result
            job = RenderJob(f'Render-{input_path.stem}-{idx}', scene, queue)
            job.start()
            queue.waitLeft(0)
            queue.join()
            t.update()


def create_movie_from_frames(images_path: Path, output_filepath: Path, frame_durations=None):
    """
    Create a movie from images (like the ones rendered by Mitsuba).
    Uses moviepy (which uses ffmpeg). Movies will have 30fps if frame_durations is None
    :param images_path: Path containing only image files
    :param output_filepath: Path to output video file
    :param frame_durations: If not None, specifies the duration of frames (1 = one frame).
    Must contain one for every file in images_path
    :return:
    """
    files = list(images_path.iterdir())
    files = sorted(files, key=lambda filepath: int(filepath.stem.split('-')[-1]))

    if frame_durations is not None:
        assert len(files) == len(frame_durations)
        clip = mpy.ImageSequenceClip([str(file) for file in files], durations=frame_durations)
    else:
        clip = mpy.ImageSequenceClip([str(file) for file in files], fps=30)

    clip.on_color().write_videofile(str(output_filepath), fps=30, audio=False)


def main(args):
    input_paths = [Path(path_str) for path_str in args.input]
    output_path = Path(args.output)
    assert output_path.parent.is_dir()
    if not output_path.is_dir():
        output_path.mkdir(parents=False)

    assert all([path.exists() for path in input_paths])
    mesh_paths = list(util.generate_mesh_paths(input_paths, output_path, selected_meshes=util.read_filelist(Path(args.scenes_list)) if args.scenes_list is not None else None))

    for input_path, output_path in mesh_paths:
        intermediate_path = Path(output_path) / f'{input_path.stem}-frames'
        if not args.norender:
            if intermediate_path.is_dir():
                res = input(f"Mesh {input_path.stem} has already been rendered, delete and re-render? Y/n")
                if res == 'n':
                    sys.exit(0)
                shutil.rmtree(intermediate_path)

    render_queue = util.prepare_queue(args.workers, remote_stream=[SocketStream(remote, 7554) for remote in args.remote] if args.remote is not None else None)
    for input_path, output_path in mesh_paths:
        intermediate_path = Path(output_path) / f'{input_path.stem}-frames'
        if not args.norender:
            intermediate_path.mkdir()

        if args.cameras is None:
            bbox = pointcloud.get_pointcloud_bbox(pointcloud.load_from_ply(input_path))
            trajectory = cameras.create_trajectory_on_bbsphere(bbox,
                                                               initial_positioning_vector=Vector3(0, 1, 1),
                                                               rotation_around=util.axis_unit_vector('z'),
                                                               num_cameras=args.frames, radius_multiplier=3.,
                                                               tilt=Transform.rotate(util.axis_unit_vector('x'), 20.))

        else:
            cameras_path = Path(args.cameras)
            if cameras_path.is_dir():
                camera_filepath = cameras_path / f'{input_path.stem}.xml'
            else:
                camera_filepath = cameras_path

            assert camera_filepath.is_file(), f"Camera file {camera_filepath} has to exist"

            key_poses = cameras.read_meshlab_sensor_transforms(camera_filepath)
            trajectory = cameras.create_interpolated_trajectory(key_poses, method=args.interpolation, num_total_cameras=args.frames)

        if not args.norender:
            render_trajectory_frames((input_path, intermediate_path), trajectory, render_queue,
                                     args.shutter_time, args.width, args.height, args.fov, args.samples)
        if not args.novideo:
            create_movie_from_frames(intermediate_path, output_path / f'{input_path.stem}.mp4')

        if not args.keep:
            shutil.rmtree(intermediate_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Render a flythrough video of a scene")
    parser.add_argument('input', nargs='+', help='Path to the ply file to render')
    parser.add_argument('-o', '--output', required=True, help='Path to write output video to')

    # Scene render parameters
    parser.add_argument('--remote', nargs='*', required=False, help='Urls of the remote render servers')
    parser.add_argument('--novideo', action='store_true', help='Only render frames, do not produce video')
    parser.add_argument('--norender', action='store_true', help='Only render video from existing frames, no rendering')
    parser.add_argument('--keep', action='store_true', help='Whether to keep the frame images')
    parser.add_argument('--scenes_list', required=False, help='Path to file containing filenames to render in base path')
    parser.add_argument('--frames', required=True, type=int, help='Number of frames to render (The video file will have 30fps)')
    parser.add_argument('--cameras', required=False, help='XML file containing meshlab cameras (or path to directory only containing such files). '
                                                          'If set, this is used for spline interpolation. Otherwise, a rotating flyover is generated')

    # Sensor parameters
    parser.add_argument('--shutter_time', default=1., type=float, help='Shutter time of the moving sensor')
    parser.add_argument('--width', default=1280, type=int, help='Width of the resulting image')
    parser.add_argument('--height', default=960, type=int, help='Height of the resulting image')
    parser.add_argument('--fov', default=60., type=float, help='Field of view of the sensor in degrees (meshlab default is 60)')
    parser.add_argument('--samples', default=128, type=int, help='Number of integrator samples per pixel')
    parser.add_argument('--interpolation', default='catmullrom', choices=['catmullrom', 'bezier'], help='Which method to use for interpolation between control points')

    # General render parameters
    parser.add_argument('--workers', required=False, default=8, type=int, help="How many local concurrent workers to use")

    main(parser.parse_args())
