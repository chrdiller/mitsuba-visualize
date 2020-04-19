from pathlib import Path
from typing import List
import numpy as np

import set_python_path
from mitsuba.core import Vector, Spectrum, Scheduler, LocalWorker, RemoteWorker, PluginManager, Appender, EInfo, Thread, Transform
from mitsuba.render import Scene, RenderQueue


def axis_unit_vector(axis: str):
    """
    Create a unit vector along the given axis
    :param axis: 3D axis x, y, or z
    :return: A Vector of length one along the specified axis
    """
    if axis == 'x':
        return Vector(1, 0, 0)
    elif axis == 'y':
        return Vector(0, 1, 0)
    elif axis == 'z':
        return Vector(0, 0, 1)
    else:
        raise ValueError("Choose between x, y, and z")


def read_filelist(filepath: Path) -> List[str]:
    """
    Read a list of files from a file (one file per line). Lines beginning with # are ignored
    :param filepath: The path of the file containing filenames
    :return: A list of filenames as strings
    """
    assert filepath.is_file()
    with open(filepath, 'r') as f:
        return [line for line in f.read().splitlines() if not line.startswith('#')]


def create_spectrum_from_rgb(r: int, g: int, b: int) -> Spectrum:
    """
    Create a Mitsuba Spectrum from r, g, and b values
    :param r: Red component
    :param g: Green component
    :param b: Blue component
    :return: A Mitsuba Spectrum
    """
    assert (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255), "Provide integer rgb values in the range 0-255"
    spectrum = Spectrum()
    spectrum.fromSRGB(float(r) / 255., float(g) / 255., float(b) / 255.)

    return spectrum


def get_predefined_spectrum(name: str) -> Spectrum:
    """
    Get a predefined spectrum from a name string.
    Currently supports: light_blue, cornflower_blue, orange
    :param name: The spectrum name
    :return: A Mitsuba Spectrum
    """
    if name == 'light_blue':
        return create_spectrum_from_rgb(160, 180, 200)
    elif name == 'cornflower_blue':
        return create_spectrum_from_rgb(100, 149, 237)
    elif name == 'orange':
        return create_spectrum_from_rgb(200, 160, 0)
    else:
        raise ValueError


def prepare_queue(num_local_workers, remote_stream=None) -> RenderQueue:
    """
    Prepare a Mitsuba render queue with the given amount of workers
    :param num_local_workers: Number of local rendering workers on the CPU (corresponding to CPU cores fully used during rendering)
    :param remote_stream: TODO
    :return: A mitsuba RenderQueue object
    """
    scheduler = Scheduler.getInstance()
    queue = RenderQueue()
    for worker_idx in range(num_local_workers):
        local_worker = LocalWorker(worker_idx, f'LocalWorker-{worker_idx}')
        scheduler.registerWorker(local_worker)
    if remote_stream is not None:
        for idx, stream in enumerate(remote_stream):
            remote_worker = RemoteWorker(f'RemoteWorker-{idx}', stream)
            scheduler.registerWorker(remote_worker)
    scheduler.start()

    return queue


def construct_simple_scene(scene_objects, sensor) -> Scene:
    """
    Construct a simple scene containing given objects and using the given sensor. Uses the path integrator and constant
    emitter
    :param scene_objects: All scene child objects to add
    :param sensor: The mitsuba sensor definition to use for this scene
    :return: The scene created, already configured and initialized
    """
    pmgr = PluginManager.getInstance()
    integrator = pmgr.create({'type': 'path'})
    emitter = pmgr.create({'type': 'constant'})

    scene = Scene()
    scene.addChild(integrator)
    scene.addChild(emitter)
    scene.addChild(sensor)
    for obj in scene_objects:
        scene.addChild(obj)

    scene.configure()
    scene.initialize()

    return scene


def convert_transform2numpy(transform: Transform) -> np.ndarray:
    """
    Get a numpy array containing the same transformation matrix values as a Mitsuba Transform object
    :param transform: Mitsuba Transform
    :return: 4x4 Numpy array representing the transformation matrix
    """
    matrix = np.zeros([4, 4], dtype=float)
    for i in range(4):
        for j in range(4):
            matrix[i, j] = transform.getMatrix()[i, j]
    return matrix


def distances_from_control_points(camera_poses: List[Transform], control_poses: List[Transform]) -> List[np.ndarray]:
    """
    Calculate distance to closest control pose for every camera pose
    :param camera_poses: Camera poses in a trajectory
    :param control_poses: Control poses to calculate distances to
    :return: The minimal distance to a control pose for every camera pose
    """
    camera_poses = [convert_transform2numpy(camera.getMatrix()) for camera in camera_poses]
    control_poses = [convert_transform2numpy(camera.getMatrix()) for camera in control_poses]
    distances = []
    for pose in camera_poses:
        curr_distances = [np.linalg.norm(pose[:3, 3] - control_pose[:3, 3]) for control_pose in control_poses]
        distances.append(np.min(curr_distances))
    return distances


def generate_mesh_paths(input_paths: List[Path], base_output_path: Path, selected_meshes: List[str] = None):
    """
    Generator for mesh paths. Takes a list of input paths, a base output path and a list of selected scenes.
    If an item in input paths is a ply file, it will be yielded. If it is a path, its subelements will be traversed and
    yielded if they are ply files and in the list of selected files
    :param input_paths: List of ply files or directories containing ply files, or a mixture of both
    :param base_output_path: The base output path. If input_paths contains directories, their last components will translate
    into a subdirectory in base_output_path
    :param selected_meshes: A list of selected meshes in the input_paths subdirectories. Elements have to end with .ply.
    Does not apply to elements in input_paths that are ply files.
    :return: Yields tuples: (path to ply file, directory output path for this file)
    """
    for input_path in input_paths:
        if input_path.is_file(): # If it is a ply file, yield
            if input_path.suffix == '.ply':
                yield input_path, base_output_path
        else: # Else, iterate over all files in the directory
            for file in input_path.iterdir():
                output_path = base_output_path / input_path.name
                if not output_path.is_dir():
                    output_path.mkdir(parents=False)
                if file.suffix == '.ply' and (selected_meshes is None or str(file.name) in selected_meshes):
                    yield file, output_path


def redirect_logger(write_function=print, log_level=EInfo, tqdm_progressbar=None):
    """
    Redirect Mitsuba's Logger output to a custom function (so it can be used with e.g. tqdm).
    Additionally, can be used to control the log level
    :param write_function: A function like print() or tqdm.write() that is used to write log messages and progess bars
    :param log_level: The Mitsuba log level (mitsuba.EError, ...)
    :param tqdm_progressbar: Optionally, pass a tqdm progress bar. The Mitsuba rendering bar will be set as that bar's description
    :return:
    """
    class RedirectedAppender(Appender):
        def __init__(self, write_function, tqdm_progressbar):
            self.write_function = write_function
            self.tqdm_progressbar = tqdm_progressbar
            super().__init__()

        def append(self, log_level, message):
            self.write_function(message)

        def logProgress(self, progress, name, formatted, eta):
            if self.tqdm_progressbar is not None:
                self.tqdm_progressbar.set_description_str(formatted.replace('\r', ''), refresh=True)
            else:
                self.write_function(f"\r{formatted}", end='')

    logger = Thread.getThread().getLogger()
    logger.clearAppenders()
    logger.addAppender(RedirectedAppender(write_function, tqdm_progressbar))
    logger.setLogLevel(log_level)


if __name__ == '__main__':
    raise NotImplementedError("Cannot call the util script directly")
