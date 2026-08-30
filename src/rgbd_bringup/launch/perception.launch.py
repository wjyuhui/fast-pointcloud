from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import re

_MAX_OPENMP_THREADS = 16
_MAX_CPU_ID = 7


def _as_bool(value):
    return value.strip().lower() == 'true'


def _parse_omp_places(places_str):
    matches = re.findall(r'\{([^}]*)\}', places_str.strip())
    cpus = []
    for raw in matches:
        token = raw.strip()
        if not token.isdigit():
            raise RuntimeError(
                "projection_omp_places must list one CPU per place, "
                "e.g. '{4},{5},{6},{7}', got {%s}" % token)
        cpus.append(int(token))
    return cpus


def launch_setup(context, *args, **kwargs):
    bringup_share = get_package_share_directory('rgbd_bringup')
    detection_share = get_package_share_directory('rgbd_detection')
    params_file = os.path.join(bringup_share, 'config', 'perception.yaml')

    model_path = LaunchConfiguration('model_path').perform(context)
    if not model_path:
        model_path = os.path.join(detection_share, 'models', 'yolov8n_rk3588_fp16.rknn')

    use_orbbec = LaunchConfiguration('use_orbbec').perform(context) == 'true'
    enable_yolo = LaunchConfiguration('enable_yolo').perform(context) == 'true'
    projection_enable_openmp = _as_bool(
        LaunchConfiguration('projection_enable_openmp').perform(context))
    projection_enable_neon = _as_bool(
        LaunchConfiguration('projection_enable_neon').perform(context))
    projection_omp_places = LaunchConfiguration('projection_omp_places').perform(context)
    threads_raw = LaunchConfiguration('projection_openmp_threads').perform(context)
    compute_cpu_raw = LaunchConfiguration('projection_compute_cpu').perform(context)
    try:
        projection_openmp_threads = int(threads_raw)
    except ValueError as exc:
        raise RuntimeError(
            'projection_openmp_threads must be an integer, got %r' % threads_raw
        ) from exc
    try:
        projection_compute_cpu = int(compute_cpu_raw)
    except ValueError as exc:
        raise RuntimeError(
            'projection_compute_cpu must be an integer, got %r' % compute_cpu_raw
        ) from exc

    if projection_openmp_threads < 1 or projection_openmp_threads > _MAX_OPENMP_THREADS:
        raise RuntimeError(
            'projection_openmp_threads must be in [1, %d], got %d' % (
                _MAX_OPENMP_THREADS, projection_openmp_threads))
    if projection_compute_cpu < 0 or projection_compute_cpu > _MAX_CPU_ID:
        raise RuntimeError(
            'projection_compute_cpu must be in [0, %d], got %d' % (
                _MAX_CPU_ID, projection_compute_cpu))
    if projection_enable_openmp and projection_enable_neon:
        raise RuntimeError(
            'V3 does not support projection OpenMP and NEON at the same time')
    place_cpus = []
    if projection_enable_openmp:
        if not projection_omp_places.strip():
            raise RuntimeError(
                'projection_omp_places must not be empty when OpenMP is enabled')
        place_cpus = _parse_omp_places(projection_omp_places)
        if len(place_cpus) != projection_openmp_threads:
            raise RuntimeError(
                'projection_omp_places count (%d) must match '
                'projection_openmp_threads (%d)' % (
                    len(place_cpus), projection_openmp_threads))

    nodes = []

    # base_link -> camera_link (mount). Orbbec publishes camera internal TF.
    nodes.append(
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_link',
            arguments=[
                '--x', '0', '--y', '0', '--z', '1.5',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'camera_link',
            ],
        )
    )

    projection_parameters = {
        'projection_enable_openmp': projection_enable_openmp,
        'projection_openmp_threads': projection_openmp_threads,
        'projection_enable_neon': projection_enable_neon,
        'projection_compute_cpu': (
            place_cpus[0] if projection_enable_openmp else projection_compute_cpu
        ),
    }

    # 只有V3启用OpenMP时才传递非空CPU数组。
    # V2/V4不传递该参数，节点使用默认空数组。
    if projection_enable_openmp:
        projection_parameters['projection_openmp_cpus'] = place_cpus

    cloud_workspace_kwargs = {
        'package': 'rgbd_pcl',
        'executable': 'cloud_workspace_node',
        'name': 'cloud_workspace_node',
        'output': 'screen',
        'parameters': [
            params_file,
            projection_parameters
        ],
    }
    cloud_workspace_kwargs['additional_env'] = {
        'OMP_DYNAMIC': 'FALSE',
        # Keep libgomp from binding the initial thread at process startup.
        # Compute-thread affinity is applied on the persistent compute thread.
        'OMP_PROC_BIND': 'FALSE',
    }
    nodes.append(Node(**cloud_workspace_kwargs))

    nodes.append(
        Node(
            package='rgbd_pcl',
            executable='pcl_obstacle_node',
            name='pcl_obstacle_node',
            output='screen',
            parameters=[params_file],
        )
    )

    nodes.append(
        Node(
            package='rgbd_bev',
            executable='bev_node',
            name='bev_node',
            output='screen',
            parameters=[params_file],
        )
    )

    if enable_yolo:
        nodes.append(
            Node(
                package='rgbd_detection',
                executable='yolo_3d_node',
                name='yolo_3d_node',
                output='screen',
                parameters=[
                    params_file,
                    {'model_path': model_path, 'enable_detection': True},
                ],
            )
        )

    actions = list(nodes)

    if use_orbbec:
        orbbec_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('orbbec_camera'),
                    'launch',
                    'gemini_330_series.launch.py',
                ])
            ]),
           launch_arguments={
          'depth_registration': 'true',
          'align_mode': 'SW',
          'align_target_stream': 'COLOR',

          'depth_width': '1280',
          'depth_height': '720',
          'depth_fps': '30',

          'color_width': '1280',
          'color_height': '720',
          'color_fps': '30',

          'color_qos': 'SENSOR_DATA',
          'depth_qos': 'SENSOR_DATA',
          'color_camera_info_qos': 'SENSOR_DATA',
          'depth_camera_info_qos': 'SENSOR_DATA',
          }.items(),
        )
        actions.insert(0, orbbec_launch)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_orbbec', default_value='true'),
        DeclareLaunchArgument('enable_yolo', default_value='true'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('projection_enable_openmp', default_value='false'),
        DeclareLaunchArgument('projection_openmp_threads', default_value='1'),
        DeclareLaunchArgument('projection_enable_neon', default_value='true'),
        DeclareLaunchArgument('projection_omp_places', default_value='{7}'),
        DeclareLaunchArgument('projection_compute_cpu', default_value='7'),
        OpaqueFunction(function=launch_setup),
    ])
