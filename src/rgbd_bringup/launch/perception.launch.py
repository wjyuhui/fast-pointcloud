from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    bringup_share = get_package_share_directory('rgbd_bringup')
    detection_share = get_package_share_directory('rgbd_detection')
    params_file = os.path.join(bringup_share, 'config', 'perception.yaml')

    model_path = LaunchConfiguration('model_path').perform(context)
    if not model_path:
        model_path = os.path.join(detection_share, 'models', 'yolov8n_rk3588_fp16.rknn')

    use_orbbec = LaunchConfiguration('use_orbbec').perform(context) == 'true'
    enable_yolo = LaunchConfiguration('enable_yolo').perform(context) == 'true'

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

    nodes.append(
        Node(
            package='rgbd_pcl',
            executable='cloud_workspace_node',
            name='cloud_workspace_node',
            output='screen',
            parameters=[params_file],
        )
    )

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
        OpaqueFunction(function=launch_setup),
    ])
