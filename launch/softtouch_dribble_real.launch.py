from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    network_interface = LaunchConfiguration('network_interface')
    policy_path = LaunchConfiguration('policy_path')
    controllers_config = LaunchConfiguration('controllers_config')
    ext_pos_corr = LaunchConfiguration('ext_pos_corr')
    enable_teleop = LaunchConfiguration('enable_teleop')
    enable_rosbag = LaunchConfiguration('enable_rosbag')

    base_launch = PathJoinSubstitution([
        FindPackageShare('motion_tracking_controller'),
        'launch',
        'real.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('network_interface'),
        DeclareLaunchArgument(
            'policy_path',
            default_value=PathJoinSubstitution([
                EnvironmentVariable('HOME'),
                'SoftTouch',
                'checkpoints',
                'g1_dribble_s3_human_iter35000',
                'softtouch_dribble_deploy.onnx',
            ]),
            description='SoftTouch dribble deployment ONNX path',
        ),
        DeclareLaunchArgument(
            'controllers_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('motion_tracking_controller'),
                'config',
                'g1',
                'softtouch_dribble_controllers.yaml',
            ]),
            description='SoftTouch dribble controller YAML',
        ),
        DeclareLaunchArgument(
            'ext_pos_corr',
            default_value='false',
            description='Forwarded to real.launch.py',
        ),
        DeclareLaunchArgument(
            'enable_teleop',
            default_value='true',
            description='Launch unitree_bringup teleop nodes',
        ),
        DeclareLaunchArgument(
            'enable_rosbag',
            default_value='false',
            description='Record all non-Unitree topics with rosbag2/mcap. Off by default for SoftTouch mocap wiring tests.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'network_interface': network_interface,
                'controllers_config': controllers_config,
                'policy_path': policy_path,
                'ext_pos_corr': ext_pos_corr,
                'enable_teleop': enable_teleop,
                'enable_rosbag': enable_rosbag,
            }.items(),
        ),
    ])
