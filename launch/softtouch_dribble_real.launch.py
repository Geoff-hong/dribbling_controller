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
    softtouch_ball_pose_topic = LaunchConfiguration('softtouch_ball_pose_topic')
    softtouch_ball_twist_topic = LaunchConfiguration('softtouch_ball_twist_topic')
    softtouch_ball_radius_m = LaunchConfiguration('softtouch_ball_radius_m')
    softtouch_obs_frame_source = LaunchConfiguration('softtouch_obs_frame_source')
    softtouch_obs_frame_pose_topic = LaunchConfiguration('softtouch_obs_frame_pose_topic')
    softtouch_base_state_source = LaunchConfiguration('softtouch_base_state_source')
    softtouch_route_cmd_mode = LaunchConfiguration('softtouch_route_cmd_mode')
    softtouch_route_vmax = LaunchConfiguration('softtouch_route_vmax')
    softtouch_obs_dump_path = LaunchConfiguration('softtouch_obs_dump_path')

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
                'g1_dribble_s3_net512_iter20000',
                'softtouch_dribble_deploy_m19999.onnx',
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
        DeclareLaunchArgument(
            'softtouch_ball_pose_topic',
            default_value='/softtouch/mocap/ball/pose',
            description='SoftTouch ball PoseStamped topic from mocap bridge.',
        ),
        DeclareLaunchArgument(
            'softtouch_ball_twist_topic',
            default_value='/softtouch/mocap/ball/twist',
            description='SoftTouch ball TwistStamped topic from mocap bridge.',
        ),
        DeclareLaunchArgument(
            'softtouch_ball_radius_m',
            default_value='0.10',
            description='SoftTouch ball radius in metres; feeds ball_radius obs as radius_m - 0.10.',
        ),
        DeclareLaunchArgument(
            'softtouch_obs_frame_source',
            default_value='topic',
            description='Use mocap policy-frame pose for ball/cmd body-frame observations on hardware.',
        ),
        DeclareLaunchArgument(
            'softtouch_obs_frame_pose_topic',
            default_value='/softtouch/mocap/chest/pose',
            description='Mocap PoseStamped topic for the policy observation frame.',
        ),
        DeclareLaunchArgument(
            'softtouch_base_state_source',
            default_value='model',
            description='Keep base_ang_vel/projected_gravity on robot estimator/IMU by default.',
        ),
        DeclareLaunchArgument(
            'softtouch_route_cmd_mode',
            default_value='',
            description='Optional SoftTouch route cmd_mode override. Use 0 for straight route.',
        ),
        DeclareLaunchArgument(
            'softtouch_route_vmax',
            default_value='',
            description='Optional SoftTouch route.route_vmax override in m/s.',
        ),
        DeclareLaunchArgument(
            'softtouch_obs_dump_path',
            default_value='',
            description='Optional per-policy-tick SoftTouch obs/action dump path.',
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
                'softtouch_ball_pose_topic': softtouch_ball_pose_topic,
                'softtouch_ball_twist_topic': softtouch_ball_twist_topic,
                'softtouch_ball_radius_m': softtouch_ball_radius_m,
                'softtouch_obs_frame_source': softtouch_obs_frame_source,
                'softtouch_obs_frame_pose_topic': softtouch_obs_frame_pose_topic,
                'softtouch_base_state_source': softtouch_base_state_source,
                'softtouch_route_cmd_mode': softtouch_route_cmd_mode,
                'softtouch_route_vmax': softtouch_route_vmax,
                'softtouch_obs_dump_path': softtouch_obs_dump_path,
            }.items(),
        ),
    ])
