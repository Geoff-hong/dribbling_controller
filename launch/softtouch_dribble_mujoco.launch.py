from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    policy_path = LaunchConfiguration('policy_path')
    controllers_config = LaunchConfiguration('controllers_config')
    mujoco_model_package = LaunchConfiguration('mujoco_model_package')
    mujoco_model_file = LaunchConfiguration('mujoco_model_file')
    mujoco_reset_state_file = LaunchConfiguration('mujoco_reset_state_file')
    mujoco_reset_hold_until_time = LaunchConfiguration('mujoco_reset_hold_until_time')
    mujoco_reset_zero_velocity = LaunchConfiguration('mujoco_reset_zero_velocity')
    enable_teleop = LaunchConfiguration('enable_teleop')
    spawn_inactive_controllers = LaunchConfiguration('spawn_inactive_controllers')
    ext_pos_corr = LaunchConfiguration('ext_pos_corr')
    softtouch_base_state_source = LaunchConfiguration('softtouch_base_state_source')
    softtouch_route_cmd_mode = LaunchConfiguration('softtouch_route_cmd_mode')
    softtouch_action_command_mode = LaunchConfiguration('softtouch_action_command_mode')
    softtouch_mujoco_reset_hold_s = LaunchConfiguration('softtouch_mujoco_reset_hold_s')
    launch_rviz = LaunchConfiguration('launch_rviz')
    rviz_config = LaunchConfiguration('rviz_config')

    base_launch = PathJoinSubstitution([
        FindPackageShare('motion_tracking_controller'),
        'launch',
        'mujoco.launch.py',
    ])

    return LaunchDescription([
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
            'mujoco_model_package',
            default_value='motion_tracking_controller',
            description='Package that contains the patched SoftTouch dribble MJCF',
        ),
        DeclareLaunchArgument(
            'mujoco_model_file',
            default_value='/mjcf/g1_softtouch_dribble.xml',
            description='Patched SoftTouch dribble MJCF path inside mujoco_model_package',
        ),
        DeclareLaunchArgument(
            'mujoco_reset_state_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('motion_tracking_controller'),
                'config',
                'g1',
                'softtouch_mujoco_reset_walkf_rf_frame0.txt',
            ]),
            description='SoftTouch MuJoCo reset-state txt path.',
        ),
        DeclareLaunchArgument(
            'mujoco_reset_hold_until_time',
            default_value='',
            description='Hold the SoftTouch MuJoCo reset state until this sim time in seconds.',
        ),
        DeclareLaunchArgument(
            'mujoco_reset_zero_velocity',
            default_value='',
            description='Optional SoftTouch MuJoCo reset.zero_velocity override.',
        ),
        DeclareLaunchArgument(
            'enable_teleop',
            default_value='false',
            description='Launch unitree_bringup teleop nodes. Off by default for headless SoftTouch sim2sim.',
        ),
        DeclareLaunchArgument(
            'spawn_inactive_controllers',
            default_value='false',
            description='Load inactive controllers from the YAML. Off by default to avoid unused StandbyController plugin requirements.',
        ),
        DeclareLaunchArgument(
            'ext_pos_corr',
            default_value='true',
            description='Use MuJoCo mid360 pose as external localization correction for StateEstimator in sim2sim.',
        ),
        DeclareLaunchArgument(
            'softtouch_base_state_source',
            default_value='',
            description='Optional SoftTouch base_state.source override: model or topic.',
        ),
        DeclareLaunchArgument(
            'softtouch_route_cmd_mode',
            default_value='',
            description='Optional SoftTouch route cmd_mode override, e.g. 0 for a straight-line route.',
        ),
        DeclareLaunchArgument(
            'softtouch_action_command_mode',
            default_value='',
            description='Optional SoftTouch action command mode override: rl_controller, position_target, or effort_pd.',
        ),
        DeclareLaunchArgument(
            'softtouch_mujoco_reset_hold_s',
            default_value='',
            description='Optional SoftTouch controller reset hold duration after publishing the MuJoCo reset request.',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='false',
            description='Launch RViz with the SoftTouch dribble route MarkerArray display.',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('motion_tracking_controller'),
                'rviz',
                'softtouch_dribble.rviz',
            ]),
            description='RViz config for SoftTouch dribble route visualization.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'controllers_config': controllers_config,
                'mujoco_model_package': mujoco_model_package,
                'mujoco_model_file': mujoco_model_file,
                'mujoco_reset_state_file': mujoco_reset_state_file,
                'mujoco_reset_hold_until_time': mujoco_reset_hold_until_time,
                'mujoco_reset_zero_velocity': mujoco_reset_zero_velocity,
                'policy_path': policy_path,
                'enable_teleop': enable_teleop,
                'spawn_inactive_controllers': spawn_inactive_controllers,
                'ext_pos_corr': ext_pos_corr,
                'softtouch_base_state_source': softtouch_base_state_source,
                'softtouch_route_cmd_mode': softtouch_route_cmd_mode,
                'softtouch_action_command_mode': softtouch_action_command_mode,
                'softtouch_mujoco_reset_hold_s': softtouch_mujoco_reset_hold_s,
            }.items(),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='softtouch_dribble_rviz',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(launch_rviz),
        ),
    ])
