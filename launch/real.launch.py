import os

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from legged_bringup.launch_utils import (
    get_controller_names, generate_temp_config, resolve_policy_paths, download_wandb_onnx, control_spawner
)


def setup_controllers(context):
    robot_type_value = LaunchConfiguration('robot_type').perform(context)
    policy_path_value = LaunchConfiguration('policy_path').perform(context)
    wandb_path_value = LaunchConfiguration('wandb_path').perform(context)
    start_step_value = LaunchConfiguration('start_step').perform(context)
    motion_length_value = LaunchConfiguration('motion_length').perform(context)
    motion_loop_value = LaunchConfiguration('motion_loop').perform(context)
    motion_time_step_stride_value = LaunchConfiguration('motion_time_step_stride').perform(context)
    policy_action_type = LaunchConfiguration('policy_action_type').perform(context)
    controller_type_value = LaunchConfiguration('controller_type').perform(context)
    controllers_config_value = LaunchConfiguration('controllers_config').perform(context)
    ext_pos_corr = LaunchConfiguration('ext_pos_corr').perform(context)
    softtouch_ball_pose_topic = LaunchConfiguration('softtouch_ball_pose_topic').perform(context)
    softtouch_ball_twist_topic = LaunchConfiguration('softtouch_ball_twist_topic').perform(context)
    softtouch_ball_radius_m = LaunchConfiguration('softtouch_ball_radius_m').perform(context)
    softtouch_obs_frame_source = LaunchConfiguration('softtouch_obs_frame_source').perform(context)
    softtouch_obs_frame_pose_topic = LaunchConfiguration('softtouch_obs_frame_pose_topic').perform(context)
    softtouch_base_state_source = LaunchConfiguration('softtouch_base_state_source').perform(context)
    softtouch_route_cmd_mode = LaunchConfiguration('softtouch_route_cmd_mode').perform(context)
    softtouch_route_vmax = LaunchConfiguration('softtouch_route_vmax').perform(context)
    softtouch_obs_dump_path = LaunchConfiguration('softtouch_obs_dump_path').perform(context)

    if not policy_path_value and wandb_path_value:
        policy_path_value = download_wandb_onnx(wandb_path_value)

    controllers_config_path = controllers_config_value or f'config/{robot_type_value}/controllers.yaml'

    kv_pairs = resolve_policy_paths(controllers_config_path, 'motion_tracking_controller')
    if controller_type_value:
        kv_pairs.append(('controller_manager.ros__parameters.walking_controller.type', controller_type_value))
    if policy_path_value:
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(policy_path_value)))
        kv_pairs.append(('walking_controller.policy.path', abs_path))
    if start_step_value:
        kv_pairs.append(('walking_controller.motion.start_step', start_step_value))
    if motion_length_value:
        kv_pairs.append(('walking_controller.motion.length', motion_length_value))
    if motion_loop_value:
        kv_pairs.append(('walking_controller.motion.loop', motion_loop_value))
    if motion_time_step_stride_value:
        kv_pairs.append(('walking_controller.motion.time_step_stride', motion_time_step_stride_value))
    if policy_action_type:
        kv_pairs.append(('walking_controller.policy.action_type', policy_action_type))
    if ext_pos_corr.lower() in ["true", "1", "yes"]:
        kv_pairs.append(('state_estimator.estimation.contact.height_sensor_noise', 1e10))
        kv_pairs.append(('state_estimator.estimation.position.topic', "/glim/odom"))
    if softtouch_ball_pose_topic:
        kv_pairs.append(('walking_controller.softtouch.ball_state.pose_topic', softtouch_ball_pose_topic))
    if softtouch_ball_twist_topic:
        kv_pairs.append(('walking_controller.softtouch.ball_state.twist_topic', softtouch_ball_twist_topic))
    if softtouch_ball_radius_m:
        kv_pairs.append(('walking_controller.softtouch.ball_state.radius_m', softtouch_ball_radius_m))
    if softtouch_obs_frame_source:
        kv_pairs.append(('walking_controller.softtouch.obs_frame.source', softtouch_obs_frame_source))
    if softtouch_obs_frame_pose_topic:
        kv_pairs.append(('walking_controller.softtouch.obs_frame.pose_topic', softtouch_obs_frame_pose_topic))
    if softtouch_base_state_source:
        kv_pairs.append(('walking_controller.softtouch.base_state.source', softtouch_base_state_source))
    if softtouch_route_cmd_mode:
        kv_pairs.append(('walking_controller.softtouch.route.cmd_mode', softtouch_route_cmd_mode))
    if softtouch_route_vmax:
        kv_pairs.append(('walking_controller.softtouch.route.route_vmax', softtouch_route_vmax))
    if softtouch_obs_dump_path:
        kv_pairs.append(('walking_controller.softtouch.debug.obs_dump_path', softtouch_obs_dump_path))
    # Real robot startup should enter standby only; MuJoCo reset requests are
    # sim-only and are intentionally disabled here even if the shared YAML
    # enables them for sim2sim.
    kv_pairs.append(('standby_controller.reset.mujoco_reset_on_activate', 'false'))

    temp_controllers_config_path = generate_temp_config(
        controllers_config_path,
        'motion_tracking_controller',
        kv_pairs
    )

    set_controllers_yaml = SetLaunchConfiguration(
        name='controllers_yaml',
        value=temp_controllers_config_path
    )

    all_controllers = get_controller_names(controllers_config_path, 'motion_tracking_controller')
    active_list = ["state_estimator", "standby_controller"]
    inactive_list = [c for c in all_controllers if c not in active_list]

    param_file = LaunchConfiguration('controllers_yaml')
    active_spawner = control_spawner(active_list, param_file=param_file)
    inactive_spawner = control_spawner(inactive_list, inactive=True, param_file=param_file)

    return [set_controllers_yaml, active_spawner, inactive_spawner]


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')
    enable_teleop = LaunchConfiguration('enable_teleop')
    enable_rosbag = LaunchConfiguration('enable_rosbag')
    urdf_name = PythonExpression(["'g1' if '", robot_type, "' == 'g1' else 'sdk1'"])

    robot_description_command = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("unitree_description"),
            "urdf",
            urdf_name,
            "robot.xacro"
        ]),
        " ", "robot_type:=", robot_type,
        " ", "simulation:=", "false",
        " ", "network_interface:=", network_interface
    ])

    robot_description = {"robot_description": robot_description_command}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {
            'publish_frequency': 500.0,
        }],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, LaunchConfiguration('controllers_yaml')],
        output="both",
        respawn=True,
    )

    controllers_opaque_func = OpaqueFunction(function=setup_controllers)

    # Exclude all Unitree topics... it should start from the same namespace, fuck Unitree!
    exclude_regex = (
        r'(/EstimatorData|/SymState(_back)?|/api/.*'
        r'|/arm/action/state|/arm_sdk'
        r'|/audio_msg|/audiosender|/config_change_status'
        r'|/dex3/(left|right)/(cmd|state)'
        r'|/frontvideostream|/gnss'
        r'|/gpt_(cmd|state)|/gptflowfeedback'
        r'|/lf/(bmsstate|dex3/(left|right)/state|lowstate|mainboardstate|'
        r'odommodestate|secondary_imu|sportmodestate)'
        r'|/low(cmd|state)|/multiplestate|/odommodestate'
        r'|/parameter_events|/public_network_status|/rosout'
        r'|/rtc/(state|status)|/secondary_imu|/selftest'
        r'|/servicestate(activate)?|/slam_info|/sportmodestate'
        r'|/utlidar/range_info|/videohub/inner'
        r'|/webrtc(req|res)|/wirelesscontroller)'
        r'|/controller_manager/introspection_data/full'
        r'|/controller_manager/statistics/full'
    )

    rosbag2 = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-s', 'mcap', '-a',  # record all topics
            '--exclude-regex', exclude_regex,  # skip those that match the regex
        ],
        output='screen',
        condition=IfCondition(enable_rosbag),
    )

    teleop = PathJoinSubstitution([
        FindPackageShare('unitree_bringup'),
        'launch',
        'teleop.launch.py'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument('network_interface'),
        DeclareLaunchArgument(
            'controllers_config',
            default_value='',
            description='Optional controller YAML path. Default: config/<robot_type>/controllers.yaml'
        ),
        DeclareLaunchArgument(
            'policy_path',
            default_value='',
            description='Absolute or ~-expanded path for walking_controller.policy.path'
        ),
        DeclareLaunchArgument(
            'controller_type',
            default_value='',
            description='Optional override for controller_manager.walking_controller.type'
        ),
        DeclareLaunchArgument(
            'start_step',
            default_value='',
            description='Optional integer start step for walking_controller.motion.start_step'
        ),
        DeclareLaunchArgument(
            'motion_length',
            default_value='',
            description='Optional reference motion length for walking_controller.motion.length'
        ),
        DeclareLaunchArgument(
            'motion_loop',
            default_value='',
            description='Optional true/false for walking_controller.motion.loop'
        ),
        DeclareLaunchArgument(
            'motion_time_step_stride',
            default_value='',
            description='Optional integer stride for walking_controller.motion.time_step_stride'
        ),
        DeclareLaunchArgument(
            'policy_action_type',
            default_value='',
            description='Optional walking_controller.policy.action_type override'
        ),
        DeclareLaunchArgument(
            'ext_pos_corr',
            default_value='false',
            description='Enable external position correction'
        ),
        DeclareLaunchArgument(
            'softtouch_ball_pose_topic',
            default_value='',
            description='Optional SoftTouch ball PoseStamped topic override'
        ),
        DeclareLaunchArgument(
            'softtouch_ball_twist_topic',
            default_value='',
            description='Optional SoftTouch ball TwistStamped topic override'
        ),
        DeclareLaunchArgument(
            'softtouch_ball_radius_m',
            default_value='',
            description='Optional SoftTouch ball radius in metres; feeds ball_radius obs as radius_m - 0.10'
        ),
        DeclareLaunchArgument(
            'softtouch_obs_frame_source',
            default_value='',
            description='Optional SoftTouch obs_frame.source override: model or topic'
        ),
        DeclareLaunchArgument(
            'softtouch_obs_frame_pose_topic',
            default_value='',
            description='Optional SoftTouch obs_frame PoseStamped topic override'
        ),
        DeclareLaunchArgument(
            'softtouch_base_state_source',
            default_value='',
            description='Optional SoftTouch base_state.source override: model or topic'
        ),
        DeclareLaunchArgument(
            'softtouch_route_cmd_mode',
            default_value='',
            description='Optional SoftTouch route cmd_mode override. Use 0 for straight route.'
        ),
        DeclareLaunchArgument(
            'softtouch_route_vmax',
            default_value='',
            description='Optional SoftTouch route.route_vmax override in m/s'
        ),
        DeclareLaunchArgument(
            'softtouch_obs_dump_path',
            default_value='',
            description='Optional per-policy-tick SoftTouch obs/action dump path'
        ),
        DeclareLaunchArgument(
            'wandb_path',
            default_value='',
            description='W&B run path to download ONNX from (used when policy_path is empty)'
        ),
        DeclareLaunchArgument(
            'enable_teleop',
            default_value='true',
            description='Launch unitree_bringup teleop nodes'
        ),
        DeclareLaunchArgument(
            'enable_rosbag',
            default_value='true',
            description='Record all non-Unitree topics with rosbag2/mcap'
        ),
        controllers_opaque_func,
        control_node,
        node_robot_state_publisher,
        rosbag2,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(teleop),
            launch_arguments={'robot_type': robot_type}.items(),
            condition=IfCondition(enable_teleop),
        )
    ])
