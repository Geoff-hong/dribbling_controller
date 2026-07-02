import os
import copy
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    IncludeLaunchDescription
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from legged_bringup.launch_utils import (
    get_controller_names, generate_temp_config, resolve_policy_paths, download_wandb_onnx
)


def mujoco_control_spawner(names, inactive=False):
    args = list(names)
    args += [
        '--controller-manager',
        '/mujoco_sim_ros2_node',
    ]
    if inactive:
        args.append('--inactive')
    return Node(
        package='controller_manager',
        executable='spawner',
        arguments=args,
        output='screen'
    )


def adapt_controller_yaml_for_mujoco(temp_config_path):
    with open(temp_config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    controller_manager_params = cfg.get('controller_manager', {}).get('ros__parameters', {})
    mujoco_node = cfg.setdefault('mujoco_sim_ros2_node', {})
    mujoco_params = mujoco_node.setdefault('ros__parameters', {})

    for key, value in controller_manager_params.items():
        mujoco_params[key] = copy.deepcopy(value)

    for name, value in controller_manager_params.items():
        if isinstance(value, dict) and 'type' in value:
            controller_params = mujoco_params.setdefault(name, {})
            controller_params.setdefault('params_file', [temp_config_path])

    with open(temp_config_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)

    return temp_config_path


def setup_controllers(context):
    robot_type_value = LaunchConfiguration('robot_type').perform(context)
    policy_path_value = LaunchConfiguration('policy_path').perform(context)
    wandb_path_value = LaunchConfiguration('wandb_path').perform(context)
    start_step_value = LaunchConfiguration('start_step').perform(context)
    motion_length_value = LaunchConfiguration('motion_length').perform(context)
    motion_loop_value = LaunchConfiguration('motion_loop').perform(context)
    motion_time_step_stride_value = LaunchConfiguration('motion_time_step_stride').perform(context)
    motion_mujoco_reset_on_activate = LaunchConfiguration('motion_mujoco_reset_on_activate').perform(context)
    motion_mujoco_reset_hold_s = LaunchConfiguration('motion_mujoco_reset_hold_s').perform(context)
    motion_mujoco_reset_topic = LaunchConfiguration('motion_mujoco_reset_topic').perform(context)
    motion_action_command_mode = LaunchConfiguration('motion_action_command_mode').perform(context)
    motion_action_policy_update_period = LaunchConfiguration('motion_action_policy_update_period').perform(context)
    motion_action_effort_scale = LaunchConfiguration('motion_action_effort_scale').perform(context)
    policy_action_type = LaunchConfiguration('policy_action_type').perform(context)
    walking_controller_update_rate = LaunchConfiguration('walking_controller_update_rate').perform(context)
    controller_type_value = LaunchConfiguration('controller_type').perform(context)
    controllers_config_value = LaunchConfiguration('controllers_config').perform(context)
    mujoco_reset_state_file = LaunchConfiguration('mujoco_reset_state_file').perform(context)
    mujoco_reset_hold_until_time = LaunchConfiguration('mujoco_reset_hold_until_time').perform(context)
    mujoco_reset_zero_velocity = LaunchConfiguration('mujoco_reset_zero_velocity').perform(context)
    softtouch_mujoco_bridge_enabled = LaunchConfiguration('softtouch_mujoco_bridge_enabled').perform(context)
    softtouch_mujoco_publish_ball_state = LaunchConfiguration('softtouch_mujoco_publish_ball_state').perform(context)
    softtouch_mujoco_apply_ball_damping = LaunchConfiguration('softtouch_mujoco_apply_ball_damping').perform(context)
    ext_pos_corr = LaunchConfiguration('ext_pos_corr').perform(context)
    softtouch_base_state_source = LaunchConfiguration('softtouch_base_state_source').perform(context)
    softtouch_route_cmd_mode = LaunchConfiguration('softtouch_route_cmd_mode').perform(context)
    softtouch_seed = LaunchConfiguration('softtouch_seed').perform(context)
    softtouch_route_length_m = LaunchConfiguration('softtouch_route_length_m').perform(context)
    softtouch_ball_angular_damping = LaunchConfiguration('softtouch_ball_angular_damping').perform(context)
    softtouch_action_command_mode = LaunchConfiguration('softtouch_action_command_mode').perform(context)
    softtouch_mujoco_reset_hold_s = LaunchConfiguration('softtouch_mujoco_reset_hold_s').perform(context)
    activate_walking_controller = LaunchConfiguration('activate_walking_controller').perform(context).lower() in [
        'true',
        '1',
        'yes',
    ]
    spawn_inactive_controllers = LaunchConfiguration('spawn_inactive_controllers').perform(context).lower() in [
        'true',
        '1',
        'yes',
    ]

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
    if motion_mujoco_reset_on_activate:
        kv_pairs.append(('walking_controller.motion.reset.mujoco_reset_on_activate', motion_mujoco_reset_on_activate))
    if motion_mujoco_reset_hold_s:
        kv_pairs.append(('walking_controller.motion.reset.mujoco_reset_hold_s', motion_mujoco_reset_hold_s))
    if motion_mujoco_reset_topic:
        kv_pairs.append(('walking_controller.motion.reset.mujoco_reset_topic', motion_mujoco_reset_topic))
    if motion_action_command_mode:
        kv_pairs.append(('walking_controller.motion.action.command_mode', motion_action_command_mode))
    if motion_action_policy_update_period:
        kv_pairs.append(('walking_controller.motion.action.policy_update_period_s', motion_action_policy_update_period))
    if motion_action_effort_scale:
        kv_pairs.append(('walking_controller.motion.action.effort_scale', motion_action_effort_scale))
    if policy_action_type:
        kv_pairs.append(('walking_controller.policy.action_type', policy_action_type))
    if walking_controller_update_rate:
        kv_pairs.append(('walking_controller.update_rate', walking_controller_update_rate))
    if mujoco_reset_state_file:
        reset_path = os.path.abspath(os.path.expanduser(os.path.expandvars(mujoco_reset_state_file)))
        kv_pairs.append(('mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.enabled', "true"))
        kv_pairs.append(('mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.reset.enabled', "true"))
        kv_pairs.append((
            'mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.reset.state_file',
            reset_path,
        ))
    if mujoco_reset_hold_until_time:
        kv_pairs.append((
            'mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.reset.hold_until_time_s',
            mujoco_reset_hold_until_time,
        ))
    if mujoco_reset_zero_velocity:
        kv_pairs.append((
            'mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.reset.zero_velocity',
            mujoco_reset_zero_velocity,
        ))
    if softtouch_mujoco_bridge_enabled:
        kv_pairs.append(('mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.enabled',
                         softtouch_mujoco_bridge_enabled))
    if softtouch_mujoco_publish_ball_state:
        kv_pairs.append(('mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.publish_ball_state',
                         softtouch_mujoco_publish_ball_state))
    if softtouch_mujoco_apply_ball_damping:
        kv_pairs.append(('mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.apply_ball_damping',
                         softtouch_mujoco_apply_ball_damping))
    if ext_pos_corr.lower() in ["true", "1", "yes"]:
        kv_pairs.append(('state_estimator.estimation.contact.height_sensor_noise', 1e10))
        kv_pairs.append(('state_estimator.estimation.position.topic', "/mid360"))
    if softtouch_base_state_source:
        kv_pairs.append(('walking_controller.softtouch.base_state.source', softtouch_base_state_source))
    if softtouch_route_cmd_mode:
        kv_pairs.append(('walking_controller.softtouch.route.cmd_mode', softtouch_route_cmd_mode))
    if softtouch_seed:
        kv_pairs.append(('walking_controller.softtouch.seed', softtouch_seed))
    if softtouch_route_length_m:
        kv_pairs.append(('walking_controller.softtouch.route.route_length_m', softtouch_route_length_m))
    if softtouch_ball_angular_damping:
        kv_pairs.append(('mujoco_sim_ros2_node.ros__parameters.softtouch_mujoco_ball_bridge.ball_angular_damping',
                         softtouch_ball_angular_damping))
    if softtouch_action_command_mode:
        kv_pairs.append(('walking_controller.softtouch.action.command_mode', softtouch_action_command_mode))
    if softtouch_mujoco_reset_hold_s:
        kv_pairs.append(('walking_controller.softtouch.reset.mujoco_reset_hold_s', softtouch_mujoco_reset_hold_s))

    temp_controllers_config_path = generate_temp_config(
        controllers_config_path,
        'motion_tracking_controller',
        kv_pairs
    )
    temp_controllers_config_path = adapt_controller_yaml_for_mujoco(temp_controllers_config_path)

    set_controllers_yaml = SetLaunchConfiguration(
        name='controllers_yaml',
        value=temp_controllers_config_path
    )

    all_controllers = get_controller_names(controllers_config_path, 'motion_tracking_controller')
    active_list = ["state_estimator"]
    if activate_walking_controller:
        active_list.append("walking_controller")
    inactive_list = [c for c in all_controllers if c not in active_list]

    active_spawner = mujoco_control_spawner(active_list)
    actions = [set_controllers_yaml, active_spawner]
    if not activate_walking_controller and "walking_controller" in inactive_list:
        actions.append(mujoco_control_spawner(["walking_controller"], inactive=True))
        inactive_list = [c for c in inactive_list if c != "walking_controller"]
    if spawn_inactive_controllers and inactive_list:
        actions.append(mujoco_control_spawner(inactive_list, inactive=True))

    return actions


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    mujoco_model_package = LaunchConfiguration('mujoco_model_package')
    mujoco_model_file = LaunchConfiguration('mujoco_model_file')
    enable_teleop = LaunchConfiguration('enable_teleop')
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
        " ", "simulation:=", "mujoco"])
    robot_description = {"robot_description": robot_description_command}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {
            'publish_frequency': 500.0,
            'use_sim_time': True
        }],
    )

    mujoco_simulator = Node(
        package='mujoco_sim_ros2',
        executable='mujoco_sim',
        name='mujoco_sim_ros2_node',
        parameters=[
            {"model_package": mujoco_model_package,
             "model_file": mujoco_model_file,
             "physics_plugins": [
                 "motion_tracking_controller/SoftTouchMujocoBallBridgePlugin",
                 "mujoco_ros2_control::MujocoRos2ControlPlugin"
             ],
             "use_sim_time": True
             },
            robot_description,
            LaunchConfiguration('controllers_yaml'),
        ],
        output='screen')

    controllers_opaque_func = OpaqueFunction(function=setup_controllers)

    teleop = PathJoinSubstitution([
        FindPackageShare('unitree_bringup'),
        'launch',
        'teleop.launch.py'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument(
            'controllers_config',
            default_value='',
            description='Optional controller YAML path. Default: config/<robot_type>/controllers.yaml'
        ),
        DeclareLaunchArgument(
            'mujoco_model_package',
            default_value='unitree_description',
            description='Package that contains the MuJoCo MJCF model file'
        ),
        DeclareLaunchArgument(
            'mujoco_model_file',
            default_value=PythonExpression(["'/mjcf/", robot_type, ".xml'"]),
            description='MuJoCo model file path inside mujoco_model_package'
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
            'motion_mujoco_reset_on_activate',
            default_value='',
            description='Optional true/false for walking_controller.motion.reset.mujoco_reset_on_activate'
        ),
        DeclareLaunchArgument(
            'motion_mujoco_reset_hold_s',
            default_value='',
            description='Optional hold duration for walking_controller motion reset requests'
        ),
        DeclareLaunchArgument(
            'motion_mujoco_reset_topic',
            default_value='',
            description='Optional topic for walking_controller motion reset requests'
        ),
        DeclareLaunchArgument(
            'motion_action_command_mode',
            default_value='',
            description='Optional MotionTrackingController action mode: rl_controller, position_target, or effort_pd'
        ),
        DeclareLaunchArgument(
            'motion_action_policy_update_period',
            default_value='',
            description='Optional MotionTrackingController policy update period in seconds'
        ),
        DeclareLaunchArgument(
            'motion_action_effort_scale',
            default_value='',
            description='Optional MotionTrackingController effort-limit scale for effort_pd mode'
        ),
        DeclareLaunchArgument(
            'policy_action_type',
            default_value='',
            description='Optional walking_controller.policy.action_type override'
        ),
        DeclareLaunchArgument(
            'walking_controller_update_rate',
            default_value='',
            description='Optional walking_controller.update_rate override in Hz'
        ),
        DeclareLaunchArgument(
            'mujoco_reset_state_file',
            default_value='',
            description='Optional SoftTouch MuJoCo reset-state txt override for the ball bridge plugin'
        ),
        DeclareLaunchArgument(
            'mujoco_reset_hold_until_time',
            default_value='',
            description='Hold the SoftTouch MuJoCo reset state until this sim time in seconds'
        ),
        DeclareLaunchArgument(
            'mujoco_reset_zero_velocity',
            default_value='',
            description='Override SoftTouch MuJoCo reset.zero_velocity'
        ),
        DeclareLaunchArgument(
            'softtouch_mujoco_bridge_enabled',
            default_value='',
            description='Override SoftTouch MuJoCo bridge enabled flag'
        ),
        DeclareLaunchArgument(
            'softtouch_mujoco_publish_ball_state',
            default_value='',
            description='Override SoftTouch MuJoCo bridge publish_ball_state flag'
        ),
        DeclareLaunchArgument(
            'softtouch_mujoco_apply_ball_damping',
            default_value='',
            description='Override SoftTouch MuJoCo bridge apply_ball_damping flag'
        ),
        DeclareLaunchArgument(
            'ext_pos_corr',
            default_value='false',
            description='Enable external position correction'
        ),
        DeclareLaunchArgument(
            'softtouch_base_state_source',
            default_value='',
            description='Optional SoftTouch base_state.source override: model or topic'
        ),
        DeclareLaunchArgument(
            'softtouch_route_cmd_mode',
            default_value='',
            description='Optional SoftTouch route cmd_mode override, e.g. 0 for a straight-line route'
        ),
        DeclareLaunchArgument(
            'softtouch_seed',
            default_value='',
            description='Optional SoftTouch route RNG seed override (softtouch.seed). '
                        'Different seeds give different but reproducible routes.'
        ),
        DeclareLaunchArgument(
            'softtouch_route_length_m',
            default_value='',
            description='Optional SoftTouch route length (m) override. The robot dribbles to '
                        'the route end and then stops, so a shorter route = a shorter episode '
                        '(e.g. ~18 m ~= 10 s at vmax 2 m/s). Used by the DR sweep.'
        ),
        DeclareLaunchArgument(
            'softtouch_ball_angular_damping',
            default_value='',
            description='Optional SoftTouch MuJoCo ball bridge ball_angular_damping override '
                        '(= 4*I). Used by the DR sweep to keep spin-decay consistent with the '
                        'randomized ball mass/radius.'
        ),
        DeclareLaunchArgument(
            'softtouch_action_command_mode',
            default_value='',
            description='Optional SoftTouch action command mode override: rl_controller, position_target, or effort_pd'
        ),
        DeclareLaunchArgument(
            'softtouch_mujoco_reset_hold_s',
            default_value='',
            description='Optional SoftTouch controller reset hold duration after publishing /softtouch/mujoco_reset'
        ),
        DeclareLaunchArgument(
            'spawn_inactive_controllers',
            default_value='true',
            description='Load controllers from the YAML that are not in the active controller list'
        ),
        DeclareLaunchArgument(
            'activate_walking_controller',
            default_value='true',
            description='Activate walking_controller during launch; set false for staged controller activation'
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
        controllers_opaque_func,
        mujoco_simulator,
        node_robot_state_publisher,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(teleop),
            launch_arguments={'robot_type': robot_type}.items(),
            condition=IfCondition(enable_teleop),
        )
    ])
