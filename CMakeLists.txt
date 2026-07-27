cmake_minimum_required(VERSION 3.8)
project(motion_tracking_controller)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


find_package(Eigen3 REQUIRED)
find_package(mujoco_ament REQUIRED)

ament_auto_add_library(
    ${PROJECT_NAME}
    SHARED
    DIRECTORY src
)

# prevent pluginlib from using boost
target_compile_definitions(${PROJECT_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
pluginlib_export_plugin_description_file(controller_interface motion_tracking_controller.xml)
pluginlib_export_plugin_description_file(mujoco_sim_ros2 mujoco_physics_plugins.xml)

target_include_directories(
    ${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
ament_target_dependencies(${PROJECT_NAME} mujoco_ament)

install(
    PROGRAMS
    tools/ros_utils/bridge_softtouch_mocap_ball.py
    tools/onnx_export/export_softtouch_dribble_policy.py
    tools/reset_export/export_softtouch_mujoco_reset_state.py
    tools/ros1_mocap_udp_sender.py
    tools/ros1_unlabeled_ball_udp_sender.py
    tools/ros2_mocap_pose_viz.py
    tools/ros2_mocap_udp_receiver.py
    tools/softtouch_ball_relative_window.py
    tools/softtouch_real_run.sh
    DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(
    INSTALL_TO_SHARE config launch mjcf rviz tools
)
