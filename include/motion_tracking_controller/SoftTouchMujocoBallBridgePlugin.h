#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mujoco_sim_ros2/mujoco_physics_plugin.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace legged {

class SoftTouchMujocoBallBridgePlugin final : public mujoco_sim_ros2::MujocoPhysicsPlugin {
 public:
  void Configure(rclcpp::Node::SharedPtr& node, rclcpp::NodeOptions cm_node_option, mjModel* model, mjData* data) override;
  void Reset(mjModel* model, mjData* data) override;
  void Update(mjModel* model, mjData* data) override;

 private:
  struct SensorRef {
    std::string name;
    int id = -1;
    int adr = -1;
    int dim = 0;
  };

  struct ResetState {
    std::string rootJointName = "floating_base_joint";
    std::string ballJointName = "softtouch_ball_freejoint";
    std::vector<double> rootPosition;
    std::vector<double> rootQuaternion;
    std::vector<double> rootLinearVelocity;
    std::vector<double> rootAngularVelocityBody;
    std::vector<double> ballPosition;
    std::vector<double> ballQuaternion;
    std::vector<double> ballLinearVelocity;
    std::vector<double> ballAngularVelocity;
    std::vector<std::string> jointNames;
    std::vector<double> jointPosition;
    std::vector<double> jointVelocity;
  };

  void resolveSensors(mjModel* model);
  bool hasBallReset() const;
  void applyBallDamping(mjModel* model) const;
  ResetState loadResetState(const std::string& path) const;
  void applyResetState(mjModel* model, mjData* data, bool logReset = true);
  void publishBallState(const mjData* data);
  void resolveBaseJoint(mjModel* model);
  void publishBaseState(const mjData* data);
  void handleResetRequest(const std_msgs::msg::Float64::SharedPtr msg);
  void resolveRouteDots(mjModel* model);
  void handleRouteMarkers(const visualization_msgs::msg::MarkerArray::SharedPtr msg);
  void updateRouteDots(const mjModel* model, mjData* data);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr posePub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twistPub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr basePosePub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr baseTwistPub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr resetRequestSub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr routeMarkerSub_;

  // Route-line visualization in the MuJoCo viewer: mocap 'route_dot_*' bodies are
  // positioned along the cmd route parsed from /softtouch/dribble/markers.
  std::string routeMarkerTopic_ = "/softtouch/dribble/markers";
  std::mutex routeMutex_;
  std::vector<std::array<double, 3>> routePoints_;  // latest LINE_STRIP points (world)
  std::vector<int> routeDotMocapIds_;               // mocap id per route_dot body (-1 if absent)

  bool enabled_ = false;
  std::string frameId_ = "world";
  std::string poseTopic_ = "/softtouch/ball/pose";
  std::string twistTopic_ = "/softtouch/ball/twist";
  std::string ballJointName_ = "softtouch_ball_freejoint";
  std::string positionSensorName_ = "softtouch_ball_pos";
  std::string quaternionSensorName_ = "softtouch_ball_quat";
  std::string linearVelocitySensorName_ = "softtouch_ball_linvel";
  std::string angularVelocitySensorName_ = "softtouch_ball_angvel";
  bool publishBallStateEnabled_ = true;
  bool applyBallDampingEnabled_ = true;
  bool publishBaseState_ = false;
  std::string baseJointName_ = "floating_base_joint";
  std::string basePoseTopic_ = "/softtouch/base/pose";
  std::string baseTwistTopic_ = "/softtouch/base/twist";
  int baseQposAdr_ = -1;
  int baseDofAdr_ = -1;
  double publishPeriod_ = 0.01;
  double lastPublishTime_ = -1.0e30;
  double ballTranslationalDamping_ = 0.0;
  double ballAngularDamping_ = 0.006256;
  bool resetEnabled_ = false;
  bool resetApplyOnConfigure_ = true;
  bool resetApplyOnReset_ = true;
  bool resetZeroVelocity_ = false;
  double resetHoldUntilTime_ = 0.0;
  bool resetHoldLogged_ = false;
  std::string resetRequestTopic_ = "/softtouch/mujoco_reset";
  std::mutex resetRequestMutex_;
  bool resetRequested_ = false;
  double resetRequestHoldDuration_ = 0.0;
  bool resetStateLoaded_ = false;
  std::string resetStatePath_;
  ResetState resetState_;

  SensorRef positionSensor_;
  SensorRef quaternionSensor_;
  SensorRef linearVelocitySensor_;
  SensorRef angularVelocitySensor_;
};

}  // namespace legged
