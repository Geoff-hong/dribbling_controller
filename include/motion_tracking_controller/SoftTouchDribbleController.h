#pragma once

#include "motion_tracking_controller/SoftTouchDribbleCommand.h"

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <legged_rl_controllers/RlController.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <deque>
#include <fstream>
#include <limits>
#include <memory>

namespace legged {

class SoftTouchDribbleController : public RlController {
 public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

 protected:
  bool parserCommand(const std::string& name) override;
  bool parserObservation(const std::string& name) override;

 private:
  void loadSoftTouchConfig();
  void configureJointTargetClip();
  void ensureCommandTerm();
  void subscribeBallState();
  void subscribeBaseState();
  void subscribeObservationFrameState();
  void handleBallPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void handleBallTwist(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void handleBasePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void handleBaseTwist(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void handleObservationFramePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  scalar_t stampToSeconds(const builtin_interfaces::msg::Time& stamp) const;
  void configureJointMappings();
  vector_t policyVectorToControlOrder(const vector_t& value) const;
  vector_t readPolicyJointPosition() const;
  vector_t readPolicyJointVelocity() const;
  bool hasFreshPolicyInputs() const;
  void configureObservationLayout();
  void resetObservationHistory();
  vector_t buildPolicyObservation();
  controller_interface::return_type holdCurrentPositionTarget();
  controller_interface::return_type updatePositionTarget(const rclcpp::Time& time, const rclcpp::Duration& period,
                                                         vector_t& policyObs);
  controller_interface::return_type updateEffortPd(const rclcpp::Time& time, const rclcpp::Duration& period,
                                                   vector_t& policyObs);
  void publishVisualization(const rclcpp::Time& time);
  void publishPolicyDebug(const vector_t& policyObs);
  void publishPolicyDebugSchema();

  SoftTouchDribbleCommandCfg cfg_;
  SoftTouchDribbleOnnxPolicy::SharedPtr softtouchPolicy_;
  SoftTouchDribbleCommandTerm::SharedPtr commandTerm_;
  bool commandTermRegistered_ = false;
  bool resetRouteOnActivate_ = true;
  bool resetPolicyMemoryOnActivate_ = true;
  bool mujocoResetOnActivate_ = false;
  std::string mujocoResetTopic_ = "/softtouch/mujoco_reset";
  scalar_t mujocoResetHoldDuration_ = 0.0;
  scalar_t skipPolicyUntilTime_ = 0.0;
  bool pendingStartupPolicyReset_ = false;
  bool pendingFreshInputRouteReset_ = false;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mujocoResetPub_;
  std::string actionCommandMode_ = "rl_controller";
  scalar_t actionPolicyPeriod_ = 0.02;
  vector_t effortLimit_;
  vector_t effortTargetPolicyOrder_;
  scalar_t lastEffortPolicyUpdateTime_ = -std::numeric_limits<scalar_t>::infinity();
  bool hasEffortTarget_ = false;
  std::vector<size_t> policyToControlIndex_;
  std::vector<size_t> policyToModelJointIndex_;
  bool clipJointTarget_ = true;
  scalar_t jointTargetLimitFactor_ = 0.9;
  size_t actorHistoryLength_ = 1;
  size_t actorSingleFrameDim_ = 0;
  size_t actorObservationDim_ = 0;
  size_t decoderStateDim_ = 0;
  std::deque<vector_t> actorFrameHistory_;
  // sim2sim parity debug: per-policy-tick obs/action dump (empty = off)
  std::string obsDumpPath_;
  std::shared_ptr<std::ofstream> obsDump_;
  int obsDumpSeq_ = 0;

  std::string ballPoseTopic_ = "/softtouch/ball/pose";
  std::string ballTwistTopic_ = "/softtouch/ball/twist";
  scalar_t ballRadiusM_ = 0.10;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ballPoseSub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr ballTwistSub_;
  std::string basePoseTopic_ = "/softtouch/base/pose";
  std::string baseTwistTopic_ = "/softtouch/base/twist";
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr basePoseSub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr baseTwistSub_;
  std::string observationFramePoseTopic_ = "/softtouch/obs_frame/pose";
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr observationFramePoseSub_;

  bool visualize_ = true;
  std::string visualizationTopic_ = "/softtouch/dribble/markers";
  std::string visualizationFrameId_ = "world";
  scalar_t visualizationPeriod_ = 0.05;
  scalar_t targetArrowLength_ = 0.45;
  scalar_t ballMarkerRadius_ = 0.09;
  int routeMaxPoints_ = 160;
  rclcpp::Time lastVisualizationTime_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualizationPub_;

  bool debugPublishPolicyIo_ = true;
  std::string debugObservationTopic_ = "/softtouch/policy/observation";
  std::string debugObservationSchemaTopic_ = "/softtouch/policy/observation_schema";
  std::string debugRawActionTopic_ = "/softtouch/policy/raw_action";
  std::string debugLatentActionTopic_ = "/softtouch/policy/latent_action";
  std::string debugJointTargetTopic_ = "/softtouch/policy/joint_target";
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debugObservationPub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debugObservationSchemaPub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debugRawActionPub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debugLatentActionPub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debugJointTargetPub_;
};

}  // namespace legged
