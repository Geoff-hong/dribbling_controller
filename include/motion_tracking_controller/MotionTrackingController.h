#pragma once

#include <legged_rl_controllers/RlController.h>
#include <std_msgs/msg/float64.hpp>

#include <limits>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/common.h"

namespace legged {
class MotionTrackingController : public RlController {
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
  void configurePolicyJointMapping();
  vector_t readPolicyJointPosition() const;
  vector_t readPolicyJointVelocity() const;
  void configureActionMode();
  vector_t policyVectorToControlOrder(const vector_t& value) const;
  vector_t makeJointTarget(const vector_t& rawAction) const;
  controller_interface::return_type updatePositionTarget(const rclcpp::Time& time, const rclcpp::Duration& period,
                                                         vector_t& policyObs);
  controller_interface::return_type updateEffortPd(const rclcpp::Time& time, const rclcpp::Duration& period,
                                                   vector_t& policyObs);

  MotionCommandCfg cfg_;
  MotionCommandTerm::SharedPtr commandTerm_;
  bool mujocoResetOnActivate_ = false;
  std::string mujocoResetTopic_ = "/softtouch/mujoco_reset";
  scalar_t mujocoResetHoldDuration_ = 0.0;
  scalar_t skipPolicyUntilTime_ = 0.0;
  bool pendingStartupPolicyReset_ = false;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mujocoResetPub_;

  std::vector<size_t> policyToModelJointIndex_;
  std::vector<size_t> policyToControlIndex_;
  std::string actionCommandMode_ = "rl_controller";
  scalar_t actionPolicyPeriod_ = 0.02;
  scalar_t effortLimitScale_ = 1.0;
  vector_t effortLimit_;
  vector_t actionTargetPolicyOrder_;
  scalar_t lastActionPolicyUpdateTime_ = -std::numeric_limits<scalar_t>::infinity();
  bool hasActionTarget_ = false;
};

}  // namespace legged
