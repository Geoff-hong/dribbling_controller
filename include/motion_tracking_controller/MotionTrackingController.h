#pragma once

#include <legged_rl_controllers/OnnxController.h>

#include "motion_tracking_controller/DataLogger.h"
#include "motion_tracking_controller/common.h"

namespace legged {
class MotionTrackingController : public OnnxController {
 public:
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  bool parserCommand(const std::string& name) override;
  bool parserObservation(const std::string& name) override;

  MotionCommandCfg cfg_;
  DataLogger::SharedPtr dataLogger_;
};

}  // namespace legged
