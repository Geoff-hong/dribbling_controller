#pragma once

#include "legged_rl_controllers/OnnxController.h"
#include "motion_tracking_controller/common.h"

namespace legged {
class MotionTrackingController : public OnnxController {
 public:
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  bool parserCommand(const std::string& name) override;
  bool parserObservation(const std::string& name) override;

  MotionCommandCfg cfg_;
};

}  // namespace legged
