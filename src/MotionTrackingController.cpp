#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {

controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  get_node()->get_parameter("motion.path", cfg_.path);
  get_node()->get_parameter("motion.reference_body", cfg_.referenceBody);
  get_node()->get_parameter("motion.joint_names", cfg_.jointNames);
  get_node()->get_parameter("motion.body_names", cfg_.bodyNames);

  return OnnxController::on_configure(previous_state);
}

bool MotionTrackingController::parserCommand(const std::string& name) {
  if (OnnxController::parserCommand(name)) {
    return true;
  }
  if (name == "motion") {
    const auto term = std::make_shared<MotionCommandTerm>(leggedModel_->getLeggedModel(), cfg_);
    commandManager_->addTerm(term);
    return term->loadMotionFile();
  }
  return false;
}

bool MotionTrackingController::parserObservation(const std::string& name) {
  if (OnnxController::parserObservation(name)) {
    return true;
  }
  if (name == "motion") {
    const auto term = std::make_shared<MotionObservation>(leggedModel_->getLeggedModel(), cfg_);
    observationManager_->addTerm(term);
    return true;
  }
  return false;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)
