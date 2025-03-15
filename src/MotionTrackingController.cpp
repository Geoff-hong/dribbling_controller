#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {
controller_interface::return_type MotionTrackingController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (OnnxController::update(time, period) != controller_interface::return_type::OK) {
    return controller_interface::return_type::ERROR;
  }

  dataLogger_->update(time);

  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  get_node()->get_parameter("motion.path", cfg_.path);
  get_node()->get_parameter("motion.reference_body", cfg_.referenceBody);
  get_node()->get_parameter("motion.joint_names", cfg_.jointNames);
  get_node()->get_parameter("motion.body_names", cfg_.bodyNames);

  if (OnnxController::on_configure(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  dataLogger_ = std::make_shared<DataLogger>(leggedModel_->getLeggedModel(), onnxPolicy_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (OnnxController::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  dataLogger_->writeAndClear();

  return controller_interface::CallbackReturn::SUCCESS;
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
