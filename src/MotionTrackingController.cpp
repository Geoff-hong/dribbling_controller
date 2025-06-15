#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {
controller_interface::return_type MotionTrackingController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (OnnxController::update(time, period) != controller_interface::return_type::OK) {
    return controller_interface::return_type::ERROR;
  }

  // dataLogger_->update(time);

  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  get_node()->get_parameter("motion.reference_body", cfg_.referenceBody);
  get_node()->get_parameter("motion.body_names", cfg_.bodyNames);

  std::string policyPath{};
  get_node()->get_parameter("policy.path", policyPath);
  policy_ = std::make_shared<MotionOnnxPolicy>(policyPath);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("MotionTrackingController"), "Load Onnx model from" << policyPath << " successfully !");

  return RlController::on_configure(previous_state);
}

controller_interface::CallbackReturn MotionTrackingController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (OnnxController::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // dataLogger_ = std::make_shared<DataLogger>(leggedModel(), policy_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (OnnxController::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // dataLogger_->writeAndClear();

  return controller_interface::CallbackReturn::SUCCESS;
}

bool MotionTrackingController::parserCommand(const std::string& name) {
  if (OnnxController::parserCommand(name)) {
    return true;
  }
  if (name == "motion") {
    commandTerm_ = std::make_shared<MotionCommandTerm>(cfg_, std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_));
    commandManager_->addTerm(commandTerm_);
    return true;
  }
  return false;
}

bool MotionTrackingController::parserObservation(const std::string& name) {
  if (OnnxController::parserObservation(name)) {
    return true;
  }
  if (name == "motion_ref_pos_b") {
    observationManager_->addTerm(std::make_shared<MotionReferencePosition>(commandTerm_));
  } else if (name == "motion_ref_ori_b") {
    observationManager_->addTerm(std::make_shared<MotionReferenceOrientation>(commandTerm_));
  } else if (name == "robot_body_pos") {
    observationManager_->addTerm(std::make_shared<RobotBodyPosition>(commandTerm_));
  } else if (name == "robot_body_ori") {
    observationManager_->addTerm(std::make_shared<RobotBodyOrientation>(commandTerm_));
  } else {
    return false;
  }
  return true;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)
