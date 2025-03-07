#include "motion_tracking_controller/MotionTrackingController.h"

namespace legged {
bool MotionTrackingController::parserObservation(const std::string& name) {
  std::cerr << "MotionTrackingController::parserObservation got called with name: " << name << std::endl;
  if (OnnxController::parserObservation(name)) {
    return true;
  }
  if (name == "my_observation") {
    // observationManager_->addTerm(std::make_shared<MyObservationTerm>(leggedModel));
  } else {
    return false;
  }
  return true;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)
