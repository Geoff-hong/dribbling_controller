//
// Created by qiayuanl on 5/14/25.
//

#include "motion_tracking_controller/MotionOnnxPolicy.h"

#include <iostream>

namespace legged {

void MotionOnnxPolicy::reset() {
  OnnxPolicy::reset();
  timeStep_ = startStep_;
  forward(vector_t::Zero(getObservationSize()));
}

vector_t MotionOnnxPolicy::forward(const vector_t& observations) {
  tensor2d_t timeStep(1, 1);
  timeStep(0, 0) = static_cast<tensor_element_t>(timeStep_++);
  inputTensors_[name2Index_.at("time_step")] = timeStep;
  OnnxPolicy::forward(observations);

  jointPosition_ = outputTensors_[name2Index_.at("joint_pos")].row(0).cast<scalar_t>();
  jointVelocity_ = outputTensors_[name2Index_.at("joint_vel")].row(0).cast<scalar_t>();
  bodyPositions_.clear();
  bodyOrientations_.clear();

  auto body_pos_w = outputTensors_[name2Index_.at("body_pos_w")].cast<scalar_t>();
  auto body_quat_w = outputTensors_[name2Index_.at("body_quat_w")].cast<scalar_t>();

  for (size_t i = 0; i < body_pos_w.rows(); ++i) {
    vector3_t pos = body_pos_w.row(i);
    vector_t quat = body_quat_w.row(i);
    quaternion_t ori;
    ori.w() = quat(0);
    ori.coeffs().head(3) = quat.tail(3);
    bodyPositions_.push_back(pos);
    bodyOrientations_.push_back(ori);
  }
  return getLastAction();
}

}  // namespace legged
