//
// Created by qiayuanl on 3/7/25.
//

#pragma once

#include "motion_tracking_controller/MotionOnnxPolicy.h"
#include "motion_tracking_controller/common.h"

#include <legged_rl_controllers/CommandManager.h>

#include <utility>

namespace legged {

class MotionCommandTerm : public CommandTerm {
 public:
  using SharedPtr = std::shared_ptr<MotionCommandTerm>;

  MotionCommandTerm(MotionCommandCfg cfg, MotionOnnxPolicy::SharedPtr motionPolicy)
      : cfg_(std::move(cfg)), motionPolicy_(std::move(motionPolicy)), referenceRobotIndex_(0), referenceMotionIndex_(0) {}

  vector_t getValue() override;
  void reset() override;

  MotionCommandCfg getCfg() const { return cfg_; }
  vector3_t getReferencePositionLocal() const;
  vector_t getReferenceOrientationLocal() const;
  vector_t getRobotBodyPositionLocal() const;
  vector_t getRobotBodyOrientationLocal() const;

 protected:
  size_t getSize() const override { return 2 * model_->getNumJoints(); }

  MotionCommandCfg cfg_;
  MotionOnnxPolicy::SharedPtr motionPolicy_;

  size_t referenceRobotIndex_, referenceMotionIndex_;
  std::vector<size_t> bodyIndices_{};
  pinocchio::SE3 worldToInit_;
};

template <typename Scalar>
Eigen::Quaternion<Scalar> yawQuaternion(const Eigen::Quaternion<Scalar>& q) {
  Scalar yaw = std::atan2(Scalar(2) * (q.w() * q.z() + q.x() * q.y()), Scalar(1) - Scalar(2) * (q.y() * q.y() + q.z() * q.z()));
  Scalar half_yaw = yaw * Scalar(0.5);
  Eigen::Quaternion<Scalar> ret(std::cos(half_yaw), Scalar(0), Scalar(0), std::sin(half_yaw));
  return ret.normalized();
}
}  // namespace legged
