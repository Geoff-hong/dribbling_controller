//
// Created by qiayuanl on 3/7/25.
//

#pragma once

#include "motion_tracking_controller/common.h"

#include <legged_rl_controllers/CommandManager.h>

namespace legged {

class MotionCommandTerm : public CommandTerm {
 public:
  using SharedPtr = std::shared_ptr<MotionCommandTerm>;

  explicit MotionCommandTerm(const MotionCommandCfg& cfg) : cfg_(cfg), motionIndex_(0), referenceBodyIndex_(0) {}
  bool loadMotionFile();

  vector_t getValue() override;
  void reset() override;

  MotionCommandCfg getCfg() const { return cfg_; }
  vector3_t getReferencePositionLocal() const;
  vector_t getReferenceOrientationGlobal() const;
  vector_t getRobotBodyPositionLocal() const;
  vector_t getRobotBodyOrientationLocal() const;

 protected:
  size_t getSize() const override { return 2 * cfg_.jointNames.size(); }

  MotionCommandCfg cfg_;

  size_t motionIndex_, referenceBodyIndex_;
  std::vector<size_t> bodyIndices_{};
  pinocchio::SE3 worldToInit_;

  std::vector<vector3_t> referencePosition_;
  std::vector<quaternion_t> referenceOrientation_;
  std::vector<vector_t> jointPosition_;
  std::vector<vector_t> jointVelocity_;
  std::vector<std::vector<vector3_t>> bodyPositions_;
};

template <typename Scalar>
Eigen::Quaternion<Scalar> yawQuaternion(const Eigen::Quaternion<Scalar>& q) {
  Scalar yaw = std::atan2(Scalar(2) * (q.w() * q.z() + q.x() * q.y()), Scalar(1) - Scalar(2) * (q.y() * q.y() + q.z() * q.z()));
  Scalar half_yaw = yaw * Scalar(0.5);
  Eigen::Quaternion<Scalar> ret(std::cos(half_yaw), Scalar(0), Scalar(0), std::sin(half_yaw));
  return ret.normalized();
}
}  // namespace legged
