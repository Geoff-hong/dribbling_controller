//
// Created by qiayuanl on 3/7/25.
//

#include "motion_tracking_controller/MotionCommand.h"

namespace legged {

vector_t MotionCommandTerm::getValue() {
  return (vector_t(getSize()) << motionPolicy_->getJointPosition(), motionPolicy_->getJointVelocity()).finished();
}

void MotionCommandTerm::reset() {
  const auto& pinModel = model_->getPinModel();
  anchorRobotIndex_ = pinModel.getFrameId(cfg_.anchorBody);
  if (anchorRobotIndex_ >= pinModel.nframes) {
    throw std::runtime_error("Anchor body " + cfg_.anchorBody + " not found.");
  }
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
    if (bodyIndices_.back() >= pinModel.nframes) {
      throw std::runtime_error("Frame " + bodyName + " not found.");
    }
  }
  anchorMotionIndex_ = cfg_.bodyNames.size();
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    if (cfg_.bodyNames[i] == cfg_.anchorBody) {
      anchorMotionIndex_ = i;
      break;
    }
  }
  if (anchorMotionIndex_ == cfg_.bodyNames.size()) {
    throw std::runtime_error("Anchor body " + cfg_.anchorBody + " not found in body names.");
  }

  // Move the whole motion frame s.t. the first frame of the motion is aligned with the current robot in position and yaw orientation.
  pinocchio::SE3 initToAnchor(motionPolicy_->getBodyOrientations()[anchorMotionIndex_], motionPolicy_->getBodyPositions()[anchorMotionIndex_]);
  pinocchio::SE3 worldToAnchor = model_->getPinData().oMf[anchorRobotIndex_];
  initToAnchor.rotation() = yawQuaternion(quaternion_t(initToAnchor.rotation()));
  worldToAnchor.rotation() = yawQuaternion(quaternion_t(worldToAnchor.rotation()));

  worldToInit_ = worldToAnchor * initToAnchor.inverse();

  std::cerr << initToAnchor << std::endl;
  std::cerr << worldToAnchor << std::endl;
  std::cerr << worldToInit_ << std::endl;
}

vector3_t MotionCommandTerm::getAnchorPositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];

  const auto& anchorPos = motionPolicy_->getBodyPositions()[anchorMotionIndex_];
  return anchorPoseReal.actInv(worldToInit_.act(anchorPos));
}

vector_t MotionCommandTerm::getAnchorOrientationLocal() const {
  const auto& anchorPoseReal = model_->getPinData().oMf[anchorRobotIndex_];
  const pinocchio::SE3 anchorOri(motionPolicy_->getBodyOrientations()[anchorMotionIndex_], vector3_t::Zero());
  const auto rot = anchorPoseReal.actInv(worldToInit_.act(anchorOri)).rotation();
  vector_t rot6(6);
  rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
  return rot6;
}

vector_t MotionCommandTerm::getRobotBodyPositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];
  vector_t value(3 * cfg_.bodyNames.size());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal = anchorPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(3 * i, 3) = bodyPoseLocal.translation();
  }
  return value;
}

vector_t MotionCommandTerm::getRobotBodyOrientationLocal() const {
  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];
  vector_t value(6 * cfg_.bodyNames.size());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& rot = anchorPoseReal.actInv(data.oMf[bodyIndices_[i]]).rotation();
    vector_t rot6(6);
    rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    value.segment(i * 6, 6) = rot6;
  }
  return value;
}

}  // namespace legged
