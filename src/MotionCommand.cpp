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
  referenceRobotIndex_ = pinModel.getFrameId(cfg_.referenceBody);
  if (referenceRobotIndex_ >= pinModel.nframes) {
    throw std::runtime_error("Reference body " + cfg_.referenceBody + " not found.");
  }
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
    if (bodyIndices_.back() >= pinModel.nframes) {
      throw std::runtime_error("Frame " + bodyName + " not found.");
    }
  }
  referenceMotionIndex_ = cfg_.bodyNames.size();
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    if (cfg_.bodyNames[i] == cfg_.referenceBody) {
      referenceMotionIndex_ = i;
      break;
    }
  }
  if (referenceMotionIndex_ == cfg_.bodyNames.size()) {
    throw std::runtime_error("Reference body " + cfg_.referenceBody + " not found in body names.");
  }

  // Move the whole motion frame s.t. the first frame of the motion is aligned with the current robot in position and yaw orientation.
  pinocchio::SE3 initToRef(motionPolicy_->getBodyOrientations()[referenceMotionIndex_],
                           motionPolicy_->getBodyPositions()[referenceMotionIndex_]);
  pinocchio::SE3 worldToRef = model_->getPinData().oMf[referenceRobotIndex_];
  initToRef.rotation() = yawQuaternion(quaternion_t(initToRef.rotation()));
  worldToRef.rotation() = yawQuaternion(quaternion_t(worldToRef.rotation()));

  worldToInit_ = worldToRef * initToRef.inverse();

  std::cerr << initToRef << std::endl;
  std::cerr << worldToRef << std::endl;
  std::cerr << worldToInit_ << std::endl;
}

vector3_t MotionCommandTerm::getReferencePositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceRobotIndex_];

  const auto& refPos = motionPolicy_->getBodyPositions()[referenceMotionIndex_];
  return refPoseReal.actInv(worldToInit_.act(refPos));
}

vector_t MotionCommandTerm::getReferenceOrientationLocal() const {
  const auto& refPoseReal = model_->getPinData().oMf[referenceRobotIndex_];
  const pinocchio::SE3 refOri(motionPolicy_->getBodyOrientations()[referenceMotionIndex_], vector3_t::Zero());
  const auto rot = refPoseReal.actInv(worldToInit_.act(refOri)).rotation();
  vector_t rot6(6);
  rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
  return rot6;
}

vector_t MotionCommandTerm::getRobotBodyPositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceRobotIndex_];
  vector_t value(3 * cfg_.bodyNames.size());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal = refPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(3 * i, 3) = bodyPoseLocal.translation();
  }
  return value;
}

vector_t MotionCommandTerm::getRobotBodyOrientationLocal() const {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceRobotIndex_];
  vector_t value(6 * cfg_.bodyNames.size());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& rot = refPoseReal.actInv(data.oMf[bodyIndices_[i]]).rotation();
    vector_t rot6(6);
    rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    value.segment(i * 6, 6) = rot6;
  }
  return value;
}

}  // namespace legged
