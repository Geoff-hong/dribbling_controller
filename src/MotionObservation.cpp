//
// Created by qiayuanl on 3/7/25.
//
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {

void RobotReferenceObservation::setModel(const LeggedModel::SharedPtr& model) {
  ObservationTerm::setModel(model);
  const auto& pinModel = model_->getPinModel();

  referenceBodyIndex_ = pinModel.getFrameId(cfg_.referenceBody);
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
    if (bodyIndices_.back() >= pinModel.nframes) {
      throw std::runtime_error("Frame " + bodyName + " not found.");
    }
  }
}

vector_t RobotReferenceOrientation::evaluate() {
  const auto& refPoseReal = model_->getPinData().oMf[referenceBodyIndex_];
  return rotationToVectorWxyz(refPoseReal.rotation());
}

vector_t RobotBodyPosition::evaluate() {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceBodyIndex_];
  vector_t value(getSize());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal = refPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(3 * i, 3) = bodyPoseLocal.translation();
  }
  return value;
}

vector_t RobotBodyOrientation::evaluate() {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceBodyIndex_];
  vector_t value(getSize());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal = refPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(i * 4, 4) = rotationToVectorWxyz(bodyPoseLocal.rotation());
  }
  return value;
}

MotionReferencePosition::MotionReferencePosition(const MotionCommandTerm::SharedPtr& commandTerm) : commandTerm_(commandTerm) {}

vector_t MotionReferencePosition::evaluate() {
  return commandTerm_->getReferencePositionLocal();
}

}  // namespace legged
