//
// Created by qiayuanl on 3/7/25.
//
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {

MotionObservation::MotionObservation(const std::shared_ptr<LeggedModel>& leggedModel, const MotionCommandCfg& cfg)
    : ObservationTerm(leggedModel), cfg_(cfg) {
  const auto& pinModel = model_->getPinModel();

  referenceBodyIndex_ = pinModel.getFrameId(cfg_.referenceBody);
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
  }
}

vector_t MotionObservation::evaluate() {
  vector_t value(getSize());
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceBodyIndex_];
  auto ori = quaternion_t(refPoseReal.rotation());
  // Reference orientation
  value.segment(0, 4) = quaternionToVectorWxyz(ori);

  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPose = data.oMf[bodyIndices_[i]];
    const auto& bodyPoseLocal = refPoseReal.actInv(bodyPose);
    // Body local position
    value.segment(4 + i * 3, 3) = bodyPoseLocal.translation();
    // Body local orientation
    value.segment(4 + 3 * cfg_.bodyNames.size() + i * 4, 4) = quaternionToVectorWxyz(quaternion_t(bodyPoseLocal.rotation()));
  }

  return value;
}

size_t MotionObservation::getSize() const {
  return 3 + 3 * cfg_.bodyNames.size() + 4 * cfg_.bodyNames.size();
}

}  // namespace legged
