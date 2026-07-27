#pragma once

#include <legged_rl_controllers/OnnxPolicy.h>

#include "motion_tracking_controller/SoftTouchDribbleCommon.h"

#include <memory>
#include <string>
#include <vector>

namespace legged {

class SoftTouchDribbleOnnxPolicy : public OnnxPolicy {
 public:
  using SharedPtr = std::shared_ptr<SoftTouchDribbleOnnxPolicy>;
  explicit SoftTouchDribbleOnnxPolicy(const std::string& modelPath) : OnnxPolicy(modelPath) {}

  void reset() override;
  vector_t forward(const vector_t& observations) override;
  void parseMetadata() override;

  const std::vector<std::string>& getJointNames() const { return jointNames_; }
  const vector_t& getDefaultJointPosition() const { return defaultJointPosition_; }
  const vector_t& getActionScale() const { return actionScale_; }
  const vector_t& getRawAction() const { return previousRawAction_; }
  const vector_t& getLatentAction() const { return previousLatentAction_; }
  const vector_t& getJointTarget() const { return jointTarget_; }
  const vector_t& getCurrentRawAction() const { return rawAction_; }
  const vector_t& getCurrentLatentAction() const { return latentAction_; }
  const std::vector<std::string>& getObservationNames() const { return observationNames_; }
  const std::vector<size_t>& getObservationDims() const { return observationDims_; }
  const std::vector<size_t>& getStoredObservationHistoryLengths() const { return observationHistoryLengths_; }
  size_t getLatentDim() const { return latentDim_; }
  size_t getActorObservationDim() const { return actorObservationDim_; }
  size_t getActorHistoryLength() const { return actorHistoryLength_; }
  size_t getDecoderStateDim() const { return decoderStateDim_; }
  const std::string& getObsFrameBodyName() const { return obsFrameBodyName_; }
  const vector3_t& getObsFrameOffset() const { return obsFrameOffset_; }
  // Per-observation-term history lengths from ONNX metadata (v2 history policy).
  // Empty if the model predates the field (caller then keeps default history=1).
  std::vector<size_t> getObservationHistoryLengths() const;
  void setJointTargetClip(const vector_t& lower, const vector_t& upper, scalar_t factor);
  void disableJointTargetClip();

 protected:
  vector_t parseVectorMetadata(const std::string& key, size_t expectedSize);

  std::vector<std::string> jointNames_;
  vector_t defaultJointPosition_;
  vector_t actionScale_;
  vector_t rawAction_;
  vector_t latentAction_;
  vector_t previousRawAction_;
  vector_t previousLatentAction_;
  vector_t jointTarget_;
  vector_t jointTargetLower_;
  vector_t jointTargetUpper_;
  std::vector<std::string> observationNames_;
  std::vector<size_t> observationDims_;
  std::vector<size_t> observationHistoryLengths_;
  size_t latentDim_ = kSoftTouchDribbleLatentDim;
  size_t actorObservationDim_ = 0;
  size_t actorHistoryLength_ = 1;
  size_t decoderStateDim_ = 0;
  std::string obsFrameBodyName_ = "pelvis";
  vector3_t obsFrameOffset_ = vector3_t::Zero();
  scalar_t jointTargetLimitFactor_ = 0.9;
  bool clipJointTarget_ = false;
};

}  // namespace legged
