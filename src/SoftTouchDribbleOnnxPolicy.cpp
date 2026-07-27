#include "motion_tracking_controller/SoftTouchDribbleOnnxPolicy.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace legged {
namespace {

vector_t vectorFromStdVector(const std::vector<scalar_t>& values) {
  vector_t out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out(static_cast<Eigen::Index>(i)) = values[i];
  }
  return out;
}

}  // namespace

void SoftTouchDribbleOnnxPolicy::reset() {
  OnnxPolicy::reset();
  rawAction_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  latentAction_ = vector_t::Zero(static_cast<Eigen::Index>(latentDim_));
  previousRawAction_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  previousLatentAction_ = vector_t::Zero(static_cast<Eigen::Index>(latentDim_));
  jointTarget_ = defaultJointPosition_.size() == static_cast<Eigen::Index>(kSoftTouchDribbleNumJoints)
                     ? defaultJointPosition_
                     : vector_t::Zero(kSoftTouchDribbleNumJoints);
}

vector_t SoftTouchDribbleOnnxPolicy::forward(const vector_t& observations) {
  // Drive the expected width from the ONNX obs input so v1 (172) and v2 history
  // (920) policies both work without recompiling.
  const auto expectedObsSize = static_cast<Eigen::Index>(getObservationSize());
  if (observations.size() != expectedObsSize) {
    throw std::runtime_error("SoftTouchDribbleOnnxPolicy expected " + std::to_string(expectedObsSize) +
                             "-D observation, got " + std::to_string(observations.size()));
  }
  OnnxPolicy::forward(observations);

  rawAction_ = outputTensors_[name2Index_.at("actions")].row(0).cast<scalar_t>();
  latentAction_ = outputTensors_[name2Index_.at("latent_action")].row(0).cast<scalar_t>();
  if (latentAction_.size() != static_cast<Eigen::Index>(latentDim_)) {
    throw std::runtime_error("SoftTouch latent_action output has size " + std::to_string(latentAction_.size()) +
                             ", expected metadata latent_dim " + std::to_string(latentDim_));
  }
  jointTarget_ = makeSoftTouchJointTarget(rawAction_, defaultJointPosition_, actionScale_);
  if (clipJointTarget_) {
    for (Eigen::Index i = 0; i < jointTarget_.size(); ++i) {
      const scalar_t lower = jointTargetLower_(i);
      const scalar_t upper = jointTargetUpper_(i);
      if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper)) {
        continue;
      }
      const scalar_t center = scalar_t(0.5) * (lower + upper);
      const scalar_t halfWidth = scalar_t(0.5) * (upper - lower) * jointTargetLimitFactor_;
      jointTarget_(i) = std::max(center - halfWidth, std::min(jointTarget_(i), center + halfWidth));
    }
  }
  previousRawAction_ = rawAction_;
  previousLatentAction_ = latentAction_;

  return jointTarget_;
}

void SoftTouchDribbleOnnxPolicy::parseMetadata() {
  OnnxPolicy::parseMetadata();
  jointNames_ = parseCsv<std::string>(getMetadataStr("joint_names"));
  defaultJointPosition_ = parseVectorMetadata("default_joint_pos", kSoftTouchDribbleNumJoints);
  actionScale_ = parseVectorMetadata("action_scale", kSoftTouchDribbleNumJoints);
  try {
    latentDim_ = std::stoul(getMetadataStr("latent_dim"));
  } catch (const std::exception&) {
    latentDim_ = kSoftTouchDribbleLatentDim;
  }
  if (latentDim_ == 0) {
    throw std::runtime_error("SoftTouch ONNX metadata latent_dim must be positive.");
  }
  rawAction_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  latentAction_ = vector_t::Zero(static_cast<Eigen::Index>(latentDim_));
  previousRawAction_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  previousLatentAction_ = vector_t::Zero(static_cast<Eigen::Index>(latentDim_));
  jointTarget_ = defaultJointPosition_;
  jointTargetLower_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  jointTargetUpper_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  observationNames_.clear();
  observationDims_.clear();
  observationHistoryLengths_.clear();
  try {
    actorObservationDim_ = std::stoul(getMetadataStr("actor_observation_dim"));
  } catch (const std::exception&) {
    actorObservationDim_ = 0;
  }
  try {
    actorHistoryLength_ = std::stoul(getMetadataStr("actor_history_length"));
  } catch (const std::exception&) {
    actorHistoryLength_ = 1;
  }
  try {
    decoderStateDim_ = std::stoul(getMetadataStr("decoder_state_dim"));
  } catch (const std::exception&) {
    decoderStateDim_ = 0;
  }
  try {
    observationNames_ = parseCsv<std::string>(getMetadataStr("observation_names"));
  } catch (const std::exception&) {
    observationNames_.clear();
  }
  try {
    observationDims_ = parseCsv<size_t>(getMetadataStr("observation_dims"));
  } catch (const std::exception&) {
    observationDims_.clear();
  }
  observationHistoryLengths_ = getObservationHistoryLengths();

  obsFrameBodyName_ = "pelvis";
  obsFrameOffset_ = vector3_t::Zero();
  try {
    const auto bodyName = getMetadataStr("obs_frame_body");
    if (!bodyName.empty()) {
      obsFrameBodyName_ = bodyName;
    }
  } catch (const std::exception&) {
    obsFrameBodyName_ = "pelvis";
  }
  try {
    const auto offset = parseCsv<scalar_t>(getMetadataStr("obs_frame_offset"));
    if (offset.size() == 3) {
      obsFrameOffset_ = vector3_t(offset[0], offset[1], offset[2]);
    }
  } catch (const std::exception&) {
    obsFrameOffset_ = vector3_t::Zero();
  }

  if (jointNames_.size() != kSoftTouchDribbleNumJoints) {
    throw std::runtime_error("SoftTouch ONNX joint_names metadata has " + std::to_string(jointNames_.size()) +
                             " joints, expected 29.");
  }
  if (!observationNames_.empty() && observationNames_.size() != observationDims_.size()) {
    throw std::runtime_error("SoftTouch ONNX observation_names and observation_dims metadata sizes do not match.");
  }

  std::cout << '\t' << "softtouch_policy_kind: " << getMetadataStr("policy_kind") << '\n';
  std::cout << '\t' << "softtouch_latent_dim: " << latentDim_ << '\n';
  std::cout << '\t' << "softtouch_obs_frame_body: " << obsFrameBodyName_ << '\n';
  std::cout << '\t' << "softtouch_obs_frame_offset: " << obsFrameOffset_.transpose() << '\n';
  std::cout << '\t' << "softtouch_joint_names: " << jointNames_ << '\n';
}

vector_t SoftTouchDribbleOnnxPolicy::parseVectorMetadata(const std::string& key, size_t expectedSize) {
  const auto values = parseCsv<scalar_t>(getMetadataStr(key));
  if (values.size() != expectedSize) {
    throw std::runtime_error("SoftTouch ONNX metadata " + key + " has " + std::to_string(values.size()) +
                             " values, expected " + std::to_string(expectedSize));
  }
  return vectorFromStdVector(values);
}

std::vector<size_t> SoftTouchDribbleOnnxPolicy::getObservationHistoryLengths() const {
  std::vector<size_t> lengths;
  try {
    lengths = parseCsv<size_t>(getMetadataStr("observation_history_lengths"));
  } catch (const std::exception&) {
    return {};  // model predates the field -> caller keeps default history=1
  }
  // Only trust it when it lines up 1:1 with observation_names (i.e. one length per
  // ObservationManager term); otherwise indexing terms by it would be unsafe.
  try {
    if (parseCsv<std::string>(getMetadataStr("observation_names")).size() != lengths.size()) {
      return {};
    }
  } catch (const std::exception&) {
    return {};
  }
  return lengths;
}

void SoftTouchDribbleOnnxPolicy::setJointTargetClip(const vector_t& lower, const vector_t& upper, scalar_t factor) {
  if (lower.size() != static_cast<Eigen::Index>(kSoftTouchDribbleNumJoints) ||
      upper.size() != static_cast<Eigen::Index>(kSoftTouchDribbleNumJoints)) {
    throw std::runtime_error("SoftTouch joint target clip limits must have 29 entries.");
  }
  if (!(factor > scalar_t(0.0) && factor <= scalar_t(1.0))) {
    throw std::runtime_error("SoftTouch joint target clip factor must be in (0, 1].");
  }
  jointTargetLower_ = lower;
  jointTargetUpper_ = upper;
  jointTargetLimitFactor_ = factor;
  clipJointTarget_ = true;
}

void SoftTouchDribbleOnnxPolicy::disableJointTargetClip() {
  clipJointTarget_ = false;
}

}  // namespace legged
