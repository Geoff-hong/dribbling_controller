//
// Created by qiayuanl on 3/7/25.
//

#include "motion_tracking_controller/MotionCommand.h"

namespace legged {

vector_t MotionCommandTerm::getValue() {
  vector_t value(getSize());

  if (motionIndex_ < referencePosition_.size() - 1) {
    motionIndex_++;
  }

  const auto& refJointPos = jointPosition_[motionIndex_];
  const auto& refJointVel = jointVelocity_[motionIndex_];

  value.head(cfg_.jointNames.size()) = refJointPos;
  value.tail(cfg_.jointNames.size()) = refJointVel;
  return value;
}

bool MotionCommandTerm::loadMotionFile() {
  std::ifstream file(cfg_.path);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << cfg_.path << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    std::vector<double> rowValues;
    while (std::getline(lineStream, cell, ',')) {
      try {
        double value = std::stod(cell);
        rowValues.push_back(value);
      } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid number in CSV: " << cell << std::endl;
        return false;
      } catch (const std::out_of_range& e) {
        std::cerr << "Number out of range in CSV: " << cell << std::endl;
        return false;
      }
    }
    if (!rowValues.empty()) {
      size_t bodyIndex = 0;
      vector3_t pos;
      quaternion_t ori;
      vector_t joint(cfg_.jointNames.size());
      bodyPositions_.push_back(std::vector<vector3_t>(cfg_.bodyNames.size()));

      for (size_t i = 0; i < rowValues.size(); ++i) {
        if (i < 3) {
          pos[i] = rowValues[i];
        } else if (i == 3) {
          ori.w() = rowValues[i];
        } else if (i < 7) {
          ori.coeffs()[i - 4] = rowValues[i];
        } else if (i < 7 + cfg_.bodyNames.size() * 3) {
          pos[i - 7 - bodyIndex * 3] = rowValues[i];
        } else if (i < 7 + cfg_.bodyNames.size() * 3 + cfg_.jointNames.size()) {
          joint[i - 7 - cfg_.bodyNames.size() * 3] = rowValues[i];
        } else if (i < 7 + cfg_.bodyNames.size() * 3 + 2 * cfg_.jointNames.size()) {
          joint[i - 7 - cfg_.bodyNames.size() * 3 - cfg_.jointNames.size()] = rowValues[i];
        } else {
          std::cerr << "Too many values in CSV" << std::endl;
          return false;
        }
        if (i == 3 - 1) {
          referencePosition_.push_back(pos);
        } else if (i == 7 - 1) {
          referenceOrientation_.push_back(ori);
        } else if (i == 7 + (bodyIndex + 1) * 3 - 1 && bodyIndex < cfg_.bodyNames.size()) {
          bodyPositions_.back()[bodyIndex] = pos;
          bodyIndex++;
        } else if (i == 7 + cfg_.bodyNames.size() * 3 + cfg_.jointNames.size() - 1) {
          jointPosition_.push_back(joint);
        } else if (i == 7 + cfg_.bodyNames.size() * 3 + 2 * cfg_.jointNames.size() - 1) {
          jointVelocity_.push_back(joint);
        }
      }
    }
  }
  file.close();

  if (referencePosition_.size() != referenceOrientation_.size() || referencePosition_.size() != bodyPositions_.size() ||
      referencePosition_.size() != jointPosition_.size() || referencePosition_.size() != jointVelocity_.size()) {
    std::cerr << "Error loading motion: length of time sequence is different" << std::endl;
    return false;
  }

  const auto& pinModel = model_->getPinModel();
  referenceBodyIndex_ = pinModel.getFrameId(cfg_.referenceBody);
  if (referenceBodyIndex_ >= pinModel.nframes) {
    std::cerr << "Frame " + cfg_.referenceBody + " not found." << std::endl;
    return false;
  }
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
    if (bodyIndices_.back() >= pinModel.nframes) {
      throw std::runtime_error("Frame " + bodyName + " not found.");
    }
  }
  return true;
}

void MotionCommandTerm::reset() {
  motionIndex_ = 0;
  const pinocchio::SE3 initToRef(referenceOrientation_[motionIndex_], referencePosition_[motionIndex_]);
  const pinocchio::SE3 worldToRef = model_->getPinData().oMf[referenceBodyIndex_];
  worldToInit_ = worldToRef * initToRef.inverse();
  worldToInit_.rotation() = yawQuaternion(quaternion_t(worldToInit_.rotation()));
}

vector3_t MotionCommandTerm::getReferencePositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceBodyIndex_];

  const auto& refPos = referencePosition_[motionIndex_];
  return refPoseReal.actInv(worldToInit_.act(refPos));
}

vector_t MotionCommandTerm::getReferenceOrientationLocal() const {
  const auto& refPoseReal = model_->getPinData().oMf[referenceBodyIndex_];
  const pinocchio::SE3 refOri(referenceOrientation_[motionIndex_], vector3_t::Zero());
  const auto rot = refPoseReal.actInv(worldToInit_.act(refOri)).rotation();
  vector_t rot6(6);
  rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
  return rot6;
}

vector_t MotionCommandTerm::getRobotBodyPositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceBodyIndex_];
  vector_t value(3 * cfg_.bodyNames.size());
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal = refPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(3 * i, 3) = bodyPoseLocal.translation();
  }
  return value;
}

vector_t MotionCommandTerm::getRobotBodyOrientationLocal() const {
  const auto& data = model_->getPinData();
  const auto& refPoseReal = data.oMf[referenceBodyIndex_];
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
