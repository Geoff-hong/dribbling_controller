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

  MotionCommandTerm(const LeggedModel::SharedPtr& leggedModel, const MotionCommandCfg& cfg)
      : CommandTerm(leggedModel), cfg_(cfg), motionIndex_(0), referenceBodyIndex_(0) {}
  bool loadMotionFile();

  vector_t getValue() override;
  void reset() override;

  vector3_t getReferencePositionLocal();

 protected:
  size_t getSize() const { return 2 * cfg_.jointNames.size(); }

  MotionCommandCfg cfg_;

  size_t motionIndex_, referenceBodyIndex_;
  vector3_t positionOffset_;

  std::vector<vector3_t> referencePosition_;
  std::vector<quaternion_t> referenceOrientation_;
  std::vector<vector_t> jointPosition_;
  std::vector<vector_t> jointVelocity_;
  std::vector<std::vector<vector3_t>> bodyPositions_;
};
}  // namespace legged
