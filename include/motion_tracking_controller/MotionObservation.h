//
// Created by qiayuanl on 3/7/25.
//

#pragma once

#include <legged_rl_controllers/ObservationManager.h>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/common.h"

namespace legged {

class RobotReferenceObservation : public ObservationTerm {
 public:
  RobotReferenceObservation(LeggedModel::SharedPtr leggedModel, MotionCommandCfg cfg);

 protected:
  MotionCommandCfg cfg_;
  size_t referenceBodyIndex_;
  std::vector<size_t> bodyIndices_;
};

class RobotReferenceOrientation final : public RobotReferenceObservation {
 public:
  using RobotReferenceObservation::RobotReferenceObservation;
  size_t getSize() const override { return 4; }

 protected:
  vector_t evaluate() override;
};

class RobotBodyPosition final : public RobotReferenceObservation {
 public:
  using RobotReferenceObservation::RobotReferenceObservation;
  size_t getSize() const override { return 3 * bodyIndices_.size(); }

 protected:
  vector_t evaluate() override;
};

class RobotBodyOrientation final : public RobotReferenceObservation {
 public:
  using RobotReferenceObservation::RobotReferenceObservation;
  size_t getSize() const override { return 4 * bodyIndices_.size(); }

 protected:
  vector_t evaluate() override;
};

class MotionReferencePosition final : public ObservationTerm {
 public:
  MotionReferencePosition(LeggedModel::SharedPtr model, MotionCommandTerm::SharedPtr commandTerm);
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override;
  MotionCommandTerm::SharedPtr commandTerm_;
};

}  // namespace legged
