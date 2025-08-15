//
// Created by qiayuanl on 3/7/25.
//

#pragma once

#include <legged_rl_controllers/ObservationManager.h>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/common.h"

namespace legged {

class MotionObservation : public ObservationTerm {
 public:
  explicit MotionObservation(const MotionCommandTerm::SharedPtr& commandTerm) : commandTerm_(commandTerm) {}

 protected:
  MotionCommandTerm::SharedPtr commandTerm_;
};

class MotionAnchorPosition final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override { return commandTerm_->getAnchorPositionLocal(); }
};

class MotionAnchorOrientation final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 6; }

 protected:
  vector_t evaluate() override { return commandTerm_->getAnchorOrientationLocal(); }
};

class RobotBodyPosition final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 3 * commandTerm_->getCfg().bodyNames.size(); }

 protected:
  vector_t evaluate() override { return commandTerm_->getRobotBodyPositionLocal(); }
};

class RobotBodyOrientation final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 6 * commandTerm_->getCfg().bodyNames.size(); }

 protected:
  vector_t evaluate() override { return commandTerm_->getRobotBodyOrientationLocal(); }
};

}  // namespace legged
