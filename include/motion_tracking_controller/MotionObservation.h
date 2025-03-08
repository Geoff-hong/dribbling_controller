//
// Created by qiayuanl on 3/7/25.
//

#pragma once

#include <legged_rl_controllers/ObservationManager.h>

#include "motion_tracking_controller/common.h"

namespace legged {

class MotionObservation : public ObservationTerm {
 public:
  MotionObservation(const std::shared_ptr<LeggedModel>& leggedModel, const MotionCommandCfg& cfg);
  size_t getSize() const override;

 protected:
  vector_t evaluate() override;

  MotionCommandCfg cfg_;
  size_t referenceBodyIndex_;
  std::vector<size_t> bodyIndices_;
};

}  // namespace legged
