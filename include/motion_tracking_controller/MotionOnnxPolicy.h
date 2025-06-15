//
// Created by qiayuanl on 5/14/25.
//

#pragma once

#include <legged_rl_controllers/OnnxPolicy.h>

namespace legged {

class MotionOnnxPolicy : public OnnxPolicy {
 public:
  using SharedPtr = std::shared_ptr<MotionOnnxPolicy>;
  using OnnxPolicy::OnnxPolicy;

  void reset() override;
  vector_t forward(const vector_t& observations) override;

  vector_t getJointPosition() const { return jointPosition_; }
  vector_t getJointVelocity() const { return jointVelocity_; }
  std::vector<vector3_t> getBodyPositions() const { return bodyPositions_; }
  std::vector<quaternion_t> getBodyOrientations() const { return bodyOrientations_; }

 protected:
  size_t time_step_ = 0;
  vector_t jointPosition_;
  vector_t jointVelocity_;
  std::vector<vector3_t> bodyPositions_;
  std::vector<quaternion_t> bodyOrientations_;
};

}  // namespace legged
