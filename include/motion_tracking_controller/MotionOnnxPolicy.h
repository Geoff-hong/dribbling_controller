//
// Created by qiayuanl on 5/14/25.
//

#pragma once

#include <legged_rl_controllers/OnnxPolicy.h>

namespace legged {

class MotionOnnxPolicy : public OnnxPolicy {
 public:
  using SharedPtr = std::shared_ptr<MotionOnnxPolicy>;
  MotionOnnxPolicy(const std::string& modelPath, size_t startStep, size_t motionLength = 0, bool loopMotion = false,
                   size_t timeStepStride = 1)
      : OnnxPolicy(modelPath),
        startStep_(startStep),
        motionLength_(motionLength),
        loopMotion_(loopMotion),
        timeStepStride_(timeStepStride == 0 ? 1 : timeStepStride) {}

  void reset() override;
  vector_t forward(const vector_t& observations) override;

  std::string getAnchorBodyName() const { return anchorBodyName_; }
  std::vector<std::string> getBodyNames() const { return bodyNames_; }

  vector_t getJointPosition() const { return jointPosition_; }
  vector_t getJointVelocity() const { return jointVelocity_; }
  std::vector<vector3_t> getBodyPositions() const { return bodyPositions_; }
  std::vector<quaternion_t> getBodyOrientations() const { return bodyOrientations_; }

  void parseMetadata() override;

 protected:
  size_t timeStep_ = 0, startStep_ = 0;
  size_t motionLength_ = 0;
  bool loopMotion_ = false;
  size_t timeStepStride_ = 1;
  vector_t jointPosition_;
  vector_t jointVelocity_;
  std::vector<vector3_t> bodyPositions_;
  std::vector<quaternion_t> bodyOrientations_;
  std::string anchorBodyName_;
  std::vector<std::string> bodyNames_;
};

}  // namespace legged
