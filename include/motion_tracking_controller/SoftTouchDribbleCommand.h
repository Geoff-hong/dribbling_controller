#pragma once

#include "motion_tracking_controller/SoftTouchDribbleCommon.h"
#include "motion_tracking_controller/SoftTouchDribbleOnnxPolicy.h"

#include <legged_rl_controllers/CommandManager.h>

#include <mutex>
#include <string>
#include <utility>

namespace legged {

struct SoftTouchDribbleCommandCfg {
  std::string baseName = "pelvis";
  std::string baseStateSource = "model";
  SoftTouchDribbleRouteConfig route;
  int cmdMode = 4;
  uint32_t seed = 42;
  scalar_t resetBallForward = 0.65;
  scalar_t resetBallZ = 0.09;
  scalar_t ballTimeout = 0.10;
  scalar_t baseTimeout = 0.10;
};

struct SoftTouchDribbleBallState {
  vector3_t positionWorld = vector3_t::Zero();
  vector3_t linearVelocityWorld = vector3_t::Zero();
  scalar_t positionStamp = 0.0;
  scalar_t linearVelocityStamp = 0.0;
  bool hasPosition = false;
  bool hasVelocity = false;
};

struct SoftTouchDribbleBaseState {
  vector3_t positionWorld = vector3_t::Zero();
  quaternion_t orientationWorld = quaternion_t::Identity();
  vector3_t angularVelocityBody = vector3_t::Zero();
  scalar_t poseStamp = 0.0;
  scalar_t twistStamp = 0.0;
  bool hasPose = false;
  bool hasTwist = false;
};

class SoftTouchDribbleCommandTerm : public CommandTerm {
 public:
  using SharedPtr = std::shared_ptr<SoftTouchDribbleCommandTerm>;

  SoftTouchDribbleCommandTerm(SoftTouchDribbleCommandCfg cfg, SoftTouchDribbleOnnxPolicy::SharedPtr policy)
      : cfg_(std::move(cfg)), policy_(std::move(policy)), route_(cfg_.route, cfg_.seed) {}

  vector_t getValue() override;
  void reset() override;

  const SoftTouchDribbleCommandCfg& getCfg() const { return cfg_; }
  SoftTouchDribbleOnnxPolicy::SharedPtr getPolicy() const { return policy_; }

  void setBallPosition(const vector3_t& positionWorld, scalar_t stamp);
  void setBallVelocity(const vector3_t& linearVelocityWorld, scalar_t stamp);
  void setBasePose(const vector3_t& positionWorld, const quaternion_t& orientationWorld, scalar_t stamp);
  void setBaseAngularVelocity(const vector3_t& angularVelocityBody, scalar_t stamp);
  void setNow(scalar_t now) { now_ = now; }
  bool hasFreshBallState() const;
  bool hasFreshBaseState() const;

  void refreshRouteCommand();
  SoftTouchDribbleCommand getRouteCommand();
  const std::vector<vector2_t>& getRoutePoints() const { return route_.getRoutePoints(); }
  int getRouteFilledSegments() const { return route_.getRouteFilledSegments(); }
  vector3_t getBallPositionWorld() const;
  vector3_t getBallLinearVelocityWorld() const;
  vector3_t getPelvisPositionWorld() const;
  quaternion_t getPelvisOrientationWorld() const;
  vector3_t getBaseAngularVelocityBody() const;
  // sim2sim parity debug: sample stamps for measuring the topic-hop staleness
  scalar_t getBallPositionStamp() const { return getBallState().positionStamp; }
  scalar_t getBasePoseStamp() const { return getBaseState().poseStamp; }

 protected:
  size_t getSize() const override { return 6; }

 private:
  SoftTouchDribbleBallState getBallState() const;
  SoftTouchDribbleBaseState getBaseState() const;
  bool isFresh(scalar_t stamp) const;
  bool isFresh(scalar_t stamp, scalar_t timeout) const;
  bool hasFreshPosition(const SoftTouchDribbleBallState& ball) const;
  bool hasFreshVelocity(const SoftTouchDribbleBallState& ball) const;
  bool hasFreshPose(const SoftTouchDribbleBaseState& base) const;
  bool hasFreshTwist(const SoftTouchDribbleBaseState& base) const;
  bool useBaseTopic() const;
  vector3_t fallbackBallPositionWorld() const;

  SoftTouchDribbleCommandCfg cfg_;
  SoftTouchDribbleOnnxPolicy::SharedPtr policy_;
  SoftTouchDribbleRoute route_;
  SoftTouchDribbleCommand cachedCommand_;
  bool cachedCommandValid_ = false;
  size_t baseFrameIndex_ = 0;
  scalar_t now_ = 0.0;

  mutable std::mutex ballMutex_;
  SoftTouchDribbleBallState ball_;
  mutable std::mutex baseMutex_;
  SoftTouchDribbleBaseState base_;
};

}  // namespace legged
