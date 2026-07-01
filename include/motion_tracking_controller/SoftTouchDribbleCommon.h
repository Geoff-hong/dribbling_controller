#pragma once

#include <legged_model/common.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace legged {

static constexpr size_t kSoftTouchDribbleNumJoints = 29;
static constexpr size_t kSoftTouchDribbleLatentDim = 8;
// 2026-06-17 DR run dropped the 4 world-frame actor obs terms (cmd_dir_w,
// next_cmd_dir_w, pelvis_pos_xy_w, pelvis_yaw_cossin_w) -> 90 - 8 = 82.
static constexpr size_t kSoftTouchDribbleActorObsDim = 82;
static constexpr size_t kSoftTouchDribbleDecoderStateDim = 90;
static constexpr size_t kSoftTouchDribbleObsDim = kSoftTouchDribbleActorObsDim + kSoftTouchDribbleDecoderStateDim;

using vector2_t = Eigen::Matrix<scalar_t, 2, 1>;

struct SoftTouchDribbleCommand {
  vector2_t targetDirWorld = vector2_t::UnitX();
  scalar_t targetSpeed = 0.0;
  vector2_t nextTargetDirWorld = vector2_t::UnitX();
  scalar_t nextTargetSpeed = 0.0;
  scalar_t crossTrackDistance = 0.0;
  int ballSegmentIndex = 0;
};

struct SoftTouchDribbleState {
  vector3_t pelvisPositionWorld = vector3_t::Zero();
  quaternion_t pelvisOrientationWorld = quaternion_t::Identity();
  vector3_t baseAngularVelocityBody = vector3_t::Zero();
  vector_t jointPosition;
  vector_t jointVelocity;
  vector3_t ballPositionWorld = vector3_t::Zero();
  vector3_t ballLinearVelocityWorld = vector3_t::Zero();
  vector_t lastLatentAction;
  vector_t lastDecodedAction;
};

struct SoftTouchDribbleRouteConfig {
  scalar_t routeLength = 20.0;
  scalar_t routeSegmentLength = 0.25;
  scalar_t routeLookahead = 0.8;
  scalar_t routePreviewArc = 1.0;
  scalar_t routeCurvatureMin = 0.0;
  scalar_t routeCurvatureMax = 0.0;
  scalar_t routeSFlipArc = 2.5;
  scalar_t routeHumanKappaCap = 0.5;
  scalar_t routeHumanPersist = 0.6;
  scalar_t routeHumanWeaveMin = 0.4;
  scalar_t routeHumanWeaveMax = 1.0;
  scalar_t routeHumanBigProbability = 0.09;
  scalar_t routeHumanBigAngleMinDeg = 40.0;
  scalar_t routeHumanBigAngleMaxDeg = 180.0;
  scalar_t routeKvScale = 0.75;
  scalar_t routeVmax = 2.0;
  bool routeLazyExtend = true;
  int routeInitSegments = 9;
  int routeExtendChunk = 1;
  int routeExtendAheadMarginSegments = 10;
};

class SoftTouchDribbleRoute {
 public:
  explicit SoftTouchDribbleRoute(SoftTouchDribbleRouteConfig cfg = SoftTouchDribbleRouteConfig{}, uint32_t seed = 42);

  void seed(uint32_t seed);
  SoftTouchDribbleCommand reset(const vector2_t& originWorld, const vector2_t& forwardWorld, int cmdMode = 4);
  SoftTouchDribbleCommand update(const vector2_t& ballPositionWorld);

  const SoftTouchDribbleRouteConfig& getConfig() const { return cfg_; }
  const std::vector<vector2_t>& getRoutePoints() const { return routePoints_; }
  const std::vector<scalar_t>& getRouteSpeed() const { return routeSpeed_; }
  int getRouteFilledSegments() const { return routeFilledSegments_; }
  int getLastBallSegmentIndex() const { return lastBallSegmentIndex_; }

 private:
  scalar_t uniform(scalar_t low, scalar_t high);
  scalar_t uniform01();
  static vector2_t safeUnit(const vector2_t& value);
  vector2_t routePointAtArcLength(scalar_t arcLength) const;
  void buildRouteChunk(int numSegments, bool init, const vector2_t& originWorld = vector2_t::Zero(),
                       const vector2_t& forwardWorld = vector2_t::UnitX());
  std::vector<scalar_t> sampleHumanKappa(int numSegments);
  void extendRoutesIfNeeded();

  SoftTouchDribbleRouteConfig cfg_;
  std::mt19937 rng_;
  std::vector<vector2_t> routePoints_;
  std::vector<scalar_t> routeSpeed_;
  int routeFilledSegments_ = 0;
  scalar_t routeEndHeading_ = 0.0;
  scalar_t humanDribbleSign_ = 1.0;
  scalar_t humanDribbleBigRemain_ = 0.0;
  scalar_t humanDribbleBigSign_ = 1.0;
  int lastBallSegmentIndex_ = -1;
  int cmdMode_ = 4;
  scalar_t cmdSign_ = 1.0;
};

scalar_t yawFromQuaternion(const quaternion_t& quat);
vector3_t projectedGravityBody(const quaternion_t& pelvisOrientationWorld);
vector_t makeSoftTouchDribbleObservation(const SoftTouchDribbleState& state, const SoftTouchDribbleCommand& command,
                                         const vector_t& defaultJointPosition);
vector_t makeSoftTouchJointTarget(const vector_t& rawAction, const vector_t& defaultJointPosition, const vector_t& actionScale);
std::vector<std::string> softTouchDribbleJointNames();

}  // namespace legged
