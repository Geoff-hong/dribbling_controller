#include "motion_tracking_controller/SoftTouchDribbleCommon.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace legged {
namespace {

void requireSize(const vector_t& value, Eigen::Index size, const std::string& name) {
  if (value.size() != size) {
    throw std::runtime_error(name + " has size " + std::to_string(value.size()) + ", expected " + std::to_string(size));
  }
}

vector3_t rotateWorldToBody(const quaternion_t& bodyOrientationWorld, const vector3_t& valueWorld) {
  return bodyOrientationWorld.conjugate() * valueWorld;
}

int clampInt(int value, int lo, int hi) {
  return std::max(lo, std::min(value, hi));
}

}  // namespace

scalar_t yawFromQuaternion(const quaternion_t& quat) {
  const auto q = quat.normalized();
  return std::atan2(static_cast<scalar_t>(2.0) * (q.w() * q.z() + q.x() * q.y()),
                   static_cast<scalar_t>(1.0) - static_cast<scalar_t>(2.0) * (q.y() * q.y() + q.z() * q.z()));
}

vector3_t projectedGravityBody(const quaternion_t& pelvisOrientationWorld) {
  return rotateWorldToBody(pelvisOrientationWorld.normalized(), vector3_t(0.0, 0.0, -1.0));
}

vector_t makeSoftTouchDribbleObservation(const SoftTouchDribbleState& state, const SoftTouchDribbleCommand& command,
                                         const vector_t& defaultJointPosition) {
  requireSize(defaultJointPosition, kSoftTouchDribbleNumJoints, "defaultJointPosition");
  requireSize(state.jointPosition, kSoftTouchDribbleNumJoints, "state.jointPosition");
  requireSize(state.jointVelocity, kSoftTouchDribbleNumJoints, "state.jointVelocity");
  requireSize(state.lastLatentAction, kSoftTouchDribbleLatentDim, "state.lastLatentAction");
  requireSize(state.lastDecodedAction, kSoftTouchDribbleNumJoints, "state.lastDecodedAction");

  const quaternion_t pelvisOrientationWorld = state.pelvisOrientationWorld.normalized();
  const vector_t jointPositionRel = state.jointPosition - defaultJointPosition;
  const vector3_t ballPositionBody =
      rotateWorldToBody(pelvisOrientationWorld, state.ballPositionWorld - state.pelvisPositionWorld);
  const vector3_t ballLinearVelocityBody = rotateWorldToBody(pelvisOrientationWorld, state.ballLinearVelocityWorld);
  const vector3_t targetDirBody =
      rotateWorldToBody(pelvisOrientationWorld, vector3_t(command.targetDirWorld.x(), command.targetDirWorld.y(), 0.0));

  vector_t obs(kSoftTouchDribbleObsDim);
  Eigen::Index cursor = 0;
  obs.segment(cursor, 3) = state.baseAngularVelocityBody;
  cursor += 3;
  obs.segment(cursor, 3) = projectedGravityBody(pelvisOrientationWorld);
  cursor += 3;
  obs.segment(cursor, kSoftTouchDribbleNumJoints) = jointPositionRel;
  cursor += kSoftTouchDribbleNumJoints;
  obs.segment(cursor, kSoftTouchDribbleNumJoints) = state.jointVelocity;
  cursor += kSoftTouchDribbleNumJoints;
  obs.segment(cursor, kSoftTouchDribbleLatentDim) = state.lastLatentAction;
  cursor += kSoftTouchDribbleLatentDim;
  obs.segment(cursor, 3) = ballPositionBody;
  cursor += 3;
  obs.segment(cursor, 3) = ballLinearVelocityBody;
  cursor += 3;
  obs.segment(cursor, 2) = targetDirBody.head<2>();
  cursor += 2;
  obs(cursor++) = command.targetSpeed;
  obs(cursor++) = command.nextTargetSpeed;

  obs.segment(cursor, 3) = state.baseAngularVelocityBody;
  cursor += 3;
  obs.segment(cursor, kSoftTouchDribbleNumJoints) = jointPositionRel;
  cursor += kSoftTouchDribbleNumJoints;
  obs.segment(cursor, kSoftTouchDribbleNumJoints) = state.jointVelocity;
  cursor += kSoftTouchDribbleNumJoints;
  obs.segment(cursor, kSoftTouchDribbleNumJoints) = state.lastDecodedAction;
  cursor += kSoftTouchDribbleNumJoints;

  if (cursor != static_cast<Eigen::Index>(kSoftTouchDribbleObsDim)) {
    throw std::runtime_error("SoftTouch dribble observation cursor mismatch.");
  }
  return obs;
}

vector_t makeSoftTouchJointTarget(const vector_t& rawAction, const vector_t& defaultJointPosition, const vector_t& actionScale) {
  requireSize(rawAction, kSoftTouchDribbleNumJoints, "rawAction");
  requireSize(defaultJointPosition, kSoftTouchDribbleNumJoints, "defaultJointPosition");
  requireSize(actionScale, kSoftTouchDribbleNumJoints, "actionScale");
  return defaultJointPosition + actionScale.cwiseProduct(rawAction);
}

SoftTouchDribbleRoute::SoftTouchDribbleRoute(SoftTouchDribbleRouteConfig cfg, uint32_t seed)
    : cfg_(cfg), rng_(seed) {
  const int numSegments =
      std::max(1, static_cast<int>(std::llround(cfg_.routeLength / std::max(cfg_.routeSegmentLength, scalar_t(1.0e-9)))));
  routePoints_.assign(static_cast<size_t>(numSegments + 1), vector2_t::Zero());
  routeSpeed_.assign(static_cast<size_t>(numSegments), 0.0);
}

void SoftTouchDribbleRoute::seed(uint32_t seedValue) {
  rng_.seed(seedValue);
}

SoftTouchDribbleCommand SoftTouchDribbleRoute::reset(const vector2_t& originWorld, const vector2_t& forwardWorld, int cmdMode) {
  cmdMode_ = cmdMode;
  cmdSign_ = (cmdMode_ == 2) ? -1.0 : 1.0;
  lastBallSegmentIndex_ = -1;
  const int maxSegments = static_cast<int>(routeSpeed_.size());
  const int initSegments =
      cfg_.routeLazyExtend ? clampInt(cfg_.routeInitSegments, 1, maxSegments) : maxSegments;
  buildRouteChunk(initSegments, true, originWorld, safeUnit(forwardWorld));
  return update(originWorld);
}

SoftTouchDribbleCommand SoftTouchDribbleRoute::update(const vector2_t& ballPositionWorld) {
  extendRoutesIfNeeded();

  const int filled = std::max(1, routeFilledSegments_);
  scalar_t bestDistanceSquared = std::numeric_limits<scalar_t>::infinity();
  scalar_t bestT = 0.0;
  int bestSegment = 0;
  vector2_t bestProjection = routePoints_[0];

  for (int i = 0; i < filled; ++i) {
    const vector2_t a = routePoints_[static_cast<size_t>(i)];
    const vector2_t b = routePoints_[static_cast<size_t>(i + 1)];
    const vector2_t ab = b - a;
    const scalar_t ab2 = std::max(ab.squaredNorm(), scalar_t(1.0e-9));
    const scalar_t t = std::max(scalar_t(0.0), std::min(scalar_t(1.0), (ballPositionWorld - a).dot(ab) / ab2));
    const vector2_t projection = a + t * ab;
    const scalar_t distanceSquared = (ballPositionWorld - projection).squaredNorm();
    if (distanceSquared < bestDistanceSquared) {
      bestDistanceSquared = distanceSquared;
      bestT = t;
      bestSegment = i;
      bestProjection = projection;
    }
  }

  lastBallSegmentIndex_ = bestSegment;
  const scalar_t crossTrack = (ballPositionWorld - bestProjection).norm();
  const scalar_t sStar = (static_cast<scalar_t>(bestSegment) + bestT) * cfg_.routeSegmentLength;
  const int nextSpeedIndex =
      clampInt(static_cast<int>(std::floor((sStar + cfg_.routeLookahead) / cfg_.routeSegmentLength)), 0,
               static_cast<int>(routeSpeed_.size()) - 1);

  SoftTouchDribbleCommand command;
  command.targetSpeed = routeSpeed_[static_cast<size_t>(bestSegment)];
  command.nextTargetSpeed = routeSpeed_[static_cast<size_t>(nextSpeedIndex)];
  command.targetDirWorld = safeUnit(routePointAtArcLength(sStar + cfg_.routeLookahead) - ballPositionWorld);
  command.nextTargetDirWorld =
      safeUnit(routePointAtArcLength(sStar + cfg_.routeLookahead + cfg_.routePreviewArc) - ballPositionWorld);
  command.crossTrackDistance = crossTrack;
  command.ballSegmentIndex = bestSegment;
  return command;
}

scalar_t SoftTouchDribbleRoute::uniform(scalar_t low, scalar_t high) {
  std::uniform_real_distribution<scalar_t> dist(low, high);
  return dist(rng_);
}

scalar_t SoftTouchDribbleRoute::uniform01() {
  return uniform(0.0, 1.0);
}

vector2_t SoftTouchDribbleRoute::safeUnit(const vector2_t& value) {
  const scalar_t norm = value.norm();
  if (norm < scalar_t(1.0e-9)) {
    return vector2_t::UnitX();
  }
  return value / norm;
}

vector2_t SoftTouchDribbleRoute::routePointAtArcLength(scalar_t arcLength) const {
  const scalar_t maxF = std::max(scalar_t(0.0), static_cast<scalar_t>(routeFilledSegments_) - scalar_t(1.0e-4));
  const scalar_t f = std::min(std::max(arcLength / cfg_.routeSegmentLength, scalar_t(0.0)), maxF);
  const int i = clampInt(static_cast<int>(f), 0, static_cast<int>(routePoints_.size()) - 2);
  const scalar_t frac = f - static_cast<scalar_t>(i);
  return routePoints_[static_cast<size_t>(i)] +
         frac * (routePoints_[static_cast<size_t>(i + 1)] - routePoints_[static_cast<size_t>(i)]);
}

void SoftTouchDribbleRoute::buildRouteChunk(int numSegments, bool init, const vector2_t& originWorld,
                                            const vector2_t& forwardWorld) {
  const scalar_t ds = cfg_.routeSegmentLength;
  const int segmentOffset = init ? 0 : routeFilledSegments_;
  const vector2_t origin = init ? originWorld : routePoints_[static_cast<size_t>(segmentOffset)];
  scalar_t thetaStart = routeEndHeading_;

  if (init) {
    thetaStart = std::atan2(forwardWorld.y(), forwardWorld.x());
    humanDribbleSign_ = uniform01() < 0.5 ? 1.0 : -1.0;
    humanDribbleBigRemain_ = 0.0;
    humanDribbleBigSign_ = 1.0;
    routePoints_[0] = origin;
  }

  std::vector<scalar_t> kappa(static_cast<size_t>(numSegments), 0.0);
  const scalar_t kappaMagnitude = uniform(cfg_.routeCurvatureMin, cfg_.routeCurvatureMax + scalar_t(1.0e-12));
  if (cmdMode_ == 1 || cmdMode_ == 2) {
    std::fill(kappa.begin(), kappa.end(), kappaMagnitude * cmdSign_);
  } else if (cmdMode_ == 3) {
    const int flipSegments = std::max(1, static_cast<int>(std::llround(cfg_.routeSFlipArc / ds)));
    for (int i = 0; i < numSegments; ++i) {
      const int globalSegment = segmentOffset + i;
      const scalar_t wave = ((globalSegment / flipSegments) % 2 == 0) ? 1.0 : -1.0;
      kappa[static_cast<size_t>(i)] = kappaMagnitude * cmdSign_ * wave;
    }
  } else if (cmdMode_ == 4) {
    kappa = sampleHumanKappa(numSegments);
  }

  scalar_t heading = thetaStart;
  vector2_t point = origin;
  for (int i = 0; i < numSegments; ++i) {
    point += vector2_t(std::cos(heading), std::sin(heading)) * ds;
    const int pointIndex = segmentOffset + 1 + i;
    const int speedIndex = segmentOffset + i;
    routePoints_[static_cast<size_t>(pointIndex)] = point;
    const scalar_t kabs = std::max(std::abs(kappa[static_cast<size_t>(i)]), scalar_t(1.0e-3));
    routeSpeed_[static_cast<size_t>(speedIndex)] = std::min(cfg_.routeVmax, std::sqrt(cfg_.routeKvScale / kabs));
    heading += kappa[static_cast<size_t>(i)] * ds;
  }

  routeEndHeading_ = heading;
  routeFilledSegments_ = segmentOffset + numSegments;
}

std::vector<scalar_t> SoftTouchDribbleRoute::sampleHumanKappa(int numSegments) {
  constexpr scalar_t kPi = 3.14159265358979323846;
  constexpr scalar_t kDegToRad = kPi / 180.0;
  const scalar_t bigAngleMin = cfg_.routeHumanBigAngleMinDeg * kDegToRad;
  const scalar_t bigAngleMax = cfg_.routeHumanBigAngleMaxDeg * kDegToRad;
  std::vector<scalar_t> out(static_cast<size_t>(numSegments), 0.0);

  for (int i = 0; i < numSegments; ++i) {
    bool inBig = humanDribbleBigRemain_ > 0.0;
    if (!inBig && uniform01() < cfg_.routeHumanBigProbability) {
      const scalar_t angle = uniform(bigAngleMin, bigAngleMax);
      humanDribbleBigRemain_ = std::max(scalar_t(2.0), std::ceil(angle / (cfg_.routeHumanKappaCap * cfg_.routeSegmentLength)));
      humanDribbleBigSign_ = uniform01() < 0.5 ? 1.0 : -1.0;
      inBig = true;
    }
    if (uniform01() > cfg_.routeHumanPersist) {
      humanDribbleSign_ = -humanDribbleSign_;
    }
    const scalar_t magnitude = uniform(cfg_.routeHumanWeaveMin, cfg_.routeHumanWeaveMax) * cfg_.routeHumanKappaCap;
    out[static_cast<size_t>(i)] = inBig ? humanDribbleBigSign_ * cfg_.routeHumanKappaCap : humanDribbleSign_ * magnitude;
    if (inBig) {
      humanDribbleBigRemain_ -= 1.0;
    }
  }

  return out;
}

void SoftTouchDribbleRoute::extendRoutesIfNeeded() {
  if (!cfg_.routeLazyExtend || lastBallSegmentIndex_ < 0) {
    return;
  }
  const int maxSegments = static_cast<int>(routeSpeed_.size());
  const int headroom = routeFilledSegments_ - lastBallSegmentIndex_;
  if (routeFilledSegments_ >= maxSegments || headroom >= cfg_.routeExtendAheadMarginSegments) {
    return;
  }
  const int numSegments = std::min(cfg_.routeExtendChunk, maxSegments - routeFilledSegments_);
  if (numSegments > 0) {
    buildRouteChunk(numSegments, false);
  }
}

std::vector<std::string> softTouchDribbleJointNames() {
  return {
      "left_hip_pitch_joint",     "right_hip_pitch_joint",    "waist_yaw_joint",
      "left_hip_roll_joint",      "right_hip_roll_joint",     "waist_roll_joint",
      "left_hip_yaw_joint",       "right_hip_yaw_joint",      "waist_pitch_joint",
      "left_knee_joint",          "right_knee_joint",         "left_shoulder_pitch_joint",
      "right_shoulder_pitch_joint", "left_ankle_pitch_joint",   "right_ankle_pitch_joint",
      "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
      "right_ankle_roll_joint",   "left_shoulder_yaw_joint",  "right_shoulder_yaw_joint",
      "left_elbow_joint",         "right_elbow_joint",        "left_wrist_roll_joint",
      "right_wrist_roll_joint",   "left_wrist_pitch_joint",   "right_wrist_pitch_joint",
      "left_wrist_yaw_joint",     "right_wrist_yaw_joint",
  };
}

}  // namespace legged
