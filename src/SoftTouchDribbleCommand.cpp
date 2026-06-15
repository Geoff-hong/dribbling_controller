#include "motion_tracking_controller/SoftTouchDribbleCommand.h"

#include <algorithm>
#include <stdexcept>

namespace legged {
namespace {

vector2_t forwardXyFromOrientation(const quaternion_t& orientationWorld) {
  vector3_t forward3 = orientationWorld.normalized() * vector3_t::UnitX();
  vector2_t forward(forward3.x(), forward3.y());
  const scalar_t norm = forward.norm();
  if (norm < scalar_t(1.0e-9)) {
    return vector2_t::UnitX();
  }
  return forward / norm;
}

}  // namespace

vector_t SoftTouchDribbleCommandTerm::getValue() {
  const auto command = getRouteCommand();
  vector_t value(getSize());
  value << command.targetDirWorld, command.targetSpeed, command.nextTargetDirWorld, command.nextTargetSpeed;
  return value;
}

void SoftTouchDribbleCommandTerm::reset() {
  const auto& pinModel = model_->getPinModel();
  baseFrameIndex_ = pinModel.getFrameId(cfg_.baseName);
  if (baseFrameIndex_ >= pinModel.nframes) {
    throw std::runtime_error("SoftTouch base frame " + cfg_.baseName + " not found.");
  }

  const vector3_t ball = getBallPositionWorld();
  const auto pelvisOrientation = getPelvisOrientationWorld();
  route_.seed(cfg_.seed);
  cachedCommand_ = route_.reset(ball.head<2>(), forwardXyFromOrientation(pelvisOrientation), cfg_.cmdMode);
  cachedCommandValid_ = true;
}

void SoftTouchDribbleCommandTerm::setBallPosition(const vector3_t& positionWorld, scalar_t stamp) {
  std::lock_guard<std::mutex> lock(ballMutex_);
  if (ball_.hasPosition && stamp < ball_.positionStamp) {
    return;
  }
  ball_.positionWorld = positionWorld;
  ball_.positionStamp = stamp;
  ball_.hasPosition = true;
}

void SoftTouchDribbleCommandTerm::setBallVelocity(const vector3_t& linearVelocityWorld, scalar_t stamp) {
  std::lock_guard<std::mutex> lock(ballMutex_);
  if (ball_.hasVelocity && stamp < ball_.linearVelocityStamp) {
    return;
  }
  ball_.linearVelocityWorld = linearVelocityWorld;
  ball_.linearVelocityStamp = stamp;
  ball_.hasVelocity = true;
}

void SoftTouchDribbleCommandTerm::setBasePose(const vector3_t& positionWorld, const quaternion_t& orientationWorld,
                                             scalar_t stamp) {
  std::lock_guard<std::mutex> lock(baseMutex_);
  if (base_.hasPose && stamp < base_.poseStamp) {
    return;
  }
  base_.positionWorld = positionWorld;
  base_.orientationWorld = orientationWorld.normalized();
  base_.poseStamp = stamp;
  base_.hasPose = true;
}

void SoftTouchDribbleCommandTerm::setBaseAngularVelocity(const vector3_t& angularVelocityBody, scalar_t stamp) {
  std::lock_guard<std::mutex> lock(baseMutex_);
  if (base_.hasTwist && stamp < base_.twistStamp) {
    return;
  }
  base_.angularVelocityBody = angularVelocityBody;
  base_.twistStamp = stamp;
  base_.hasTwist = true;
}

bool SoftTouchDribbleCommandTerm::hasFreshBallState() const {
  const auto ball = getBallState();
  return hasFreshPosition(ball) && hasFreshVelocity(ball);
}

bool SoftTouchDribbleCommandTerm::hasFreshBaseState() const {
  if (!useBaseTopic()) {
    return true;
  }
  const auto base = getBaseState();
  return hasFreshPose(base) && hasFreshTwist(base);
}

void SoftTouchDribbleCommandTerm::refreshRouteCommand() {
  const vector3_t ball = getBallPositionWorld();
  cachedCommand_ = route_.update(ball.head<2>());
  cachedCommandValid_ = true;
}

SoftTouchDribbleCommand SoftTouchDribbleCommandTerm::getRouteCommand() {
  if (!cachedCommandValid_) {
    refreshRouteCommand();
  }
  return cachedCommand_;
}

vector3_t SoftTouchDribbleCommandTerm::getBallPositionWorld() const {
  const auto ball = getBallState();
  if (hasFreshPosition(ball)) {
    return ball.positionWorld;
  }
  return fallbackBallPositionWorld();
}

vector3_t SoftTouchDribbleCommandTerm::getBallLinearVelocityWorld() const {
  const auto ball = getBallState();
  if (hasFreshVelocity(ball)) {
    return ball.linearVelocityWorld;
  }
  return vector3_t::Zero();
}

vector3_t SoftTouchDribbleCommandTerm::getPelvisPositionWorld() const {
  if (useBaseTopic()) {
    const auto base = getBaseState();
    if (hasFreshPose(base)) {
      return base.positionWorld;
    }
  }
  const auto& pinData = model_->getPinData();
  return pinData.oMf[baseFrameIndex_].translation();
}

quaternion_t SoftTouchDribbleCommandTerm::getPelvisOrientationWorld() const {
  if (useBaseTopic()) {
    const auto base = getBaseState();
    if (hasFreshPose(base)) {
      return base.orientationWorld.normalized();
    }
  }
  const auto& pinData = model_->getPinData();
  return quaternion_t(pinData.oMf[baseFrameIndex_].rotation()).normalized();
}

vector3_t SoftTouchDribbleCommandTerm::getBaseAngularVelocityBody() const {
  if (useBaseTopic()) {
    const auto base = getBaseState();
    if (hasFreshTwist(base)) {
      return base.angularVelocityBody;
    }
  }
  return model_->getGeneralizedVelocity().segment<3>(3);
}

SoftTouchDribbleBallState SoftTouchDribbleCommandTerm::getBallState() const {
  std::lock_guard<std::mutex> lock(ballMutex_);
  return ball_;
}

SoftTouchDribbleBaseState SoftTouchDribbleCommandTerm::getBaseState() const {
  std::lock_guard<std::mutex> lock(baseMutex_);
  return base_;
}

bool SoftTouchDribbleCommandTerm::isFresh(scalar_t stamp) const {
  return isFresh(stamp, cfg_.ballTimeout);
}

bool SoftTouchDribbleCommandTerm::isFresh(scalar_t stamp, scalar_t timeout) const {
  if (now_ <= scalar_t(0.0)) {
    return true;
  }
  return now_ - stamp <= timeout;
}

bool SoftTouchDribbleCommandTerm::hasFreshPosition(const SoftTouchDribbleBallState& ball) const {
  return ball.hasPosition && isFresh(ball.positionStamp);
}

bool SoftTouchDribbleCommandTerm::hasFreshVelocity(const SoftTouchDribbleBallState& ball) const {
  return ball.hasVelocity && isFresh(ball.linearVelocityStamp);
}

bool SoftTouchDribbleCommandTerm::hasFreshPose(const SoftTouchDribbleBaseState& base) const {
  return base.hasPose && isFresh(base.poseStamp, cfg_.baseTimeout);
}

bool SoftTouchDribbleCommandTerm::hasFreshTwist(const SoftTouchDribbleBaseState& base) const {
  return base.hasTwist && isFresh(base.twistStamp, cfg_.baseTimeout);
}

bool SoftTouchDribbleCommandTerm::useBaseTopic() const {
  return cfg_.baseStateSource == "topic";
}

vector3_t SoftTouchDribbleCommandTerm::fallbackBallPositionWorld() const {
  const vector3_t pelvis = getPelvisPositionWorld();
  const vector2_t forward = forwardXyFromOrientation(getPelvisOrientationWorld());
  return vector3_t(pelvis.x() + cfg_.resetBallForward * forward.x(), pelvis.y() + cfg_.resetBallForward * forward.y(),
                   cfg_.resetBallZ);
}

}  // namespace legged
