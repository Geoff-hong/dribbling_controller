#include "motion_tracking_controller/SoftTouchDribbleObservation.h"

#include <cmath>

namespace legged {
namespace {

vector3_t rotateWorldToBody(const quaternion_t& bodyOrientationWorld, const vector3_t& valueWorld) {
  return bodyOrientationWorld.normalized().conjugate() * valueWorld;
}

vector_t toVector(const vector2_t& value) {
  vector_t out(2);
  out << value.x(), value.y();
  return out;
}

vector_t toVector(const vector3_t& value) {
  vector_t out(3);
  out << value.x(), value.y(), value.z();
  return out;
}

vector_t scalarVector(scalar_t value) {
  vector_t out(1);
  out(0) = value;
  return out;
}

}  // namespace

vector_t SoftTouchBaseAngularVelocity::evaluate() {
  return toVector(commandTerm_->getBaseAngularVelocityBody());
}

vector_t SoftTouchProjectedGravity::evaluate() {
  return toVector(projectedGravityBody(commandTerm_->getPelvisOrientationWorld()));
}

vector_t SoftTouchJointPosition::evaluate() {
  return model_->getGeneralizedPosition().tail(model_->getJointNames().size());
}

vector_t SoftTouchJointPosition::modify(const vector_t& observation) {
  vector_t modifiedObservation(getSize());
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    modifiedObservation[static_cast<Eigen::Index>(i)] = observation[model_->getJointIndex(jointNameInPolicy_[i])];
  }
  return modifiedObservation - defaultJointPosition_;
}

vector_t SoftTouchJointVelocity::evaluate() {
  return model_->getGeneralizedVelocity().tail(model_->getJointNames().size());
}

vector_t SoftTouchLastLatentAction::evaluate() {
  return commandTerm_->getPolicy()->getLatentAction();
}

vector_t SoftTouchBallPositionBody::evaluate() {
  const quaternion_t pelvisOrientation = commandTerm_->getPelvisOrientationWorld();
  const vector3_t rel = commandTerm_->getBallPositionWorld() - commandTerm_->getPelvisPositionWorld();
  return toVector(rotateWorldToBody(pelvisOrientation, rel));
}

vector_t SoftTouchBallLinearVelocityBody::evaluate() {
  return toVector(rotateWorldToBody(commandTerm_->getPelvisOrientationWorld(), commandTerm_->getBallLinearVelocityWorld()));
}

vector_t SoftTouchTargetDirectionBody::evaluate() {
  const auto command = commandTerm_->getRouteCommand();
  const vector3_t dirBody =
      rotateWorldToBody(commandTerm_->getPelvisOrientationWorld(), vector3_t(command.targetDirWorld.x(), command.targetDirWorld.y(), 0.0));
  return toVector(vector2_t(dirBody.x(), dirBody.y()));
}

vector_t SoftTouchTargetSpeed::evaluate() {
  return scalarVector(commandTerm_->getRouteCommand().targetSpeed);
}

vector_t SoftTouchCommandDirectionWorld::evaluate() {
  return toVector(commandTerm_->getRouteCommand().targetDirWorld);
}

vector_t SoftTouchNextCommandDirectionWorld::evaluate() {
  return toVector(commandTerm_->getRouteCommand().nextTargetDirWorld);
}

vector_t SoftTouchNextTargetSpeed::evaluate() {
  return scalarVector(commandTerm_->getRouteCommand().nextTargetSpeed);
}

vector_t SoftTouchPelvisPositionXyWorld::evaluate() {
  const vector3_t pelvis = commandTerm_->getPelvisPositionWorld();
  return toVector(vector2_t(pelvis.x(), pelvis.y()));
}

vector_t SoftTouchPelvisYawCosSinWorld::evaluate() {
  const scalar_t yaw = yawFromQuaternion(commandTerm_->getPelvisOrientationWorld());
  vector_t out(2);
  out << std::cos(yaw), std::sin(yaw);
  return out;
}

vector_t SoftTouchLastDecodedAction::evaluate() {
  return commandTerm_->getPolicy()->getRawAction();
}

}  // namespace legged
