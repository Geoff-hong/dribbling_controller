#pragma once

#include "motion_tracking_controller/SoftTouchDribbleCommand.h"

#include <legged_rl_controllers/ObservationManager.h>

#include <utility>

namespace legged {

class SoftTouchDribbleObservation : public ObservationTerm {
 public:
  explicit SoftTouchDribbleObservation(SoftTouchDribbleCommandTerm::SharedPtr commandTerm)
      : commandTerm_(std::move(commandTerm)) {}

 protected:
  SoftTouchDribbleCommandTerm::SharedPtr commandTerm_;
};

class SoftTouchBaseAngularVelocity final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchProjectedGravity final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchJointPosition final : public JointObservationTerm {
 public:
  SoftTouchJointPosition(const std::vector<std::string>& jointNames, vector_t defaultJointPosition)
      : JointObservationTerm(jointNames), defaultJointPosition_(std::move(defaultJointPosition)) {}

 protected:
  vector_t evaluate() override;
  vector_t modify(const vector_t& observation) override;

 private:
  vector_t defaultJointPosition_;
};

class SoftTouchJointVelocity final : public JointObservationTerm {
 public:
  explicit SoftTouchJointVelocity(const std::vector<std::string>& jointNames) : JointObservationTerm(jointNames) {}

 protected:
  vector_t evaluate() override;
};

class SoftTouchLastLatentAction final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return kSoftTouchDribbleLatentDim; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchBallPositionBody final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchBallLinearVelocityBody final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchTargetDirectionBody final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 2; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchTargetSpeed final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 1; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchCommandDirectionWorld final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 2; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchNextCommandDirectionWorld final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 2; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchNextTargetSpeed final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 1; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchPelvisPositionXyWorld final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 2; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchPelvisYawCosSinWorld final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return 2; }

 protected:
  vector_t evaluate() override;
};

class SoftTouchLastDecodedAction final : public SoftTouchDribbleObservation {
 public:
  using SoftTouchDribbleObservation::SoftTouchDribbleObservation;
  size_t getSize() const override { return kSoftTouchDribbleNumJoints; }

 protected:
  vector_t evaluate() override;
};

}  // namespace legged
