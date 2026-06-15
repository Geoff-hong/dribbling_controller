#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace legged {
namespace {

std::vector<double> getDoubleArrayParam(const rclcpp_lifecycle::LifecycleNode::SharedPtr& node, const std::string& name,
                                        const std::vector<double>& fallback) {
  if (!node->has_parameter(name)) {
    return fallback;
  }
  return node->get_parameter(name).as_double_array();
}

vector_t defaultMotionEffortLimit(size_t size) {
  if (size != 29) {
    return vector_t::Constant(static_cast<Eigen::Index>(size), std::numeric_limits<scalar_t>::infinity());
  }
  vector_t out(29);
  out << 88.0, 88.0, 88.0, 139.0, 139.0, 50.0, 88.0, 88.0, 50.0, 139.0, 139.0, 25.0, 25.0, 50.0, 50.0,
      25.0, 25.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0, 5.0;
  return out;
}

void requireVectorSize(const vector_t& value, Eigen::Index expectedSize, const std::string& name) {
  if (value.size() != expectedSize) {
    throw std::runtime_error(name + " has size " + std::to_string(value.size()) + ", expected " +
                             std::to_string(expectedSize));
  }
}

}  // namespace

controller_interface::CallbackReturn MotionTrackingController::on_init() {
  if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    auto_declare("motion.start_step", 0);
    auto_declare("motion.length", 0);
    auto_declare("motion.loop", false);
    auto_declare("motion.time_step_stride", 1);
    auto_declare<bool>("motion.reset.mujoco_reset_on_activate", false);
    auto_declare<std::string>("motion.reset.mujoco_reset_topic", "/softtouch/mujoco_reset");
    auto_declare<double>("motion.reset.mujoco_reset_hold_s", 0.0);
    auto_declare<std::string>("motion.action.command_mode", "rl_controller");
    auto_declare<double>("motion.action.policy_update_period_s", 0.02);
    auto_declare<double>("motion.action.effort_scale", 1.0);
    auto_declare("motion.action.effort_limit", std::vector<double>{});
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  const auto policyPath = get_node()->get_parameter("policy.path").as_string();
  const auto startStepParam = get_node()->get_parameter("motion.start_step").as_int();
  const auto motionLengthParam = get_node()->get_parameter("motion.length").as_int();
  const auto loopMotion = get_node()->get_parameter("motion.loop").as_bool();
  const auto timeStepStrideParam = get_node()->get_parameter("motion.time_step_stride").as_int();
  if (startStepParam < 0 || motionLengthParam < 0 || timeStepStrideParam <= 0) {
    RCLCPP_ERROR(get_node()->get_logger(), "motion.start_step/motion.length must be non-negative and motion.time_step_stride must be positive.");
    return controller_interface::CallbackReturn::ERROR;
  }
  mujocoResetOnActivate_ = get_node()->get_parameter("motion.reset.mujoco_reset_on_activate").as_bool();
  mujocoResetTopic_ = get_node()->get_parameter("motion.reset.mujoco_reset_topic").as_string();
  mujocoResetHoldDuration_ = static_cast<scalar_t>(get_node()->get_parameter("motion.reset.mujoco_reset_hold_s").as_double());
  if (mujocoResetHoldDuration_ < scalar_t(0.0)) {
    RCLCPP_ERROR(get_node()->get_logger(), "motion.reset.mujoco_reset_hold_s must be non-negative.");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (mujocoResetOnActivate_) {
    mujocoResetPub_ = get_node()->create_publisher<std_msgs::msg::Float64>(mujocoResetTopic_, 1);
  } else {
    mujocoResetPub_.reset();
  }
  const auto startStep = static_cast<size_t>(startStepParam);
  const auto motionLength = static_cast<size_t>(motionLengthParam);
  const auto timeStepStride = static_cast<size_t>(timeStepStrideParam);

  policy_ = std::make_shared<MotionOnnxPolicy>(policyPath, startStep, motionLength, loopMotion, timeStepStride);
  policy_->init();

  auto policy = std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_);
  cfg_.anchorBody = policy->getAnchorBodyName();
  cfg_.bodyNames = policy->getBodyNames();
  RCLCPP_INFO_STREAM(rclcpp::get_logger("MotionTrackingController"), "Load Onnx model from " << policyPath << " successfully !");
  RCLCPP_INFO_STREAM(rclcpp::get_logger("MotionTrackingController"),
                     "Motion time_step start=" << startStep << ", length=" << motionLength
                                               << ", loop=" << (loopMotion ? "true" : "false")
                                               << ", stride=" << timeStepStride);
  try {
    configureActionMode();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Motion action config failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  const auto ret = RlController::on_configure(previous_state);
  if (ret != controller_interface::CallbackReturn::SUCCESS) {
    return ret;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  configurePolicyJointMapping();
  hasActionTarget_ = false;
  lastActionPolicyUpdateTime_ = -std::numeric_limits<scalar_t>::infinity();
  if (mujocoResetOnActivate_ && mujocoResetPub_) {
    std_msgs::msg::Float64 msg;
    msg.data = static_cast<double>(mujocoResetHoldDuration_);
    mujocoResetPub_->publish(msg);
    skipPolicyUntilTime_ = static_cast<scalar_t>(get_node()->get_clock()->now().seconds()) + mujocoResetHoldDuration_;
    pendingStartupPolicyReset_ = true;
    RCLCPP_INFO_STREAM(get_node()->get_logger(), "MotionTrackingController requested MuJoCo reset on activate; skipping policy until sim time "
                                                      << skipPolicyUntilTime_);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type MotionTrackingController::update(const rclcpp::Time& time,
                                                                   const rclcpp::Duration& period) {
  if (pendingStartupPolicyReset_ && static_cast<scalar_t>(time.seconds()) < skipPolicyUntilTime_) {
    return ControllerBase::update(time, period);
  }
  if (pendingStartupPolicyReset_) {
    if (policy_) {
      policy_->reset();
    }
    if (commandTerm_) {
      commandTerm_->reset();
    }
    hasActionTarget_ = false;
    lastActionPolicyUpdateTime_ = -std::numeric_limits<scalar_t>::infinity();
    pendingStartupPolicyReset_ = false;
  }
  vector_t policyObs;
  controller_interface::return_type ret = controller_interface::return_type::ERROR;
  try {
    if (actionCommandMode_ == "effort_pd") {
      ret = updateEffortPd(time, period, policyObs);
    } else if (actionCommandMode_ == "position_target") {
      ret = updatePositionTarget(time, period, policyObs);
    } else {
      ret = RlController::update(time, period);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "MotionTrackingController update failed: %s", e.what());
    return controller_interface::return_type::ERROR;
  }
  return ret;
}

bool MotionTrackingController::parserCommand(const std::string& name) {
  if (RlController::parserCommand(name)) {
    return true;
  }
  if (name == "motion") {
    commandTerm_ = std::make_shared<MotionCommandTerm>(cfg_, std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_));
    commandManager_->addTerm(commandTerm_);
    return true;
  }
  return false;
}

bool MotionTrackingController::parserObservation(const std::string& name) {
  if (RlController::parserObservation(name)) {
    return true;
  }
  if (name == "motion_ref_pos_b" || name == "motion_anchor_pos_b") {
    observationManager_->addTerm(std::make_shared<MotionAnchorPosition>(commandTerm_));
  } else if (name == "motion_ref_ori_b" || name == "motion_anchor_ori_b") {
    observationManager_->addTerm(std::make_shared<MotionAnchorOrientation>(commandTerm_));
  } else if (name == "motion_body_pos_error_b") {
    observationManager_->addTerm(std::make_shared<MotionBodyPositionError>(commandTerm_));
  } else if (name == "robot_body_pos") {
    observationManager_->addTerm(std::make_shared<RobotBodyPosition>(commandTerm_));
  } else if (name == "robot_body_ori") {
    observationManager_->addTerm(std::make_shared<RobotBodyOrientation>(commandTerm_));
  } else {
    return false;
  }
  return true;
}

void MotionTrackingController::configurePolicyJointMapping() {
  if (!policy_) {
    return;
  }
  const auto policyJointNames = policy_->getJointNames();
  policyToModelJointIndex_.clear();
  policyToModelJointIndex_.reserve(policyJointNames.size());
  for (const auto& jointName : policyJointNames) {
    policyToModelJointIndex_.push_back(leggedModel()->getJointIndex(jointName));
  }
  policyToControlIndex_.assign(policyJointNames.size(), std::numeric_limits<size_t>::max());
  std::unordered_map<std::string, size_t> controlIndexByName;
  for (size_t i = 0; i < jointNameInControl_.size(); ++i) {
    controlIndexByName.emplace(jointNameInControl_[i], i);
  }
  for (size_t policyIndex = 0; policyIndex < policyJointNames.size(); ++policyIndex) {
    const auto it = controlIndexByName.find(policyJointNames[policyIndex]);
    if (it != controlIndexByName.end()) {
      policyToControlIndex_[policyIndex] = it->second;
    }
  }
  if (actionCommandMode_ == "position_target" || actionCommandMode_ == "effort_pd") {
    for (size_t policyIndex = 0; policyIndex < policyToControlIndex_.size(); ++policyIndex) {
      if (policyToControlIndex_[policyIndex] == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Controller command joint order is missing motion policy joint '" +
                                 policyJointNames[policyIndex] + "'.");
      }
    }
  }
  RCLCPP_INFO_STREAM(get_node()->get_logger(), "Motion action command mode: " << actionCommandMode_);
}

void MotionTrackingController::configureActionMode() {
  if (!policy_) {
    throw std::runtime_error("policy is not configured.");
  }
  actionCommandMode_ = get_node()->get_parameter("motion.action.command_mode").as_string();
  if (actionCommandMode_ != "rl_controller" && actionCommandMode_ != "position_target" && actionCommandMode_ != "effort_pd") {
    throw std::runtime_error("motion.action.command_mode must be 'rl_controller', 'position_target', or 'effort_pd'.");
  }
  actionPolicyPeriod_ = static_cast<scalar_t>(get_node()->get_parameter("motion.action.policy_update_period_s").as_double());
  if (actionPolicyPeriod_ < scalar_t(0.0)) {
    throw std::runtime_error("motion.action.policy_update_period_s must be non-negative.");
  }
  effortLimitScale_ = static_cast<scalar_t>(get_node()->get_parameter("motion.action.effort_scale").as_double());
  if (!(effortLimitScale_ > scalar_t(0.0)) || !std::isfinite(effortLimitScale_)) {
    throw std::runtime_error("motion.action.effort_scale must be finite and positive.");
  }

  effortLimit_ = defaultMotionEffortLimit(policy_->getActionSize());
  const auto effortLimit = getDoubleArrayParam(get_node(), "motion.action.effort_limit", {});
  if (!effortLimit.empty()) {
    if (effortLimit.size() != policy_->getActionSize()) {
      throw std::runtime_error("motion.action.effort_limit must have one value per policy action.");
    }
    effortLimit_.resize(static_cast<Eigen::Index>(effortLimit.size()));
    for (size_t i = 0; i < effortLimit.size(); ++i) {
      effortLimit_(static_cast<Eigen::Index>(i)) = std::abs(static_cast<scalar_t>(effortLimit[i]));
    }
  }
  effortLimit_ *= effortLimitScale_;
  actionTargetPolicyOrder_ = policy_->getDefaultJointPositions();
  hasActionTarget_ = false;
  lastActionPolicyUpdateTime_ = -std::numeric_limits<scalar_t>::infinity();
}

vector_t MotionTrackingController::policyVectorToControlOrder(const vector_t& value) const {
  requireVectorSize(value, static_cast<Eigen::Index>(policyToControlIndex_.size()), "policy vector");
  if (jointNameInControl_.empty()) {
    throw std::runtime_error("Controller command joint order is empty.");
  }
  vector_t out = vector_t::Zero(static_cast<Eigen::Index>(jointNameInControl_.size()));
  for (size_t policyIndex = 0; policyIndex < policyToControlIndex_.size(); ++policyIndex) {
    const size_t controlIndex = policyToControlIndex_[policyIndex];
    if (controlIndex == std::numeric_limits<size_t>::max() || controlIndex >= jointNameInControl_.size()) {
      throw std::runtime_error("Policy to control joint mapping is incomplete.");
    }
    out(static_cast<Eigen::Index>(controlIndex)) = value(static_cast<Eigen::Index>(policyIndex));
  }
  return out;
}

vector_t MotionTrackingController::makeJointTarget(const vector_t& rawAction) const {
  const auto defaultJointPositions = policy_->getDefaultJointPositions();
  const auto actionScale = policy_->getActionScale();
  requireVectorSize(rawAction, static_cast<Eigen::Index>(defaultJointPositions.size()), "rawAction");
  requireVectorSize(actionScale, static_cast<Eigen::Index>(defaultJointPositions.size()), "actionScale");
  return defaultJointPositions + actionScale.cwiseProduct(rawAction);
}

controller_interface::return_type MotionTrackingController::updatePositionTarget(const rclcpp::Time& time,
                                                                                 const rclcpp::Duration& period,
                                                                                 vector_t& policyObs) {
  if (!observationManager_ || !policy_) {
    return controller_interface::return_type::ERROR;
  }
  const auto baseResult = ControllerBase::update(time, period);
  if (baseResult != controller_interface::return_type::OK) {
    return baseResult;
  }

  const scalar_t now = static_cast<scalar_t>(time.seconds());
  const bool shouldUpdatePolicy =
      !hasActionTarget_ || actionPolicyPeriod_ <= scalar_t(0.0) ||
      (now - lastActionPolicyUpdateTime_) >= (actionPolicyPeriod_ - scalar_t(1.0e-9));
  if (shouldUpdatePolicy) {
    policyObs = observationManager_->getValue();
    actionTargetPolicyOrder_ = makeJointTarget(policy_->forward(policyObs));
    lastActionPolicyUpdateTime_ = now;
    hasActionTarget_ = true;
  } else if (policyObs.size() != static_cast<Eigen::Index>(policy_->getObservationSize())) {
    policyObs = observationManager_->getValue();
  }

  const vector_t targetControlOrder = policyVectorToControlOrder(actionTargetPolicyOrder_);
  const vector_t kpControlOrder = policyVectorToControlOrder(policy_->getJointStiffness());
  const vector_t kdControlOrder = policyVectorToControlOrder(policy_->getJointDamping());
  const vector_t zeros = vector_t::Zero(static_cast<Eigen::Index>(jointNameInControl_.size()));
  setPositions(targetControlOrder);
  setVelocities(zeros);
  setStiffnesses(kpControlOrder);
  setDampings(kdControlOrder);
  setEfforts(zeros);
  desiredPosition_ = targetControlOrder;
  return controller_interface::return_type::OK;
}

controller_interface::return_type MotionTrackingController::updateEffortPd(const rclcpp::Time& time,
                                                                           const rclcpp::Duration& period,
                                                                           vector_t& policyObs) {
  if (!observationManager_ || !policy_) {
    return controller_interface::return_type::ERROR;
  }
  const auto baseResult = ControllerBase::update(time, period);
  if (baseResult != controller_interface::return_type::OK) {
    return baseResult;
  }

  const scalar_t now = static_cast<scalar_t>(time.seconds());
  const bool shouldUpdatePolicy =
      !hasActionTarget_ || actionPolicyPeriod_ <= scalar_t(0.0) ||
      (now - lastActionPolicyUpdateTime_) >= (actionPolicyPeriod_ - scalar_t(1.0e-9));
  if (shouldUpdatePolicy) {
    policyObs = observationManager_->getValue();
    actionTargetPolicyOrder_ = makeJointTarget(policy_->forward(policyObs));
    lastActionPolicyUpdateTime_ = now;
    hasActionTarget_ = true;
  } else if (policyObs.size() != static_cast<Eigen::Index>(policy_->getObservationSize())) {
    policyObs = observationManager_->getValue();
  }

  const vector_t q = readPolicyJointPosition();
  const vector_t qd = readPolicyJointVelocity();
  const vector_t kp = policy_->getJointStiffness();
  const vector_t kd = policy_->getJointDamping();
  requireVectorSize(actionTargetPolicyOrder_, q.size(), "actionTargetPolicyOrder");
  requireVectorSize(kp, q.size(), "jointStiffness");
  requireVectorSize(kd, q.size(), "jointDamping");
  requireVectorSize(effortLimit_, q.size(), "effortLimit");

  vector_t tau = kp.cwiseProduct(actionTargetPolicyOrder_ - q) - kd.cwiseProduct(qd);
  for (Eigen::Index i = 0; i < tau.size(); ++i) {
    const scalar_t limit = effortLimit_(i);
    if (std::isfinite(limit) && limit > scalar_t(0.0)) {
      tau(i) = std::max(-limit, std::min(limit, tau(i)));
    }
  }

  const vector_t targetControlOrder = policyVectorToControlOrder(actionTargetPolicyOrder_);
  const vector_t tauControlOrder = policyVectorToControlOrder(tau);
  const vector_t zeros = vector_t::Zero(static_cast<Eigen::Index>(jointNameInControl_.size()));
  setPositions(targetControlOrder);
  setVelocities(zeros);
  setStiffnesses(zeros);
  setDampings(zeros);
  setEfforts(tauControlOrder);
  desiredPosition_ = targetControlOrder;
  return controller_interface::return_type::OK;
}

vector_t MotionTrackingController::readPolicyJointPosition() const {
  vector_t out(static_cast<Eigen::Index>(policyToModelJointIndex_.size()));
  const auto joints = leggedModel()->getGeneralizedPosition().tail(leggedModel()->getJointNames().size());
  for (size_t i = 0; i < policyToModelJointIndex_.size(); ++i) {
    out(static_cast<Eigen::Index>(i)) = joints(static_cast<Eigen::Index>(policyToModelJointIndex_[i]));
  }
  return out;
}

vector_t MotionTrackingController::readPolicyJointVelocity() const {
  vector_t out(static_cast<Eigen::Index>(policyToModelJointIndex_.size()));
  const auto joints = leggedModel()->getGeneralizedVelocity().tail(leggedModel()->getJointNames().size());
  for (size_t i = 0; i < policyToModelJointIndex_.size(); ++i) {
    out(static_cast<Eigen::Index>(i)) = joints(static_cast<Eigen::Index>(policyToModelJointIndex_[i]));
  }
  return out;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)
