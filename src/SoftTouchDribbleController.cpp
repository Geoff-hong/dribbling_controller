#include "motion_tracking_controller/SoftTouchDribbleController.h"

#include "motion_tracking_controller/SoftTouchDribbleObservation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <rclcpp/qos.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace legged {
namespace {

std::vector<double> getDoubleArrayParam(const rclcpp_lifecycle::LifecycleNode::SharedPtr& node, const std::string& name,
                                        const std::vector<double>& fallback) {
  if (!node->has_parameter(name)) {
    return fallback;
  }
  return node->get_parameter(name).as_double_array();
}

geometry_msgs::msg::Point makePoint(const vector3_t& value) {
  geometry_msgs::msg::Point point;
  point.x = value.x();
  point.y = value.y();
  point.z = value.z();
  return point;
}

geometry_msgs::msg::Point makePoint(scalar_t x, scalar_t y, scalar_t z) {
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a) {
  std_msgs::msg::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

vector_t defaultSoftTouchEffortLimit() {
  vector_t out(kSoftTouchDribbleNumJoints);
  out << 88.0, 88.0, 88.0, 139.0, 139.0, 50.0, 88.0, 88.0, 50.0, 139.0, 139.0, 25.0, 25.0, 50.0, 50.0,
      25.0, 25.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0, 5.0;
  return out;
}

}  // namespace

controller_interface::CallbackReturn SoftTouchDribbleController::on_init() {
  if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    auto_declare<std::string>("softtouch.base_name", "pelvis");
    auto_declare<std::string>("softtouch.base_state.source", "model");
    auto_declare<double>("softtouch.base_state.timeout_s", 0.10);
    auto_declare<std::string>("softtouch.base_state.pose_topic", "/softtouch/base/pose");
    auto_declare<std::string>("softtouch.base_state.twist_topic", "/softtouch/base/twist");
    auto_declare<int>("softtouch.seed", 42);
    auto_declare<int>("softtouch.route.cmd_mode", 4);
    auto_declare<double>("softtouch.ball_state.timeout_s", 0.10);
    auto_declare<std::string>("softtouch.ball_state.pose_topic", "/softtouch/ball/pose");
    auto_declare<std::string>("softtouch.ball_state.twist_topic", "/softtouch/ball/twist");
    auto_declare<double>("softtouch.reset.ball_forward_m", 0.65);
    auto_declare<double>("softtouch.reset.ball_z_m", 0.09);
    auto_declare<bool>("softtouch.reset.reset_route_on_activate", true);
    auto_declare<bool>("softtouch.reset.reset_policy_memory_on_activate", true);
    auto_declare<bool>("softtouch.reset.mujoco_reset_on_activate", false);
    auto_declare<std::string>("softtouch.reset.mujoco_reset_topic", "/softtouch/mujoco_reset");
    auto_declare<double>("softtouch.reset.mujoco_reset_hold_s", 0.0);
    auto_declare<std::string>("softtouch.action.command_mode", "rl_controller");
    auto_declare("softtouch.action.effort_limit", std::vector<double>{});
    auto_declare<double>("softtouch.action.policy_update_period_s", 0.02);
    auto_declare<std::string>("softtouch.debug.obs_dump_path", "");
    auto_declare<bool>("softtouch.action.clip_target_to_joint_range", true);
    auto_declare<double>("softtouch.action.joint_limit_factor", 0.9);
    auto_declare("softtouch.action.joint_limit_lower", std::vector<double>{});
    auto_declare("softtouch.action.joint_limit_upper", std::vector<double>{});
    auto_declare<double>("softtouch.route.route_length_m", 20.0);
    auto_declare<double>("softtouch.route.route_seg_len_m", 0.25);
    auto_declare<double>("softtouch.route.route_lookahead_m", 0.8);
    auto_declare<double>("softtouch.route.route_preview_arc_m", 1.0);
    auto_declare<double>("softtouch.route.route_human_kappa_cap", 0.5);
    auto_declare<double>("softtouch.route.route_human_persist", 0.6);
    auto_declare("softtouch.route.route_human_weave_mag", std::vector<double>{0.4, 1.0});
    auto_declare<double>("softtouch.route.route_human_big_prob", 0.09);
    auto_declare("softtouch.route.route_human_big_angle_deg", std::vector<double>{40.0, 180.0});
    auto_declare<double>("softtouch.route.route_kvscale", 0.75);
    auto_declare<double>("softtouch.route.route_vmax", 2.0);
    auto_declare<bool>("softtouch.route.route_lazy_extend", true);
    auto_declare<int>("softtouch.route.route_init_segments", 9);
    auto_declare<int>("softtouch.route.route_extend_chunk", 1);
    auto_declare<int>("softtouch.route.route_extend_ahead_margin_segs", 10);
    auto_declare<bool>("softtouch.visualization.enabled", true);
    auto_declare<std::string>("softtouch.visualization.topic", "/softtouch/dribble/markers");
    auto_declare<std::string>("softtouch.visualization.frame_id", "world");
    auto_declare<double>("softtouch.visualization.publish_period_s", 0.05);
    auto_declare<double>("softtouch.visualization.target_arrow_length", 0.45);
    auto_declare<double>("softtouch.visualization.ball_marker_radius", 0.09);
    auto_declare<int>("softtouch.visualization.route_max_points", 160);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "SoftTouchDribbleController init failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SoftTouchDribbleController::on_configure(
    const rclcpp_lifecycle::State& previous_state) {
  ballPoseSub_.reset();
  ballTwistSub_.reset();
  basePoseSub_.reset();
  baseTwistSub_.reset();
  mujocoResetPub_.reset();
  visualizationPub_.reset();
  commandTerm_.reset();
  commandTermRegistered_ = false;
  lastVisualizationTime_ = rclcpp::Time(0);

  const auto policyPath = get_node()->get_parameter("policy.path").as_string();

  softtouchPolicy_ = std::make_shared<SoftTouchDribbleOnnxPolicy>(policyPath);
  softtouchPolicy_->init();
  policy_ = softtouchPolicy_;

  loadSoftTouchConfig();
  if (visualize_) {
    visualizationPub_ = get_node()->create_publisher<visualization_msgs::msg::MarkerArray>(visualizationTopic_, 1);
  }
  if (mujocoResetOnActivate_) {
    mujocoResetPub_ = get_node()->create_publisher<std_msgs::msg::Float64>(mujocoResetTopic_, 1);
  }

  RCLCPP_INFO_STREAM(get_node()->get_logger(), "Load SoftTouch dribble ONNX model from " << policyPath << " successfully.");
  const auto result = RlController::on_configure(previous_state);
  if (result != controller_interface::CallbackReturn::SUCCESS) {
    return result;
  }
  // v2 history policies: stack per-term observation history (e.g. 10-frame actor
  // history) from ONNX metadata. Empty for v1 -> terms keep their default length 1.
  const auto historyLengths = softtouchPolicy_->getObservationHistoryLengths();
  if (!historyLengths.empty() && observationManager_) {
    observationManager_->setHistoryLengths(historyLengths);
  }
  // Fail fast at configure time if the assembled observation width does not match
  // the ONNX obs input (wrong term set / history layout) instead of throwing later.
  if (observationManager_ &&
      observationManager_->getSize() != softtouchPolicy_->getObservationSize()) {
    RCLCPP_ERROR_STREAM(get_node()->get_logger(),
                        "SoftTouch observation width " << observationManager_->getSize()
                        << " != ONNX obs input " << softtouchPolicy_->getObservationSize()
                        << ". Check observation_names / history metadata vs the policy.");
    return controller_interface::CallbackReturn::ERROR;
  }
  configureJointTargetClip();
  subscribeBallState();
  subscribeBaseState();
  return result;
}

controller_interface::CallbackReturn SoftTouchDribbleController::on_activate(
    const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  try {
    configureJointMappings();
    if (resetPolicyMemoryOnActivate_ && softtouchPolicy_) {
      softtouchPolicy_->reset();
      hasEffortTarget_ = false;
    }
    if (resetRouteOnActivate_) {
      ensureCommandTerm();
      commandTerm_->setNow(static_cast<scalar_t>(get_node()->get_clock()->now().seconds()));
      commandTerm_->reset();
    }
    if (mujocoResetOnActivate_ && mujocoResetPub_) {
      std_msgs::msg::Float64 msg;
      msg.data = static_cast<double>(mujocoResetHoldDuration_);
      mujocoResetPub_->publish(msg);
      skipPolicyUntilTime_ = static_cast<scalar_t>(get_node()->get_clock()->now().seconds()) + mujocoResetHoldDuration_;
      pendingStartupPolicyReset_ = true;
      RCLCPP_INFO_STREAM(get_node()->get_logger(), "SoftTouch requested MuJoCo reset on activate; skipping policy until sim time "
                                                        << skipPolicyUntilTime_);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "SoftTouchDribbleController activation reset failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SoftTouchDribbleController::on_deactivate(
    const rclcpp_lifecycle::State& previous_state) {
  return RlController::on_deactivate(previous_state);
}

controller_interface::return_type SoftTouchDribbleController::update(const rclcpp::Time& time,
                                                                     const rclcpp::Duration& period) {
  if (commandTerm_) {
    commandTerm_->setNow(static_cast<scalar_t>(time.seconds()));
    if (pendingStartupPolicyReset_ && static_cast<scalar_t>(time.seconds()) < skipPolicyUntilTime_) {
      publishVisualization(time);
      return controller_interface::return_type::OK;
    }
    if (pendingStartupPolicyReset_) {
      if (softtouchPolicy_) {
        softtouchPolicy_->reset();
      }
      hasEffortTarget_ = false;
      if (resetRouteOnActivate_) {
        commandTerm_->reset();
      }
      pendingStartupPolicyReset_ = false;
    }
    if (!commandTerm_->hasFreshBallState()) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                           "SoftTouch ball state is missing or stale; using fallback position and/or zero velocity.");
    }
    if (!commandTerm_->hasFreshBaseState()) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                           "SoftTouch base state topic is missing or stale; using StateEstimator model base state.");
    }
    commandTerm_->refreshRouteCommand();
  }
  vector_t policyObs;
  controller_interface::return_type result = controller_interface::return_type::ERROR;
  if (actionCommandMode_ == "effort_pd") {
    result = updateEffortPd(time, period, policyObs);
  } else if (actionCommandMode_ == "position_target") {
    result = updatePositionTarget(time, period, policyObs);
  } else {
    result = RlController::update(time, period);
  }
  publishVisualization(time);
  return result;
}

bool SoftTouchDribbleController::parserCommand(const std::string& name) {
  if (RlController::parserCommand(name)) {
    return true;
  }
  if (name == "softtouch_dribble" || name == "dribble_route") {
    ensureCommandTerm();
    return true;
  }
  return false;
}

bool SoftTouchDribbleController::parserObservation(const std::string& name) {
  if (RlController::parserObservation(name)) {
    return true;
  }

  ensureCommandTerm();
  if (name == "base_ang_vel" || name == "decoder_base_ang_vel") {
    observationManager_->addTerm(std::make_shared<SoftTouchBaseAngularVelocity>(commandTerm_));
  } else if (name == "projected_gravity") {
    observationManager_->addTerm(std::make_shared<SoftTouchProjectedGravity>(commandTerm_));
  } else if (name == "joint_pos" || name == "decoder_joint_pos") {
    observationManager_->addTerm(
        std::make_shared<SoftTouchJointPosition>(softtouchPolicy_->getJointNames(), softtouchPolicy_->getDefaultJointPosition()));
  } else if (name == "joint_vel" || name == "decoder_joint_vel") {
    observationManager_->addTerm(std::make_shared<SoftTouchJointVelocity>(softtouchPolicy_->getJointNames()));
  } else if (name == "last_latent_action") {
    observationManager_->addTerm(std::make_shared<SoftTouchLastLatentAction>(commandTerm_));
  } else if (name == "ball_pos_b") {
    observationManager_->addTerm(std::make_shared<SoftTouchBallPositionBody>(commandTerm_));
  } else if (name == "ball_lin_vel_b") {
    observationManager_->addTerm(std::make_shared<SoftTouchBallLinearVelocityBody>(commandTerm_));
  } else if (name == "ball_radius") {
    // Deployed ball rests at z = radius, so cfg_.resetBallZ is the ball radius.
    observationManager_->addTerm(std::make_shared<SoftTouchBallRadius>(cfg_.resetBallZ));
  } else if (name == "target_dir_b") {
    observationManager_->addTerm(std::make_shared<SoftTouchTargetDirectionBody>(commandTerm_));
  } else if (name == "target_speed") {
    observationManager_->addTerm(std::make_shared<SoftTouchTargetSpeed>(commandTerm_));
  } else if (name == "cmd_dir_w") {
    observationManager_->addTerm(std::make_shared<SoftTouchCommandDirectionWorld>(commandTerm_));
  } else if (name == "next_cmd_dir_w") {
    observationManager_->addTerm(std::make_shared<SoftTouchNextCommandDirectionWorld>(commandTerm_));
  } else if (name == "next_target_speed") {
    observationManager_->addTerm(std::make_shared<SoftTouchNextTargetSpeed>(commandTerm_));
  } else if (name == "pelvis_pos_xy_w") {
    observationManager_->addTerm(std::make_shared<SoftTouchPelvisPositionXyWorld>(commandTerm_));
  } else if (name == "pelvis_yaw_cossin_w") {
    observationManager_->addTerm(std::make_shared<SoftTouchPelvisYawCosSinWorld>(commandTerm_));
  } else if (name == "last_decoded_action") {
    observationManager_->addTerm(std::make_shared<SoftTouchLastDecodedAction>(commandTerm_));
  } else {
    return false;
  }
  return true;
}

void SoftTouchDribbleController::loadSoftTouchConfig() {
  const auto node = get_node();
  cfg_.baseName = node->get_parameter("softtouch.base_name").as_string();
  cfg_.baseStateSource = node->get_parameter("softtouch.base_state.source").as_string();
  cfg_.baseTimeout = node->get_parameter("softtouch.base_state.timeout_s").as_double();
  cfg_.seed = static_cast<uint32_t>(std::max<int64_t>(0, node->get_parameter("softtouch.seed").as_int()));
  cfg_.cmdMode = static_cast<int>(node->get_parameter("softtouch.route.cmd_mode").as_int());
  cfg_.ballTimeout = node->get_parameter("softtouch.ball_state.timeout_s").as_double();
  cfg_.resetBallForward = node->get_parameter("softtouch.reset.ball_forward_m").as_double();
  cfg_.resetBallZ = node->get_parameter("softtouch.reset.ball_z_m").as_double();
  basePoseTopic_ = node->get_parameter("softtouch.base_state.pose_topic").as_string();
  baseTwistTopic_ = node->get_parameter("softtouch.base_state.twist_topic").as_string();
  resetRouteOnActivate_ = node->get_parameter("softtouch.reset.reset_route_on_activate").as_bool();
  resetPolicyMemoryOnActivate_ = node->get_parameter("softtouch.reset.reset_policy_memory_on_activate").as_bool();
  mujocoResetOnActivate_ = node->get_parameter("softtouch.reset.mujoco_reset_on_activate").as_bool();
  mujocoResetTopic_ = node->get_parameter("softtouch.reset.mujoco_reset_topic").as_string();
  mujocoResetHoldDuration_ = node->get_parameter("softtouch.reset.mujoco_reset_hold_s").as_double();
  clipJointTarget_ = node->get_parameter("softtouch.action.clip_target_to_joint_range").as_bool();
  jointTargetLimitFactor_ = node->get_parameter("softtouch.action.joint_limit_factor").as_double();
  if (!(jointTargetLimitFactor_ > scalar_t(0.0) && jointTargetLimitFactor_ <= scalar_t(1.0))) {
    throw std::runtime_error("softtouch.action.joint_limit_factor must be in (0, 1].");
  }
  actionCommandMode_ = node->get_parameter("softtouch.action.command_mode").as_string();
  if (actionCommandMode_ != "rl_controller" && actionCommandMode_ != "position_target" && actionCommandMode_ != "effort_pd") {
    throw std::runtime_error("softtouch.action.command_mode must be 'rl_controller', 'position_target', or 'effort_pd'.");
  }
  actionPolicyPeriod_ = node->get_parameter("softtouch.action.policy_update_period_s").as_double();
  obsDumpPath_ = node->get_parameter("softtouch.debug.obs_dump_path").as_string();
  obsDump_.reset();
  obsDumpSeq_ = 0;
  if (actionPolicyPeriod_ < scalar_t(0.0)) {
    throw std::runtime_error("softtouch.action.policy_update_period_s must be non-negative.");
  }
  effortLimit_ = defaultSoftTouchEffortLimit();
  effortTargetPolicyOrder_ = vector_t::Zero(kSoftTouchDribbleNumJoints);
  hasEffortTarget_ = false;
  lastEffortPolicyUpdateTime_ = -std::numeric_limits<scalar_t>::infinity();
  const auto effortLimit = getDoubleArrayParam(node, "softtouch.action.effort_limit", {});
  if (!effortLimit.empty()) {
    if (effortLimit.size() != kSoftTouchDribbleNumJoints) {
      throw std::runtime_error("softtouch.action.effort_limit must have 29 entries when set.");
    }
    for (size_t i = 0; i < effortLimit.size(); ++i) {
      effortLimit_(static_cast<Eigen::Index>(i)) = std::abs(effortLimit[i]);
    }
  }
  cfg_.route.routeLength = node->get_parameter("softtouch.route.route_length_m").as_double();
  cfg_.route.routeSegmentLength = node->get_parameter("softtouch.route.route_seg_len_m").as_double();
  cfg_.route.routeLookahead = node->get_parameter("softtouch.route.route_lookahead_m").as_double();
  cfg_.route.routePreviewArc = node->get_parameter("softtouch.route.route_preview_arc_m").as_double();
  cfg_.route.routeHumanKappaCap = node->get_parameter("softtouch.route.route_human_kappa_cap").as_double();
  cfg_.route.routeHumanPersist = node->get_parameter("softtouch.route.route_human_persist").as_double();
  const auto weave = getDoubleArrayParam(node, "softtouch.route.route_human_weave_mag", {0.4, 1.0});
  if (weave.size() >= 2) {
    cfg_.route.routeHumanWeaveMin = weave[0];
    cfg_.route.routeHumanWeaveMax = weave[1];
  }
  cfg_.route.routeHumanBigProbability = node->get_parameter("softtouch.route.route_human_big_prob").as_double();
  const auto bigAngle = getDoubleArrayParam(node, "softtouch.route.route_human_big_angle_deg", {40.0, 180.0});
  if (bigAngle.size() >= 2) {
    cfg_.route.routeHumanBigAngleMinDeg = bigAngle[0];
    cfg_.route.routeHumanBigAngleMaxDeg = bigAngle[1];
  }
  cfg_.route.routeKvScale = node->get_parameter("softtouch.route.route_kvscale").as_double();
  cfg_.route.routeVmax = node->get_parameter("softtouch.route.route_vmax").as_double();
  cfg_.route.routeLazyExtend = node->get_parameter("softtouch.route.route_lazy_extend").as_bool();
  cfg_.route.routeInitSegments = static_cast<int>(node->get_parameter("softtouch.route.route_init_segments").as_int());
  cfg_.route.routeExtendChunk = static_cast<int>(node->get_parameter("softtouch.route.route_extend_chunk").as_int());
  cfg_.route.routeExtendAheadMarginSegments =
      static_cast<int>(node->get_parameter("softtouch.route.route_extend_ahead_margin_segs").as_int());
  ballPoseTopic_ = node->get_parameter("softtouch.ball_state.pose_topic").as_string();
  ballTwistTopic_ = node->get_parameter("softtouch.ball_state.twist_topic").as_string();
  visualize_ = node->get_parameter("softtouch.visualization.enabled").as_bool();
  visualizationTopic_ = node->get_parameter("softtouch.visualization.topic").as_string();
  visualizationFrameId_ = node->get_parameter("softtouch.visualization.frame_id").as_string();
  visualizationPeriod_ = node->get_parameter("softtouch.visualization.publish_period_s").as_double();
  targetArrowLength_ = node->get_parameter("softtouch.visualization.target_arrow_length").as_double();
  ballMarkerRadius_ = node->get_parameter("softtouch.visualization.ball_marker_radius").as_double();
  routeMaxPoints_ = static_cast<int>(node->get_parameter("softtouch.visualization.route_max_points").as_int());
}

void SoftTouchDribbleController::configureJointMappings() {
  if (!softtouchPolicy_) {
    throw std::runtime_error("SoftTouch policy is not configured.");
  }
  const auto policyJointNames = softtouchPolicy_->getJointNames();
  if (policyJointNames.size() != kSoftTouchDribbleNumJoints) {
    throw std::runtime_error("SoftTouch policy joint order must have 29 joints.");
  }
  if (jointNameInControl_.empty()) {
    throw std::runtime_error("Controller command joint order is empty after RlController configure.");
  }

  std::unordered_map<std::string, size_t> policyIndexByName;
  for (size_t i = 0; i < policyJointNames.size(); ++i) {
    policyIndexByName.emplace(policyJointNames[i], i);
  }

  policyToControlIndex_.assign(kSoftTouchDribbleNumJoints, std::numeric_limits<size_t>::max());
  for (size_t controlIndex = 0; controlIndex < jointNameInControl_.size(); ++controlIndex) {
    const auto it = policyIndexByName.find(jointNameInControl_[controlIndex]);
    if (it == policyIndexByName.end()) {
      continue;
    }
    policyToControlIndex_[it->second] = controlIndex;
  }
  for (size_t policyIndex = 0; policyIndex < policyToControlIndex_.size(); ++policyIndex) {
    if (policyToControlIndex_[policyIndex] == std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("Controller command joint order is missing SoftTouch joint '" + policyJointNames[policyIndex] + "'.");
    }
  }

  policyToModelJointIndex_.resize(kSoftTouchDribbleNumJoints);
  const auto model = leggedModel();
  for (size_t i = 0; i < policyJointNames.size(); ++i) {
    policyToModelJointIndex_[i] = model->getJointIndex(policyJointNames[i]);
  }
  RCLCPP_INFO_STREAM(get_node()->get_logger(), "SoftTouch action command mode: " << actionCommandMode_);
}

vector_t SoftTouchDribbleController::policyVectorToControlOrder(const vector_t& value) const {
  if (value.size() != static_cast<Eigen::Index>(kSoftTouchDribbleNumJoints)) {
    throw std::runtime_error("SoftTouch policy vector has unexpected size.");
  }
  vector_t out(kSoftTouchDribbleNumJoints);
  for (size_t policyIndex = 0; policyIndex < kSoftTouchDribbleNumJoints; ++policyIndex) {
    out(static_cast<Eigen::Index>(policyToControlIndex_[policyIndex])) = value(static_cast<Eigen::Index>(policyIndex));
  }
  return out;
}

vector_t SoftTouchDribbleController::readPolicyJointPosition() const {
  vector_t out(kSoftTouchDribbleNumJoints);
  const auto joints = leggedModel()->getGeneralizedPosition().tail(leggedModel()->getJointNames().size());
  for (size_t i = 0; i < kSoftTouchDribbleNumJoints; ++i) {
    out(static_cast<Eigen::Index>(i)) = joints(static_cast<Eigen::Index>(policyToModelJointIndex_[i]));
  }
  return out;
}

vector_t SoftTouchDribbleController::readPolicyJointVelocity() const {
  vector_t out(kSoftTouchDribbleNumJoints);
  const auto joints = leggedModel()->getGeneralizedVelocity().tail(leggedModel()->getJointNames().size());
  for (size_t i = 0; i < kSoftTouchDribbleNumJoints; ++i) {
    out(static_cast<Eigen::Index>(i)) = joints(static_cast<Eigen::Index>(policyToModelJointIndex_[i]));
  }
  return out;
}

controller_interface::return_type SoftTouchDribbleController::updatePositionTarget(const rclcpp::Time& time,
                                                                                   const rclcpp::Duration& period,
                                                                                   vector_t& policyObs) {
  if (!observationManager_ || !softtouchPolicy_) {
    return controller_interface::return_type::ERROR;
  }

  const auto baseResult = ControllerBase::update(time, period);
  if (baseResult != controller_interface::return_type::OK) {
    return baseResult;
  }

  const scalar_t now = static_cast<scalar_t>(time.seconds());
  const bool shouldUpdatePolicy =
      !hasEffortTarget_ || actionPolicyPeriod_ <= scalar_t(0.0) ||
      (now - lastEffortPolicyUpdateTime_) >= (actionPolicyPeriod_ - scalar_t(1.0e-9));
  if (shouldUpdatePolicy) {
    policyObs = observationManager_->getValue();
    effortTargetPolicyOrder_ = softtouchPolicy_->forward(policyObs);
    lastEffortPolicyUpdateTime_ = now;
    hasEffortTarget_ = true;
    if (!obsDumpPath_.empty()) {
      if (!obsDump_) {
        obsDump_ = std::make_shared<std::ofstream>(obsDumpPath_, std::ios::trunc);
        obsDump_->precision(17);
      }
      const vector3_t ball = commandTerm_->getBallPositionWorld();
      const vector3_t ballVel = commandTerm_->getBallLinearVelocityWorld();
      const vector3_t pelvis = commandTerm_->getPelvisPositionWorld();
      const quaternion_t pq = commandTerm_->getPelvisOrientationWorld();
      const vector3_t bav = commandTerm_->getBaseAngularVelocityBody();
      *obsDump_ << obsDumpSeq_++ << " " << now << " "
                << commandTerm_->getBallPositionStamp() << " "
                << commandTerm_->getBasePoseStamp() << " " << ball.transpose() << " "
                << ballVel.transpose() << " " << pelvis.transpose() << " "
                << pq.w() << " " << pq.x() << " " << pq.y() << " " << pq.z() << " "
                << bav.transpose() << " | " << policyObs.transpose() << " | "
                << effortTargetPolicyOrder_.transpose() << "\n";
      obsDump_->flush();
    }
  } else if (policyObs.size() != static_cast<Eigen::Index>(softtouchPolicy_->getObservationSize())) {
    policyObs = observationManager_->getValue();
  }

  const vector_t targetControlOrder = policyVectorToControlOrder(effortTargetPolicyOrder_);
  const vector_t kpControlOrder = policyVectorToControlOrder(softtouchPolicy_->getJointStiffness());
  const vector_t kdControlOrder = policyVectorToControlOrder(softtouchPolicy_->getJointDamping());
  const vector_t zeros = vector_t::Zero(static_cast<Eigen::Index>(kSoftTouchDribbleNumJoints));
  setPositions(targetControlOrder);
  setVelocities(zeros);
  setStiffnesses(kpControlOrder);
  setDampings(kdControlOrder);
  setEfforts(zeros);
  desiredPosition_ = targetControlOrder;
  return controller_interface::return_type::OK;
}

controller_interface::return_type SoftTouchDribbleController::updateEffortPd(const rclcpp::Time& time,
                                                                             const rclcpp::Duration& period,
                                                                             vector_t& policyObs) {
  if (!observationManager_ || !softtouchPolicy_) {
    return controller_interface::return_type::ERROR;
  }

  const auto baseResult = ControllerBase::update(time, period);
  if (baseResult != controller_interface::return_type::OK) {
    return baseResult;
  }

  const scalar_t now = static_cast<scalar_t>(time.seconds());
  const bool shouldUpdatePolicy =
      !hasEffortTarget_ || actionPolicyPeriod_ <= scalar_t(0.0) ||
      (now - lastEffortPolicyUpdateTime_) >= (actionPolicyPeriod_ - scalar_t(1.0e-9));
  if (shouldUpdatePolicy) {
    policyObs = observationManager_->getValue();
    effortTargetPolicyOrder_ = softtouchPolicy_->forward(policyObs);
    lastEffortPolicyUpdateTime_ = now;
    hasEffortTarget_ = true;
  } else if (policyObs.size() != static_cast<Eigen::Index>(softtouchPolicy_->getObservationSize())) {
    policyObs = observationManager_->getValue();
  }
  const vector_t& targetPolicyOrder = effortTargetPolicyOrder_;
  const vector_t q = readPolicyJointPosition();
  const vector_t qd = readPolicyJointVelocity();
  const vector_t kp = softtouchPolicy_->getJointStiffness();
  const vector_t kd = softtouchPolicy_->getJointDamping();

  vector_t tau = kp.cwiseProduct(targetPolicyOrder - q) - kd.cwiseProduct(qd);
  for (Eigen::Index i = 0; i < tau.size(); ++i) {
    const scalar_t limit = effortLimit_(i);
    if (std::isfinite(limit) && limit > scalar_t(0.0)) {
      tau(i) = std::max(-limit, std::min(limit, tau(i)));
    }
  }

  const vector_t targetControlOrder = policyVectorToControlOrder(targetPolicyOrder);
  const vector_t tauControlOrder = policyVectorToControlOrder(tau);
  const vector_t zeros = vector_t::Zero(static_cast<Eigen::Index>(kSoftTouchDribbleNumJoints));
  setPositions(targetControlOrder);
  setVelocities(zeros);
  setStiffnesses(zeros);
  setDampings(zeros);
  setEfforts(tauControlOrder);
  desiredPosition_ = targetControlOrder;
  return controller_interface::return_type::OK;
}

void SoftTouchDribbleController::configureJointTargetClip() {
  if (!softtouchPolicy_) {
    return;
  }
  if (!clipJointTarget_) {
    softtouchPolicy_->disableJointTargetClip();
    RCLCPP_INFO(get_node()->get_logger(), "SoftTouch joint target clipping disabled by parameter.");
    return;
  }

  const auto& jointNames = softtouchPolicy_->getJointNames();
  vector_t lower = vector_t::Constant(static_cast<Eigen::Index>(jointNames.size()), std::numeric_limits<scalar_t>::quiet_NaN());
  vector_t upper = vector_t::Constant(static_cast<Eigen::Index>(jointNames.size()), std::numeric_limits<scalar_t>::quiet_NaN());
  const auto paramLower = getDoubleArrayParam(get_node(), "softtouch.action.joint_limit_lower", {});
  const auto paramUpper = getDoubleArrayParam(get_node(), "softtouch.action.joint_limit_upper", {});

  size_t usableLimits = 0;
  if (!paramLower.empty() || !paramUpper.empty()) {
    if (paramLower.size() != jointNames.size() || paramUpper.size() != jointNames.size()) {
      throw std::runtime_error("softtouch.action.joint_limit_lower/upper must both have one entry per policy joint.");
    }
    for (size_t i = 0; i < jointNames.size(); ++i) {
      const scalar_t lo = static_cast<scalar_t>(paramLower[i]);
      const scalar_t hi = static_cast<scalar_t>(paramUpper[i]);
      if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) {
        RCLCPP_WARN_STREAM(get_node()->get_logger(), "SoftTouch parameter joint target limit for " << jointNames[i]
                                                                                                    << " is invalid; this joint will not be clipped.");
        continue;
      }
      lower(static_cast<Eigen::Index>(i)) = lo;
      upper(static_cast<Eigen::Index>(i)) = hi;
      ++usableLimits;
    }
    if (usableLimits == 0) {
      softtouchPolicy_->disableJointTargetClip();
      RCLCPP_WARN(get_node()->get_logger(),
                  "SoftTouch joint target clipping requested, but no finite parameter position limits were available.");
      return;
    }
    softtouchPolicy_->setJointTargetClip(lower, upper, jointTargetLimitFactor_);
    RCLCPP_INFO(get_node()->get_logger(),
                "SoftTouch joint target clipping enabled from parameters for %zu/%zu joints with factor %.3f.",
                usableLimits, jointNames.size(), static_cast<double>(jointTargetLimitFactor_));
    return;
  }

  const auto& hardLimits = get_hard_joint_limits();
  for (size_t i = 0; i < jointNames.size(); ++i) {
    const auto limitIt = hardLimits.find(jointNames[i]);
    if (limitIt == hardLimits.end() || !limitIt->second.has_position_limits ||
        !std::isfinite(limitIt->second.min_position) || !std::isfinite(limitIt->second.max_position) ||
        !(limitIt->second.min_position < limitIt->second.max_position)) {
      RCLCPP_WARN_STREAM(get_node()->get_logger(), "SoftTouch joint target clipping has no finite position limit for "
                                                    << jointNames[i] << "; this joint will not be clipped.");
      continue;
    }
    lower(static_cast<Eigen::Index>(i)) = static_cast<scalar_t>(limitIt->second.min_position);
    upper(static_cast<Eigen::Index>(i)) = static_cast<scalar_t>(limitIt->second.max_position);
    ++usableLimits;
  }

  if (usableLimits == 0) {
    softtouchPolicy_->disableJointTargetClip();
    RCLCPP_WARN(get_node()->get_logger(),
                "SoftTouch joint target clipping requested, but no finite ROS2 control position limits were available.");
    return;
  }

  softtouchPolicy_->setJointTargetClip(lower, upper, jointTargetLimitFactor_);
  RCLCPP_INFO(get_node()->get_logger(), "SoftTouch joint target clipping enabled for %zu/%zu joints with factor %.3f.",
              usableLimits, jointNames.size(), static_cast<double>(jointTargetLimitFactor_));
}

void SoftTouchDribbleController::ensureCommandTerm() {
  if (!commandTerm_) {
    commandTerm_ = std::make_shared<SoftTouchDribbleCommandTerm>(cfg_, softtouchPolicy_);
  }
  if (!commandTermRegistered_) {
    commandManager_->addTerm(commandTerm_);
    commandTermRegistered_ = true;
  }
}

void SoftTouchDribbleController::subscribeBallState() {
  const auto qos = rclcpp::SensorDataQoS();
  ballPoseSub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      ballPoseTopic_, qos,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { handleBallPose(msg); });
  ballTwistSub_ = get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
      ballTwistTopic_, qos,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) { handleBallTwist(msg); });
}

void SoftTouchDribbleController::subscribeBaseState() {
  if (cfg_.baseStateSource != "topic") {
    return;
  }
  const auto qos = rclcpp::SensorDataQoS();
  basePoseSub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      basePoseTopic_, qos,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { handleBasePose(msg); });
  baseTwistSub_ = get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
      baseTwistTopic_, qos,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) { handleBaseTwist(msg); });
  RCLCPP_INFO_STREAM(get_node()->get_logger(), "SoftTouch base state uses topic pose=" << basePoseTopic_
                                                                                        << " twist=" << baseTwistTopic_);
}

void SoftTouchDribbleController::handleBallPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  if (!commandTerm_) {
    return;
  }
  const vector3_t position(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  commandTerm_->setBallPosition(position, stampToSeconds(msg->header.stamp));
}

void SoftTouchDribbleController::handleBallTwist(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  if (!commandTerm_) {
    return;
  }
  const vector3_t velocity(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
  commandTerm_->setBallVelocity(velocity, stampToSeconds(msg->header.stamp));
}

void SoftTouchDribbleController::handleBasePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  if (!commandTerm_) {
    return;
  }
  const vector3_t position(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  const quaternion_t orientation(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y,
                                 msg->pose.orientation.z);
  commandTerm_->setBasePose(position, orientation, stampToSeconds(msg->header.stamp));
}

void SoftTouchDribbleController::handleBaseTwist(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  if (!commandTerm_) {
    return;
  }
  const vector3_t angularVelocityBody(msg->twist.angular.x, msg->twist.angular.y, msg->twist.angular.z);
  commandTerm_->setBaseAngularVelocity(angularVelocityBody, stampToSeconds(msg->header.stamp));
}

scalar_t SoftTouchDribbleController::stampToSeconds(const builtin_interfaces::msg::Time& stamp) const {
  if (stamp.sec == 0 && stamp.nanosec == 0) {
    return static_cast<scalar_t>(get_node()->get_clock()->now().seconds());
  }
  return static_cast<scalar_t>(stamp.sec) + static_cast<scalar_t>(stamp.nanosec) * scalar_t(1.0e-9);
}

void SoftTouchDribbleController::publishVisualization(const rclcpp::Time& time) {
  if (!visualize_ || !visualizationPub_ || !commandTerm_) {
    return;
  }
  if (visualizationPeriod_ > scalar_t(0.0) && lastVisualizationTime_.nanoseconds() > 0 &&
      (time - lastVisualizationTime_).seconds() < visualizationPeriod_) {
    return;
  }
  lastVisualizationTime_ = time;

  const vector3_t ball = commandTerm_->getBallPositionWorld();
  const auto command = commandTerm_->getRouteCommand();

  visualization_msgs::msg::Marker ballMarker;
  ballMarker.header.frame_id = visualizationFrameId_;
  ballMarker.header.stamp = time;
  ballMarker.ns = "softtouch_dribble";
  ballMarker.id = 0;
  ballMarker.type = visualization_msgs::msg::Marker::SPHERE;
  ballMarker.action = visualization_msgs::msg::Marker::ADD;
  ballMarker.pose.position = makePoint(ball);
  ballMarker.pose.orientation.w = 1.0;
  ballMarker.scale.x = 2.0 * ballMarkerRadius_;
  ballMarker.scale.y = 2.0 * ballMarkerRadius_;
  ballMarker.scale.z = 2.0 * ballMarkerRadius_;
  ballMarker.color = makeColor(1.0F, 0.45F, 0.0F, 0.9F);

  visualization_msgs::msg::Marker routeMarker;
  routeMarker.header.frame_id = visualizationFrameId_;
  routeMarker.header.stamp = time;
  routeMarker.ns = "softtouch_dribble";
  routeMarker.id = 1;
  routeMarker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  routeMarker.action = visualization_msgs::msg::Marker::ADD;
  routeMarker.pose.orientation.w = 1.0;
  routeMarker.scale.x = 0.035;
  routeMarker.color = makeColor(0.1F, 0.8F, 1.0F, 0.85F);
  const auto& points = commandTerm_->getRoutePoints();
  const int filledSegments = std::min(commandTerm_->getRouteFilledSegments(), static_cast<int>(points.size()) - 1);
  const int pointCount = std::max(0, std::min(filledSegments + 1, std::max(2, routeMaxPoints_)));
  routeMarker.points.reserve(static_cast<size_t>(pointCount));
  for (int i = 0; i < pointCount; ++i) {
    const auto& point = points[static_cast<size_t>(i)];
    routeMarker.points.push_back(makePoint(point.x(), point.y(), ballMarkerRadius_));
  }

  auto makeArrow = [&](int id, const vector2_t& dir, scalar_t z, const std_msgs::msg::ColorRGBA& color) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = visualizationFrameId_;
    marker.header.stamp = time;
    marker.ns = "softtouch_dribble";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.035;
    marker.scale.y = 0.085;
    marker.scale.z = 0.085;
    marker.color = color;
    marker.points.push_back(makePoint(ball.x(), ball.y(), z));
    marker.points.push_back(makePoint(ball.x() + targetArrowLength_ * dir.x(), ball.y() + targetArrowLength_ * dir.y(), z));
    return marker;
  };

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(ballMarker);
  markers.markers.push_back(routeMarker);
  markers.markers.push_back(makeArrow(2, command.targetDirWorld, ball.z() + 0.18, makeColor(0.1F, 1.0F, 0.25F, 0.95F)));
  markers.markers.push_back(
      makeArrow(3, command.nextTargetDirWorld, ball.z() + 0.28, makeColor(1.0F, 0.85F, 0.05F, 0.85F)));
  visualizationPub_->publish(markers);
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::SoftTouchDribbleController, controller_interface::ControllerInterface)
