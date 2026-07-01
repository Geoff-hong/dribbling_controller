#include "motion_tracking_controller/SoftTouchMujocoBallBridgePlugin.h"

#include <builtin_interfaces/msg/time.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/qos.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace legged {
namespace {

constexpr const char* kParamPrefix = "softtouch_mujoco_ball_bridge.";
// = 4 * ball_rotational_inertia; reproduces PhysX angular_damping=4.0 spin-decay rate.
// (MuJoCo dof_damping is a torque coeff, not PhysX's 1/s rate.) See controllers.yaml.
// 2026-06-17 ball r=0.10/m=0.391 -> 4*I=0.006256 (was 0.0044064 for 0.09/0.34).
constexpr double kDefaultMujocoBallAngularDamping = 0.006256;

bool readBoolParam(const rclcpp::Node::SharedPtr& node, const std::string& name, bool fallback) {
  if (!node->has_parameter(name)) {
    node->declare_parameter<bool>(name, fallback);
  }
  return node->get_parameter(name).as_bool();
}

double readDoubleParam(const rclcpp::Node::SharedPtr& node, const std::string& name, double fallback) {
  if (!node->has_parameter(name)) {
    node->declare_parameter<double>(name, fallback);
  }
  return node->get_parameter(name).as_double();
}

std::string readStringParam(const rclcpp::Node::SharedPtr& node, const std::string& name, const std::string& fallback) {
  if (!node->has_parameter(name)) {
    node->declare_parameter<std::string>(name, fallback);
  }
  return node->get_parameter(name).as_string();
}

builtin_interfaces::msg::Time stampFromSeconds(double seconds) {
  const auto nanoseconds = static_cast<int64_t>(std::llround(std::max(0.0, seconds) * 1.0e9));
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / 1000000000LL);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000LL);
  return stamp;
}

void requireFinite(double value, const std::string& name) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("SoftTouch MuJoCo ball bridge read non-finite " + name + ".");
  }
}

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<double> readDoubles(std::istringstream& stream, size_t expected, const std::string& key) {
  std::vector<double> values;
  values.reserve(expected);
  double value = 0.0;
  while (stream >> value) {
    values.push_back(value);
  }
  if (values.size() != expected) {
    throw std::runtime_error("SoftTouch MuJoCo reset key '" + key + "' has " + std::to_string(values.size()) +
                             " values, expected " + std::to_string(expected) + ".");
  }
  for (double entry : values) {
    requireFinite(entry, key);
  }
  return values;
}

std::vector<std::string> readStrings(std::istringstream& stream, const std::string& key) {
  std::vector<std::string> values{std::istream_iterator<std::string>{stream}, std::istream_iterator<std::string>{}};
  if (values.empty()) {
    throw std::runtime_error("SoftTouch MuJoCo reset key '" + key + "' is empty.");
  }
  return values;
}

void requireVectorSet(const std::vector<double>& values, size_t expected, const std::string& key) {
  if (values.size() != expected) {
    throw std::runtime_error("SoftTouch MuJoCo reset missing or invalid key '" + key + "'.");
  }
}

bool isNoneName(const std::string& value) {
  return value.empty() || value == "none" || value == "None" || value == "NONE" || value == "__none__" || value == "-";
}

}  // namespace

void SoftTouchMujocoBallBridgePlugin::Configure(rclcpp::Node::SharedPtr& node, rclcpp::NodeOptions cm_node_option,
                                                mjModel* model, mjData* data) {
  (void)cm_node_option;
  node_ = node;

  enabled_ = readBoolParam(node_, std::string(kParamPrefix) + "enabled", false);
  frameId_ = readStringParam(node_, std::string(kParamPrefix) + "frame_id", frameId_);
  poseTopic_ = readStringParam(node_, std::string(kParamPrefix) + "pose_topic", poseTopic_);
  twistTopic_ = readStringParam(node_, std::string(kParamPrefix) + "twist_topic", twistTopic_);
  ballJointName_ = readStringParam(node_, std::string(kParamPrefix) + "ball_joint", ballJointName_);
  positionSensorName_ = readStringParam(node_, std::string(kParamPrefix) + "position_sensor", positionSensorName_);
  quaternionSensorName_ = readStringParam(node_, std::string(kParamPrefix) + "quaternion_sensor", quaternionSensorName_);
  linearVelocitySensorName_ =
      readStringParam(node_, std::string(kParamPrefix) + "linear_velocity_sensor", linearVelocitySensorName_);
  angularVelocitySensorName_ =
      readStringParam(node_, std::string(kParamPrefix) + "angular_velocity_sensor", angularVelocitySensorName_);
  publishBallStateEnabled_ =
      readBoolParam(node_, std::string(kParamPrefix) + "publish_ball_state", publishBallStateEnabled_);
  applyBallDampingEnabled_ =
      readBoolParam(node_, std::string(kParamPrefix) + "apply_ball_damping", publishBallStateEnabled_);
  publishBaseState_ = readBoolParam(node_, std::string(kParamPrefix) + "publish_base_state", publishBaseState_);
  baseJointName_ = readStringParam(node_, std::string(kParamPrefix) + "base_joint", baseJointName_);
  basePoseTopic_ = readStringParam(node_, std::string(kParamPrefix) + "base_pose_topic", basePoseTopic_);
  baseTwistTopic_ = readStringParam(node_, std::string(kParamPrefix) + "base_twist_topic", baseTwistTopic_);
  const double publishRate = readDoubleParam(node_, std::string(kParamPrefix) + "publish_rate_hz", 100.0);
  publishPeriod_ = publishRate > 0.0 ? 1.0 / publishRate : 0.0;
  ballTranslationalDamping_ = readDoubleParam(node_, std::string(kParamPrefix) + "ball_translational_damping", 0.0);
  ballAngularDamping_ =
      readDoubleParam(node_, std::string(kParamPrefix) + "ball_angular_damping", kDefaultMujocoBallAngularDamping);
  resetEnabled_ = readBoolParam(node_, std::string(kParamPrefix) + "reset.enabled", false);
  resetStatePath_ = readStringParam(node_, std::string(kParamPrefix) + "reset.state_file", "");
  resetApplyOnConfigure_ = readBoolParam(node_, std::string(kParamPrefix) + "reset.apply_on_configure", true);
  resetApplyOnReset_ = readBoolParam(node_, std::string(kParamPrefix) + "reset.apply_on_reset", true);
  resetZeroVelocity_ = readBoolParam(node_, std::string(kParamPrefix) + "reset.zero_velocity", false);
  resetHoldUntilTime_ = readDoubleParam(node_, std::string(kParamPrefix) + "reset.hold_until_time_s", 0.0);
  resetRequestTopic_ = readStringParam(node_, std::string(kParamPrefix) + "reset.request_topic", resetRequestTopic_);
  resetHoldLogged_ = false;

  if (!enabled_) {
    RCLCPP_INFO(node_->get_logger(), "SoftTouch MuJoCo ball bridge disabled.");
    return;
  }

  if (model == nullptr) {
    throw std::runtime_error("SoftTouch MuJoCo ball bridge received a null mjModel.");
  }
  if (publishBallStateEnabled_) {
    resolveSensors(model);
  }
  if (publishBaseState_ || resetEnabled_) {
    resolveBaseJoint(model);
  }
  if (applyBallDampingEnabled_) {
    applyBallDamping(model);
  }
  if (resetEnabled_) {
    if (resetStatePath_.empty()) {
      throw std::runtime_error("SoftTouch MuJoCo reset is enabled but reset.state_file is empty.");
    }
    resetState_ = loadResetState(resetStatePath_);
    resetStateLoaded_ = true;
    if (resetApplyOnConfigure_) {
      applyResetState(model, data);
    }
  }

  const auto qos = rclcpp::SensorDataQoS();
  if (publishBallStateEnabled_) {
    posePub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(poseTopic_, qos);
    twistPub_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(twistTopic_, qos);
  }
  if (resetEnabled_) {
    resetRequestSub_ = node_->create_subscription<std_msgs::msg::Float64>(
        resetRequestTopic_, 1, [this](const std_msgs::msg::Float64::SharedPtr msg) { handleResetRequest(msg); });
  }
  if (publishBaseState_) {
    basePosePub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(basePoseTopic_, qos);
    baseTwistPub_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(baseTwistTopic_, qos);
  }

  // Route-line visualization inside the MuJoCo viewer: drive 'route_dot_*' mocap
  // bodies from the controller's route MarkerArray.
  resolveRouteDots(model);
  if (!routeDotMocapIds_.empty()) {
    routeMarkerTopic_ = readStringParam(node_, std::string(kParamPrefix) + "route_marker_topic", routeMarkerTopic_);
    routeMarkerSub_ = node_->create_subscription<visualization_msgs::msg::MarkerArray>(
        routeMarkerTopic_, 1,
        [this](const visualization_msgs::msg::MarkerArray::SharedPtr msg) { handleRouteMarkers(msg); });
    RCLCPP_INFO_STREAM(node_->get_logger(), "SoftTouch route viz: " << routeDotMocapIds_.size()
                                                                    << " mocap dots from " << routeMarkerTopic_);
  }

  if (publishBallStateEnabled_) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "SoftTouch MuJoCo ball bridge publishes "
                                                << positionSensorName_ << "/" << linearVelocitySensorName_ << " to "
                                                << poseTopic_ << " and " << twistTopic_);
  } else {
    RCLCPP_INFO(node_->get_logger(), "SoftTouch MuJoCo ball bridge running with ball publishing disabled.");
  }
  if (publishBaseState_) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "SoftTouch MuJoCo base truth publishes " << baseJointName_ << " to "
                                                                                     << basePoseTopic_ << " and "
                                                                                     << baseTwistTopic_);
  }
  if (resetEnabled_) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "SoftTouch MuJoCo reset requests listen on " << resetRequestTopic_);
  }
}

void SoftTouchMujocoBallBridgePlugin::Reset(mjModel* model, mjData* data) {
  if (enabled_) {
    if (applyBallDampingEnabled_) {
      applyBallDamping(model);
    }
    if (resetEnabled_ && resetApplyOnReset_) {
      if (!resetStateLoaded_) {
        resetState_ = loadResetState(resetStatePath_);
        resetStateLoaded_ = true;
      }
      applyResetState(model, data);
    }
  }
}

void SoftTouchMujocoBallBridgePlugin::Update(mjModel* model, mjData* data) {
  if (!enabled_ || data == nullptr) {
    return;
  }
  bool requested = false;
  double requestHoldDuration = 0.0;
  {
    std::lock_guard<std::mutex> lock(resetRequestMutex_);
    requested = resetRequested_;
    requestHoldDuration = resetRequestHoldDuration_;
    resetRequested_ = false;
  }
  if (requested) {
    if (!resetStateLoaded_) {
      resetState_ = loadResetState(resetStatePath_);
      resetStateLoaded_ = true;
    }
    applyResetState(model, data);
    resetHoldUntilTime_ = data->time + std::max(0.0, requestHoldDuration);
    resetHoldLogged_ = false;
  }
  if (resetEnabled_ && resetHoldUntilTime_ > 0.0 && data->time <= resetHoldUntilTime_) {
    if (!resetStateLoaded_) {
      resetState_ = loadResetState(resetStatePath_);
      resetStateLoaded_ = true;
    }
    applyResetState(model, data, false);
    if (!resetHoldLogged_) {
      RCLCPP_INFO_STREAM(node_->get_logger(), "SoftTouch MuJoCo reset hold active until sim time "
                                                  << resetHoldUntilTime_ << " s.");
      resetHoldLogged_ = true;
    }
  }
  if (publishPeriod_ > 0.0 && (data->time - lastPublishTime_) < publishPeriod_) {
    return;
  }
  lastPublishTime_ = data->time;
  updateRouteDots(model, data);
  if (publishBallStateEnabled_) {
    publishBallState(data);
  }
  publishBaseState(data);
}

void SoftTouchMujocoBallBridgePlugin::resolveRouteDots(mjModel* model) {
  routeDotMocapIds_.clear();
  if (model == nullptr) {
    return;
  }
  for (int i = 0;; ++i) {
    const std::string name = "route_dot_" + std::to_string(i);
    const int bodyId = mj_name2id(model, mjOBJ_BODY, name.c_str());
    if (bodyId < 0) {
      break;
    }
    routeDotMocapIds_.push_back(model->body_mocapid[bodyId]);
  }
}

void SoftTouchMujocoBallBridgePlugin::handleRouteMarkers(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
  std::vector<std::array<double, 3>> points;
  for (const auto& marker : msg->markers) {
    if (marker.type != visualization_msgs::msg::Marker::LINE_STRIP) {
      continue;  // the route is the LINE_STRIP marker (ns softtouch_dribble, id 1)
    }
    points.reserve(marker.points.size());
    for (const auto& p : marker.points) {
      points.push_back({p.x, p.y, p.z});
    }
    break;
  }
  std::lock_guard<std::mutex> lock(routeMutex_);
  routePoints_ = std::move(points);
}

void SoftTouchMujocoBallBridgePlugin::updateRouteDots(const mjModel* model, mjData* data) {
  (void)model;
  if (routeDotMocapIds_.empty() || data == nullptr) {
    return;
  }
  std::vector<std::array<double, 3>> points;
  {
    std::lock_guard<std::mutex> lock(routeMutex_);
    points = routePoints_;
  }
  const int numDots = static_cast<int>(routeDotMocapIds_.size());
  const int numPoints = static_cast<int>(points.size());
  for (int i = 0; i < numDots; ++i) {
    const int mocapId = routeDotMocapIds_[i];
    if (mocapId < 0) {
      continue;
    }
    double* dst = data->mocap_pos + 3 * mocapId;
    if (numPoints < 2) {
      dst[0] = 0.0;
      dst[1] = 0.0;
      dst[2] = -1.0;  // no route yet -> park underground (hidden)
      continue;
    }
    // sample the route evenly across the available points (integer-rounded)
    const int idx = (numDots > 1) ? (i * (numPoints - 1) + (numDots - 1) / 2) / (numDots - 1) : 0;
    dst[0] = points[static_cast<size_t>(idx)][0];
    dst[1] = points[static_cast<size_t>(idx)][1];
    dst[2] = points[static_cast<size_t>(idx)][2];
  }
}

void SoftTouchMujocoBallBridgePlugin::resolveSensors(mjModel* model) {
  auto resolve = [&](const std::string& name, int expectedDim) {
    SensorRef ref;
    ref.name = name;
    ref.id = mj_name2id(model, mjOBJ_SENSOR, name.c_str());
    if (ref.id < 0) {
      throw std::runtime_error("SoftTouch MuJoCo ball bridge cannot find sensor '" + name + "'.");
    }
    ref.adr = model->sensor_adr[ref.id];
    ref.dim = model->sensor_dim[ref.id];
    if (ref.dim != expectedDim) {
      throw std::runtime_error("SoftTouch MuJoCo ball bridge sensor '" + name + "' has dim " + std::to_string(ref.dim) +
                               ", expected " + std::to_string(expectedDim) + ".");
    }
    return ref;
  };

  positionSensor_ = resolve(positionSensorName_, 3);
  quaternionSensor_ = resolve(quaternionSensorName_, 4);
  linearVelocitySensor_ = resolve(linearVelocitySensorName_, 3);
  angularVelocitySensor_ = resolve(angularVelocitySensorName_, 3);
}

void SoftTouchMujocoBallBridgePlugin::applyBallDamping(mjModel* model) const {
  if (model == nullptr) {
    return;
  }
  const int jointId = mj_name2id(model, mjOBJ_JOINT, ballJointName_.c_str());
  if (jointId < 0) {
    throw std::runtime_error("SoftTouch MuJoCo ball bridge cannot find freejoint '" + ballJointName_ + "'.");
  }
  const int dofAdr = model->jnt_dofadr[jointId];
  for (int i = 0; i < 3; ++i) {
    model->dof_damping[dofAdr + i] = ballTranslationalDamping_;
  }
  for (int i = 3; i < 6; ++i) {
    model->dof_damping[dofAdr + i] = ballAngularDamping_;
  }
}

void SoftTouchMujocoBallBridgePlugin::resolveBaseJoint(mjModel* model) {
  const int jointId = mj_name2id(model, mjOBJ_JOINT, baseJointName_.c_str());
  if (jointId < 0) {
    throw std::runtime_error("SoftTouch MuJoCo ball bridge cannot find base freejoint '" + baseJointName_ + "'.");
  }
  if (model->jnt_type[jointId] != mjJNT_FREE) {
    throw std::runtime_error("SoftTouch MuJoCo base joint '" + baseJointName_ + "' is not a freejoint.");
  }
  baseQposAdr_ = model->jnt_qposadr[jointId];
  baseDofAdr_ = model->jnt_dofadr[jointId];
}

SoftTouchMujocoBallBridgePlugin::ResetState SoftTouchMujocoBallBridgePlugin::loadResetState(const std::string& path) const {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("SoftTouch MuJoCo reset cannot open state file '" + path + "'.");
  }

  ResetState state;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    std::istringstream stream(line);
    std::string key;
    stream >> key;
    if (key == "source_policy" || key == "source_motion" || key == "motion_frame" || key == "target_yaw_deg" ||
        key == "joint_noise") {
      continue;
    }
    if (key == "root_joint") {
      stream >> state.rootJointName;
    } else if (key == "ball_joint") {
      stream >> state.ballJointName;
    } else if (key == "root_pos") {
      state.rootPosition = readDoubles(stream, 3, key);
    } else if (key == "root_quat") {
      state.rootQuaternion = readDoubles(stream, 4, key);
    } else if (key == "root_lin_vel") {
      state.rootLinearVelocity = readDoubles(stream, 3, key);
    } else if (key == "root_ang_vel_body") {
      state.rootAngularVelocityBody = readDoubles(stream, 3, key);
    } else if (key == "ball_pos") {
      state.ballPosition = readDoubles(stream, 3, key);
    } else if (key == "ball_quat") {
      state.ballQuaternion = readDoubles(stream, 4, key);
    } else if (key == "ball_lin_vel") {
      state.ballLinearVelocity = readDoubles(stream, 3, key);
    } else if (key == "ball_ang_vel") {
      state.ballAngularVelocity = readDoubles(stream, 3, key);
    } else if (key == "joint_names") {
      state.jointNames = readStrings(stream, key);
    } else if (key == "joint_pos") {
      state.jointPosition = readDoubles(stream, 29, key);
    } else if (key == "joint_vel") {
      state.jointVelocity = readDoubles(stream, 29, key);
    } else {
      throw std::runtime_error("SoftTouch MuJoCo reset state file '" + path + "' has unknown key '" + key + "'.");
    }
  }

  requireVectorSet(state.rootPosition, 3, "root_pos");
  requireVectorSet(state.rootQuaternion, 4, "root_quat");
  requireVectorSet(state.rootLinearVelocity, 3, "root_lin_vel");
  requireVectorSet(state.rootAngularVelocityBody, 3, "root_ang_vel_body");
  if (!isNoneName(state.ballJointName)) {
    requireVectorSet(state.ballPosition, 3, "ball_pos");
    requireVectorSet(state.ballQuaternion, 4, "ball_quat");
    requireVectorSet(state.ballLinearVelocity, 3, "ball_lin_vel");
    requireVectorSet(state.ballAngularVelocity, 3, "ball_ang_vel");
  }
  if (state.jointNames.size() != state.jointPosition.size() || state.jointNames.size() != state.jointVelocity.size()) {
    throw std::runtime_error("SoftTouch MuJoCo reset state has inconsistent joint_names/joint_pos/joint_vel lengths.");
  }

  return state;
}

bool SoftTouchMujocoBallBridgePlugin::hasBallReset() const {
  return resetStateLoaded_ && !isNoneName(resetState_.ballJointName);
}

void SoftTouchMujocoBallBridgePlugin::applyResetState(mjModel* model, mjData* data, bool logReset) {
  if (model == nullptr || data == nullptr) {
    throw std::runtime_error("SoftTouch MuJoCo reset received a null mjModel/mjData.");
  }
  if (!resetStateLoaded_) {
    throw std::runtime_error("SoftTouch MuJoCo reset state has not been loaded.");
  }

  std::fill(data->qpos, data->qpos + model->nq, mjtNum(0));
  std::fill(data->qvel, data->qvel + model->nv, mjtNum(0));
  if (data->qacc_warmstart != nullptr) {
    std::fill(data->qacc_warmstart, data->qacc_warmstart + model->nv, mjtNum(0));
  }
  if (data->qacc != nullptr) {
    std::fill(data->qacc, data->qacc + model->nv, mjtNum(0));
  }
  if (data->qfrc_applied != nullptr) {
    std::fill(data->qfrc_applied, data->qfrc_applied + model->nv, mjtNum(0));
  }
  if (data->xfrc_applied != nullptr) {
    std::fill(data->xfrc_applied, data->xfrc_applied + 6 * model->nbody, mjtNum(0));
  }
  if (data->ctrl != nullptr) {
    std::fill(data->ctrl, data->ctrl + model->nu, mjtNum(0));
  }
  if (data->act != nullptr) {
    std::fill(data->act, data->act + model->na, mjtNum(0));
  }

  const auto writeFreeJoint = [&](const std::string& jointName, const std::vector<double>& pos,
                                  const std::vector<double>& quat, const std::vector<double>& linVel,
                                  const std::vector<double>& angVelBody) {
    const int jointId = mj_name2id(model, mjOBJ_JOINT, jointName.c_str());
    if (jointId < 0) {
      throw std::runtime_error("SoftTouch MuJoCo reset cannot find freejoint '" + jointName + "'.");
    }
    if (model->jnt_type[jointId] != mjJNT_FREE) {
      throw std::runtime_error("SoftTouch MuJoCo reset joint '" + jointName + "' is not a freejoint.");
    }
    const int qposAdr = model->jnt_qposadr[jointId];
    const int dofAdr = model->jnt_dofadr[jointId];
    for (int i = 0; i < 3; ++i) {
      data->qpos[qposAdr + i] = pos[static_cast<size_t>(i)];
      data->qvel[dofAdr + i] = resetZeroVelocity_ ? mjtNum(0) : linVel[static_cast<size_t>(i)];
    }
    for (int i = 0; i < 4; ++i) {
      data->qpos[qposAdr + 3 + i] = quat[static_cast<size_t>(i)];
    }
    for (int i = 0; i < 3; ++i) {
      data->qvel[dofAdr + 3 + i] = resetZeroVelocity_ ? mjtNum(0) : angVelBody[static_cast<size_t>(i)];
    }
  };

  writeFreeJoint(resetState_.rootJointName, resetState_.rootPosition, resetState_.rootQuaternion,
                 resetState_.rootLinearVelocity, resetState_.rootAngularVelocityBody);
  if (hasBallReset()) {
    writeFreeJoint(resetState_.ballJointName, resetState_.ballPosition, resetState_.ballQuaternion,
                   resetState_.ballLinearVelocity, resetState_.ballAngularVelocity);
  }

  for (size_t i = 0; i < resetState_.jointNames.size(); ++i) {
    const std::string& jointName = resetState_.jointNames[i];
    const int jointId = mj_name2id(model, mjOBJ_JOINT, jointName.c_str());
    if (jointId < 0) {
      throw std::runtime_error("SoftTouch MuJoCo reset cannot find policy joint '" + jointName + "'.");
    }
    const int qposAdr = model->jnt_qposadr[jointId];
    const int dofAdr = model->jnt_dofadr[jointId];
    data->qpos[qposAdr] = resetState_.jointPosition[i];
    data->qvel[dofAdr] = resetZeroVelocity_ ? mjtNum(0) : resetState_.jointVelocity[i];
  }

  mj_forward(model, data);
  lastPublishTime_ = -1.0e30;
  if (logReset) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "SoftTouch MuJoCo reset applied from " << resetStatePath_ << " root=("
                                                                                   << resetState_.rootPosition[0] << ", "
                                                                                   << resetState_.rootPosition[1] << ", "
                                                                                   << resetState_.rootPosition[2] << ") ball="
                                                                                   << (hasBallReset() ? "enabled" : "disabled")
                                                                                   << " zero_velocity="
                                                                                   << (resetZeroVelocity_ ? "true" : "false"));
  }
}

void SoftTouchMujocoBallBridgePlugin::publishBallState(const mjData* data) {
  const mjtNum* pos = data->sensordata + positionSensor_.adr;
  const mjtNum* quat = data->sensordata + quaternionSensor_.adr;
  const mjtNum* linvel = data->sensordata + linearVelocitySensor_.adr;
  const mjtNum* angvel = data->sensordata + angularVelocitySensor_.adr;

  for (int i = 0; i < 3; ++i) {
    requireFinite(pos[i], positionSensor_.name);
    requireFinite(linvel[i], linearVelocitySensor_.name);
    requireFinite(angvel[i], angularVelocitySensor_.name);
  }
  for (int i = 0; i < 4; ++i) {
    requireFinite(quat[i], quaternionSensor_.name);
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frameId_;
  pose.header.stamp = stampFromSeconds(data->time);
  pose.pose.position.x = pos[0];
  pose.pose.position.y = pos[1];
  pose.pose.position.z = pos[2];
  pose.pose.orientation.w = quat[0];
  pose.pose.orientation.x = quat[1];
  pose.pose.orientation.y = quat[2];
  pose.pose.orientation.z = quat[3];

  geometry_msgs::msg::TwistStamped twist;
  twist.header = pose.header;
  twist.twist.linear.x = linvel[0];
  twist.twist.linear.y = linvel[1];
  twist.twist.linear.z = linvel[2];
  twist.twist.angular.x = angvel[0];
  twist.twist.angular.y = angvel[1];
  twist.twist.angular.z = angvel[2];

  posePub_->publish(pose);
  twistPub_->publish(twist);
}

void SoftTouchMujocoBallBridgePlugin::publishBaseState(const mjData* data) {
  if (!publishBaseState_ || !basePosePub_ || !baseTwistPub_ || baseQposAdr_ < 0 || baseDofAdr_ < 0 || data == nullptr) {
    return;
  }

  const mjtNum* qpos = data->qpos + baseQposAdr_;
  const mjtNum* qvel = data->qvel + baseDofAdr_;
  for (int i = 0; i < 7; ++i) {
    requireFinite(qpos[i], baseJointName_ + ".qpos");
  }
  for (int i = 0; i < 6; ++i) {
    requireFinite(qvel[i], baseJointName_ + ".qvel");
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frameId_;
  pose.header.stamp = stampFromSeconds(data->time);
  pose.pose.position.x = qpos[0];
  pose.pose.position.y = qpos[1];
  pose.pose.position.z = qpos[2];
  pose.pose.orientation.w = qpos[3];
  pose.pose.orientation.x = qpos[4];
  pose.pose.orientation.y = qpos[5];
  pose.pose.orientation.z = qpos[6];

  geometry_msgs::msg::TwistStamped twist;
  twist.header = pose.header;
  twist.twist.linear.x = qvel[0];
  twist.twist.linear.y = qvel[1];
  twist.twist.linear.z = qvel[2];
  twist.twist.angular.x = qvel[3];
  twist.twist.angular.y = qvel[4];
  twist.twist.angular.z = qvel[5];

  basePosePub_->publish(pose);
  baseTwistPub_->publish(twist);
}

void SoftTouchMujocoBallBridgePlugin::handleResetRequest(const std_msgs::msg::Float64::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(resetRequestMutex_);
  resetRequested_ = true;
  resetRequestHoldDuration_ = msg ? std::max(0.0, msg->data) : 0.0;
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::SoftTouchMujocoBallBridgePlugin, mujoco_sim_ros2::MujocoPhysicsPlugin)
