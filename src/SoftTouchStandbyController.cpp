#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <legged_controllers/ControllerBase.h>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/float64.hpp>

namespace legged {
namespace {

vector_t readVectorParam(const rclcpp_lifecycle::LifecycleNode::SharedPtr& node, const std::string& name,
                         const std::size_t expectedSize) {
  const auto values = node->get_parameter(name).as_double_array();
  if (values.size() != expectedSize) {
    throw std::runtime_error(name + " must have " + std::to_string(expectedSize) + " entries, got " +
                             std::to_string(values.size()) + ".");
  }
  vector_t out(static_cast<Eigen::Index>(values.size()));
  for (std::size_t i = 0; i < values.size(); ++i) {
    out(static_cast<Eigen::Index>(i)) = static_cast<scalar_t>(values[i]);
  }
  return out;
}

}  // namespace

class SoftTouchStandbyController : public ControllerBase {
 public:
  controller_interface::CallbackReturn on_init() override {
    if (ControllerBase::on_init() != controller_interface::CallbackReturn::SUCCESS) {
      return controller_interface::CallbackReturn::ERROR;
    }
    try {
      auto_declare("joint_names", std::vector<std::string>{});
      auto_declare("default_position", std::vector<double>{});
      auto_declare("kp", std::vector<double>{});
      auto_declare("kd", std::vector<double>{});
      auto_declare<double>("total_time", 2.0);
      auto_declare<bool>("reset.mujoco_reset_on_activate", false);
      auto_declare<std::string>("reset.mujoco_reset_topic", "/softtouch/mujoco_reset");
      auto_declare<double>("reset.mujoco_reset_hold_s", 0.0);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "SoftTouchStandbyController init failed: %s", e.what());
      return controller_interface::CallbackReturn::ERROR;
    }
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State&) override {
    try {
      jointNameInControl_ = get_node()->get_parameter("joint_names").as_string_array();
      if (jointNameInControl_.empty()) {
        throw std::runtime_error("joint_names is empty.");
      }
      const std::size_t n = jointNameInControl_.size();
      defaultPosition_ = readVectorParam(get_node(), "default_position", n);
      kp_ = readVectorParam(get_node(), "kp", n);
      kd_ = readVectorParam(get_node(), "kd", n);
      totalTime_ = static_cast<scalar_t>(std::max(0.0, get_node()->get_parameter("total_time").as_double()));
      mujocoResetOnActivate_ = get_node()->get_parameter("reset.mujoco_reset_on_activate").as_bool();
      mujocoResetTopic_ = get_node()->get_parameter("reset.mujoco_reset_topic").as_string();
      mujocoResetHoldDuration_ = static_cast<scalar_t>(std::max(0.0, get_node()->get_parameter("reset.mujoco_reset_hold_s").as_double()));
      if (mujocoResetOnActivate_) {
        mujocoResetPub_ = get_node()->create_publisher<std_msgs::msg::Float64>(mujocoResetTopic_, 1);
      } else {
        mujocoResetPub_.reset();
      }
      startPosition_ = defaultPosition_;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "SoftTouchStandbyController configure failed: %s", e.what());
      return controller_interface::CallbackReturn::ERROR;
    }
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previousState) override {
    if (ControllerBase::on_activate(previousState) != controller_interface::CallbackReturn::SUCCESS) {
      return controller_interface::CallbackReturn::ERROR;
    }
    try {
      const auto model = leggedModel();
      const auto joints = model->getGeneralizedPosition().tail(model->getJointNames().size());
      startPosition_.resize(static_cast<Eigen::Index>(jointNameInControl_.size()));
      if (mujocoResetOnActivate_ && mujocoResetPub_) {
        startPosition_ = defaultPosition_;
        std_msgs::msg::Float64 msg;
        msg.data = static_cast<double>(mujocoResetHoldDuration_);
        mujocoResetPub_->publish(msg);
        RCLCPP_INFO_STREAM(get_node()->get_logger(), "SoftTouch standby requested MuJoCo reset on activate; hold_s="
                                                        << mujocoResetHoldDuration_);
      } else {
        for (std::size_t i = 0; i < jointNameInControl_.size(); ++i) {
          startPosition_(static_cast<Eigen::Index>(i)) = joints(static_cast<Eigen::Index>(
              model->getJointIndex(jointNameInControl_[i])));
        }
      }
      firstUpdate_ = true;
      commandIndexBuilt_ = false;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "SoftTouchStandbyController activation failed: %s", e.what());
      return controller_interface::CallbackReturn::ERROR;
    }
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override {
    const auto baseResult = ControllerBase::update(time, period);
    if (baseResult != controller_interface::return_type::OK) {
      return baseResult;
    }
    if (!commandIndexBuilt_) {
      buildCommandIndex();
    }
    if (firstUpdate_) {
      startTime_ = time;
      firstUpdate_ = false;
    }
    const scalar_t elapsed = static_cast<scalar_t>((time - startTime_).seconds());
    const scalar_t alpha = totalTime_ <= 0.0 ? 1.0 : std::clamp(elapsed / totalTime_, static_cast<scalar_t>(0.0),
                                                                static_cast<scalar_t>(1.0));
    const vector_t target = (static_cast<scalar_t>(1.0) - alpha) * startPosition_ + alpha * defaultPosition_;

    for (std::size_t i = 0; i < jointNameInControl_.size(); ++i) {
      const auto& jointName = jointNameInControl_[i];
      const Eigen::Index idx = static_cast<Eigen::Index>(i);
      setCommand(jointName, "position", target(idx));
      setCommand(jointName, "velocity", 0.0);
      setCommand(jointName, "effort", 0.0);
      setCommand(jointName, "stiffness", kp_(idx));
      setCommand(jointName, "damping", kd_(idx));
    }
    return controller_interface::return_type::OK;
  }

 private:
  static std::string commandKey(const std::string& jointName, const std::string& interfaceName) {
    return jointName + "/" + interfaceName;
  }

  void buildCommandIndex() {
    commandIndexByName_.clear();
    for (std::size_t i = 0; i < command_interfaces_.size(); ++i) {
      commandIndexByName_[command_interfaces_[i].get_name()] = i;
    }
    for (const auto& jointName : jointNameInControl_) {
      for (const auto& interfaceName : {"position", "velocity", "effort", "stiffness", "damping"}) {
        const auto key = commandKey(jointName, interfaceName);
        if (commandIndexByName_.find(key) == commandIndexByName_.end()) {
          throw std::runtime_error("SoftTouchStandbyController missing command interface '" + key + "'.");
        }
      }
    }
    commandIndexBuilt_ = true;
  }

  void setCommand(const std::string& jointName, const std::string& interfaceName, scalar_t value) {
    const auto it = commandIndexByName_.find(commandKey(jointName, interfaceName));
    if (it == commandIndexByName_.end()) {
      throw std::runtime_error("SoftTouchStandbyController missing command interface '" +
                               commandKey(jointName, interfaceName) + "'.");
    }
    [[maybe_unused]] const bool ok = command_interfaces_[it->second].set_value(static_cast<double>(value));
  }

  vector_t defaultPosition_;
  vector_t kp_;
  vector_t kd_;
  vector_t startPosition_;
  std::unordered_map<std::string, std::size_t> commandIndexByName_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mujocoResetPub_;
  std::string mujocoResetTopic_ = "/softtouch/mujoco_reset";
  rclcpp::Time startTime_;
  scalar_t totalTime_ = 2.0;
  scalar_t mujocoResetHoldDuration_ = 0.0;
  bool mujocoResetOnActivate_ = false;
  bool firstUpdate_ = true;
  bool commandIndexBuilt_ = false;
};

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::SoftTouchStandbyController, controller_interface::ControllerInterface)
