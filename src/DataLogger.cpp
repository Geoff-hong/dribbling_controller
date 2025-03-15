//
// Created by qiayuanl on 3/14/25.
//

#include <utility>

#include "motion_tracking_controller/DataLogger.h"

namespace legged {
DataLogger::DataLogger(LeggedModel::SharedPtr model, OnnxPolicy::SharedPtr policy, scalar_t frequency, size_t maxBufferSize)
    : model_(std::move(model)), policy_(std::move(policy)), lastUpdate_(0, 0, RCL_ROS_TIME), period_(1. / frequency) {
  buffer.reserve(maxBufferSize);
}

void DataLogger::update(const rclcpp::Time& time) {
  if ((time - lastUpdate_).seconds() < period_ || buffer.size() >= buffer.capacity()) {
    return;
  }

  lastUpdate_ = time;
  std::vector<scalar_t> data;

  for (const auto& i : model_->getGeneralizedPosition()) {
    data.push_back(i);
  }
  for (const auto& i : model_->getGeneralizedVelocity()) {
    data.push_back(i);
  }
  for (const auto& i : policy_->getLastOutput()) {
    data.push_back(i);
  }
  buffer.push_back(data);
}

void DataLogger::writeAndClear() {
  std::ofstream result("logged_data.csv");
  for (auto& data : buffer) {
    for (size_t j = 0; j < data.size(); j++) {
      result << data[j];
      if (j < data.size() - 1) {
        result << ",";
      }
    }
    result << std::endl;
  }
  result.close();

  std::ofstream info("logged_data.yaml");
  info << "nq: " << model_->getPinModel().nq << std::endl;
  info << "nv: " << model_->getPinModel().nv << std::endl;
  info << "njoints: " << model_->getPinModel().njoints - 2 << std::endl;
  info << "joint_names: [";
  for (int i = 2; i < model_->getPinModel().njoints; i++) {
    info << model_->getPinModel().names[i] << ",";
  }
  info << "]" << std::endl;
  info.close();
}

}  // namespace legged
