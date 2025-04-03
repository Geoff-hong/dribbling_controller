//
// Created by qiayuanl on 3/14/25.
//

#pragma once

#include <legged_model/LeggedModel.h>
#include <legged_rl_controllers/Policy.h>
#include <rclcpp/time.hpp>

namespace legged {

class DataLogger {
 public:
  using SharedPtr = std::shared_ptr<DataLogger>;

  DataLogger(LeggedModel::SharedPtr model, Policy::SharedPtr policy, scalar_t frequency = 200, size_t maxBufferSize = 1e5);
  void update(const rclcpp::Time& time);
  void writeAndClear();

 private:
  std::vector<std::vector<scalar_t>> buffer;
  LeggedModel::SharedPtr model_;
  Policy::SharedPtr policy_;
  rclcpp::Time lastUpdate_;
  scalar_t period_;
};

}  // namespace legged
