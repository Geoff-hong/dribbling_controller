//
// Created by qiayuanl on 3/7/25.
//

#pragma once

#include <legged_model/common.h>
#include <string>
#include <vector>

namespace legged {

struct MotionCommandCfg {
  std::string referenceBody;
  std::vector<std::string> bodyNames;
  int startStep = 0;
};

inline vector_t rotationToVectorWxyz(const quaternion_t& ori) {
  vector_t vec(4);
  vec(0) = ori.w();
  vec.segment(1, 3) = ori.coeffs().head(3);
  return vec;
}

inline vector_t rotationToVectorWxyz(const matrix3_t& ori) {
  return rotationToVectorWxyz(quaternion_t(ori));
}

}  // namespace legged
