// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle2onnx/mapper/activation/relu6.h"

namespace paddle2onnx {
REGISTER_MAPPER(relu6, Relu6Mapper)

void Relu6Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  float min = 0.0;
  float threshold = 6.0;
  if (HasAttr("threshold")) {
    GetAttr("threshold", &threshold);
  }
  helper_->Clip(input_info[0].name, output_info[0].name, min, threshold, input_info[0].dtype);
}
}