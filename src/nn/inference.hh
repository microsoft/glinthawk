#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace glinthawk {

struct MatrixBuffer
{
  float* ptr { nullptr };
  int32_t len { 0 };
};

struct InferenceState
{
  int token { 1 };
  int token_pos { 0 };
  int next_layer { 0 };
  MatrixBuffer activations {};
};

struct InferenceResult
{
  InferenceState inference_state {};
  std::optional<std::string> word { std::nullopt };
};

}
