#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace glinthawk {

struct MatrixBuffer
{
  std::unique_ptr<float[]> ptr { nullptr };
  int32_t len { 0 };

  MatrixBuffer() = default;
  MatrixBuffer( const std::unique_ptr<float[]>& other_ptr, const int32_t other_len )
    : ptr( std::make_unique<float[]>( other_len ) )
    , len( other_len )
  {
    std::memcpy( ptr.get(), other_ptr.get(), len * sizeof( float ) );
  }
};

struct InferenceState
{
  int32_t token { 1 };
  int32_t token_pos { 0 };
  int32_t next_layer { 0 };
  MatrixBuffer activations {};

  InferenceState() = default;
  InferenceState( const std::string_view serialized );
  std::string serialize();
};

struct InferenceResult
{
  InferenceState inference_state {};
  std::optional<std::string> word { std::nullopt };
};

} // namespace glinthawk
