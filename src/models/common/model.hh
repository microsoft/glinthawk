#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef GLINTHAWK_CUDA_ENABLED
#include <cuda_fp16.h>
#else
#define __half uint16_t
#endif

#include "util/digest.hh"

namespace glinthawk {

using PromptID = glinthawk::util::digest::SHA256Hash;
using ModelID = uint32_t;

} // namespace glinthawk

namespace glinthawk::models {

class DataType
{
public:
  enum class Type : uint8_t
  {
    Float16,
    Float32
  };

public:
  DataType( const Type t )
    : dtype( t )
  {
  }

  size_t size() const
  {
    switch ( dtype ) {
      case Type::Float16:
        return 2;
      case Type::Float32:
        return 4;
    }

    throw std::runtime_error( "invalid dtype" );
  }

  Type dtype;
};

static_assert( sizeof( DataType ) == sizeof( DataType::Type ) );

struct DataBuffer
{
  DataType dtype { DataType::Type::Float32 };
  std::unique_ptr<uint8_t[]> ptr { nullptr };
  uint64_t len { 0 };

  DataBuffer() = default;
  DataBuffer( const DataType other_dtype, std::unique_ptr<uint8_t[]>&& other_ptr, const uint64_t other_len )
    : dtype( other_dtype )
    , ptr( std::move( other_ptr ) )
    , len( other_len )
  {
  }
};

class InferenceState
{
private:
  PromptID prompt_id_ {};
  ModelID model_id_ { 0 };

  uint32_t token_ { 1 };
  uint32_t token_pos_ { 0 };
  uint32_t next_layer_ { 0 };
  float temperature_ { 0.0f };

  DataBuffer activations_ {};

  size_t serialized_size() const;

public:
  InferenceState( const PromptID prompt_id,
                  const ModelID model_id,
                  const uint32_t token,
                  const uint32_t token_pos,
                  const uint32_t next_layer,
                  const float temperature,
                  DataBuffer&& activations )
    : prompt_id_( prompt_id )
    , model_id_( model_id )
    , token_( token )
    , token_pos_( token_pos )
    , next_layer_( next_layer )
    , temperature_( temperature )
    , activations_( std::move( activations ) )
  {
  }

  InferenceState( const std::string_view serialized_state );
  std::string serialize() const;
  std::string to_string() const;

  PromptID prompt_id() const { return prompt_id_; }
  ModelID model_id() const { return model_id_; }

  uint32_t token() const { return token_; }
  uint32_t token_pos() const { return token_pos_; }
  uint32_t next_layer() const { return next_layer_; }
  float temperature() const { return temperature_; }

  const DataBuffer& activations() const { return activations_; }
};

template<typename Context>
class Model
{
public:
  virtual ~Model() {}
  virtual InferenceState forward( const InferenceState& inference_state, Context& context ) = 0;
};

} // namespace glinthawk::models
