#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef GLINTHAWK_CUDA_ENABLED
#include <cuda_fp16.h>
#else
#define __half uint16_t
#endif

#include "net/address.hh"
#include "util/digest.hh"

namespace glinthawk {

using PromptID = glinthawk::util::digest::SHA256Hash;
using ModelID = uint32_t;

} // namespace glinthawk

namespace glinthawk::models {

class SerializedDataType
{
public:
  enum class Type : uint8_t
  {
    Float16,
    Float32
  };

public:
  SerializedDataType( const Type t )
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

static_assert( sizeof( SerializedDataType ) == sizeof( SerializedDataType::Type ) );

struct DataBuffer
{
  SerializedDataType dtype { SerializedDataType::Type::Float32 };
  std::unique_ptr<uint8_t[]> ptr { nullptr };
  uint64_t len { 0 };

  DataBuffer() = default;
  DataBuffer( const SerializedDataType other_dtype, std::unique_ptr<uint8_t[]>&& other_ptr, const uint64_t other_len )
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
  uint32_t prompt_length_ { 1 };
  float temperature_ { 0.0f };
  bool finished_ { false };

  DataBuffer activations_ {};

  // mapping from layer to worker address for this inference state
  std::map<uint32_t, glinthawk::net::Address> layer_workers_ {};

  size_t serialized_size() const;

public:
  InferenceState() {}

  InferenceState( const std::string_view serialized_state );
  std::string serialize() const;
  std::string to_string() const;

  PromptID prompt_id() const { return prompt_id_; }
  ModelID model_id() const { return model_id_; }

  uint32_t token() const { return token_; }
  uint32_t token_pos() const { return token_pos_; }
  uint32_t next_layer() const { return next_layer_; }
  uint32_t prompt_length() const { return prompt_length_; }
  float temperature() const { return temperature_; }
  bool finished() const { return finished_; }
  const decltype( layer_workers_ )& layer_workers() const { return layer_workers_; }

  void set_prompt_id( const PromptID prompt_id ) { prompt_id_ = prompt_id; }
  void set_model_id( const ModelID model_id ) { model_id_ = model_id; }
  void set_token( const uint32_t token ) { token_ = token; }
  void set_token_pos( const uint32_t token_pos ) { token_pos_ = token_pos; }
  void set_next_layer( const uint32_t next_layer ) { next_layer_ = next_layer; }
  void set_prompt_length( const uint32_t prompt_length ) { prompt_length_ = prompt_length; }
  void set_temperature( const float temperature ) { temperature_ = temperature; }
  void set_activations( DataBuffer&& activations ) { activations_ = std::move( activations ); }
  void set_layer_workers( decltype( layer_workers_ )&& layer_workers ) { layer_workers_ = layer_workers; }
  void set_finished() { finished_ = true; }

  glinthawk::net::Address next_worker() const;
  void erase_from_workers( const uint32_t next_layer );
  const DataBuffer& activations() const { return activations_; }
};

template<typename Context>
class Model
{
public:
  virtual ~Model() = default;

  virtual void dummy_forward( InferenceState& inference_state ) = 0;
  virtual bool is_finished( const InferenceState& inference_state ) = 0;

  virtual InferenceState forward( InferenceState&& inference_state, std::shared_ptr<Context> context ) = 0;

  virtual std::vector<InferenceState> forward( std::vector<InferenceState>&& inference_states,
                                               const std::vector<std::shared_ptr<Context>>& contexts )
    = 0;
};

} // namespace glinthawk::models
