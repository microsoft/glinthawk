#pragma once

#include <iostream>
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

enum class DataType : uint8_t
{
  Float16 = 0,
  Float32 = 1
};

size_t DataTypeSize( const DataType dtype );

struct DataBuffer
{
private:
  std::unique_ptr<uint8_t[]> ptr_ { nullptr };

  /// Length of the buffer in bytes
  uint64_t len_ { 0 };

public:
  DataBuffer() = default;

  DataBuffer( const size_t n )
    : ptr_( std::make_unique<uint8_t[]>( n ) )
    , len_( n )
  {
  }

  template<typename T>
  DataBuffer( const size_t n, const T* data )
    : ptr_( std::make_unique<uint8_t[]>( n ) )
    , len_( n )
  {
    memcpy( ptr_.get(), data, n );
  }

  DataBuffer( std::unique_ptr<uint8_t[]>&& other_ptr, const uint64_t other_len )
    : ptr_( std::move( other_ptr ) )
    , len_( other_len )
  {
  }

  uint64_t len() const { return len_; }
  uint8_t* data() { return ptr_.get(); }
  const uint8_t* data() const { return ptr_.get(); }
};

} // namespace glinthawk

namespace glinthawk::models {

class InferenceState
{
public:
  enum class Stage : uint8_t
  {
    PreAttention,
    Attention,
    PostAttention
  };

private:
  PromptID prompt_id_ {};
  ModelID model_id_ { 0 };

  uint32_t token_ { 1 };
  uint32_t token_pos_ { 0 };
  uint32_t next_layer_ { 0 };
  Stage next_stage_ { Stage::PreAttention };
  uint32_t prompt_length_ { 1 };
  float temperature_ { 0.0f };
  bool finished_ { false };

  DataType dtype_ { DataType::Float32 };
  DataBuffer activations_ {};

  // mapping from layer to worker address for this inference state
  std::map<uint32_t, glinthawk::net::Address> layer_workers_ {};

  size_t serialized_size() const;

public:
  InferenceState() = default;

  InferenceState( const DataType dtype )
    : dtype_( dtype )
  {
  }

  InferenceState( const std::string_view serialized_state );

  /* movable, not copyable */
  InferenceState( const InferenceState& other ) = delete;
  InferenceState& operator=( const InferenceState& other ) = delete;
  InferenceState( InferenceState&& other ) = default;
  InferenceState& operator=( InferenceState&& other ) = default;

  std::string serialize() const;
  std::string to_string() const;

  PromptID prompt_id() const { return prompt_id_; }
  ModelID model_id() const { return model_id_; }

  uint32_t token() const { return token_; }
  uint32_t token_pos() const { return token_pos_; }
  uint32_t next_layer() const { return next_layer_; }
  Stage next_stage() const { return next_stage_; }
  uint32_t prompt_length() const { return prompt_length_; }
  float temperature() const { return temperature_; }
  bool finished() const { return finished_; }
  const decltype( layer_workers_ )& layer_workers() const { return layer_workers_; }
  DataType dtype() const { return dtype_; }

  DataBuffer& activations() { return activations_; }
  const DataBuffer& activations() const { return activations_; }

  void set_prompt_id( const PromptID prompt_id ) { prompt_id_ = prompt_id; }
  void set_model_id( const ModelID model_id ) { model_id_ = model_id; }
  void set_token( const uint32_t token ) { token_ = token; }
  void set_token_pos( const uint32_t token_pos ) { token_pos_ = token_pos; }
  void set_next_layer( const uint32_t next_layer ) { next_layer_ = next_layer; }
  void set_next_stage( const Stage next_stage ) { next_stage_ = next_stage; }
  void set_prompt_length( const uint32_t prompt_length ) { prompt_length_ = prompt_length; }
  void set_temperature( const float temperature ) { temperature_ = temperature; }
  void set_activations( DataBuffer&& activations ) { activations_ = std::move( activations ); }
  void set_layer_workers( const decltype( layer_workers_ )& layer_workers ) { layer_workers_ = layer_workers; }
  void set_finished() { finished_ = true; }

  glinthawk::net::Address next_worker() const;
  void erase_from_workers( const uint32_t next_layer );
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

  virtual InferenceState pre_attention_forward( InferenceState&& inference_state, std::shared_ptr<Context> context )
    = 0;

  virtual std::vector<InferenceState> pre_attention_forward( std::vector<InferenceState>&& inference_states,
                                                             const std::vector<std::shared_ptr<Context>>& contexts )
    = 0;

  virtual InferenceState attention_forward( InferenceState&& inference_state, std::shared_ptr<Context> context ) = 0;

  virtual std::vector<InferenceState> attention_forward( std::vector<InferenceState>&& inference_states,
                                                         const std::vector<std::shared_ptr<Context>>& contexts )
    = 0;

  virtual InferenceState post_attention_forward( InferenceState&& inference_state ) = 0;

  virtual std::vector<InferenceState> post_attention_forward( std::vector<InferenceState>&& inference_states ) = 0;
};

} // namespace glinthawk::models

std::ostream& operator<<( std::ostream& os, const glinthawk::DataType& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::DataBuffer& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::models::InferenceState::Stage& v );
