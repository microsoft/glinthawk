#pragma once

#include <glog/logging.h>
#include <span>
#include <string_view>

#include "../llama2/variants.hh"
#include "model.hh"

namespace glinthawk::models {

// NOTE(sadjad): right now, inference state is designed to be used by Llama and Llama-like models. We need to work out
// the generality later.
template<typename Config>
requires llama2::ModelConfig<Config>
class BatchedInferenceState
{
public:
  using Stage = InferenceState::Stage;

private:
  struct __attribute__( ( packed ) ) PromptData
  {
    PromptID prompt_id;
    uint32_t token;
    uint32_t token_pos;
    uint8_t temperature; // compact temprature, between [0, 255]; has to be divided by 255.0f
    uint32_t prompt_length;
  };

  struct __attribute__( ( packed ) ) Metadata
  {
    uint32_t batch_size {};
    DataType dtype { DataType::Float32 };
    RouteID route_id {};
    ModelID model_id {};
    bool finished { false };
    uint32_t next_layer { 0 };
    Stage next_stage { Stage::PreAttention };

    bool has_activations { false };
    bool has_queries { false };
    bool has_kvs { false };
  };

  Metadata metadata_;
  std::vector<PromptData> prompts_( batch_size_ );

  // we use contiguous memory for activations, queries, and key-values; allowing one-shot copying.
  DataBuffer activations_ {};
  DataBuffer queries_ {};
  DataBuffer kvs_ {};

  size_t activation_len() const { return Config::dim * DataTypeSize( dtype_ ); }
  size_t q_len() const { return Config::dim * DataTypeSize( dtype_ ); }
  size_t kv_len() const { return 2 * Config::kv_dim * DataTypeSize( dtype_ ); }

  uint8_t* activation_ptr( const size_t i ) { return activations_.get() + i * activation_len(); }
  uint8_t* q_ptr( const size_t i ) { return queries_.get() + i * q_len(); }
  uint8_t* kv_ptr( const size_t i ) { return kvs_.get() + i * kv_len(); }

public:
  BatchedInferenceState( uint32_t batch_size, DataType dtype, RouteID route_id, ModelID model_id )
    : batch_size_( batch_size )
    , dtype_( dtype )
    , route_id_( route_id )
    , model_id_( model_id )
    , activations_( batch_size * Config::dim * DataTypeSize( dtype_ ) )
    , queries_( batch_size * Config::dim * DataTypeSize( dtype_ ) )
    , kvs_( batch_size * Config::kv_dim * DataTypeSize( dtype_ ) )
  {
  }

  // TODO(sadjad) eventually we got to get rid of the default constructor.
  BatchedInferenceState()
    : BatchedInferenceState( 0, DataType::Float32, 0, 0 )
  {
  }

  // serialization and deserialization of this monstrosity
  BatchedInferenceState( const std::string_view serialized_state );
  std::string serialize() const;

  // movable, but not copyable
  BatchedInferenceState( const BatchedInferenceState& other ) = delete;
  BatchedInferenceState& operator=( const BatchedInferenceState& other ) = delete;
  BatchedInferenceState( BatchedInferenceState&& other ) = default;
  BatchedInferenceState& operator=( BatchedInferenceState&& other ) = default;

  // metadata setters
  void set_dtype( DataType dtype ) { metadata_.dtype_ = dtype; }
  void set_route_id( RouteID route_id ) { metadata_.route_id = route_id; }
  void set_model_id( ModelID model_id ) { metadata_.model_id = model_id; }
  void set_finished( bool finished ) { metadata_.finished = finished; }
  void set_next_layer( uint32_t next_layer ) { metadata_.next_layer = next_layer; }
  void set_next_stage( Stage next_stage ) { metadata_.next_stage = next_stage; }
  void set_has_activations( bool has_activations ) { metadata_.has_activations = has_activations; }
  void set_has_queries( bool has_queries ) { metadata_.has_queries = has_queries; }
  void set_has_kvs( bool has_kvs ) { metadata_.has_kvs = has_kvs; }

  // metadata getters
  uint32_t batch_size() const { return metadata_.batch_size; }
  DataType dtype() const { return metadata_.dtype; }
  RouteID route_id() const { return metadata_.route_id; }
  ModelID model_id() const { return metadata_.model_id; }
  bool finished() const { return metadata_.finished; }
  uint32_t next_layer() const { return metadata_.next_layer; }
  Stage next_stage() const { return metadata_.next_stage; }
  bool has_activations() const { return metadata_.has_activations; }
  bool has_queries() const { return metadata_.has_queries; }
  bool has_kvs() const { return metadata_.has_kvs; }

  // prompt setters
  void set_prompt( const size_t i,
                   PromptID prompt_id,
                   uint32_t token,
                   uint32_t token_pos,
                   float temperature,
                   uint32_t prompt_length )
  {
    prompts_[i].prompt_id = prompt_id;
    prompts_[i].token = token;
    prompts_[i].token_pos = token_pos;
    prompts_[i].temperature = static_cast<uint8_t>( temperature * 255.0f );
    prompts_[i].prompt_length = prompt_length;
  }

  // prompt getters
  PromptID prompt_id( const size_t i ) const { return prompts_[i].prompt_id; }
  uint32_t token( const size_t i ) const { return prompts_[i].token; }
  uint32_t token_pos( const size_t i ) const { return prompts_[i].token_pos; }
  uint32_t prompt_length( const size_t i ) const { return prompts_[i].prompt_length; }
  float temperature( const size_t i ) const { return prompts_[i].temperature / 255.0f; }

  // The memory is owned by the inference state; be careful with the lifetime of the returned spans.
  std::span<uint8_t> activations( const size_t i ) { return { activation_ptr( i ), activation_len() }; }
  std::span<uint8_t> q( const size_t i ) { return { q_ptr( i ), q_len() }; }
  std::span<uint8_t> kv( const size_t i ) { return { kv_ptr( i ), kv_len() }; }

  std::span<const uint8_t> activations( const size_t i ) const { return { activation_ptr( i ), activation_len() }; }
  std::span<const uint8_t> q( const size_t i ) const { q_ptr( i ), q_len() };
  std::span<const uint8_t> kv( const size_t i ) const { return { kv_ptr( i ), kv_len() }; }
};

template<typename Config>
BatchedInferenceState<Config>::BatchedInferenceState( const std::string_view serialized_state )
{
  auto ptr = serialized_state.data();

  // we need to make sure that the serialized state is at least as big as the metadata
  size_t expected_size = sizeof( Metadata );
  CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain metadata";

  // copy the metadata
  std::memcpy( &metadata_, ptr, sizeof( Metadata ) );
  ptr += sizeof( Metadata );

  // check if the serialized state is big enough to contain the prompts
  expected_size += metadata_.batch_size * sizeof( PromptData );
  CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain prompts";

  // resize the prompts
  prompts_.resize( metadata_.batch_size );

  // copy the prompts
  std::memcpy( prompts_.data(), ptr, metadata_.batch_size * sizeof( PromptData ) );
  ptr += metadata_.batch_size * sizeof( PromptData );

  if ( has_activations() ) {
    expected_size += metadata_.batch_size * activation_len();
    // check if the serialized state is big enough to contain the activations
    CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain activations";

    // copy the activations
    std::memcpy( activations_.get(), ptr, metadata_.batch_size * activation_len() );
    ptr += metadata_.batch_size * activation_len();
  }

  if ( has_queries() ) {
    // check if the serialized state is big enough to contain the queries
    expected_size += metadata_.batch_size * q_len();
    CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain queries";

    // copy the queries
    std::memcpy( queries_.get(), ptr, metadata_.batch_size * q_len() );
  }

  if ( has_kvs() ) {
    // check if the serialized state is big enough to contain the key-values
    expected_size += metadata_.batch_size * kv_len();
    CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain key-values";

    // copy the key-values
    std::memcpy( kvs_.get(), ptr, metadata_.batch_size * kv_len() );
  }
}

template<typename Config>
std::string serialize() const
{
  std::string serialized_state;
  const size_t expected_size = sizeof( Metadata ) + metadata_.batch_size * sizeof( PromptData )
                               + ( has_activations() ? metadata_.batch_size * activation_len() : 0 )
                               + ( has_queries() ? metadata_.batch_size * q_len() : 0 )
                               + ( has_kvs() ? metadata_.batch_size * kv_len() : 0 );

  // reserve enough space for the state
  serialized_state.reserve( expected_size );

  // copy the metadata
  serialized_state.append( reinterpret_cast<const char*>( &metadata_ ), sizeof( Metadata ) );
  serialized_state.append( reinterpret_cast<const char*>( prompts_.data() ),
                           metadata_.batch_size * sizeof( PromptData ) );

  if ( has_activations() ) {
    serialized_state.append( reinterpret_cast<const char*>( activations_.get() ),
                             metadata_.batch_size * activation_len() );
  }

  if ( has_queries() ) {
    serialized_state.append( reinterpret_cast<const char*>( queries_.get() ), metadata_.batch_size * q_len() );
  }

  if ( has_kvs() ) {
    serialized_state.append( reinterpret_cast<const char*>( kvs_.get() ), metadata_.batch_size * kv_len() );
  }

  return serialized_state;
}

} // namespace glinthawk::models
