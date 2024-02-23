#pragma once

#include <algorithm>
#include <glog/logging.h>
#include <span>
#include <sstream>
#include <string>
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
  struct __attribute__( ( packed ) ) Metadata
  {
    uint32_t batch_size {};
    DataType dtype { DataType::Float32 };
    RouteID route_id {};
    ModelID model_id {};
    uint32_t next_layer { 0 };
    Stage next_stage { Stage::PreAttention };

    bool has_activations { false };
    bool has_queries { false };
    bool has_kvs { false };

    uint32_t discarded_contexts { 0 };
  };

  struct __attribute__( ( packed ) ) DiscardedContext
  {
    PromptID prompt_id {};
  };

  struct __attribute__( ( packed ) ) PromptData
  {
    bool active { true };

    PromptID prompt_id {};
    uint32_t token {};
    uint32_t token_pos {};
    uint8_t temperature {}; // compact temprature, between [0, 255]; has to be divided by 255.0f before use.
    uint32_t prompt_length {};
    bool finished { false };
  };

  Metadata metadata_ {};
  std::vector<DiscardedContext> discarded_contexts_ {};
  std::vector<PromptData> prompts_ {};

  // we use contiguous memory for activations, queries, and key-values; allowing one-shot copying.
  DataBuffer activations_ {};
  DataBuffer queries_ {};
  DataBuffer kvs_ {};

  size_t activation_len() const { return Config::dim * DataTypeSize( metadata_.dtype ); }
  size_t q_len() const { return Config::dim * DataTypeSize( metadata_.dtype ); }
  size_t kv_len() const { return 2 * Config::kv_dim * DataTypeSize( metadata_.dtype ); }

  uint8_t* activation_ptr( const size_t i ) { return activations_.data() + i * activation_len(); }
  uint8_t* q_ptr( const size_t i ) { return queries_.data() + i * q_len(); }
  uint8_t* kv_ptr( const size_t i ) { return kvs_.data() + i * kv_len(); }

public:
  BatchedInferenceState( uint32_t batch_size,
                         DataType dtype,
                         RouteID route_id,
                         ModelID model_id,
                         const bool state_has_activations,
                         const bool state_has_queries,
                         const bool state_has_kvs );

  // TODO(sadjad) eventually we got to get rid of the default constructor.
  BatchedInferenceState()
    : BatchedInferenceState( 0, DataType::Float32, {}, {}, {}, {}, {} )
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
  void set_next_layer( uint32_t next_layer ) { metadata_.next_layer = next_layer; }
  void set_next_stage( Stage next_stage ) { metadata_.next_stage = next_stage; }
  void set_has_activations( bool has_activations ) { metadata_.has_activations = has_activations; }
  void set_has_queries( bool has_queries ) { metadata_.has_queries = has_queries; }
  void set_has_kvs( bool has_kvs ) { metadata_.has_kvs = has_kvs; }

  void clear_discards()
  {
    discarded_contexts_.clear();
    metadata_.discarded_contexts = 0;
  }

  // metadata getters
  uint32_t batch_size() const { return metadata_.batch_size; }
  DataType dtype() const { return metadata_.dtype; }
  RouteID route_id() const { return metadata_.route_id; }
  ModelID model_id() const { return metadata_.model_id; }
  uint32_t next_layer() const { return metadata_.next_layer; }
  Stage next_stage() const { return metadata_.next_stage; }
  bool has_activations() const { return metadata_.has_activations; }
  bool has_queries() const { return metadata_.has_queries; }
  bool has_kvs() const { return metadata_.has_kvs; }
  uint32_t discarded_contexts() const { return metadata_.discarded_contexts; }

  const PromptID& discarded_prompt_id( const size_t i ) const { return discarded_contexts_.at( i ).prompt_id; }

  // prompt setters
  void set_prompt( const size_t i,
                   PromptID prompt_id,
                   uint32_t token,
                   uint32_t token_pos,
                   float temperature,
                   uint32_t prompt_length );

  // prompt getters
  PromptID prompt_id( const size_t i ) const { return prompts_[i].prompt_id; }
  uint32_t token( const size_t i ) const { return prompts_[i].token; }
  uint32_t token_pos( const size_t i ) const { return prompts_[i].token_pos; }
  uint32_t prompt_length( const size_t i ) const { return prompts_[i].prompt_length; }
  float temperature( const size_t i ) const { return prompts_[i].temperature / 255.0f; }
  bool finished( const size_t i ) const { return prompts_[i].finished; }
  bool active( const size_t i ) const { return prompts_[i].active; }

  // prompt setters
  void set_prompt_id( const size_t i, PromptID prompt_id ) { prompts_[i].prompt_id = prompt_id; }
  void set_token( const size_t i, uint32_t token ) { prompts_[i].token = token; }
  void set_token_pos( const size_t i, uint32_t token_pos ) { prompts_[i].token_pos = token_pos; }
  void set_prompt_length( const size_t i, uint32_t prompt_length ) { prompts_[i].prompt_length = prompt_length; }
  void set_temperature( const size_t i, float t ) { prompts_[i].temperature = static_cast<uint8_t>( t * 255.0f ); }
  void set_finished( const size_t i ) { prompts_[i].finished = true; }
  void set_discarded( const size_t i );

  // The memory is owned by the inference state; be careful with the lifetime of the returned spans.
  std::span<uint8_t> activations( const size_t i ) { return { activation_ptr( i ), activation_len() }; }
  std::span<uint8_t> q( const size_t i ) { return { q_ptr( i ), q_len() }; }
  std::span<uint8_t> kv( const size_t i ) { return { kv_ptr( i ), kv_len() }; }

  std::span<const uint8_t> activations( const size_t i ) const { return { activation_ptr( i ), activation_len() }; }
  std::span<const uint8_t> q( const size_t i ) const { return { q_ptr( i ), q_len() }; }
  std::span<const uint8_t> kv( const size_t i ) const { return { kv_ptr( i ), kv_len() }; }

  DataBuffer& activations() { return activations_; }
  DataBuffer& queries() { return queries_; }
  DataBuffer& kvs() { return kvs_; }

  const DataBuffer& activations() const { return activations_; }
  const DataBuffer& queries() const { return queries_; }
  const DataBuffer& kvs() const { return kvs_; }

  void allocate_activations();
  void allocate_queries();
  void allocate_kvs();

  void deallocate_activations();
  void deallocate_queries();
  void deallocate_kvs();

  size_t free_slots() const
  {
    return std::count_if( prompts_.begin(), prompts_.end(), []( const auto& p ) { return !p.active; } );
  }

  /// @brief Replace the inactive prompts in the current state with the active prompts from the other state.
  /// @param other The state to replenish from.
  /// @note Modifies both the current state and the other state by moving activations from the other state to current.
  /// @return True if all inactive prompts were replaced, false otherwise.
  bool replenish_from( BatchedInferenceState& other );

  std::string debug_string( const bool prompt_details ) const;
};

template<typename Config>
BatchedInferenceState<Config>::BatchedInferenceState( uint32_t batch_size,
                                                      DataType dtype,
                                                      RouteID route_id,
                                                      ModelID model_id,
                                                      const bool state_has_activations,
                                                      const bool state_has_queries,
                                                      const bool state_has_kvs )
{
  metadata_.batch_size = batch_size;
  metadata_.dtype = dtype;
  metadata_.route_id = route_id;
  metadata_.model_id = model_id;
  metadata_.has_activations = state_has_activations;
  metadata_.has_queries = state_has_queries;
  metadata_.has_kvs = state_has_kvs;

  prompts_.resize( metadata_.batch_size );

  if ( state_has_activations ) {
    allocate_activations();
  }

  if ( state_has_queries ) {
    allocate_queries();
  }

  if ( state_has_kvs ) {
    allocate_kvs();
  }
}

template<typename Config>
BatchedInferenceState<Config>::BatchedInferenceState( const std::string_view serialized_state )
{
  auto ptr = serialized_state.data();

  // we need to make sure that the serialized state is at least as big as the metadata
  size_t expected_size = sizeof( Metadata );
  CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain metadata";

  metadata_ = *reinterpret_cast<const Metadata*>( ptr );
  ptr += sizeof( Metadata );

  expected_size += sizeof( DiscardedContext ) * metadata_.discarded_contexts;
  expected_size += sizeof( PromptData ) * metadata_.batch_size;

  discarded_contexts_.resize( metadata_.discarded_contexts );
  std::memcpy( reinterpret_cast<char*>( discarded_contexts_.data() ),
               ptr,
               metadata_.discarded_contexts * sizeof( DiscardedContext ) );
  ptr += metadata_.discarded_contexts * sizeof( DiscardedContext );

  prompts_.resize( metadata_.batch_size );

  std::memcpy( reinterpret_cast<char*>( prompts_.data() ), ptr, metadata_.batch_size * sizeof( PromptData ) );
  ptr += metadata_.batch_size * sizeof( PromptData );

  if ( has_activations() ) {
    expected_size += metadata_.batch_size * activation_len();
    CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain activations";

    activations_ = DataBuffer( metadata_.batch_size * activation_len() );
    std::memcpy( activations_.data(), ptr, metadata_.batch_size * activation_len() );
    ptr += metadata_.batch_size * activation_len();
  }

  if ( has_queries() ) {
    expected_size += metadata_.batch_size * q_len();
    CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain queries";

    queries_ = DataBuffer( metadata_.batch_size * q_len() );
    std::memcpy( queries_.data(), ptr, metadata_.batch_size * q_len() );
    ptr += metadata_.batch_size * q_len();
  }

  if ( has_kvs() ) {
    expected_size += metadata_.batch_size * kv_len();
    CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain key-values";

    kvs_ = DataBuffer( metadata_.batch_size * kv_len() );
    std::memcpy( kvs_.data(), ptr, metadata_.batch_size * kv_len() );
    ptr += metadata_.batch_size * kv_len();
  }

  CHECK_EQ( ptr, serialized_state.data() + serialized_state.size() )
    << "Serialized state contains more data than expected";
}

template<typename Config>
std::string BatchedInferenceState<Config>::serialize() const
{
  std::string serialized_state;
  const size_t expected_size = sizeof( Metadata )                                                    /* metadata */
                               + sizeof( DiscardedContext ) * discarded_contexts_.size()             /* discards */
                               + sizeof( PromptData ) * metadata_.batch_size                         /* prompts */
                               + ( has_activations() ? metadata_.batch_size * activation_len() : 0 ) /* activations */
                               + ( has_queries() ? metadata_.batch_size * q_len() : 0 )              /* queries */
                               + ( has_kvs() ? metadata_.batch_size * kv_len() : 0 );                /* kvs */

  // reserve enough space for the state
  serialized_state.reserve( expected_size );

  serialized_state.append( reinterpret_cast<const char*>( &metadata_ ), sizeof( Metadata ) );

  serialized_state.append( reinterpret_cast<const char*>( discarded_contexts_.data() ),
                           discarded_contexts_.size() * sizeof( DiscardedContext ) );

  serialized_state.append( reinterpret_cast<const char*>( prompts_.data() ),
                           metadata_.batch_size * sizeof( PromptData ) );

  if ( has_activations() ) {
    serialized_state.append( reinterpret_cast<const char*>( activations_.data() ),
                             metadata_.batch_size * activation_len() );
  }

  if ( has_queries() ) {
    serialized_state.append( reinterpret_cast<const char*>( queries_.data() ), metadata_.batch_size * q_len() );
  }

  if ( has_kvs() ) {
    serialized_state.append( reinterpret_cast<const char*>( kvs_.data() ), metadata_.batch_size * kv_len() );
  }

  CHECK_EQ( serialized_state.size(), expected_size ) << "Serialized state size mismatch";
  return serialized_state;
}

template<typename Config>
void BatchedInferenceState<Config>::set_prompt( const size_t i,
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

template<typename Config>
void BatchedInferenceState<Config>::set_discarded( const size_t i )
{
  // XXX this function should only be called by the first worker in a chain
  CHECK( metadata_.next_stage == Stage::PreAttention ) << "Discarding prompts in a non-PreAttention stage";
  CHECK_EQ( metadata_.next_layer, 0 ) << "Discarding prompts in a non-0 layer";

  discarded_contexts_.push_back( { prompts_[i].prompt_id } );
  prompts_[i] = {};
  prompts_[i].active = false;
}

template<typename Config>
void BatchedInferenceState<Config>::allocate_activations()
{
  activations_ = { metadata_.batch_size * activation_len() };
  set_has_activations( true );
}

template<typename Config>
void BatchedInferenceState<Config>::allocate_queries()
{
  queries_ = { metadata_.batch_size * q_len() };
  set_has_queries( true );
}

template<typename Config>
void BatchedInferenceState<Config>::allocate_kvs()
{
  kvs_ = { metadata_.batch_size * kv_len() };
  set_has_kvs( true );
}

template<typename Config>
void BatchedInferenceState<Config>::deallocate_activations()
{
  activations_ = {};
  set_has_activations( false );
}

template<typename Config>
void BatchedInferenceState<Config>::deallocate_queries()
{
  queries_ = {};
  set_has_queries( false );
}

template<typename Config>
void BatchedInferenceState<Config>::deallocate_kvs()
{
  kvs_ = {};
  set_has_kvs( false );
}

template<typename Config>
bool BatchedInferenceState<Config>::replenish_from( BatchedInferenceState& other )
{
  CHECK_EQ( metadata_.batch_size, other.metadata_.batch_size ) << "States with different batch sizes";
  CHECK( metadata_.dtype == other.metadata_.dtype ) << "States with different data types";
  CHECK_EQ( metadata_.route_id, other.metadata_.route_id ) << "States with different route IDs";
  CHECK_EQ( metadata_.model_id, other.metadata_.model_id ) << "States with different model IDs";
  CHECK_EQ( metadata_.next_layer, other.metadata_.next_layer ) << "States with different next layers";
  CHECK( metadata_.next_stage == other.metadata_.next_stage ) << "States with different next stages";
  CHECK_EQ( metadata_.has_activations, other.metadata_.has_activations ) << "States with different activation states";
  CHECK_EQ( metadata_.has_queries, other.metadata_.has_queries ) << "States with different query states";
  CHECK_EQ( metadata_.has_kvs, other.metadata_.has_kvs ) << "States with different key-value states";

  // copy the discard list
  metadata_.discarded_contexts += other.metadata_.discarded_contexts;
  discarded_contexts_.insert(
    discarded_contexts_.end(), other.discarded_contexts_.begin(), other.discarded_contexts_.end() );

  other.clear_discards();

  size_t other_idx = 0;
  for ( size_t my_idx = 0; my_idx < prompts_.size(); my_idx++ ) {
    auto& prompt = prompts_[my_idx];

    if ( prompt.active ) {
      continue;
    }

    // we need to replace this with an active prompt from the other state
    while ( not other.prompts_[other_idx].active ) {
      other_idx++;

      if ( other_idx >= other.prompts_.size() ) {
        // no more active prompts in the other state
        return false;
      }
    }

    prompt = other.prompts_[other_idx];

    if ( metadata_.has_activations ) {
      std::memcpy( activation_ptr( my_idx ), other.activation_ptr( other_idx ), activation_len() );
    }

    if ( metadata_.has_queries ) {
      std::memcpy( q_ptr( my_idx ), other.q_ptr( other_idx ), q_len() );
    }

    if ( metadata_.has_kvs ) {
      std::memcpy( kv_ptr( my_idx ), other.kv_ptr( other_idx ), kv_len() );
    }

    other.prompts_[other_idx] = {};
    other.prompts_[other_idx].active = false;
    other_idx++;
  }

  return true;
}

template<typename Config>
std::string BatchedInferenceState<Config>::debug_string( const bool prompt_details ) const
{
  std::ostringstream oss;
  oss << "BatchedInferenceState(" << "batch_size=" << metadata_.batch_size << ", " << "dtype=" << metadata_.dtype
      << ", " << "route_id=" << metadata_.route_id << ", " << "model_id=" << metadata_.model_id << ", "
      << "next_layer=" << metadata_.next_layer << ", " << "next_stage=" << metadata_.next_stage << ", "
      << "has_activations=" << metadata_.has_activations << ", " << "activations.len=" << activations_.len() << ", "
      << "has_queries=" << metadata_.has_queries << ", " << "queries.len=" << queries_.len() << ", "
      << "has_kvs=" << metadata_.has_kvs << ", " << "kvs.len=" << kvs_.len() << ", " << "discarded_contexts=[ ";

  for ( const auto& d : discarded_contexts_ ) {
    oss << " " << d.prompt_id.base58digest().substr( 0, 8 );
  }

  oss << "], ";

  if ( prompt_details ) {
    oss << "prompts=[";

    for ( const auto& p : prompts_ ) {
      oss << " (" << p.prompt_id.base58digest().substr( 0, 8 ) << ", " << p.token << ", " << p.token_pos << ", "
          << ( p.temperature / 255.0f ) << ", " << p.prompt_length << ", " << p.finished << ") ";
    }

    oss << "]";
  } else {
    oss << "prompts.len=" << prompts_.size();
  }

  return oss.str();
}

} // namespace glinthawk::models
