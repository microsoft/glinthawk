#pragma once

#include <algorithm>
#include <glog/logging.h>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

#include "../llama2/variants.hh"
#include "../types.hh"

namespace glinthawk::models {

// TODO(pouya): double check if tier_rank is implemented correctly. (this file is huge!).
// TODO(pouya): double check if is_sharded implemented correctly.
// TODO(pouya): double check if merge is implemented correctly.
// TODO(pouya): revamp how discarded context works.
// TODO(pouya): design a lazy merge.
// TODO(pouya): revamp soft_split to be optimal for TierRouter.

struct __attribute__( ( packed ) ) StateMetadata
{
  uint64_t id {};

  uint32_t batch_size {};
  DataType dtype { DataType::Float32 };
  RouteID route_id {};
  ModelID model_id {};
  uint32_t next_layer { 0 };
  InferenceStage next_stage { InferenceStage::PreAttention };

  bool has_activations { false };
  bool has_queries { false };
  bool has_kvs { false };
  bool is_sharded { false };

  uint32_t discarded_contexts { 0 };
};

/// @brief This struct is used to keep track of prompts that are done, and the workers can safely discard them.
struct __attribute__( ( packed ) ) DiscardedContext
{
  // TODO(pouya): add tier ranks here
  PromptID prompt_id {};
};

struct __attribute__( ( packed ) ) PromptData
{
  bool active { false };

  PromptID prompt_id {};
  uint32_t token {};
  uint32_t token_pos {};
  uint8_t temperature {}; // compact temprature, between [0, 255]; has to be divided by 255.0f before use.
  uint32_t prompt_length {};
  bool finished { false };
  uint8_t rank_tier_1 {}; // Denotes which group in tier 1 this prompt needs to be forwarded to, 0-indexed, -1
                          // means it belongs in tier 2, if both are -1 this prompt has not been routed yet
  uint8_t rank_tier_2 {}; // Denotes which group in tier 2 this prompt needs to be forwarded to, 0-indexed, -1
                          // means it belongs in tier 1, if both are -1 this prompt has not been routed yet
};

template<typename T>
concept StateConcept = requires( T state, const T cstate, const std::string cstr ) {
  { T( cstr ) } -> std::same_as<T>;
  { state.serialize() } -> std::same_as<std::string>;

  { state.empty() } -> std::same_as<bool>;

  { state.set_id( 0 ) };
  { state.set_dtype( DataType::Float32 ) };
  { state.set_route_id( {} ) };
  { state.set_model_id( {} ) };
  { state.set_next_layer( 0 ) };
  { state.set_next_stage( InferenceStage::PreAttention ) };
  { state.set_has_activations( false ) };
  { state.set_has_queries( false ) };
  { state.set_has_kvs( false ) };
  { state.set_is_sharded( false ) };
  { state.clear_discards() };

  { cstate.id() } -> std::same_as<uint64_t>;
  { cstate.batch_size() } -> std::same_as<uint32_t>;
  { cstate.dtype() } -> std::same_as<DataType>;
  { cstate.route_id() } -> std::same_as<RouteID>;
  { cstate.model_id() } -> std::same_as<ModelID>;
  { cstate.next_layer() } -> std::same_as<uint32_t>;
  { cstate.next_stage() } -> std::same_as<InferenceStage>;
  { cstate.has_activations() } -> std::same_as<bool>;
  { cstate.has_queries() } -> std::same_as<bool>;
  { cstate.has_kvs() } -> std::same_as<bool>;
  { cstate.is_sharded() } -> std::same_as<bool>;
  { state.all_rank_assigned() } -> std::same_as<bool>;
  { cstate.discarded_contexts() } -> std::same_as<uint32_t>;

  { state.discarded_prompt_id( 0 ) } -> std::same_as<const PromptID&>;

  { state.set_prompt( 0, {}, 0, 0, 0.0f, 0, 0, 0 ) };
  { state.prompt_id( 0 ) } -> std::same_as<PromptID>;
  { state.token( 0 ) } -> std::same_as<uint32_t>;
  { state.token_pos( 0 ) } -> std::same_as<uint32_t>;
  { state.prompt_length( 0 ) } -> std::same_as<uint32_t>;
  { state.temperature( 0 ) } -> std::same_as<float>;
  { state.finished( 0 ) } -> std::same_as<bool>;
  { state.rank_tier_1( 0 ) } -> std::same_as<uint8_t>;
  { state.rank_tier_2( 0 ) } -> std::same_as<uint8_t>;
  { state.rank_assigned( 0 ) } -> std::same_as<bool>;
  { state.active( 0 ) } -> std::same_as<bool>;

  { state.set_prompt_id( 0, {} ) };
  { state.set_token( 0, 0 ) };
  { state.set_token_pos( 0, 0 ) };
  { state.set_prompt_length( 0, 0 ) };
  { state.set_temperature( 0, 0.0f ) };
  { state.set_finished( 0 ) };
  { state.set_rank_tier_1( 0, 0 ) };
  { state.set_rank_tier_2( 0, 0 ) };

  { state.discard( 0 ) };

  { state.activations( 0 ) } -> std::same_as<std::span<uint8_t>>;
  { state.q( 0 ) } -> std::same_as<std::span<uint8_t>>;
  { state.kv( 0 ) } -> std::same_as<std::span<uint8_t>>;

  { state.activations() } -> std::same_as<DataBuffer&>;
  { state.queries() } -> std::same_as<DataBuffer&>;
  { state.kvs() } -> std::same_as<DataBuffer&>;

  { cstate.activations( 0 ) } -> std::same_as<std::span<const uint8_t>>;
  { cstate.q( 0 ) } -> std::same_as<std::span<const uint8_t>>;
  { cstate.kv( 0 ) } -> std::same_as<std::span<const uint8_t>>;

  { cstate.activations() } -> std::same_as<const DataBuffer&>;
  { cstate.queries() } -> std::same_as<const DataBuffer&>;
  { cstate.kvs() } -> std::same_as<const DataBuffer&>;

  { state.allocate_activations() };
  { state.allocate_queries() };
  { state.allocate_kvs() };

  { state.deallocate_activations() };
  { state.deallocate_queries() };
  { state.deallocate_kvs() };

  { state.free_slots() } -> std::same_as<size_t>;

  { state.replenish_from( state ) } -> std::same_as<bool>;
  { state.split( 0 ) } -> std::same_as<std::pair<T, T>>;
  { state.merge( std::move( state ) ) };

  { cstate.debug_string() } -> std::same_as<std::string>;
  { cstate.debug_string( true ) } -> std::same_as<std::string>;
};

// Forward declaration for BatchedInferenceStateSpan

template<typename Config>
requires llama2::ModelConfig<Config>
class BatchedInferenceStateSpan;

// NOTE(sadjad): right now, inference state is designed to be used by Llama and Llama-like models. We need to work out
// the generality later.
template<typename Config>
requires llama2::ModelConfig<Config>
class BatchedInferenceState
{
private:
  StateMetadata metadata_ {};
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

  bool empty() const { return metadata_.batch_size == 0; }

  // metadata setters
  void set_id( const uint64_t id ) { metadata_.id = id; }
  void set_dtype( const DataType dtype ) { metadata_.dtype = dtype; }
  void set_route_id( const RouteID route_id ) { metadata_.route_id = route_id; }
  void set_model_id( const ModelID model_id ) { metadata_.model_id = model_id; }
  void set_next_layer( const uint32_t next_layer ) { metadata_.next_layer = next_layer; }
  void set_next_stage( const InferenceStage next_stage ) { metadata_.next_stage = next_stage; }
  void set_has_activations( const bool has_activations ) { metadata_.has_activations = has_activations; }
  void set_has_queries( const bool has_queries ) { metadata_.has_queries = has_queries; }
  void set_has_kvs( const bool has_kvs ) { metadata_.has_kvs = has_kvs; }
  void set_is_sharded( const bool is_sharded ) { metadata_.is_sharded = is_shared; }
  void clear_discards();

  // metadata getters
  uint64_t id() const { return metadata_.id; }
  uint32_t batch_size() const { return metadata_.batch_size; }
  DataType dtype() const { return metadata_.dtype; }
  RouteID route_id() const { return metadata_.route_id; }
  ModelID model_id() const { return metadata_.model_id; }
  uint32_t next_layer() const { return metadata_.next_layer; }
  InferenceStage next_stage() const { return metadata_.next_stage; }
  bool has_activations() const { return metadata_.has_activations; }
  bool has_queries() const { return metadata_.has_queries; }
  bool has_kvs() const { return metadata_.has_kvs; }
  bool is_sharded() const { return metadata_.is_sharded; }
  bool all_rank_assigned() const
  {
    for ( int i = 0; i < metadata_.batch_size; i++ ) {
      if ( not rank_assigned( i ) ) {
        return false;
      }
    }
    return true;
  }
  uint32_t discarded_contexts() const { return metadata_.discarded_contexts; }

  const PromptID& discarded_prompt_id( const size_t i ) const { return discarded_contexts_.at( i ).prompt_id; }

  // prompt setters
  void set_prompt( const size_t i,
                   PromptID prompt_id,
                   uint32_t token,
                   uint32_t token_pos,
                   float temperature,
                   uint32_t prompt_length,
                   uint8_t rank_tier_1,
                   uint8_t rank_tier_2 );

  // prompt getters
  PromptID prompt_id( const size_t i ) const { return prompts_[i].prompt_id; }
  uint32_t token( const size_t i ) const { return prompts_[i].token; }
  uint32_t token_pos( const size_t i ) const { return prompts_[i].token_pos; }
  uint32_t prompt_length( const size_t i ) const { return prompts_[i].prompt_length; }
  float temperature( const size_t i ) const { return prompts_[i].temperature / 255.0f; }
  bool finished( const size_t i ) const { return prompts_[i].finished; }
  uint8_t rank_tier_1( const size_t i ) const { return prompts_[i].rank_tier_1; }
  uint8_t rank_tier_2( const size_t i ) const { return prompts_[i].rank_tier_2; }
  bool rank_assigned( const size_t i ) const
  {
    // TODO(pouya): -1 on uint8_t is bad practice!
    return prompts_[i].rank_tier_1 != -1 or prompts_[i].rank_tier_2 != -1;
  }
  bool active( const size_t i ) const { return prompts_[i].active; }

  // prompt setters
  void set_prompt_id( const size_t i, PromptID prompt_id ) { prompts_[i].prompt_id = prompt_id; }
  void set_token( const size_t i, uint32_t token ) { prompts_[i].token = token; }
  void set_token_pos( const size_t i, uint32_t token_pos ) { prompts_[i].token_pos = token_pos; }
  void set_prompt_length( const size_t i, uint32_t prompt_length ) { prompts_[i].prompt_length = prompt_length; }
  void set_temperature( const size_t i, float t ) { prompts_[i].temperature = static_cast<uint8_t>( t * 255.0f ); }
  void set_finished( const size_t i ) { prompts_[i].finished = true; }
  void set_rank_tier_1( const size_t i, uint8_t rank_tier_1 ) { prompts_[i].rank_tier_1 = rank_tier_1; }
  void set_rank_tier_2( const size_t i, uint8_t rank_tier_2 ) { prompts_[i].rank_tier_2 = rank_tier_2; }

  void discard( const size_t i );

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

  size_t free_slots() const;

  /// @brief Replace the inactive prompts in the current state with the active prompts from the other state.
  /// @param other The state to replenish from.
  /// @note Modifies both the current state and the other state by moving activations from the other state to current.
  /// @return True if all inactive prompts were replaced, false otherwise.
  bool replenish_from( BatchedInferenceState& other );

  /// @brief Split the current state into two states of size n and batch_size - n.
  /// @param n The size of the first state.
  /// @return A pair of states.
  std::pair<BatchedInferenceState, BatchedInferenceState> split( const size_t n );

  /// @brief Like `split`, but creates spans of the current state instead of a new state.
  /// @param n The size of the first state span.
  /// @return A pair of StateSpans.
  std::pair<BatchedInferenceStateSpan<Config>, BatchedInferenceStateSpan<Config>> soft_split( const size_t n );

  /// @brief Merge the current state with another state. The difference between this and replenish_from is that this
  /// function merges the states, while replenish_from only replaces inactive prompts. The second state after merging
  /// will always be empty.
  /// @param other
  /// @return The merged state.
  void merge( BatchedInferenceState&& other );

  static BatchedInferenceState&& merge_states( std::vector<std::reference_wrapper<BatchedInferenceState>> vec_state );

  std::string debug_string( const bool prompt_details = false ) const;
};

static_assert( StateConcept<BatchedInferenceState<models::llama2::configs::Stories_110M>> );

template<typename Config>
requires llama2::ModelConfig<Config>
class BatchedInferenceStateSpan
{
private:
  // This is a sin; we will fix it later.
  BatchedInferenceState<Config>& state_;

  size_t off_;
  size_t n_;

public:
  BatchedInferenceStateSpan( BatchedInferenceState<Config>& state, size_t off, size_t n )
    : state_( state )
    , off_( off )
    , n_( n )
  {
    CHECK_LE( off + n, state_.batch_size() ) << "Span out of bounds";
  }

  // A span can only be created from an existing state object, and cannot be serialized either.
  // XXX(sadjad): catch this during compile time.
  BatchedInferenceStateSpan( const std::string_view serialized_state ) { throw std::runtime_error( "not available" ); }
  std::string serialize() const { throw std::runtime_error( "not available" ); }

  bool empty() const { return n_ == 0; }

  void set_id( const uint64_t id ) { state_.set_id( id ); }
  void set_dtype( const DataType dtype ) { state_.set_dtype( dtype ); }
  void set_route_id( const RouteID route_id ) { state_.set_route_id( route_id ); }
  void set_model_id( const ModelID model_id ) { state_.set_model_id( model_id ); }
  void set_next_layer( const uint32_t next_layer ) { state_.set_next_layer( next_layer ); }
  void set_next_stage( const InferenceStage next_stage ) { state_.set_next_stage( next_stage ); }
  void set_has_activations( const bool has_activations ) { state_.set_has_activations( has_activations ); }
  void set_has_queries( const bool has_queries ) { state_.set_has_queries( has_queries ); }
  void set_has_kvs( const bool has_kvs ) { state_.set_has_kvs( has_kvs ); }
  // TODO(pouya): if spans share metadata with the original, this might break tier_router
  void set_is_sharded( const bool is_sharded ) { state_.set_is_sharded( is_sharded ); }
  void clear_discards() { state_.clear_discards(); }

  uint64_t id() const { return state_.id(); }
  uint32_t batch_size() const { return n_; }
  DataType dtype() const { return state_.dtype(); }
  RouteID route_id() const { return state_.route_id(); }
  ModelID model_id() const { return state_.model_id(); }
  uint32_t next_layer() const { return state_.next_layer(); }
  InferenceStage next_stage() const { return state_.next_stage(); }
  bool has_activations() const { return state_.has_activations(); }
  bool has_queries() const { return state_.has_queries(); }
  bool has_kvs() const { return state_.has_kvs(); }
  bool is_sharded() const { return state_.is_sharded(); }
  bool all_rank_assigned() const
  {
    for ( int i = 0; i < n_; i++ ) {
      if ( not state_.rank_assigned( off_ + i ) )
        return false;
    }
    return true;
  }
  // TODO(pouya): if spans share metadata with the original, this might break tier_router
  uint32_t discarded_contexts() const { return state_.discarded_contexts(); }

  const PromptID& discarded_prompt_id( const size_t i ) const { return state_.discarded_prompt_id( i ); }

  void set_prompt( const size_t i,
                   PromptID prompt_id,
                   uint32_t token,
                   uint32_t token_pos,
                   float temperature,
                   uint32_t prompt_length,
                   uint8_t rank_tier_1,
                   uint8_t rank_tier_2 )
  {
    state_.set_prompt( off_ + i, prompt_id, token, token_pos, temperature, prompt_length, rank_tier_1, rank_tier_2 );
  }

  PromptID prompt_id( const size_t i ) const { return state_.prompt_id( off_ + i ); }
  uint32_t token( const size_t i ) const { return state_.token( off_ + i ); }
  uint32_t token_pos( const size_t i ) const { return state_.token_pos( off_ + i ); }
  uint32_t prompt_length( const size_t i ) const { return state_.prompt_length( off_ + i ); }
  float temperature( const size_t i ) const { return state_.temperature( off_ + i ); }
  bool finished( const size_t i ) const { return state_.finished( off_ + i ); }
  bool active( const size_t i ) const { return state_.active( off_ + i ); }
  uint8_t rank_tier_1( const size_t i ) const { return state_.rank_tier_1( off_ + i ); }
  uint8_t rank_tier_2( const size_t i ) const { return state_.rank_tier_2( off_ + i ); }
  bool rank_assigned( const size_t i ) const { return state_.rank_assigned( off_ + i ); }

  void set_prompt_id( const size_t i, PromptID prompt_id ) { state_.set_prompt_id( off_ + i, prompt_id ); }
  void set_token( const size_t i, uint32_t token ) { state_.set_token( off_ + i, token ); }
  void set_token_pos( const size_t i, uint32_t token_pos ) { state_.set_token_pos( off_ + i, token_pos ); }
  void set_prompt_length( const size_t i, uint32_t len ) { state_.set_prompt_length( off_ + i, len ); }
  void set_temperature( const size_t i, float t ) { state_.set_temperature( off_ + i, t ); }
  void set_finished( const size_t i ) { state_.set_finished( off_ + i ); }
  void set_rank_tier_1( const size_t i, uint8_t rank_tier_1 ) { state_.set_rank_tier_1( off_ + i, rank_tier_1 ); }
  void set_rank_tier_2( const size_t i, uint8_t rank_tier_2 ) { state_.set_rank_tier_2( off_ + i, rank_tier_2 ); }

  void discard( const size_t i ) { state_.discard( off_ + i ); }

  std::span<uint8_t> activations( const size_t i ) { return state_.activations( off_ + i ); }
  std::span<uint8_t> q( const size_t i ) { return state_.q( off_ + i ); }
  std::span<uint8_t> kv( const size_t i ) { return state_.kv( off_ + i ); }

  std::span<const uint8_t> activations( const size_t i ) const { return state_.activations( off_ + i ); }
  std::span<const uint8_t> q( const size_t i ) const { return state_.q( off_ + i ); }
  std::span<const uint8_t> kv( const size_t i ) const { return state_.kv( off_ + i ); }

  DataBuffer& activations() { return state_.activations(); }
  DataBuffer& queries() { return state_.queries(); }
  DataBuffer& kvs() { return state_.kvs(); }

  const DataBuffer& activations() const { return state_.activations(); }
  const DataBuffer& queries() const { return state_.queries(); }
  const DataBuffer& kvs() const { return state_.kvs(); }

  void allocate_activations() { state_.allocate_activations(); }
  void allocate_queries() { state_.allocate_queries(); }
  void allocate_kvs() { state_.allocate_kvs(); }

  void deallocate_activations() { state_.deallocate_activations(); }
  void deallocate_queries() { state_.deallocate_queries(); }
  void deallocate_kvs() { state_.deallocate_kvs(); }

  size_t free_slots() const { return state_.free_slots(); }

  bool replenish_from( BatchedInferenceStateSpan& other ) { throw std::runtime_error( "not available" ); }

  std::pair<BatchedInferenceStateSpan, BatchedInferenceStateSpan> split( const size_t n )
  {
    throw std::runtime_error( "not available" );
  }

  void merge( BatchedInferenceStateSpan&& other ) { throw std::runtime_error( "not available" ); }
  std::string debug_string( const bool prompt_details = false ) const { return state_.debug_string( prompt_details ); }
};

static_assert( StateConcept<BatchedInferenceStateSpan<models::llama2::configs::Stories_110M>> );

/*** Implementations ***/

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
  // TODO(pouya): check with sadjad if he prefers default initializations over this, or explicit initializations.
  metadata_.is_sharded = false;

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
  size_t expected_size = sizeof( StateMetadata );
  CHECK_GE( serialized_state.size(), expected_size ) << "Serialized state is too small to contain metadata";

  metadata_ = *reinterpret_cast<const StateMetadata*>( ptr );
  ptr += sizeof( StateMetadata );

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
  const size_t expected_size = sizeof( StateMetadata )                                               /* metadata */
                               + sizeof( DiscardedContext ) * discarded_contexts_.size()             /* discards */
                               + sizeof( PromptData ) * metadata_.batch_size                         /* prompts */
                               + ( has_activations() ? metadata_.batch_size * activation_len() : 0 ) /* activations */
                               + ( has_queries() ? metadata_.batch_size * q_len() : 0 )              /* queries */
                               + ( has_kvs() ? metadata_.batch_size * kv_len() : 0 );                /* kvs */

  // reserve enough space for the state
  serialized_state.reserve( expected_size );

  serialized_state.append( reinterpret_cast<const char*>( &metadata_ ), sizeof( StateMetadata ) );

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
                                                uint32_t prompt_length,
                                                uint8_t rank_tier_1,
                                                uint8_t rank_tier_2 )
{
  prompts_[i].prompt_id = prompt_id;
  prompts_[i].token = token;
  prompts_[i].token_pos = token_pos;
  prompts_[i].temperature = static_cast<uint8_t>( temperature * 255.0f );
  prompts_[i].prompt_length = prompt_length;
  prompts_[i].finished = false;
  prompts_[i].active = true;
  prompts_[i].rank_tier_1 = rank_tier_1;
  prompts_[i].rank_tier_2 = rank_tier_2;
}

template<typename Config>
void BatchedInferenceState<Config>::discard( const size_t i )
{
  // XXX this function should only be called by the first worker in a chain
  CHECK( metadata_.next_stage == InferenceStage::PreAttention ) << "Discarding prompts in a non-PreAttention stage";
  CHECK_EQ( metadata_.next_layer, 0 ) << "Discarding prompts in a non-0 layer";

  discarded_contexts_.push_back( { prompts_[i].prompt_id } );
  metadata_.discarded_contexts++;
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
  CHECK_EQ( metadata_.is_sharded, other.metadata_.is_sharded ) << "Sharded and Monolithic states";

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

    // We can only replenish when (1) either the other prompt is not rank assigned, and we can assign it to whatever
    // rank the discarded prompt had, or (2) they have the same rank.
    // So this condition won't trigger if 'other' is entirely unassigned. The second condition should ideally never get
    // triggered, since it is hard to reason about it before runtime.
    if ( not other.prompts_[other_idx].rank_assigned ) {
      other.prompts_[other_idx].tier_1_rank = prompt.tier_1_rank;
      other.prompts_[other_idx].tier_2_rank = prompt.tier_2_rank;
    } else {
      CHECK_EQ( other.prompts_[other_idx].tier_1_rank, prompt.tier_1_rank );
      CHECK_EQ( other.prompts_[other_idx].tier_2_rank, prompt.tier_2_rank );
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
    other_idx++;
  }

  return true;
}

template<typename Config>
void BatchedInferenceState<Config>::clear_discards()
{
  discarded_contexts_.clear();
  metadata_.discarded_contexts = 0;
}

template<typename Config>
size_t BatchedInferenceState<Config>::free_slots() const
{
  return std::count_if( prompts_.begin(), prompts_.end(), []( const auto& p ) { return not p.active; } );
}

template<typename Config>
std::pair<BatchedInferenceState<Config>, BatchedInferenceState<Config>> BatchedInferenceState<Config>::split(
  const size_t n )
{
  CHECK_LT( n, metadata_.batch_size ) << "n must be less than the batch size";
  CHECK_GT( n, 0 ) << "n must be greater than 0";

  DLOG( INFO ) << "Splitting state of size " << metadata_.batch_size << " into states of sizes " << n << " and "
               << ( metadata_.batch_size - n ) << ".";

  const size_t size_a = n;
  const size_t size_b = metadata_.batch_size - n;

  BatchedInferenceState<Config> state_a, state_b;

  state_a.metadata_ = metadata_;
  state_a.metadata_.batch_size = size_a;
  state_a.prompts_.resize( size_a );

  state_b.metadata_ = metadata_;
  state_b.metadata_.batch_size = size_b;
  state_b.prompts_.resize( size_b );

  if ( this->has_activations() ) {
    state_a.allocate_activations();
    state_b.allocate_activations();
  }

  if ( this->has_queries() ) {
    state_a.allocate_queries();
    state_b.allocate_queries();
  }

  if ( this->has_kvs() ) {
    state_a.allocate_kvs();
    state_b.allocate_kvs();
  }

  // TODO(pouya): moving all discarded contexts to one state is bad
  // moving all the discarded contexts to the first state
  state_a.discarded_contexts_ = discarded_contexts_;
  state_a.metadata_.discarded_contexts = metadata_.discarded_contexts;
  state_b.discarded_contexts_.clear();
  state_b.metadata_.discarded_contexts = 0;

  for ( size_t i = 0; i < size_a; i++ ) {
    state_a.prompts_[i] = prompts_[i];
  }

  for ( size_t i = 0; i < size_b; i++ ) {
    state_b.prompts_[i] = prompts_[i + size_a];
  }

  if ( metadata_.has_activations ) {
    std::memcpy( state_a.activations_.data(), activations_.data(), size_a * activation_len() );
    std::memcpy(
      state_b.activations_.data(), activations_.data() + size_a * activation_len(), size_b * activation_len() );
  }

  if ( metadata_.has_queries ) {
    std::memcpy( state_a.queries_.data(), queries_.data(), size_a * q_len() );
    std::memcpy( state_b.queries_.data(), queries_.data() + size_a * q_len(), size_b * q_len() );
  }

  if ( metadata_.has_kvs ) {
    std::memcpy( state_a.kvs_.data(), kvs_.data(), size_a * kv_len() );
    std::memcpy( state_b.kvs_.data(), kvs_.data() + size_a * kv_len(), size_b * kv_len() );
  }

  return std::make_pair( std::move( state_a ), std::move( state_b ) );
}

template<typename Config>
std::pair<BatchedInferenceStateSpan<Config>, BatchedInferenceStateSpan<Config>>
BatchedInferenceState<Config>::soft_split( const size_t n )
{
  CHECK_LT( n, metadata_.batch_size ) << "n must be less than the batch size";
  CHECK_GT( n, 0 ) << "n must be greater than 0";

  DLOG( INFO ) << "Splitting state of size " << metadata_.batch_size << " into states of sizes " << n << " and "
               << ( metadata_.batch_size - n ) << ".";

  BatchedInferenceStateSpan<Config> state_a( *this, 0, n );
  BatchedInferenceStateSpan<Config> state_b( *this, n, metadata_.batch_size - n );

  return std::make_pair( state_a, state_b );
}

template<typename Config>
void BatchedInferenceState<Config>::merge( BatchedInferenceState&& other )
{
  CHECK_GT( metadata_.batch_size + other.metadata_.batch_size, 0 ) << "Merging two empty states";

  // merging into an empty state
  if ( metadata_.batch_size == 0 ) {
    *this = std::move( other );
    return;
  }

  // merging an empty state
  if ( other.metadata_.batch_size == 0 ) {
    return;
  }

  CHECK( metadata_.dtype == other.metadata_.dtype ) << "States with different data types";
  CHECK_EQ( metadata_.route_id, other.metadata_.route_id ) << "States with different route IDs";
  CHECK_EQ( metadata_.model_id, other.metadata_.model_id ) << "States with different model IDs";
  CHECK_EQ( metadata_.next_layer, other.metadata_.next_layer ) << "States with different next layers";
  CHECK( metadata_.next_stage == other.metadata_.next_stage ) << "States with different next stages";
  CHECK_EQ( metadata_.has_activations, other.metadata_.has_activations ) << "States with different activation states";
  CHECK_EQ( metadata_.has_queries, other.metadata_.has_queries ) << "States with different query states";
  CHECK_EQ( metadata_.has_kvs, other.metadata_.has_kvs ) << "States with different key-value states";
  CHECK_EQ( metadata_.is_sharded, other.metadata_.is_sharded ) << "Sharded and Monolithic states";

  BatchedInferenceState new_state;
  new_state.metadata_ = metadata_;
  new_state.metadata_.batch_size += other.metadata_.batch_size;

  // TODO(pouya): how about the discarded contexts?

  new_state.prompts_.resize( metadata_.batch_size + other.metadata_.batch_size );

  // copying prompt data
  for ( size_t i = 0; i < metadata_.batch_size; i++ ) {
    new_state.prompts_[i] = prompts_[i];
  }

  for ( size_t i = 0; i < other.metadata_.batch_size; i++ ) {
    new_state.prompts_[i + metadata_.batch_size] = other.prompts_[i];
  }

  if ( metadata_.has_activations ) {
    new_state.allocate_activations();
    std::memcpy( new_state.activations_.data(), activations_.data(), metadata_.batch_size * activation_len() );
    std::memcpy( new_state.activations_.data() + metadata_.batch_size * activation_len(),
                 other.activations_.data(),
                 other.metadata_.batch_size * activation_len() );
  }

  if ( metadata_.has_queries ) {
    new_state.allocate_queries();
    std::memcpy( new_state.queries_.data(), queries_.data(), metadata_.batch_size * q_len() );
    std::memcpy( new_state.queries_.data() + metadata_.batch_size * q_len(),
                 other.queries_.data(),
                 other.metadata_.batch_size * q_len() );
  }

  if ( metadata_.has_kvs ) {
    new_state.allocate_kvs();
    std::memcpy( new_state.kvs_.data(), kvs_.data(), metadata_.batch_size * kv_len() );
    std::memcpy( new_state.kvs_.data() + metadata_.batch_size * kv_len(),
                 other.kvs_.data(),
                 other.metadata_.batch_size * kv_len() );
  }

  *this = std::move( new_state );
  other = {};
}

template<typename Config>
static BatchedInferenceState&& BatchedInferenceState<Config>::merge_states(
  std::vector<std::reference_wrapper<BatchedInferenceState>> vec_state )
{
  // TODO(pouya): how about the discarded contexts?
  CHECK_GT( vec_state.size(), 1 ) << "Merging empty or single-element list";

  BatchedInferenceState new_state;
  new_state.metadata_ = vec_state[0]->get().metadata_;
  new_state.metadata_.batch_size = 0;
  for ( int i = 0; i < vec_state.size(); i++ ) {
    new_state.metadata_.batch_size += vec_state[i]->get().metadata_.batch_size;
  }
  CHECK_GT( new_state.metadata_.batch_size, 0 ) << "Merging empty states";
  new_state.prompts_.resize( new_state.metadata_.batch_size );

  if ( new_state.metadata_.has_activations ) {
    new_state.allocate_activations();
  }

  if ( new_state.metadata_.has_queries ) {
    new_state.allocate_queries();
  }

  if ( new_state.metadata_.has_kvs ) {
    new_state.allocate_kvs();
  }

  size_t last_bi = 0;
  for ( int i = 0; i < vec_state.size(); i++ ) {
    const auto other = std::move( vec_state[i]->get() );

    // merging an empty state
    if ( other.metadata_.batch_size == 0 ) {
      other = {};
      continue;
    }

    CHECK( new_state.metadata_.dtype == other.metadata_.dtype ) << "States with different data types";
    CHECK_EQ( new_state.metadata_.route_id, other.metadata_.route_id ) << "States with different route IDs";
    CHECK_EQ( new_state.metadata_.model_id, other.metadata_.model_id ) << "States with different model IDs";
    CHECK_EQ( new_state.metadata_.next_layer, other.metadata_.next_layer ) << "States with different next layers";
    CHECK( new_state.metadata_.next_stage == other.metadata_.next_stage ) << "States with different next stages";
    CHECK_EQ( new_state.metadata_.has_activations, other.metadata_.has_activations )
      << "States with different activation states";
    CHECK_EQ( new_state.metadata_.has_queries, other.metadata_.has_queries ) << "States with different query states";
    CHECK_EQ( new_state.metadata_.has_kvs, other.metadata_.has_kvs ) << "States with different key-value states";
    CHECK_EQ( new_state.metadata_.is_sharded, other.metadata_.is_sharded ) << "Sharded and Monolithic states";

    // copying prompt data
    for ( size_t j = 0; j < other.metadata_.batch_size; j++ ) {
      new_state.prompts_[last_bi + j] = other.prompts_[j];
    }

    if ( new_state.metadata_.has_activations ) {
      std::memcpy( new_state.activations_.data() + last_bi * new_state.activation_len(),
                   other.activations_.data(),
                   other.metadata_.batch_size * new_state.activation_len() );
    }

    if ( new_state.metadata_.has_queries ) {
      std::memcpy( new_state.queries_.data() + last_bi * new_state.q_len(),
                   other.queries_.data(),
                   other.metadata_.batch_size * new_state.q_len() );
    }

    if ( new_state.metadata_.has_kvs ) {
      std::memcpy( new_state.kvs_.data() + last_bi * new_state.kv_len(),
                   other.kvs_.data(),
                   other.metadata_.batch_size * new_state.kv_len() );
    }
    last_bi += other.metadata_.batch_size;

    other = {};
  }
  return new_state;
}

template<typename Config>
std::string BatchedInferenceState<Config>::debug_string( const bool prompt_details ) const
{
  std::ostringstream oss;
  oss << "BatchedInferenceState(" << "local_id=" << metadata_.id << ", " << "batch_size=" << metadata_.batch_size
      << ", " << "dtype=" << metadata_.dtype << ", " << "route_id=" << metadata_.route_id << ", "
      << "model_id=" << metadata_.model_id << ", " << "next_layer=" << metadata_.next_layer << ", "
      << "next_stage=" << metadata_.next_stage << ", " << "has_activations=" << metadata_.has_activations << ", "
      << "activations.len=" << activations_.len() << ", " << "has_queries=" << metadata_.has_queries << ", "
      << "queries.len=" << queries_.len() << ", " << "has_kvs=" << metadata_.has_kvs << ", " << "kvs.len=" << kvs_.len()
      << ", " << "is_sharded=" << metadata_.is_sharded << "discarded_contexts=[ ";

  for ( const auto& d : discarded_contexts_ ) {
    oss << " " << d.prompt_id.base58digest().substr( 0, 8 );
  }

  oss << "], ";

  if ( not prompt_details ) {
    oss << "prompts=[";

    for ( const auto& p : prompts_ ) {
      oss << " (" << p.prompt_id.base58digest().substr( 0, 8 ) << ", " << p.token << ", " << p.token_pos << ", "
          << ( p.temperature / 255.0f ) << ", " << p.prompt_length << ", " << p.finished << ", {" << p.rank_tier_1
          << ", " << p.rank_tier_2 << "}) ";
    }

    oss << "]";
  } else {
    oss << "prompts.len=" << prompts_.size();
  }

  oss << ")";

  return oss.str();
}

} // namespace glinthawk::models
