#pragma once

#include <array>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "util/eventfd.hh"
#include "util/eventloop.hh"
#include "util/util.hh"

#include "common.hh"
#include "contextman.hh"

namespace glinthawk::compute {

// Terminology:
// 'Node': a compute machine that runs forward operations for a number of layers.
// 'Tier': A set of nodes that have similar machinery, and may focus on different types of operations,
// e.g., compute-heavy operations (Post-Attention), memory-heavy tasks (e.g., Attention)
// 'Rank': an arbitrary index between nodes in a Tier.
// 'Slice': all nodes that serve an atomic unit of layers, e.g., if each node serves 2 layers, all nodes serving layers
//          0 and 1 are a slice.
// 'Monolithic State' or 'Monolith': a BatchedInferenceState that has not been broken to smaller pieces.
// 'Sharded State' or 'Shard': an atomic piece of a monolith that is only relevant to a single node.

/// @brief
/// TierRouter is a middle-man between the worker and kernel. Generally, it's job is to break down states between nodes,
/// manage context and route states to/from kernel.
template<typename ComputeKernel, typename ModelConfig>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
class TierRouter
{
public:
  explicit TierRouter( std::unique_ptr<ComputeKernel> compute_kernel )
    : compute_kernel_( std::move( compute_kernel ) ) {};

  virtual ~TierRouter() = default;

  EventFD& event_fd() { return event_fd_; }

  virtual void push( glinthawk::models::BatchedInferenceState<ModelConfig>&& state ) = 0;

  [[nodiscard]] virtual bool pop( glinthawk::models::BatchedInferenceState<ModelConfig>& state ) = 0;

  [[nodiscard]] virtual bool is_context_available() = 0;

protected:
  // Worker doesn't see the compute kernel. Only the tier router does.
  std::unique_ptr<ComputeKernel> compute_kernel_ {};

  EventFD event_fd_ {};
};

/// @brief
/// NOTE: in this simple version of ParentTierRouter, only the parent node (tier=0, rank=0) runs non-attention
/// computation.
///
///
/// ParentTierRouter is only run on tier=0_rank=0 machines, i.e., it only runs on one node in each `slice' that may
/// serve multiple layers. The other nodes in the slice have a ChildTierRouter that pass states through with no delay.
/// There is a possibility for improving this, where the tier-to-tier gather and scatter operations are distributed.
/// ParentTierRouter distinguishes states as two types:
///     Monolithic BatchInferenceState: which is a full batch across all machines in both tiers. It only exists
///         transiently inside the TierRouter, and in the very first worker when the state is first created.
///     Sharded BatchInferenceState: which is a specific part of a monolith dedicated to a specific machine.
/// ParentTierRouter receives monolithic/sharded states through push, and returns monolithic/sharded states through an
/// outgoing queue. It sends local shards to and receives them from the compute kernel directly.
/// The worker reads an outgoing queue from ParentTierRouter to send out outbound states.
/// ParentTierRouter fully handles context management, and guarantees that if a shard arrives at the kernel, that
/// kernel has space for it.
template<typename ComputeKernel, typename ModelConfig>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
class ParentTierRouter : public TierRouter<ComputeKernel, ModelConfig>
{
public:
  ParentTierRouter( std::unique_ptr<ComputeKernel> compute_kernel,
                    const SliceConcurrency& concurrency_s,
                    std::vector<size_t>& kv_slots_tier_s,
                    size_t start_layer,
                    size_t end_layer );

  ~ParentTierRouter() override { LOG( INFO ) << "ParentTierRouter shutting down..."; };

  /// @brief
  /// 1. Push monolithic state from worker -> calls process_monolith
  /// 2. Push sharded state from worker -> process_shard, may internally call process_monolith
  void push( glinthawk::models::BatchedInferenceState<ModelConfig>&& state ) override;

  /// @brief
  /// pop is called by the worker to receive states, that are:
  /// 1. monoliths that are no longer local to the node.
  /// 2. shards that the compute kernel processed.
  /// 3. shards that resulted from a monolith being broken.
  [[nodiscard]] bool pop( glinthawk::models::BatchedInferenceState<ModelConfig>& state ) override;

  /// @brief
  /// is_context_available checks if we have context available for attention for all layers in this slice. It returns
  /// false if even one node does not have the corresponding concurrency KV slots. Worker calls this function *before*
  /// pushing new prompts, but does not need to check when receiving states over the network.
  /// This is because we are assuming all slices are alike, and if slice 0 has context, so do the others. Thus, this
  /// function is truly only relevant in slice 0.
  /// This function will only return "true" a finite number of times before all contexts are filled up. From that point,
  /// new prompts are placed in discarded prompt locations, and will have context by default.
  [[nodiscard]] bool is_context_available() override;

protected:
  using Stage = glinthawk::models::InferenceStage;
  using StateType = typename glinthawk::models::BatchedInferenceState<ModelConfig>;
  using Shards = typename glinthawk::compute::ShardContainer<ModelConfig>;

  /// @brief
  /// For now we assume:
  /// 1. All tier "i" machines are alike.
  /// 2. All slices are alike.
  /// 3. Classification is done on the same machine doing pre-att-post.
  /// 4. Batch sizes are equal across stages
  /// The ParentTierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of
  /// two concurrencies.
  const SliceConcurrency concurrency_;
  std::vector<size_t> free_contexts_ {};

  const size_t start_layer_;
  const size_t end_layer_;

  // Node, Rank -> Shards
  std::vector<std::vector<Shards>> attention_idle_shards_ {};
  // Layer, Stage -> Shards
  std::vector<Shards> non_attention_idle_shards_ {};

  std::mutex ctx_mutex_ {};
  std::mutex shards_mutex_ {};

  GlobalQueue<ModelConfig> outgoing_ {};
  EventLoop event_loop_ {};
  std::jthread event_loop_thread_;

  [[nodiscard]] inline size_t vector_index( const size_t layer, const Stage stage ) const
  {
    return ( layer - start_layer_ ) * util::to_underlying( models::InferenceStage::__COUNT__ )
           + util::to_underlying( stage );
  }

  bool inline is_served_in_this_slice( const StateType& state ) const
  {
    // TODO(pouya): this is assuming classification is done on the same machine doing pre-att-post
    return state.next_layer() >= start_layer_ and state.next_layer() <= end_layer_;
  }

  /// @brief
  /// 1. Push sharded state from kernel -> process_shard, may internally call process_monolith
  void pull_from_kernel();

  void route_shard( StateType&& state );

  void clean_queue_if_dirty( Shards& queue, size_t target_conc );

  /// @brief
  /// assign_ranks assigns tier_routing_group indices to states that have not been assigned them before. The
  /// assignment is based on the index of each prompt in the state. Input states are either fully assigned (all slots
  /// in the batch inference state were assigned before) or none were assigned. If they were unassigned, the TierRouter
  /// "reserves" context for them as well. This function is called a finite number of times before all context slots are
  /// taken, and never called again. Note that when worker replaces a discarded prompt, it keeps any assigned
  /// tier_routing_group indices. This is important to avoid situations where new prompts are assigned to tier sub
  /// groups that their neighbors do not belong in, which causes fragmentation.
  void assign_ranks( StateType& state );

  /// @brief
  /// process_monolith breaks the state to sharded states by the tier_routing_group in each prompt. Each shard must have
  /// context on the remote/local machine for it. This is guaranteed by checking with is_context_available() before
  /// scheduling "new" states that are made from scratch. Old states with new prompts in discarded spots are by default
  /// going to have context.
  void process_monolith( StateType&& state );

  /// @brief
  /// process_shard manages an internal memory to merge states of various tiers together. If it receives a sharded
  /// state, it saves it. Upon merging, it may process it the same way it does the monolith. If the monolith does not
  /// need processing (is not local to that machine anymore, e.g., is for a different slice) it is sent out via an
  /// outgoing queue that the worker communicates with. This function may be indirectly called by the compute kernel or
  /// the worker.
  void process_shard( StateType&& state );

  void run_event_loop( std::stop_token stoken );
};

template<typename ComputeKernel, typename ModelConfig>
ParentTierRouter<ComputeKernel, ModelConfig>::ParentTierRouter( std::unique_ptr<ComputeKernel> compute_kernel,
                                                                const SliceConcurrency& concurrency,
                                                                std::vector<size_t>& kv_slots_tier_s,
                                                                size_t start_layer,
                                                                size_t end_layer )
  : TierRouter<ComputeKernel, ModelConfig>( std::move( compute_kernel ) )
  , concurrency_( concurrency )
  , free_contexts_( start_layer == 0 ? kv_slots_tier_s
                                     : std::vector<size_t> { static_cast<size_t>( concurrency_.num_tiers() ), 0 } )
  , start_layer_( start_layer )
  , end_layer_( end_layer )
  , non_attention_idle_shards_( ( end_layer_ - start_layer_ + 1 ) * util::to_underlying( Stage::__COUNT__ ) )
  , event_loop_thread_( std::bind( &ParentTierRouter::run_event_loop, this, std::placeholders::_1 ) )
{
  // Only one tier 0 machine.
  CHECK_EQ( concurrency_.num_ranks( 0 ), 1 );

  // TODO(pouya): if there is only one slice, a generated batch never gets pushed to worker to report generations.
  CHECK( start_layer_ != 0 or end_layer_ != ModelConfig::n_layers - 1 );

  for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
    CHECK( kv_slots_tier_s[tier_i] > 0
           or concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ) == 0 );
    attention_idle_shards_.emplace_back( concurrency_.num_ranks( tier_i ) );
    free_contexts_[tier_i] *= concurrency_.num_ranks( tier_i );
  }

  if constexpr ( ComputeKernel::Type == KernelType::Batched or ComputeKernel::Type == KernelType::SimpleHybrid ) {
    // Batched and SimpleHybrid are not "piped" kernels, i.e., they take PreAttention states as input and output
    // PreAttention states
    for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
      CHECK_EQ( concurrency_.get( tier_i, glinthawk::models::InferenceStage::PreAttention ),
                concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ) );
      CHECK_EQ( concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ),
                concurrency_.get( tier_i, glinthawk::models::InferenceStage::PostAttention ) );
      CHECK_EQ( concurrency_.get( tier_i, glinthawk::models::InferenceStage::PostAttention ),
                concurrency_.get( tier_i, glinthawk::models::InferenceStage::Classification ) );
    }
  }

  event_loop_.set_fd_failure_callback( [] { LOG( ERROR ) << "FD failure callback called."; } );

  event_loop_.add_rule(
    "Compute Kernel",
    Direction::In,
    this->compute_kernel_->event_fd(),
    std::bind( &ParentTierRouter<ComputeKernel, ModelConfig>::pull_from_kernel, this ),
    // Not sure what interest means here. Should check with Sadjad.
    [] { return true; },
    [] { LOG( ERROR ) << "TierRouter stopped pulling from kernel."; } );
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::pull_from_kernel()
{
  TierRouter<ComputeKernel, ModelConfig>::compute_kernel_->event_fd().read_event();
  models::BatchedInferenceState<ModelConfig> state;
  while ( TierRouter<ComputeKernel, ModelConfig>::compute_kernel_->pop( state ) ) {
    // Shards will not be merge-able if they are not of the same (tier/rank - shard/monolith)
    state.set_next_tier( 0 );
    state.set_next_rank( 0 );
    if ( is_served_in_this_slice( state ) ) {
      process_shard( std::move( state ) );
    } else {
      {
        std::lock_guard lock { outgoing_.mutex };
        outgoing_.queue.emplace( std::move( state ) );
      }
      TierRouter<ComputeKernel, ModelConfig>::event_fd_.write_event();
    }
  }
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::push( models::BatchedInferenceState<ModelConfig>&& state )
{
  if ( state.is_sharded() ) {
    CHECK_EQ( state.next_tier(), 0 );
    CHECK_EQ( state.next_rank(), 0 );
    process_shard( std::move( state ) );
  } else {
    process_monolith( std::move( state ) );
  }
}

template<typename ComputeKernel, typename ModelConfig>
bool ParentTierRouter<ComputeKernel, ModelConfig>::pop( models::BatchedInferenceState<ModelConfig>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename ComputeKernel, typename ModelConfig>
bool ParentTierRouter<ComputeKernel, ModelConfig>::is_context_available()
{
  std::lock_guard lock { ctx_mutex_ };

  for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
    if ( free_contexts_[tier_i] < concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ) ) {
      return false;
    }
  }

  return true;
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::assign_ranks( StateType& state )
{
  CHECK( not state.is_sharded() ) << "Cannot assign ranks to shards since sharding is done after assigning ranks!";
  CHECK_EQ( state.batch_size(), concurrency_.full_batch() );

  bool already_assigned = state.assigned_to_node( 0 );
  for ( size_t i = 0; i < state.batch_size(); i++ ) {
    CHECK_EQ( already_assigned, state.assigned_to_node( i ) )
      << "Either all prompts are already tier-routed or none of them are.";
    if ( not state.assigned_to_node( i ) ) {
      const auto [tier_i, rank_i] = concurrency_.tier_rank( glinthawk::models::InferenceStage::Attention, i );
      state.set_kv_tier( i, tier_i );
      state.set_kv_rank( i, rank_i );
      {
        std::lock_guard lock { ctx_mutex_ };
        CHECK( free_contexts_[tier_i] > 0 );
        free_contexts_[tier_i] -= 1;
      }
    }
  }
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::route_shard( StateType&& state )
{
  CHECK( is_served_in_this_slice( state ) );
  if ( state.next_tier() == 0 and state.next_rank() == 0 ) {
    TierRouter<ComputeKernel, ModelConfig>::compute_kernel_->push( std::move( state ) );
  } else {
    {
      std::lock_guard lock { outgoing_.mutex };
      outgoing_.queue.emplace( std::move( state ) );
    }
    TierRouter<ComputeKernel, ModelConfig>::event_fd_.write_event();
  }
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::clean_queue_if_dirty( Shards& queue, size_t target_conc )
{
  StateType next_shard;
  bool have_next_shard;
  {
    std::lock_guard lock { shards_mutex_ };
    if ( queue.is_dirty() == false ) {
      return;
    }
    have_next_shard = queue.pop_ang_merge( next_shard, target_conc );
  }
  while ( have_next_shard ) {
    if ( next_shard.next_stage() == glinthawk::models::InferenceStage::Attention ) {
      next_shard.set_next_tier( next_shard.kv_tier( 0 ) );
      next_shard.set_next_rank( next_shard.kv_rank( 0 ) );
    } else {
      // Non-attention operations are performed locally
      next_shard.set_next_tier( 0 );
      next_shard.set_next_rank( 0 );
    }
    route_shard( std::move( next_shard ) );
    {
      std::lock_guard lock { shards_mutex_ };
      have_next_shard = queue.pop_ang_merge( next_shard, target_conc );
    }
  }
  queue.set_clean();
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::process_monolith( StateType&& state )
{
  // This function will only be called a finite number of times, and afterwards monoliths will no longer exist
  assign_ranks( state );
  CHECK( is_served_in_this_slice( state ) );

  std::deque<StateType> shards
    = std::move( StateType::split_states( std::move( state ), concurrency_.cutting_plan( state.next_stage() ), true ) );
  for ( StateType& shard : shards ) {
    shard.set_is_sharded( true );
    if ( shard.next_stage() == glinthawk::models::InferenceStage::Attention ) {
      shard.set_next_tier( shard.kv_tier( 0 ) );
      shard.set_next_rank( shard.kv_rank( 0 ) );
    } else {
      // Non-attention operations are done locally.
      shard.set_next_tier( 0 );
      shard.set_next_rank( 0 );
    }
    route_shard( std::move( shard ) );
  }
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::process_shard( StateType&& state )
{
  CHECK( state.all_assigned_to_nodes() ) << "Sharded states must always be already routed.";
  if ( state.next_stage() == glinthawk::models::InferenceStage::Attention ) {
    {
      std::lock_guard lock { shards_mutex_ };

      auto split_shards = std::move( StateType::split_on_kv( std::move( state ) ) );
      for ( StateType& shard : split_shards ) {
        attention_idle_shards_[shard.kv_tier( 0 )][shard.kv_rank( 0 )].push_back( std::move( shard ) );
      }
    }

    for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
      for ( int rank_i = 0; rank_i < concurrency_.num_ranks( tier_i ); rank_i++ ) {
        clean_queue_if_dirty( attention_idle_shards_[tier_i][rank_i],
                              concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ) );
      }
    }

  } else {
    const auto vi = vector_index( state.next_layer(), state.next_stage() );
    {
      std::lock_guard lock { shards_mutex_ };
      non_attention_idle_shards_[vi].push_back( std::move( state ) );
    }
    clean_queue_if_dirty( non_attention_idle_shards_[vi], concurrency_.get( 0, state.next_stage() ) );
  }
}

template<typename ComputeKernel, typename ModelConfig>
void ParentTierRouter<ComputeKernel, ModelConfig>::run_event_loop( std::stop_token stoken )
{
  while ( event_loop_.wait_next_event( 1'000 ) != EventLoop::Result::Exit ) {
    if ( stoken.stop_requested() ) {
      return;
    }
  }

  LOG( INFO ) << "ParentTierRouter event loop thread exiting.";
}

/// @brief
/// ChildTierRouter is an empty middle-man between the worker and kernel. It's job is to mimic the TierRouter do the
/// worker is oblivious to which rank it has. DummyTierRouter that pass states through with no delay.
template<typename ComputeKernel, typename ModelConfig>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
class ChildTierRouter : public TierRouter<ComputeKernel, ModelConfig>
{
public:
  explicit ChildTierRouter( std::unique_ptr<ComputeKernel> compute_kernel );

  ~ChildTierRouter() override { LOG( INFO ) << "ChildTierRouter shutting down..."; };

  /// @brief
  /// 1. Push sharded state from worker -> send state to kernel
  void push( glinthawk::models::BatchedInferenceState<ModelConfig>&& state ) override;

  /// @brief
  /// Behaves similar to TierRouter
  [[nodiscard]] bool pop( glinthawk::models::BatchedInferenceState<ModelConfig>& state ) override;

  /// @brief
  /// This should never be called.
  [[nodiscard]] bool is_context_available() override;

protected:
  EventLoop event_loop_ {};
  std::jthread event_loop_thread_;

  void run_event_loop( std::stop_token stoken );
};

template<typename ComputeKernel, typename ModelConfig>
ChildTierRouter<ComputeKernel, ModelConfig>::ChildTierRouter( std::unique_ptr<ComputeKernel> compute_kernel )
  : TierRouter<ComputeKernel, ModelConfig>( std::move( compute_kernel ) )
  , event_loop_thread_( std::bind( &ChildTierRouter::run_event_loop, this, std::placeholders::_1 ) )
{
  event_loop_.set_fd_failure_callback( [] { LOG( ERROR ) << "FD failure callback called."; } );

  event_loop_.add_rule(
    "Compute Kernel",
    Direction::In,
    this->compute_kernel_->event_fd(),
    [this] {
      TierRouter<ComputeKernel, ModelConfig>::compute_kernel_->event_fd().read_event();
      TierRouter<ComputeKernel, ModelConfig>::event_fd_.write_event();
    },
    // Not sure what interest means here. Should check with Sadjad.
    [] { return true; },
    [] { LOG( ERROR ) << "TierRouter stopped pulling from kernel."; } );
}

template<typename ComputeKernel, typename ModelConfig>
void ChildTierRouter<ComputeKernel, ModelConfig>::push( models::BatchedInferenceState<ModelConfig>&& state )
{
  TierRouter<ComputeKernel, ModelConfig>::compute_kernel_->push( std::move( state ) );
}

template<typename ComputeKernel, typename ModelConfig>
bool ChildTierRouter<ComputeKernel, ModelConfig>::pop( models::BatchedInferenceState<ModelConfig>& state )
{
  if ( TierRouter<ComputeKernel, ModelConfig>::compute_kernel_->pop( state ) ) {
    state.set_next_tier( 0 );
    state.set_next_rank( 0 );
    return true;
  } else {
    return false;
  }
}

template<typename ComputeKernel, typename ModelConfig>
bool ChildTierRouter<ComputeKernel, ModelConfig>::is_context_available()
{
  LOG( FATAL ) << "DummyTierRouter should never receive new batches. That is only going to happen in slice0, tier1, "
                  "rank0.";
  return false;
}

template<typename ComputeKernel, typename ModelConfig>
void ChildTierRouter<ComputeKernel, ModelConfig>::run_event_loop( std::stop_token stoken )
{
  while ( event_loop_.wait_next_event( 1'000 ) != EventLoop::Result::Exit ) {
    if ( stoken.stop_requested() ) {
      return;
    }
  }
  LOG( INFO ) << "ChildTierRouter event loop thread exiting.";
}

} // namespace glinthawk::compute
