#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "util/eventfd.hh"
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
template<typename ComputeKernel, typename Model>
class TierRouter
{
public:
  virtual ~TierRouter() = 0;

  EventFD& event_fd() { return event_fd_; }

  void set_up_event_loop( EventLoop event_loop )
  {
    event_loop_.add_rule( "Compute Kernel",
                          Direction::In,
                          this->compute_kernel_->event_fd(),
                          std::bind( &TierRouter<ComputeKernel, ModelConfig>::pull_from_kernel, this ) );
  }

  virtual void pull_from_kernel() = 0;

  virtual void push( glinthawk::models::BatchedInferenceState<Model::ConfigType>&& state ) = 0;

  virtual bool pop( glinthawk::models::BatchedInferenceState<Model::ConfigType>& state ) = 0;

  virtual bool is_context_available() const = 0;

protected:
  // Worker doesn't see the compute kernel. Only the tier router does.
  std::unique_ptr<ComputeKernel> compute_kernel_;

  EventFD event_fd_ {};
  GlobalQueue outgoing_;
};

/// @brief
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
// TODO(pouya): can straggler's cause an unstable failure mode in dispersing work among tiers?
template<typename ComputeKernel, typename Model>
class ParentTierRouter : public TierRouter
{
public:
  ParentTierRouter( const ComputeKernel& compute_kernel,
                    const SliceConcurrency& concurrency_s,
                    const std::vector<size_t> kv_slots_tier_s,
                    const size_t start_layer,
                    const size_t end_layer );

  virtual ~ParentTierRouter() override;

  /// @brief
  /// 1. Push sharded state from kernel -> process_shard, may internally call process_monolith
  virtual void pull_from_kernel() override;

  /// @brief
  /// 1. Push monolithic state from worker -> calls process_monolith
  /// 2. Push sharded state from worker -> process_shard, may internally call process_monolith
  virtual void push( glinthawk::models::BatchedInferenceState<Model::ConfigType>&& state ) override;

  /// @brief
  /// pop is called by the worker to receive states, that are:
  /// 1. monoliths that are no longer local to the node.
  /// 2. shards that the compute kernel processed.
  /// 3. shards that resulted from a monolith being broken.
  virtual bool pop( glinthawk::models::BatchedInferenceState<Model::ConfigType>& state ) override;

  /// @brief
  /// is_context_available checks if we have context available for attention for all layers in this slice. It returns
  /// false if even one node does not have the corresponding concurrency KV slots. Worker calls this function *before*
  /// pushing new prompts, but does not need to check when receiving states over the network.
  /// This is because we are assuming all slices are alike, and if slice 0 has context, so do the others. Thus, this
  /// function is truly only relevant in slice 0.
  /// This function will only return "true" a finite number of times before all contexts are filled up. From that point,
  /// new prompts are placed in discarded prompt locations, and will have context by default.
  virtual bool is_context_available() const override;

protected:
  using Stage = glinthawk::models::InferenceStage;
  using StateType = typename glinthawk::models::BatchedInferenceState<Model::ConfigType>;

  /// @brief
  /// For now we assume:
  /// 1. All tier "i" machines are alike.
  /// 2. All slices are alike.
  /// 3. Classification is done on the same machine doing pre-att-post.
  /// 4. Batch sizes are equal across stages
  /// The ParentTierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of
  /// two concurrencies.
  const SliceConcurrency& concurrency_;
  std::vector<size_t> free_contexts_;

  const size_t start_layer_;
  const size_t end_layer_;

  std::vector<std::vector<std::deque<StateType>>> idle_shards_;

  std::mutex ctx_mutex_;
  std::mutex shards_mutex_;

  inline size_t vector_index( const size_t layer, const Stage stage ) const
  {
    return ( layer - start_layer_ ) * util::to_underlying( models::InferenceStage::__COUNT__ )
           + util::to_underlying( stage );
  }

  bool inline is_served_in_this_slice( const StateType& state ) const
  {
    // TODO(pouya): this is assuming classification is done on the same machine doing pre-att-post
    return state.next_layer() >= start_layer_ and state.next_layer() <= end_layer_;
  }

  void place_shard( StateType&& state );

  std::optional<StateType>&& form_monolith( const size_t layer, const Stage stage );

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
};

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::ParentTierRouter( const ComputeKernel& compute_kernel,
                                                               const SliceConcurrency& concurrency,
                                                               const std::vector<size_t> kv_slots_tier_s,
                                                               const size_t start_layer,
                                                               const size_t end_layer )
  : compute_kernel_( std::make_unique( compute_kernel ) )
  , concurrency_( concurrency )
  , free_contexts_( settings.start_layer_num == 0 ? kv_slots_tier_s : { concurrency_.num_tiers(), 0 } )
  , start_layer_( start_layer )
  , end_layer_( end_layer )
  , idle_shards_( concurrency_.num_tiers(),
                  ( end_layer_ - start_layer_ + 1 ) * util::to_underlying( Stage::__COUNT__ ) )
{
  // TODO(pouya): if there is only one slice, a generated batch never gets pushed to worker to report generations.
  CHECK( start_layer_ != 0 or end_layer_ != Model::ConfigType::n_layers - 1 );

  for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
    CHECK( kv_slots_tier_s[tier_i] > 0
           or concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ) == 0 );
  }

  switch ( compute_kernel_::Type ) {
    case KernelType::Batched:
    case KernelType::SimpleHybrid:
      // Batched and SimpleHybrid are not "piped" kernels, i.e., they take PreAttention states as input and output
      // PreAttention states
      for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
        CHECK_EQ( concurrency_.get( i, glinthawk::models::InferenceStage::PreAttention ),
                  concurrency_.get( i, glinthawk::models::InferenceStage::Attention ) );
        CHECK_EQ( concurrency_.get( i, glinthawk::models::InferenceStage::Attention ),
                  concurrency_.get( i, glinthawk::models::InferenceStage::PostAttention ) );
        CHECK_EQ( concurrency_.get( i, glinthawk::models::InferenceStage::PostAttention ),
                  concurrency_.get( i, glinthawk::models::InferenceStage::Classification ) );
      }
      break;
    case KernelType::Hybrid:
    case KernelType::SimplePiped: break;
    default: LOG( FATAL ) << "No such kernel type"; break;
  }
}

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::pull_from_kernel()
{
  compute_kernel_->event_fd().read_event();
  models::BatchedInferenceState<Model::ConfigType> state;
  while ( compute_kernel_->pop( state ) ) {
    // Shards will not be merge-able if they are not of the same (state/gather, shard/monolith)
    state.set_gather();
    process_shard( std::move( state ) );
  }
}

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::push( models::BatchedInferenceState<Model::ConfigType>&& state )
{
  if ( state.is_sharded() )
    process_shard( std::move( state ) );
  else
    process_monolith( std::move( state ) );
}

template<typename ComputeKernel, typename Model>
bool ParentTierRouter<ComputeKernel, Model>::pop( models::BatchedInferenceState<Model::ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename ComputeKernel, typename Model>
bool ParentTierRouter<ComputeKernel, Model>::is_context_available() const
{
  std::lock_guard lock { ctx_mutex_ };

  for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
    if ( free_contexts_[tier_i] < concurrency_.get( tier_i, glinthawk::models::InferenceStage::Attention ) ) {
      return false;
    }
  }

  return true;
}

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::assign_ranks( StateType& state )
{
  CHECK( not state.is_sharded() ) << "Cannot assign ranks to shards since sharding is done after assinging ranks!";
  CHECK_EQ( state.batch_size(), concurrency_.full_batch() );

  bool already_assigned = state.assigned_to_node( 0 );
  for ( size_t i = 0; i < state.batch_size(); i++ ) {
    CHECK_EQ( already_assigned, state.assigned_to_node( i ) )
      << "Either all prompts are already tier-routed or none of them are.";
    if ( not state.assigned_to_node( i ) ) {
      const auto [tier_i, rank_i] = concurrency_.tier_rank( stage, i );
      state.set_tier( i, tier_i );
      state.set_rank( i, rank_i );
      {
        std::lock_guard lock { ctx_mutex_ };
        CHECK( free_contexts_[tier_i] > 0 );
        free_contexts_[tier_i] -= 1;
      }
    }
  }
}

template<typename ComputeKernel, typename Model>
std::optional<StateType>&& ParentTierRouter<ComputeKernel, Model>::form_monolith( const size_t layer,
                                                                                  const Stage stage )
{
  const auto vi = vector_index( layer, stage );
  std::deque<StateType> shards;
  {
    std::lock_guard lock { shards_mutex_ };
    for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
      if ( concurrency_.get( tier_i, stage ) > 0 and idle_shards_[vi].size() < concurrency_.num_ranks( tier_i ) ) {
        return std::nullopt;
      }
    }
    for ( int tier_i = 0; tier_i < concurrency_.num_tiers(); tier_i++ ) {
      if ( concurrency_.get( tier_i, stage ) > 0 ) {
        shards.insert( shards.end(),
                       idle_shards_[tier_i][vi].begin(),
                       idle_shards_[tier_i][vi].begin() + concurrency_.num_ranks( tier_i ) );
        idle_shards_[tier_i][vi].erase( idle_shards_[tier_i][vi].begin(),
                                        idle_shards_[tier_i][vi].begin() + concurrency_.num_ranks( tier_i ) );
      }
    }
  }
  StateType monolith = StateType::merge_states( shards );
  monolith.set_is_sharded( false );
  return std::move( { monolith } );
}

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::process_monolith( StateType&& state )
{
  assign_ranks( state );
  CHECK( is_served_in_this_slice( state ) );

  // TODO(pouya): the ranks only make sense for attention, routing should not be tied to rank for other stages
  std::deque<StateType> shards
    = StateType::split_states( states, concurrency_.cutting_plan( state.next_stage() ), true );
  for ( StateType& shard : shards ) {
    shard.set_is_sharded( true );
    shard.set_scatter();
    if ( shard.tier( 0 ) == 0 and shard.rank( 0 ) == 0 ) {
      compute_kernel_->push( std::move( shard ) );
    } else {
      std::lock_guard lock { outgoing_.mutex };
      outgoing_.emplace( std::move( shard ) );
    };
  }
  event_fd_.write_event();
}

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::place_shard( StateType&& state )
{
  CHECK( state.all_rank_assigned() ) << "Sharded states must always be already routed.";
  const auto vi = vector_index( state.next_layer(), state.next_stage() );
  {
    std::lock_guard lock { shards_mutex_ };
    idle_shards_[state.tier( 0 )][vi].push_back( std::move( state ) );
  }
}

template<typename ComputeKernel, typename Model>
void ParentTierRouter<ComputeKernel, Model>::process_shard( StateType&& state )
{
  place_shard( std::move( state ) );
  std::optional<StateType> monolithic_state = std::move( form_monolith( next_layer, next_stage ) );

  if ( monolithic_state.has_value() ) {
    process_monolith( std::move( *monolithic_state ) );
  }
}

/// @brief
/// ChildTierRouter is an empty middle-man between the worker and kernel. It's job is to mimic the TierRouter do the
/// worker is oblivious to which rank it has. DummyTierRouter that pass states through with no delay.
template<typename ComputeKernel, typename Model>
class ChildTierRouter : public TierRouter
{
public:
  ChildTierRouter( const ComputeKernel& compute_kernel );

  virtual ChildTierRouter() override;

  /// @brief
  /// 1. Push sharded state from kernel -> send state to worker
  virtual void pull_from_kernel() override;

  /// @brief
  /// 1. Push sharded state from worker -> send state to kernel
  virtual void push( glinthawk::models::BatchedInferenceState<Model::ConfigType>&& state ) override;

  /// @brief
  /// Behaves similar to TierRouter
  virtual bool pop( glinthawk::models::BatchedInferenceState<Model::ConfigType>& state ) override;

  /// @brief
  /// This should never be called.
  virtual bool is_context_available() const override;
};

template<typename ComputeKernel, typename Model>
void ChildTierRouter<ComputeKernel, Model>::ChildTierRouter( const ComputeKernel& compute_kernel )
  : compute_kernel_( std::make_unique( compute_kernel ) )
{
}

template<typename ComputeKernel, typename Model>
void ChildTierRouter<ComputeKernel, Model>::pull_from_kernel()
{
  // TODO: optimize
  compute_kernel_->event_fd().read_event();
  models::BatchedInferenceState<Model::ConfigType> state;
  while ( compute_kernel_->pop( state ) ) {
    state.set_gather();
    outgoing_.emplace( std::move( state ) );
    event_fd_.write_event();
  }
}

template<typename ComputeKernel, typename Model>
void ChildTierRouter<ComputeKernel, Model>::push( models::BatchedInferenceState<Model::ConfigType>&& state )
{
  // TODO: optimize
  compute_kernel_->push( std::move( state ) );
}

template<typename ComputeKernel, typename Model>
bool ChildTierRouter<ComputeKernel, Model>::pop( models::BatchedInferenceState<Model::ConfigType>& state )
{
  // TODO: optimize
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename ComputeKernel, typename Model>
bool ChildTierRouter<ComputeKernel, Model>::is_context_available() const
{
  LOG( FATAL ) << "DummyTierRouter should never receive new batches. That is only going to happen in slice0, tier1, "
                  "rank0.";
}

/// @brief
/// SingleTierRouter is a special case of ParentTierRouter, when we know Tier 2 is empty and Tier 1 has one node. It is
/// only different from ChildTierRouter in that (1) it does manage context, and (2) it can immediately send states back
/// to compute_kernel if they are still served by this slice.
template<typename ComputeKernel, typename Model>
class SingleTierRouter : public TierRouter
{
public:
  SingleTierRouter( const ComputeKernel& compute_kernel,
                    const Concurrency& concurrency,
                    const kv_slots,
                    const size_t start_layer,
                    const size_t end_layer );

  virtual ~SingleTierRouter() override;

  /// @brief
  /// called by an event loop to pull states from kernel.outgoing. May send them back in kernel or send them to worker.
  virtual void pull_from_kernel() override;

  /// @brief
  /// called by worker to send state to kernel
  virtual void push( glinthawk::models::BatchedInferenceState<Model::ConfigType>&& state ) override;

  /// @brief
  /// called by worker to get state from kernel
  virtual bool pop( glinthawk::models::BatchedInferenceState<Model::ConfigType>& state ) override;

  /// @brief
  /// is_context_available checks if we have context available for attention for all layers in this slice. It returns
  /// false if the node does not have the corresponding concurrency KV slots. Worker calls this function *before*
  /// pushing new prompts, but does not need to check when receiving states over the network.
  /// This is because we are assuming all slices are alike, and if slice 0 has context, so do the others. Thus, this
  /// function is truly only relevant in slice 0.
  /// This function will only return "true" a finite number of times before all contexts are filled up. From that point,
  /// new prompts are placed in discarded prompt locations, and will have context by default.
  virtual bool is_context_available() const override;

protected:
  /// @brief
  /// For now we assume:
  /// 1. All slices are alike.
  /// 2. Classification is done on the same machine doing pre-att-post.
  /// 3. Batch sizes are equal across stages
  /// The SingleTierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of
  /// two concurrencies.
  const concurrency_all_stages_;
  size_t free_contexts_;

  const size_t start_layer_;
  const size_t end_layer_;

  std::mutex ctx_mutex_;

  bool inline is_served_in_this_slice( const glinthawk::models::BatchedInferenceState<Model::ConfigType>& state ) const
  {
    // TODO(pouya): this is assuming classification is done on the same machine doing pre-att-post
    return state.next_layer() >= start_layer_ and state.next_layer() <= end_layer_;
  }
};

template<typename ComputeKernel, typename Model>
void SingleTierRouter<ComputeKernel, Model>::SingleTierRouter( const ComputeKernel& compute_kernel,
                                                               const Concurrency& concurrency,
                                                               const kv_slots,
                                                               const size_t start_layer,
                                                               const size_t end_layer )
  : compute_kernel_( std::make_unique( compute_kernel ) )
  , concurrency_all_stages_( concurrency.get( glinthawk::models::InferenceStage::PreAttention ) )
  , free_contexts_( settings.start_layer_num == 0 ? kv_slots_tier_1 : 0 )
  , start_layer_( start_layer )
  , end_layer_( end_layer )
{
  // TODO(pouya): if there is only one slice, a generated batch never gets pushed to worker to report generations.
  CHECK( start_layer_ != 0 or end_layer_ != Model::ConfigType::n_layers - 1 );

  CHECK_EQ( concurrency.get( glinthawk::models::InferenceStage::PreAttention ),
            concurrency.get( glinthawk::models::InferenceStage::Attention ) );
  CHECK_EQ( concurrency.get( glinthawk::models::InferenceStage::Attention ),
            concurrency.get( glinthawk::models::InferenceStage::PostAttention ) );
  CHECK_EQ( concurrency.get( glinthawk::models::InferenceStage::PostAttention ),
            concurrency.get( glinthawk::models::InferenceStage::Classification ) );
}

template<typename ComputeKernel, typename Model>
void SingleTierRouter<ComputeKernel, Model>::pull_from_kernel()
{
  // TODO: optimize
  compute_kernel_->event_fd().read_event();
  models::BatchedInferenceState<Model::ConfigType> state;
  while ( compute_kernel_->pop( state ) ) {
    if ( is_served_in_this_slice( state ) ) {
      compute_kernel_->push( std::move( state ) );
    } else {
      state.set_gather();
      {
        std::lock_guard lock { outgoing_.mutex };
        outgoing_.emplace( std::move( state ) );
      }
      event_fd_.write_event();
    }
  }
}

template<typename ComputeKernel, typename Model>
void SingleTierRouter<ComputeKernel, Model>::push( models::BatchedInferenceState<Model::ConfigType>&& state )
{
  CHECK_EQ( state.batch_size(), concurrency_all_stages_ );
  bool already_assigned = state.rank_assigned( 0 );
  for ( size_t i = 0; i < state.batch_size(); i++ ) {
    CHECK_EQ( already_assigned, state.rank_assigned( i ) )
      << "Either all prompts are already tier-routed or none of them are.";
    if ( not state.assigned_to_node( i ) ) {
      state.set_tier( 0 );
      state.set_rank( 0 );
      {
        std::lock_guard lock { ctx_mutex_ };
        CHECK( free_contexts_ > 0 );
        free_contexts_ -= 1;
      }
    }
  }
  state.set_scatter();
  compute_kernel_->push( std::move( state ) );
}

template<typename ComputeKernel, typename Model>
bool SingleTierRouter<ComputeKernel, Model>::pop( models::BatchedInferenceState<Model::ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename ComputeKernel, typename Model>
bool SingleTierRouter<ComputeKernel, Model>::is_context_available() const
{
  std::lock_guard lock { ctx_mutex_ };
  return free_contexts_ >= concurrency_all_stages_;
}

} // namespace glinthawk::compute
