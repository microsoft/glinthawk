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
// 'Tier': A set of nodes that either mainly 1. focus on compute-heavy operations (e.g., Post-Attention) or 2. focus
//         predominantly on memory-heavy tasks (e.g., Attention)
// 'Rank': an arbitrary index between nodes in a Tier.
// 'Slice': all nodes that serve an atomic unit of layers, e.g., if each node serves 2 layers, all nodes serving layers
//          0 and 1 are a slice.
// 'Monolithic State' or 'Monolith': a BatchedInferenceState that has not been broken to smaller pieces.
// 'Sharded State' or 'Shard': an atomic piece of a monolith that is only relevant to a single node.

template<typename Model>
class TierRouter
{
public:
  virtual ~TierRouter() = 0;

  EventFD& event_fd() { return event_fd_; }

  // TODO(pouya): I'm using a pull model from kernel to make kernel more isolated. Talk to sadjad about this.
  // TODO(pouya): add an event_loop rule to pull this
  virtual void pull_from_kernel() = 0;

  virtual void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) = 0;

  virtual bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state ) = 0;

  virtual bool is_context_available() const = 0;

protected:
  // Worker doesn't see the compute kernel. Only the tier router does.
  std::unique_ptr<ComputeKernel> compute_kernel_;

  EventFD event_fd_ {};
  GlobalQueue outgoing_;
};

/// @brief
/// ParentTierRouter is a middle-man between the worker and kernel. It's job is to break down states between machines in
/// tiers, while ensuring context does not run out and deadlocks don't happen.
/// ParentTierRouter is only run on `rank-0' machines, i.e., it only runs on one node in each `slice' that may serve
/// multiple layers. The other nodes in the slice have a ChildTierRouter that pass states through with no delay. There
/// is a possibility for improving this, where the tier-to-tier gather and scatter operations are distributed.
/// ParentTierRouter distinguishes states as two types:
///     Monolithic BatchInferenceState: which is a full batch across all machines in both tiers.
///     Sharded BatchInferenceState: which is a specific part of a monolith dedicated to a specific machine.
/// ParentTierRouter receives monolithic/sharded states through push, and returns monolithic/sharded states through an
/// outgoing queue. It sends local shards to and receives them from the compute kernel directly.
/// The worker reads an outgoing queue from ParentTierRouter to send out outbound states.
/// ParentTierRouter fully handles context management, and guarantees that if a shard arrives at the kernel, that
/// kernel has space for it.
// TODO(pouya): fix incomplete states to be filled in worker
// TODO(pouya): fix discarded states in merge and split
// TODO(pouya): can straggler's cause an unstable failure mode in dispersing work among tiers?
// TODO(pouya): how does routing work?
// TODO(pouya): how does worker know to send a shard back to its origin?
// TODO(pouya): I'm forcing kernel to not have an outgoing queue. Talk to sadjad about this.
// TODO(pouya): context is guaranteed per some implicit BatchInferenceState ID. Maybe we should stop caring about
//  mapping from prompt ID to context, and just get mapping from BIS ID to context vector.
template<typename Model>
class ParentTierRouter : public TierRouter
{
public:
  ParentTierRouter( const ComputeKernel& compute_kernel,
                    const size_t n_tier_1,
                    const Concurrency& concurrency_tier_1,
                    const kv_slots_tier_1,
                    const size_t n_tier_2,
                    const Concurrency& concurrency_tier_2,
                    const kv_slots_tier_2,
                    const typename Model::SettingsType& settings );

  virtual ~ParentTierRouter() override;

  /// @brief
  /// 1. Push sharded state from kernel -> process_shard, may internally call process_monolith
  virtual void pull_from_kernel() override;

  /// @brief
  /// 1. Push monolithic state from worker -> calls process_monolith
  /// 2. Push sharded state from worker -> process_shard, may internally call process_monolith
  virtual void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) override;

  /// @brief
  /// pop is called by the worker to receive states, that are:
  /// 1. monoliths that are no longer local to the node.
  /// 2. shards that the compute kernel processed.
  /// 3. shards that resulted from a monolith being broken.
  virtual bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state ) override;

  /// @brief
  /// is_context_available checks if we have context available for attention for all layers in this slice. It returns
  /// false if even one node does not have the corresponding concurrency KV slots. Worker calls this function *before*
  /// pushing new prompts, but does not need to check when receiving states over the network.
  /// This is because we are assuming all slices are alike, and if slice 0 has context, so do the others. Thus, this
  /// function is truly only relevant in slice 0.
  /// This function will only return "true" a finite number of times before all contexts are filled up. From that point,
  /// new prompts are placed in discarded prompt locations, and will have context by default.
  virtual bool is_context_available() const override;

private:
  using ConfigType = typename Model::ConfigType;
  using Stage = glinthawk::models::InferenceStage;
  using StateType = typename glinthawk::models::BatchedInferenceState<ConfigType>;
  using RefStateType = typename std::reference_wrapper<StateType>;

  /// @brief
  /// For now we assume:
  /// 1. All tier 1 machines are alike.
  /// 2. All tier 2 machines are alike.
  /// 3. All slices are alike.
  /// 4. Classification is done on the same machine doing pre-att-post.
  /// 5. Batch sizes are equal across stages
  /// The ParentTierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of
  /// two concurrencies.
  const size_t n_tier_1_;
  const Concurrency concurrency_tier_1_;
  std::vector<size_t> free_contexts_tier_1_;
  const size_t n_tier_2_;
  const Concurrency concurrency_tier_2_;
  std::vector<size_t> free_contexts_tier_2_;

  std::array<std::vector<size_t>, util::to_underlying( Stage::__COUNT__ )> sharding_batch_sizes;

  const size_t start_layer_;
  const size_t end_layer_;

  // TODO(pouya): double check if reference_wrapper is necessary here
  std::vector<std::deque<RefStateType>> tier_1_idle_shards_;
  std::vector<std::deque<RefStateType>> tier_2_idle_shards_;

  std::vector<size_t> tier_1_idle_shard_counts_;
  std::vector<size_t> tier_2_idle_shard_counts_;

  std::mutex ctx_mutex_;
  std::mutex shards_mutex_;

  inline size_t vector_index( const size_t layer, const Stage stage ) const
  {
    return ( layer - start_layer_ ) * util::to_underlying( models::InferenceStage::__COUNT__ )
           + util::to_underlying( stage );
  }

  std::pair<uint8_t, uint8_t> tier_index( const size_t batch_index, const Stage stage ) const
  {
    if ( i < n_tier_1_ * concurrency_tier_1_.get( stage ) ) {
      return std::make_pair( i / concurrency_tier_1_.get( stage ), -1 );
    } else {
      return std::make_pair( -1,
                             ( i - n_tier_1_ * concurrency_tier_1_.get( stage ) ) / concurrency_tier_2_.get( stage ) );
    }
  }

  bool inline is_served_in_this_slice( const StateType& state ) const
  {
    // TODO(pouya): this is assuming classification is done on the same machine doing pre-att-post
    return state.next_layer() >= start_layer_ and state.next_layer() <= end_layer_;
  }

  bool can_form_monolith( const size_t layer, const Stage stage ) const;

  StateType&& form_monolith( const size_t layer, const Stage stage );

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

template<typename Model>
void ParentTierRouter<Model>::ParentTierRouter( const ComputeKernel& compute_kernel,
                                                const size_t n_tier_1,
                                                const Concurrency& concurrency_tier_1,
                                                const kv_slots_tier_1,
                                                const size_t n_tier_2,
                                                const Concurrency& concurrency_tier_2,
                                                const kv_slots_tier_2,
                                                const typename Model::SettingsType& settings )
  : compute_kernel_( std::make_unique( compute_kernel ) )
  , n_tier_1_( n_tier_1 )
  , concurrency_tier_1_( concurrency_tier_1 )
  , free_contexts_tier_1_( n_tier_1, settings.start_layer_num == 0 ? kv_slots_tier_1 : 0 )
  , n_tier_2_( n_tier_2 )
  , concurrency_tier_2_( concurrency_tier_2 )
  , free_contexts_tier_2_( n_tier_2, settings.start_layer_num == 0 ? kv_slots_tier_2 : 0 )
  , start_layer_( settings.start_layer_num )
  , end_layer_( settings.end_layer_num )
  , tier_1_idle_shards_( n_tier_1 )
  , tier_2_idle_shards_( n_tier_2 )
  , tier_1_idle_shard_counts_( n_tier_1, 0 )
  , tier_2_idle_shard_counts_( n_tier_2, 0 )
{
  // TODO(pouya): if there is only one slice, a generated batch never gets pushed to worker to report generations.
  CHECK( start_layer_ != 0 or end_layer_ != ConfigType::n_layers - 1 );

  for ( int i = 0; i < sharding_batch_sizes.size(); i++ ) {
    sharding_batch_sizes[i] = std::vector<size_t>( n_tier_1 + n_tier_2 );
  }

  for ( int i = 0; i < n_tier_1; i++ ) {
    sharding_batch_sizes[0][i] = concurrency_tier_1_.get( glinthawk::models::InferenceStage::PreAttention );
    sharding_batch_sizes[1][i] = concurrency_tier_1_.get( glinthawk::models::InferenceStage::Attention );
    sharding_batch_sizes[2][i] = concurrency_tier_1_.get( glinthawk::models::InferenceStage::PostAttention );
    sharding_batch_sizes[3][i] = concurrency_tier_1_.get( glinthawk::models::InferenceStage::Classification );
  }

  for ( int i = n_tier_1; i < n_tier_1 + n_tier_2; i++ ) {
    sharding_batch_sizes[0][i] = concurrency_tier_2_.get( glinthawk::models::InferenceStage::PreAttention );
    sharding_batch_sizes[1][i] = concurrency_tier_2_.get( glinthawk::models::InferenceStage::Attention );
    sharding_batch_sizes[2][i] = concurrency_tier_2_.get( glinthawk::models::InferenceStage::PostAttention );
    sharding_batch_sizes[3][i] = concurrency_tier_2_.get( glinthawk::models::InferenceStage::Classification );
  }

  for ( int i = 1; i < util::to_underlying( Stage::__COUNT__ ); i++ ) {
    CHECK_EQ( std::accumulate( sharding_batch_sizes[i - 1].begin(), sharding_batch_sizes[i - 1].end(), 0 ),
              std::accumulate( sharding_batch_sizes[i].begin(), sharding_batch_sizes[i].end(), 0 ) );
  }

  CHECK( kv_slots_tier_1 > 0 or concurrency_tier_1_.get( glinthawk::models::InferenceStage::Attention ) == 0 );
  CHECK( kv_slots_tier_2 > 0 or concurrency_tier_2_.get( glinthawk::models::InferenceStage::Attention ) == 0 );

  CHECK_LT( n_tier_1, 256 );
  CHECK_LT( n_tier_2, 256 );

  switch ( compute_kernel_::Type ) {
    case KernelType::Batched:
    case KernelType::SimpleHybrid:
      // Batched and SimpleHybrid are not "piped" kernels, i.e., they take PreAttention states as input and output
      // PreAttention states
      CHECK_EQ( concurrency_tier_1_.get( glinthawk::models::InferenceStage::PreAttention ),
                concurrency_tier_1_.get( glinthawk::models::InferenceStage::Attention ) );
      CHECK_EQ( concurrency_tier_1_.get( glinthawk::models::InferenceStage::Attention ),
                concurrency_tier_1_.get( glinthawk::models::InferenceStage::PostAttention ) );
      CHECK_EQ( concurrency_tier_1_.get( glinthawk::models::InferenceStage::PostAttention ),
                concurrency_tier_1_.get( glinthawk::models::InferenceStage::Classification ) );

      CHECK_EQ( concurrency_tier_2_.get( glinthawk::models::InferenceStage::PreAttention ),
                concurrency_tier_2_.get( glinthawk::models::InferenceStage::Attention ) );
      CHECK_EQ( concurrency_tier_2_.get( glinthawk::models::InferenceStage::Attention ),
                concurrency_tier_2_.get( glinthawk::models::InferenceStage::PostAttention ) );
      CHECK_EQ( concurrency_tier_2_.get( glinthawk::models::InferenceStage::PostAttention ),
                concurrency_tier_2_.get( glinthawk::models::InferenceStage::Classification ) );
      break;
    case KernelType::Hybrid:
    case KernelType::SimplePiped: break;
    default: LOG( FATAL ) << "No such kernel type"; break;
  }
}

template<typename Model>
void ParentTierRouter<Model>::pull_from_kernel( models::BatchedInferenceState<ConfigType>&& state )
{
  compute_kernel_->event_fd().read_event();
  models::BatchedInferenceState<ConfigType> state;
  while ( compute_kernel_->pop( state ) ) {
    process_shard( std::move( state ) );
  }
}

template<typename Model>
void ParentTierRouter<Model>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  if ( state.is_sharded() )
    process_shard( std::move( state ) );
  else
    process_monolith( std::move( state ) );
}

template<typename Model>
bool ParentTierRouter<Model>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename Model>
bool ParentTierRouter<Model>::is_context_available() const
{
  std::lock_guard lock { ctx_mutex_ };

  // TODO(pouya): this is somewhat pointless. Since we are assuming identical tiers, and assigning ranks identically,
  //  we only need to keep one free_contexts variable per tier.
  for ( size_t i = 0; i < n_tier_1_; i++ ) {
    if ( free_contexts_tier_1_[i] < concurrency_tier_1_.get( glinthawk::models::InferenceStage::Attention ) ) {
      return false;
    }
  }
  for ( size_t i = 0; i < n_tier_2_; i++ ) {
    if ( free_contexts_tier_2_[i] < concurrency_tier_2_.get( glinthawk::models::InferenceStage::Attention ) ) {
      return false;
    }
  }

  return true;
}

template<typename Model>
bool ParentTierRouter<Model>::can_form_monolith( const size_t layer, const Stage stage )
{
  std::lock_guard lock { shards_mutex_ };

  return tier_1_idle_shard_counts_[vector_index( layer, stage )] > n_tier_1_ * concurrency_tier_1_.get( stage )
         and tier_2_idle_shard_counts_[vector_index( layer, stage )] > n_tier_2_ * concurrency_tier_2_.get( stage )
}

template<typename Model>
void ParentTierRouter<Model>::assign_ranks( StateType& state )
{
  CHECK( not state.is_sharded() );
  CHECK_EQ( state.batch_size(),
            n_tier_1_ * concurrency_tier_1_.get( stage ) + n_tier_2_ * concurrency_tier_2_.get( stage ) );
  bool already_assigned = state.rank_assigned( 0 );
  for ( size_t i = 0; i < state.batch_size(); i++ ) {
    CHECK_EQ( already_assigned, state.rank_assigned( i ) )
      << "Either all prompts are already tier-routed or none of them are.";
    if ( not state.rank_assigned( i ) ) {
      const auto t_ind = tier_index( i, stage );
      state.set_rank_tier_1( t_ind.first );
      state.set_rank_tier_2( t_ind.second );
      if ( t_ind.second == -1 ) {
        std::lock_guard lock { ctx_mutex_ };
        CHECK_GE( free_contexts_tier_1_[t_ind.first], concurrency_tier_1_.get( Stage::Attention ) );
        free_contexts_tier_1_[t_ind.first] -= concurrency_tier_1_.get( Stage::Attention );
      } else {
        std::lock_guard lock { ctx_mutex_ };
        CHECK_GE( free_contexts_tier_2_[t_ind.second], concurrency_tier_2_.get( Stage::Attention ) );
        free_contexts_tier_2_[t_ind.second] -= concurrency_tier_2_.get( Stage::Attention );
      }
    }
  }
}

template<typename Model>
StateType&& ParentTierRouter<Model>::form_monolith( const size_t layer, const Stage stage )
{
  const auto vi = vector_index( layer, stage );
  std::vector<RefStateType> shards;
  {
    std::lock_guard lock { shards_mutex_ };
    if ( concurrency_tier_1_.get( stage ) > 0 ) {
      for ( size_t i = 0; i < n_tier_1_; i++ ) {
        shards.push_back( tier_1_idle_shards_[vi].front() );
        tier_1_idle_shards_[vi].pop_front();
        tier_1_idle_shard_counts_[vi] -= concurrency_tier_1_.get( stage );
      }
    }
    if ( concurrency_tier_2_.get( stage ) > 0 ) {
      for ( size_t i = 0; i < n_tier_2_; i++ ) {
        shards.push_back( tier_2_idle_shards_[vi].front() );
        tier_2_idle_shards_[vi].pop_front();
        tier_2_idle_shard_counts_[vi] -= concurrency_tier_2_.get( stage );
      }
    }
  }
  // TODO(pouya): not sure if a "lazy" merge isn't a better option, especially if we're going to send it over the wire
  //  next. We can do an unlazify() func in compute_kernel->incoming thread for cases where it is going to the kernel.
  StateType monolith = StateType::merge_states( shards );
  monolith.set_is_sharded( false );
  return monolith;
}

template<typename Model>
void ParentTierRouter<Model>::process_monolith( StateType&& state )
{
  assign_ranks( state );
  if ( not is_served_in_this_slice( state ) ) {
    {
      // TODO(pouya): not sure if it makes sense to merge monolith in the current slice if the next stage is in another
      //  slice. In general, optimizing the data transfer graph between tiers in different slices is something we should
      //  think about.
      std::lock_guard lock { outgoing_.mutex };
      outgoing_.emplace( std::move( state ) );
    }
  } else {
    std::deque<RefStateType> shards
      = state.split_states( sharding_batch_sizes[util::to_underlying( state.next_stage() )], true );
    if ( concurrency_tier_1_.get( state.next_stage() ) > 0 ) {
      shards.front()->get().set_is_sharded( true );
      compute_kernel_->push( std::move( shards.front() ) );
      shards.pop_front();
    }
    if ( shards.size() > 0 ) {
      std::lock_guard lock { outgoing_.mutex };
      while ( shards.size() > 0 ) {
        shards.front()->get().set_is_sharded( true );
        outgoing_.emplace( std::move( shards.front() ) );
        shards.pop_front();
      }
    }
  }
  event_fd_.write_event();
}

template<typename Model>
void ParentTierRouter<Model>::process_shard( StateType&& state )
{
  CHECK( state.all_rank_assigned() ) << "Sharded states must always be already routed.";

  const auto next_stage = state.next_stage();
  const auto next_layer = state.next_layer();
  const auto vi = vector_index( next_layer, next_stage );
  {
    std::lock_guard lock { shards_mutex_ };
    if ( state.get_rank_tier_1( 0 ) > -1 ) {
      tier_1_idle_shard_counts_[vi] += state.batch_size();
      tier_1_idle_shards_[vi].push_back( RefStateType( std::move( state ) ) );
    } else {
      tier_2_idle_shard_counts_[vi] += state.batch_size();
      tier_2_idle_shards_[vi].push_back( RefStateType( std::move( state ) ) );
    }
  }

  if ( can_form_monolith( next_layer, next_stage ) ) {
    StateType monolithic_state = form_monolith( next_layer, next_stage );
    process_monolith( std::move( monolithic_state ) );
  }
}

/// @brief
/// ChildTierRouter is an empty middle-man between the worker and kernel. It's job is to mimic the TierRouter do the
/// worker is oblivious to which rank it has. DummyTierRouter that pass states through with no delay.
template<typename Model>
class ChildTierRouter : public TierRouter
{
public:
  ChildTierRouter( const ComputeKernel& compute_kernel,
                   const size_t n_tier_1,
                   const Concurrency& concurrency_tier_1,
                   const kv_slots_tier_1,
                   const size_t n_tier_2,
                   const Concurrency& concurrency_tier_2,
                   const kv_slots_tier_2,
                   const typename Model::SettingsType& settings );

  virtual ChildTierRouter() override;

  /// @brief
  /// 1. Push sharded state from kernel -> send state to worker
  virtual void pull_from_kernel() override;

  /// @brief
  /// 1. Push sharded state from worker -> send state to kernel
  virtual void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) override;

  /// @brief
  /// Behaves similar to TierRouter
  virtual bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state ) override;

  /// @brief
  /// This should never be called.
  virtual bool is_context_available() const override;
};

template<typename Model>
void ChildTierRouter<Model>::ChildTierRouter( const ComputeKernel& compute_kernel,
                                              const size_t n_tier_1,
                                              const Concurrency& concurrency_tier_1,
                                              const kv_slots_tier_1,
                                              const size_t n_tier_2,
                                              const Concurrency& concurrency_tier_2,
                                              const kv_slots_tier_2,
                                              const typename Model::SettingsType& settings )
  : compute_kernel_( std::make_unique( compute_kernel ) )
{
}

template<typename Model>
void ChildTierRouter<Model>::pull_from_kernel()
{
  compute_kernel_->event_fd().read_event();
  models::BatchedInferenceState<ConfigType> state;
  while ( compute_kernel_->pop( state ) ) {
    outgoing_.emplace( std::move( state ) );
    event_fd_.write_event();
  }
}

template<typename Model>
void ChildTierRouter<Model>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  compute_kernel_->push( std::move( state ) );
}

template<typename Model>
bool ChildTierRouter<Model>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename Model>
bool ChildTierRouter<Model>::is_context_available() const
{
  LOG( FATAL ) << "DummyTierRouter should never receive new batches. That is only going to happen in slice0, tier1, "
                  "rank0.";
}

} // namespace glinthawk::compute
