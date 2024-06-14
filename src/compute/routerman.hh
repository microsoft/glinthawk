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

/// @brief
/// TierRouter is a middle-man between the worker and kernel. It's job is to break down states between machines in
/// tiers, while ensuring context does not run out and deadlocks don't happen.
/// TierRouter is only run on `rank-0' machines, i.e., it only runs on one node in each `slice' that may serve multiple
/// layers. The other nodes in the slice have a dummy TierRouter that pass states through with no delay. There is a
/// possibility for improving this, where the tier-to-tier gather and scatter operations are distributed.
/// TierRouter distinguishes states as two types:
///     Parent BatchInferenceState: which is a full batch across all machines in both tiers.
///     Child BatchInferenceState: which is a specific part of a parent dedicated to a specific machine.
/// TierRouter receives parent/child states through push, and returns child through an outgoing queue. It sends to and
/// receives local child states from the compute kernel directly.
/// The worker reads an outgoing queue from TierRouter to send out outbound parent states.
/// TierRouter fully handles context management, and guarantees that if a child state arrives at the kernel, that kernel
/// has space for it.
// TODO(pouya): how does routing work?
// TODO(pouya): how does worker know to send a child state back to its origin?
// TODO(pouya): do children carry discards?
// TODO(pouya): who clears discards?
// TODO(pouya): child states are one stage past when they return, compared to when they were sent, this might meme us.
template<typename Model>
class TierRouter
{
public:
  // TODO(pouya): complete the constructor
  TierRouter();

  ~TierRouter();

  EventFD& event_fd() { return event_fd_; }

  /// @brief
  /// if the state was a parent state calls process_parent_state
  /// if the state was a child state calls process_child_state
  void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state );

  /// @brief
  /// pop is called by the worker to receive states, that (1) are parent states that are no longer local to the node, or
  /// (2) are child states that resulted from the local node pushing the last
  bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state );

private:
  using ConfigType = typename Model::ConfigType;
  using Stage = glinthawk::models::InferenceStage;
  using StateType = typename glinthawk::models::BatchedInferenceState<ConfigType>;

  /// @brief
  /// For now we assume all tier 1 machines are alike and all tier 2 machines are alike.
  /// The TierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of two
  /// concurrencies.
  // TODO(pouya): why do we need kv_slots here?
  const size_t n_tier_1_;
  const size_t kv_slots_tier_1_;
  const Concurrency concurrency_tier_1_;
  const size_t n_tier_2_;
  const size_t kv_slots_tier_2_;
  const Concurrency concurrency_tier_2_;

  const size_t start_layer_;
  const size_t end_layer_;

  std::vector<VirtualPreallocatingContextManager<Model>> cms_tier_1_;
  std::vector<VirtualPreallocatingContextManager<Model>> cms_tier_2_;

  std::vector<std::deque<StateType>> tier_1_idle_child_states_;
  std::vector<std::deque<StateType>> tier_2_idle_child_states_;

  std::vector<size_t> tier_1_idle_child_state_counts_;
  std::vector<size_t> tier_2_idle_child_state_counts_;

  std::mutex push_mutex_;
  EventFD event_fd_ {};
  GlobalQueue outgoing_;

  inline size_t vector_index( const size_t layer, const Stage stage ) const
  {
    return ( layer - start_layer_ ) * util::to_underlying( models::InferenceStage::__COUNT__ )
           + util::to_underlying( stage );
  }

  std::pair<uint8_t, uint8_t> tier_index( const size_t batch_index, const Stage stage ) const
  {
    // TODO(pouya): is the syntax right?
    if ( i < n_tier_1_ * concurrency_tier_1_.get( stage ) ) {
      return { i / concurrency_tier_1_.get( stage ), -1 };
    } else {
      return { -1, ( i - n_tier_1_ * concurrency_tier_1_.get( stage ) ) / concurrency_tier_2_.get( stage ) };
    }
  }

  bool inline is_served_in_this_slice( const StateType& state ) const
  {
    // TODO(pouya): this is assuming classification is done on the same machine doing pre-att-post
    // TODO(pouya) what if there is only one slice? the batch never gets pushed to worker to report generations.
    return state.next_layer() >= start_layer_ and state.next_layer() <= end_layer_;
  }

  bool can_form_parent( const size_t layer, const Stage stage ) const;

  // TODO(pouya): looks super redundant with the above function
  // TODO(pouya): is the signature right?
  StateType&& form_parent( const size_t layer, const Stage stage );

  /// @brief
  /// assign_sub_groups assigns tier_routing_group indices to states that have not been assigned them before. The
  /// assignment is based on the index of each prompt in the state. Input states are either fully assigned (all slots
  /// in the batch inference state were assigned before) or none were assigned. This function is called a finite number
  /// of times before all context slots are taken, and never called again. Note that when worker replaces a discarded
  /// prompt, it keeps any assigned tier_routing_group indices. This is important to avoid situations where new prompts
  /// are assigned to tier sub groups that their neighbors do not belong in, which causes fragmentation.
  void assign_sub_groups( StateType& state ) const;

  /// @brief
  /// process_parent_state breaks the `parent' state to `child' states by the tier_routing_group in each prompt. Each
  /// child is only sent out if the remote/local machine can allot context for it. Note that we do not care if some
  /// other machine has context for its own child state or not. If said context does not exist, TierRouter holds onto
  /// that child state until it does. The latter only occurs on releasing past context, which also happens in push. So,
  /// push may return older child states or return none at all.
  void process_parent_state( StateType&& state );

  /// @brief
  /// process_child_state manages an internal memory to merge states of various tiers together. If it receives a child
  /// state, it saves it. Upon merging, it may process it the same way it does the parent. If the parent does not need
  /// processing (is not local to that machine anymore, e.g., is for a different slice) it is sent out via an outgoing
  /// queue that the worker communicates with. This function may be indirectly called byh the compute kernel or the
  /// worker.
  void process_child_state( StateType&& state );
};

template<typename Model>
void TierRouter<Model>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  std::lock_guard lock { push_mutex_ };

  if ( state.is_parent() )
    process_parent_state( std::move( state ) );
  else
    process_child_state( std::move( state ) );
}

template<typename Model>
bool TierRouter<Model>::pop( models::BatchedInferenceState<ConfigType>& state )
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
bool TierRouter<Model>::can_form_parent( const size_t layer, const Stage stage )
{
  return tier_1_idle_child_state_counts_( vector_index( layer, stage ) ) > n_tier_1_ * concurrency_tier_1_.get( stage )
         and tier_2_idle_child_state_counts_( vector_index( layer, stage ) )
               > n_tier_2_ * concurrency_tier_2_.get( stage )
}

template<typename Model>
void TierRouter<Model>::assign_sub_groups( StateType& state ) const
{
  CHECK_EQ( ( state.is_parent(), true );
  CHECK_EQ( ( state.batch_size(), n_tier_1_ * concurrency_tier_1_.get( stage ) + n_tier_2_ * concurrency_tier_2_.get( stage ) );
  bool already_assigned = state.tier_routed( 0 );
  for ( size_t i = 0; i < state.batch_size(); i++ ) {
    if ( already_assigned != state.tier_routed( i ) ) {
      LOG( FATAL ) << "Either all prompts are already tier-routed or none of them are.";
    }
    if ( not state.tier_routed( i ) ) {
      const auto t_ind = tier_index( i, stage );
      state.set_tier_1_routing_group( t_ind.first );
      state.set_tier_2_routing_group( t_ind.second );
    }
  }
}

template<typename Model>
StateType&& TierRouter<Model>::form_parent( const size_t layer, const Stage stage )
{
  // TODO(pouya): merge children to parent
}

template<typename Model>
void TierRouter<Model>::process_parent_state( StateType&& state )
{
  if ( not is_served_in_this_slice( state ) ) {
    {
      std::lock_guard lock { outgoing_.mutex };
      outgoing_.emplace( std::move( state ) );
    }
    event_fd_.write_event();
  } else {
    // TODO(pouya): break the parent and route
  }
}

template<typename Model>
void TierRouter<Model>::process_child_state( StateType&& state )
{
  for ( size_t i = 0; i < state.batch_size(); i++ ) {
    if ( not state.tier_routed( i ) ) {
      LOG( FATAL ) << "Child states must always be already routed.";
    }
  }

  const auto next_stage = state.next_stage();
  const auto next_layer = state.next_layer();
  const auto vi = vector_index( next_layer, next_stage );

  if ( state.get_tier_1_routing_group( 0 ) > -1 ) {
    tier_1_idle_child_state_counts_[vi] += state.batch_size();
    tier_1_idle_child_states_[vi].push_back( std::move( state ) );
  } else {
    tier_2_idle_child_state_counts_[vi] += state.batch_size();
    tier_2_idle_child_states_[vi].push_back( std::move( state ) );
  }

  if ( can_form_parent( next_layer, next_stage ) ) {
    // TODO(pouya): no move needed here? I never learn
    StateType merged_parent_state = form_parent( next_layer, next_stage );
    process_parent_state( std::move( merged_parent_state ) );
  }
}

} // namespace glinthawk::compute
