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

template<typename Model>
class TierRouter
{
public:
  // TODO(pouya): complete the constructor
  TierRouter( const size_t n_tier_1,
              const Concurrency concurrency_tier_1,
              const kv_slots_tier_1,
              const size_t n_tier_2,
              const Concurrency concurrency_tier_2,
              const kv_slots_tier_2,
              const typename Model::SettingsType& settings,
              const std::unique_ptr<ComputeKernel> compute_kernel );

  virtual ~TierRouter() = 0;

  virtual EventFD& event_fd() = 0;

  virtual void push_from_kernel( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) = 0;

  virtual void push_from_worker( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) = 0;

  virtual bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state ) = 0;

  virtual bool is_context_available() = 0;
};

/// @brief
/// ParentTierRouter is a middle-man between the worker and kernel. It's job is to break down states between machines in
/// tiers, while ensuring context does not run out and deadlocks don't happen.
/// ParentTierRouter is only run on `rank-0' machines, i.e., it only runs on one node in each `slice' that may serve
/// multiple layers. The other nodes in the slice have a ChildTierRouter that pass states through with no delay. There
/// is a possibility for improving this, where the tier-to-tier gather and scatter operations are distributed.
/// ParentTierRouter distinguishes states as two types:
///     Parent BatchInferenceState: which is a full batch across all machines in both tiers.
///     Child BatchInferenceState: which is a specific part of a parent dedicated to a specific machine.
/// ParentTierRouter receives parent/child states through push, and returns child through an outgoing queue. It sends to
/// and receives local child states from the compute kernel directly.
/// The worker reads an outgoing queue from ParentTierRouter to send out outbound parent states.
/// ParentTierRouter fully handles context management, and guarantees that if a child state arrives at the kernel, that
/// kernel has space for it.
// TODO(pouya): can straggler's cause an unstable failure mode in dispersing work among tiers?
// TODO(pouya): how does routing work?
// TODO(pouya): how does worker know to send a child state back to its origin?
// TODO(pouya): do children carry discards?
// TODO(pouya): who clears discards?
// TODO(pouya): child states are one stage past when they return, compared to when they were sent, this might meme us.
// TODO(pouya): I'm forcing kernel to not have an outgoing queue. Talk to sadjad about this.
// TODO(pouya): I have to separate push to from_worker and from_kernel, since the ChildTierRouter can't tell where it is
//  getting the state from.
template<typename Model>
class ParentTierRouter : public glinthawk::compute::TierRouter
{
public:
  // TODO(pouya): complete the constructor, does it need virtual?
  ParentTierRouter( const size_t n_tier_1,
                    const Concurrency concurrency_tier_1,
                    const kv_slots_tier_1,
                    const size_t n_tier_2,
                    const Concurrency concurrency_tier_2,
                    const kv_slots_tier_2,
                    const typename Model::SettingsType& settings,
                    const std::unique_ptr<ComputeKernel> compute_kernel );

  virtual ~ParentTierRouter() override;

  virtual EventFD& event_fd() override { return event_fd_; }

  /// @brief
  /// 1. Push child state from kernel -> process_child_state, may internally call process_parent_state
  virtual void push_from_kernel( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) override;

  /// @brief
  /// 1. Push parent state from worker -> calls process_parent_state
  /// 2. Push child state from worker -> process_child_state, may internally call process_parent_state
  virtual void push_from_worker( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) override;

  /// @brief
  /// pop is called by the worker to receive states, that (1) are parent states that are no longer local to the node, or
  /// (2) are child states that resulted from the local node pushing the last
  virtual bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state ) override;

  /// @brief
  /// is_context_available checks if we have context available for attention for all layers in this slice. It returns
  /// false if even one node does not have the corresponding concurrency KV slots. Worker calls this function *before*
  /// pushing new prompts, but does not need to check when receiving states over the network.
  /// This is because we are assuming all slices are alike, and if slice 0 has context, so do the others. Thus, this
  /// function is truly only relevant in slice 0.
  /// This function will only return "true" a finite number of times before all contexts are filled up. From that point,
  /// new prompts are placed in discarded prompt locations, and will have context by default.
  virtual bool is_context_available() override;

private:
  using ConfigType = typename Model::ConfigType;
  using Stage = glinthawk::models::InferenceStage;
  using StateType = typename glinthawk::models::BatchedInferenceState<ConfigType>;
  // TODO(pouya): double check if reference_wrapper is necessary here
  using RefStateType = typename std::reference_wrapper<StateType>;

  /// @brief
  /// For now we assume all tier 1 machines are alike and all tier 2 machines are alike.
  /// We also assume all slices are alike.
  /// The ParentTierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of
  /// two concurrencies.
  const size_t n_tier_1_;
  const Concurrency concurrency_tier_1_;
  const size_t n_tier_2_;
  const Concurrency concurrency_tier_2_;

  const size_t start_layer_;
  const size_t end_layer_;

  // Worker doesn't see the compute kernel. Only the tier router does.
  std::unique_ptr<ComputeKernel> compute_kernel_ { nullptr };

  std::vector<VirtualPreallocatingContextManager<Model>> cms_tier_1_;
  std::vector<VirtualPreallocatingContextManager<Model>> cms_tier_2_;

  std::vector<std::deque<RefStateType>> tier_1_idle_child_states_;
  std::vector<std::deque<RefStateType>> tier_2_idle_child_states_;

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
    // TODO(pouya) what if there is only one slice? the batch never gets pushed to worker to report generations.
    return state.next_layer() >= start_layer_ and state.next_layer() <= end_layer_;
  }

  bool can_form_parent( const size_t layer, const Stage stage ) const;

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
  /// child must have context on the remote/local machine for it. This is guaranteed by checking with
  /// is_context_available() before scheduling "new" states that are made from scratch. Old states with new prompts in
  /// discarded spots are by default going to have context.
  void process_parent_state( StateType&& state );

  /// @brief
  /// process_child_state manages an internal memory to merge states of various tiers together. If it receives a child
  /// state, it saves it. Upon merging, it may process it the same way it does the parent. If the parent does not need
  /// processing (is not local to that machine anymore, e.g., is for a different slice) it is sent out via an outgoing
  /// queue that the worker communicates with. This function may be indirectly called by the compute kernel or the
  /// worker.
  void process_child_state( StateType&& state );
};

template<typename Model>
void ParentTierRouter<Model>::push_from_kernel( models::BatchedInferenceState<ConfigType>&& state )
{
  // TODO(pouya): we hold push_mutex here, and outgoing_mutex and compute_kernel incoming mutex later. Need to check for
  //  deadlocks or revise mutex style.
  std::lock_guard lock { push_mutex_ };

  process_child_state( std::move( state ) );
}

template<typename Model>
void ParentTierRouter<Model>::push_from_worker( models::BatchedInferenceState<ConfigType>&& state )
{
  // TODO(pouya): we hold push_mutex here, and outgoing_mutex and compute_kernel incoming mutex later. Need to check for
  //  deadlocks or revise mutex style.
  std::lock_guard lock { push_mutex_ };

  if ( state.is_parent() )
    process_parent_state( std::move( state ) );
  else
    process_child_state( std::move( state ) );
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
bool ParentTierRouter<Model>::is_context_available()
{
  for ( size_t i = 0; i < n_tier_1_; i++ ) {
    if ( cms_tier_1_[i].free() < concurrency_tier_1_.get( glinthawk::models::InferenceStage::Attention ) ) {
      return false;
    }
  }
  for ( size_t i = 0; i < n_tier_2_; i++ ) {
    if ( cms_tier_2_[i].free() < concurrency_tier_2_.get( glinthawk::models::InferenceStage::Attention ) ) {
      return false;
    }
  }

  return true;
}

template<typename Model>
bool ParentTierRouter<Model>::can_form_parent( const size_t layer, const Stage stage )
{
  return tier_1_idle_child_state_counts_[vector_index( layer, stage )] > n_tier_1_ * concurrency_tier_1_.get( stage )
         and tier_2_idle_child_state_counts_[vector_index( layer, stage )]
               > n_tier_2_ * concurrency_tier_2_.get( stage )
}

template<typename Model>
void ParentTierRouter<Model>::assign_sub_groups( StateType& state ) const
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
StateType&& ParentTierRouter<Model>::form_parent( const size_t layer, const Stage stage )
{
  const auto vi = vector_index( layer, stage );
  // TODO(pouya): this is probably super inefficient. We should be able to "reserve" the correct size beforehand.
  // TODO(pouya): merge creates a new state underneath, that should be unnecessary if we have the full size allocated.
  // TODO(pouya): not sure if a "lazy" merge isn't a better option, especially if we're going to send it over the wire
  //  next. We can do an unlazify() func in compute_kernel->incoming thread for cases where it is going to the kernel.
  StateType base_state;
  // TODO(pouya): while not sending states with batch_size=0 is efficient, this is spaghetti code. Better code would be
  //  to compile a list of states to merge, and then use a single function to merge them.
  if ( concurrency_tier_1_.get( stage ) > 0 ) {
    base_state = std::move( tier_1_idle_child_states_[vi].front()->get() );
    tier_1_idle_child_states_[vi].pop_front();
    tier_1_idle_child_state_counts_[vi] -= concurrency_tier_1_.get( stage );
    for ( size_t i = 1; i < n_tier_1_; i++ ) {
      base_state.merge( std::move( tier_1_idle_child_states_[vi].front()->get() ) );
      tier_1_idle_child_states_[vi].pop_front();
      tier_1_idle_child_state_counts_[vi] -= concurrency_tier_1_.get( stage );
    }
    if ( concurrency_tier_2_.get( stage ) > 0 ) {
      for ( size_t i = 0; i < n_tier_2_; i++ ) {
        base_state.merge( std::move( tier_2_idle_child_states_[vi].front()->get() ) );
        tier_2_idle_child_states_[vi].pop_front();
        tier_2_idle_child_state_counts_[vi] -= concurrency_tier_2_.get( stage );
      }
    }
  } else {
    CHECK_GT( concurrency_tier_2_.get( stage ), 0 );
    base_state = std::move( tier_2_idle_child_states_[vi].front()->get() );
    tier_2_idle_child_states_[vi].pop_front();
    tier_2_idle_child_state_counts_[vi] -= concurrency_tier_2_.get( stage );
    for ( size_t i = 1; i < n_tier_2_; i++ ) {
      base_state.merge( std::move( tier_2_idle_child_states_[vi].front()->get() ) );
      tier_2_idle_child_states_[vi].pop_front();
      tier_2_idle_child_state_counts_[vi] -= concurrency_tier_2_.get( stage );
    }
  }
  return base_state;
}

template<typename Model>
void ParentTierRouter<Model>::process_parent_state( StateType&& state )
{
  assign_sub_groups( state );
  if ( not is_served_in_this_slice( state ) ) {
    {
      // TODO(pouya): not sure if it makes sense to merge parent in the current slice if the next stage is in another
      //  slice. In general, optimizing the data transfer graph between tiers in different slices is something we should
      //  think about.
      std::lock_guard lock { outgoing_.mutex };
      outgoing_.emplace( std::move( state ) );
    }
    event_fd_.write_event();
  } else {
    // TODO(pouya): this is yet again spaghetti code. A better split function should take a list of batch sizes and
    //  split them. If some batch_size=0, it should ignore it. We can also cache these size lists for efficiency.
    if ( concurrency_tier_1_.get( state.next_stage() ) > 0 ) {

      auto [state_rank_0_t1, state] = state.split( concurrency_tier_1_.get( state.next_stage() ) );
      CHECK_EQ( cms_tier_1_[0].get_contexts( state_rank_0_t1 ), true );
      compute_kernel_->push( std::move( state_rank_0_t1 ) );
      {
        std::lock_guard lock { outgoing_.mutex };
        for ( size_t i = 1; i < n_tier_1_; i++ ) {
          auto [state_rank_i_t1, state] = state.split( concurrency_tier_1_.get( state.next_stage() ) );
          CHECK_EQ( cms_tier_1_[i].get_contexts( state_rank_i_t1 ), true );
          outgoing_.emplace( std::move( state_rank_i_t1 ) );
          // TODO: one write_event per state?
          event_fd_.write_event();
        }
        if ( concurrency_tier_2_.get( state.next_stage() ) > 0 ) {
          for ( size_t i = 0; i < n_tier_2_ - 1; i++ ) {
            auto [state_rank_i_t2, state] = state.split( concurrency_tier_2_.get( state.next_stage() ) );
            CHECK_EQ( cms_tier_2_[i].get_contexts( state_rank_i_t2 ), true );
            outgoing_.emplace( std::move( state_rank_i_t2 ) );
            event_fd_.write_event();
          }
          outgoing_.emplace( std::move( state ) );
          event_fd_.write_event();
        }
      }
    } else {
      {
        std::lock_guard lock { outgoing_.mutex };
        CHECK_GT( concurrency_tier_2_.get( state.next_stage() ), 0 );
        for ( size_t i = 0; i < n_tier_2_ - 1; i++ ) {
          auto [state_rank_i_t2, state] = state.split( concurrency_tier_2_.get( state.next_stage() ) );
          CHECK_EQ( cms_tier_2_[i].get_contexts( state_rank_i_t2 ), true );
          outgoing_.emplace( std::move( state_rank_i_t2 ) );
          event_fd_.write_event();
        }
        CHECK_EQ( cms_tier_2_[n_tier_2_ - 1].get_contexts( state ), true );
        outgoing_.emplace( std::move( state ) );
        event_fd_.write_event();
      }
    }
  }
}

template<typename Model>
void ParentTierRouter<Model>::process_child_state( StateType&& state )
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
    tier_1_idle_child_states_[vi].push_back( RefStateType( std::move( state ) ) );
  } else {
    tier_2_idle_child_state_counts_[vi] += state.batch_size();
    tier_2_idle_child_states_[vi].push_back( RefStateType( std::move( state ) ) );
  }

  if ( can_form_parent( next_layer, next_stage ) ) {
    // TODO(pouya): no move needed here? I never learn
    StateType merged_parent_state = form_parent( next_layer, next_stage );
    process_parent_state( std::move( merged_parent_state ) );
  }
}

/// @brief
/// ChildTierRouter is an empty middle-man between the worker and kernel. It's job is to mimic the TierRouter do the
/// worker is oblivious to which rank it has. DummyTierRouter that pass states through with no delay.
template<typename Model>
class ChildTierRouter : public glinthawk::compute::TierRouter
{
public:
  // TODO(pouya): complete the constructor, does it need virtual?
  ChildTierRouter( const size_t n_tier_1,
                   const Concurrency concurrency_tier_1,
                   const kv_slots_tier_1,
                   const size_t n_tier_2,
                   const Concurrency concurrency_tier_2,
                   const kv_slots_tier_2,
                   const typename Model::SettingsType& settings,
                   const std::unique_ptr<ComputeKernel> compute_kernel );

  virtual ChildTierRouter() override;

  virtual EventFD& event_fd() override { return event_fd_; }

  /// @brief
  /// 1. Push child state from kernel -> send state to worker
  virtual void push_from_kernel( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) override;

  /// @brief
  /// 1. Push child state from worker -> send state to kernel
  virtual void push_from_worker( glinthawk::models::BatchedInferenceState<ConfigType>&& state ) override;

  /// @brief
  /// Behaves similar to TierRouter
  virtual bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state ) override;

  /// @brief
  /// This should never be called.
  virtual bool is_context_available() override;

private:
  // Worker doesn't see the compute kernel. Only the tier router does.
  std::unique_ptr<ComputeKernel> compute_kernel_ { nullptr };

  EventFD event_fd_ {};
  GlobalQueue outgoing_;
};

template<typename Model>
void ChildTierRouter<Model>::push_from_kernel( models::BatchedInferenceState<ConfigType>&& state )
{
  outgoing_.emplace( std::move( state ) );
  event_fd_.write_event();
}

template<typename Model>
void ChildTierRouter<Model>::push_from_worker( models::BatchedInferenceState<ConfigType>&& state )
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
bool ChildTierRouter<Model>::is_context_available()
{
  LOG( FATAL ) << "DummyTierRouter should never receive new batches. That is only going to happen in slice0, tier1, "
                  "rank0.";
}

} // namespace glinthawk::compute
