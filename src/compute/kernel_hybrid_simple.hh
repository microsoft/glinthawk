#pragma once

#include <array>
#include <atomic>
#include <barrier>
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
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "monitoring/measurement.hh"
#include "util/eventfd.hh"
#include "util/util.hh"

#include "common.hh"
#include "contextman.hh"

namespace glinthawk::compute {

template<typename ModelA, typename ModelB>
requires std::same_as<typename ModelA::ConfigType, typename ModelB::ConfigType>
class SimpleHybridComputeKernel
{
public:
  using Stage = glinthawk::models::InferenceStage;

  using ConfigType = typename ModelA::ConfigType;
  using StateType = glinthawk::models::BatchedInferenceState<ConfigType>;
  using ContextPtrA = std::shared_ptr<typename ModelA::ContextType>;
  using ContextPtrB = std::shared_ptr<typename ModelB::ContextType>;

  static constexpr KernelType Type = KernelType::SimpleHybrid;

public:
  template<typename... Args>
  SimpleHybridComputeKernel( const uint32_t concurrency, Args&&... args );

  ~SimpleHybridComputeKernel();

  void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state );
  bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state );
  EventFD& event_fd() { return event_fd_; }

private:
  template<typename M>
  requires std::same_as<M, ModelA> || std::same_as<M, ModelB>
  struct ModelData
  {
    ModelData( std::unique_ptr<M>&& in_model );

    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;
  };

  // The number of concurrent prompts that are processed by the model; this value will be used across all stages.
  // If you need different concurrency values for each stage, please use `HybridComputeKernel`.
  size_t concurrency_ {};

  // X -> [pre_a(X) -> att_b(X) -> post_a(X)] * n_layers -> classify_a(X)
  ModelData<ModelA> a_;
  ModelData<ModelB> b_;

  // a state with some empty slots
  std::optional<StateType> incomplete_state_ {};

  // This file descriptor is used to notify the kernel user that there are new states in the outgoing queue.
  EventFD event_fd_ {};
  std::atomic<bool> running_ { true };

  // <context management>
  uint64_t current_local_state_id_ {}; // keeping track of the populated contexts for the states
  void release_discarded_contexts( const StateType& state );
  // </context management>

  // <queues>
  // global queues:
  struct GlobalQueue
  {
    std::queue<StateType> queue;
    std::mutex mutex;
    std::condition_variable cv;
  };

  // incoming -> (waiting|processing) x N -> outgoing
  GlobalQueue incoming_;
  GlobalQueue waiting_;
  GlobalQueue outgoing_;
  // </queues>

  // Used to synchronize the model threads. Every time this barrier is reached, the processing mode is toggled,
  // swapping the states processed by ModelA and ModelB.
  std::barrier<> sync_point { 2 };

  // Whether the states are being processed at the moment
  std::atomic<bool> is_processing_states_ { false };

  // The states that are currently being processed
  std::array<StateType, 2> active_states_ {};
  std::array<std::vector<ContextPtrB>, 2> active_contexts_ {};

  // The states that will be processed next
  std::array<StateType, 2> next_states_ {};
  std::array<std::vector<ContextPtrB>, 2> next_contexts_ {};

  // Number of contexts that have been released, but not yet refilled
  std::atomic<size_t> released_contexts_ {};

  // <threads>
  std::vector<std::thread> threads_;

  template<typename M>
  void execution_thread_func( ModelData<M>& model_data );

  void bookkeeping_thread_func();
  void backlog_thread_func();
  // </threads>

  void model_forward( StateType& state );

  template<typename CtxType>
  void model_forward( StateType& state, std::vector<CtxType>& contexts );
};

template<typename ModelA, typename ModelB>
template<typename M>
SimpleHybridComputeKernel<ModelA, ModelB>::ModelData<M>::ModelData( std::unique_ptr<M>&& in_model )
  : model( std::move( in_model ) )
  , context_manager( model->settings() )
{
}

template<typename ModelA, typename ModelB>
template<typename... Args>
SimpleHybridComputeKernel<ModelA, ModelB>::SimpleHybridComputeKernel( const uint32_t concurrency, Args&&... args )
  : concurrency_( concurrency / 2 )
  , a_( std::make_unique<ModelA>( std::forward<Args>( args )... ) )
  , b_( std::make_unique<ModelB>( std::forward<Args>( args )... ) )
{
  threads_.emplace_back( &SimpleHybridComputeKernel::backlog_thread_func, this );
  threads_.emplace_back( &SimpleHybridComputeKernel::bookkeeping_thread_func, this );
  threads_.emplace_back( &SimpleHybridComputeKernel::execution_thread_func<ModelA>, this, std::ref( a_ ) );
  threads_.emplace_back( &SimpleHybridComputeKernel::execution_thread_func<ModelB>, this, std::ref( b_ ) );
}

template<typename ModelA, typename ModelB>
SimpleHybridComputeKernel<ModelA, ModelB>::~SimpleHybridComputeKernel()
{
  running_ = false;
  for ( auto& t : threads_ ) {
    t.join();
  }
}

template<typename ModelA, typename ModelB>
void SimpleHybridComputeKernel<ModelA, ModelB>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  auto push_to_queue = [this]( StateType&& s ) {
    s.set_id( current_local_state_id_++ );

    {
      std::lock_guard lock { incoming_.mutex };
      incoming_.queue.push( std::move( s ) );
    }

    incoming_.cv.notify_one();
  };

  // (1) discard the contexts we have to discard
  if ( state.discarded_contexts() ) {
    release_discarded_contexts( state );
  }

  // (2) is this the last layer? if so, we can get rid of the discard list.
  if ( a_.model->settings().end_layer_num == ConfigType::n_layers - 1 ) {
    state.clear_discards();
  }

  // XXX maybe we should do all this outside of the kernel
  // (3) do we have an incomplete state to merge this with?
  if ( incomplete_state_.has_value() ) {
    auto& new_state = incomplete_state_.value();
    new_state.replenish_from( state );

    if ( new_state.free_slots() == 0 ) {
      push_to_queue( std::move( new_state ) );
      incomplete_state_.reset();
    } else {
      // if there's a free slot, it means that the input state must be now empty and we can discard it
      CHECK_EQ( state.free_slots(), state.batch_size() );
      return;
    }
  }

  // (4) there's something left in the input state; let's see if we can push it to the incoming queue
  if ( state.free_slots() == 0 ) {
    // all slots are filled; push it to the incoming queue
    push_to_queue( std::move( state ) );
  } else if ( state.free_slots() < state.batch_size() ) {
    // some slots are filled; keep it as an incomplete state
    incomplete_state_ = std::move( state );
  } else {
    // all slots are empty; discard it
  }
}

template<typename ModelA, typename ModelB>
bool SimpleHybridComputeKernel<ModelA, ModelB>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.front() );
  outgoing_.queue.pop();
  return true;
}

template<typename ModelA, typename ModelB>
void SimpleHybridComputeKernel<ModelA, ModelB>::release_discarded_contexts( const StateType& state )
{
  size_t freed_contexts = 0;

  for ( size_t i = 0; i < state.discarded_contexts(); i++ ) {
    auto& prompt_id = state.discarded_prompt_id( i );
    if ( b_.context_manager.release_context( prompt_id ) ) {
      freed_contexts++;
    }
  }

  if ( freed_contexts ) {
    released_contexts_.fetch_add( freed_contexts );
    waiting_.cv.notify_one();
  }
}

template<typename ModelA, typename ModelB>
void SimpleHybridComputeKernel<ModelA, ModelB>::model_forward( StateType& state )
{
  StateType output;

  auto& model = *a_.model;

  const auto next_stage = state.next_stage();
  const auto next_layer = state.next_layer();
  const auto model_end_layer = model.settings().end_layer_num;

  if ( next_stage == Stage::PostAttention ) {
    output = model.post_attention_forward( std::move( state ) );

    if ( output.next_stage() == Stage::PreAttention and next_layer <= model_end_layer ) {
      // since we serve the next layer, let's do pre-attention right here
      output = model.pre_attention_forward( std::move( output ) );
    } else if ( output.next_stage() == Stage::Classification and next_layer == ConfigType::n_layers - 1
                and next_layer == model_end_layer ) {
      output = model.classify_forward( std::move( output ) );
    }
  } else if ( next_stage == Stage::PreAttention ) {
    output = model.pre_attention_forward( std::move( state ) );
  } else if ( next_stage == Stage::Classification ) {
    output = model.classify_forward( std::move( state ) );
  } else {
    LOG( FATAL ) << "Invalid stage: " << next_stage;
  }

  state = std::move( output );
}

template<typename ModelA, typename ModelB>
template<typename CtxType>
void SimpleHybridComputeKernel<ModelA, ModelB>::model_forward( StateType& state, std::vector<CtxType>& contexts )
{
  StateType output;
  auto& model = *b_.model;
  const auto next_stage = state.next_stage();

  if ( next_stage == Stage::Attention ) {
    output = model.attention_forward( std::move( state ), contexts );
  } else {
    LOG( FATAL ) << "Invalid stage: " << next_stage;
  }

  state = std::move( output );
}

template<typename ModelA, typename ModelB>
template<typename M>
void SimpleHybridComputeKernel<ModelA, ModelB>::execution_thread_func(
  typename SimpleHybridComputeKernel<ModelA, ModelB>::template ModelData<M>& model_data )
{
  constexpr size_t model_idx = std::is_same<M, ModelA>() ? 0 : 1;
  auto& model = *model_data.model;
  const auto N = model.settings().n_layers_loaded();

  while ( running_ ) {
    is_processing_states_.wait( false );

    for ( size_t iteration = 0; iteration < 2 * N + 2; iteration++ ) {
      sync_point.arrive_and_wait();

      // During even iterations, ModelA processes pre/post-attention for [0] and ModelB does attention for [1]. During
      // odd iterations, ModelA does pre/post-attention for [1] and ModelB does attention for [0].
      const auto active_state_index = ( model_idx == 0 ) ? ( iteration % 2 ) : ( ( iteration + 1 ) % 2 );
      StateType& input_state = active_states_[active_state_index];

      // run the corresponding forward function
      const auto next_stage = input_state.next_stage();
      const bool should_skip = ( model_idx == 1 and next_stage != Stage::Attention )
                               or ( model_idx == 0 and next_stage == Stage::Attention );

      if ( not should_skip ) {
        switch ( model_idx ) {
          case 0: model_forward( input_state ); break;
          case 1: model_forward( input_state, active_contexts_[active_state_index] ); break;
        }
      }
    }

    if constexpr ( model_idx == 0 ) {
      // We don't need to wait for the other thread, since it has nothing to do in the last iteration.
      active_states_[0].merge( std::move( active_states_[1] ) );

      {
        std::lock_guard lock { outgoing_.mutex };
        outgoing_.queue.push( std::move( active_states_[0] ) );
      }

      // notify the user about the new outgoing state
      event_fd_.write_event();

      // we're done with processing the input; let's signal the bookkeeping thread
      is_processing_states_.store( false );
      is_processing_states_.notify_all();
    }
  }
}

template<typename ModelA, typename ModelB>
void SimpleHybridComputeKernel<ModelA, ModelB>::bookkeeping_thread_func()
{
  while ( running_ ) {
    StateType incoming_state;

    {
      std::unique_lock lock { incoming_.mutex };
      incoming_.cv.wait( lock, [this] { return !incoming_.queue.empty(); } );
      incoming_state = std::move( incoming_.queue.front() );
      incoming_.queue.pop();
    }

    CHECK_EQ( incoming_state.batch_size(), concurrency_ * 2 ) << "Batch size mismatch.";

    // First, let's see if we have enough space for contexts.
    std::vector<ContextPtrB> incoming_contexts;
    incoming_contexts.reserve( incoming_state.batch_size() );

    for ( size_t i = 0; i < incoming_state.batch_size(); i++ ) {
      auto ctx = b_.context_manager.get_context( incoming_state.prompt_id( i ) );

      if ( not ctx ) {
        // We didn't have enough space for the contexts, this is going to the waiting queue.
        break;
      }

      incoming_contexts.push_back( std::move( ctx ) );
    }

    if ( incoming_contexts.size() != incoming_state.batch_size() ) {
      // We didn't have enough space for the contexts, this is going to the waiting queue.
      std::unique_lock lock { waiting_.mutex };
      waiting_.queue.push( std::move( incoming_state ) );
      continue;
    }

    // If we're processing the active_states_ at the moment, we prepare the next batch and put it in next_states_.
    // It will be swapped for active_states_ when the processing threads are done.
    const bool is_processing = is_processing_states_.load();
    decltype( active_states_ )& states_to_fill = ( not is_processing ) ? active_states_ : next_states_;
    decltype( active_contexts_ )& contexts_to_fill = ( not is_processing ) ? active_contexts_ : next_contexts_;

    // Split the incoming state into two parts.
    // XXX(sadjad): We should make this zero-copy.
    std::tie( states_to_fill[0], states_to_fill[1] ) = incoming_state.split( concurrency_ );

    contexts_to_fill[0] = { std::make_move_iterator( incoming_contexts.begin() ),
                            std::make_move_iterator( incoming_contexts.begin() + concurrency_ ) };

    contexts_to_fill[1] = { std::make_move_iterator( incoming_contexts.begin() + concurrency_ ),
                            std::make_move_iterator( incoming_contexts.end() ) };

    // Wait until the processing threads are done with the current batch
    is_processing_states_.wait( true );

    if ( is_processing ) {
      // We were processing states when we received the new batch. Swap the active and next states.
      active_states_ = std::move( next_states_ );
      active_contexts_ = std::move( next_contexts_ );
    }

    // Notify the processing threads that they can start processing the new batch
    is_processing_states_.store( true );
    is_processing_states_.notify_all();
  }
}

template<typename ModelA, typename ModelB>
void SimpleHybridComputeKernel<ModelA, ModelB>::backlog_thread_func()
{
  StateType waiting_state;

  while ( running_ ) {
    {
      std::unique_lock lock { waiting_.mutex };
      waiting_.cv.wait( lock, [this] {
        return !waiting_.queue.empty() and waiting_.queue.front().batch_size() <= released_contexts_.load();
      } );

      waiting_state = std::move( waiting_.queue.front() );
      waiting_.queue.pop();
      released_contexts_.fetch_sub( waiting_state.batch_size() );
    }

    // We have enough space for the contexts; let's put this on the incoming queue so it gets processed.
    {
      std::lock_guard lock { incoming_.mutex };
      incoming_.queue.push( std::move( waiting_state ) );
      incoming_.cv.notify_one();
    }
  }
}

} // namespace glinthawk::compute
