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
#include "monitoring/util.hh"
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

  // The global measurement object that is used to record the time spent in various parts of the kernel.
  Measurement& __stats__ { global_measurement() };

  // The number of concurrent prompts that are processed by the model; this value will be used across all stages.
  // If you need different concurrency values for each stage, please use `HybridComputeKernel`.
  size_t concurrency_ {};

  // X -> [pre_a(X) -> att_b(X) -> post_a(X)] * n_layers -> classify_a(X)
  ModelData<ModelA> a_;
  ModelData<ModelB> b_;

  // This file descriptor is used to notify the kernel user that there are new states in the outgoing queue.
  EventFD event_fd_ {};
  std::atomic<bool> running_ { true };

  // <context management>
  // TODO(pouya): understand what this is
  uint64_t current_local_state_id_ {}; // keeping track of the populated contexts for the states

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

  // <threads>
  std::vector<std::thread> threads_;

  template<typename M>
  void execution_thread_func( ModelData<M>& model_data );

  void bookkeeping_thread_func();
  // </threads>

  void model_step_forward( StateType& state );

  template<typename CtxType>
  void model_step_forward( StateType& state, std::vector<CtxType>& contexts );
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
  // TODO(pouya): this id doesn't seem to do anything here
  state.set_id( current_local_state_id_++ );

  {
    std::lock_guard lock { incoming_.mutex };
    incoming_.queue.push( std::move( state ) );
  }

  incoming_.cv.notify_one();
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
void SimpleHybridComputeKernel<ModelA, ModelB>::model_step_forward( StateType& state )
{
  StateType output;

  auto& model = *a_.model;

  switch ( state.next_stage() ) {
    case Stage::PostAttention:
      timeit<IntDistributions::KernelPostAttentionForwardTime>( __stats__,
                                                                [&] { model.forward_post_attention( state ); } );

      if ( state.next_stage() == Stage::PreAttention and state.next_layer() <= model.settings().end_layer_num ) {
        // since we serve the next layer, let's do pre-attention right here
        timeit<IntDistributions::KernelPreAttentionForwardTime>( __stats__,
                                                                 [&] { model.forward_pre_attention( state ); } );
      } else if ( state.next_stage() == Stage::Classification and state.next_layer() == ConfigType::n_layers - 1
                  and state.next_layer() == model.settings().end_layer_num ) {
        timeit<IntDistributions::KernelClassificationForwardTime>( __stats__,
                                                                   [&] { model.forward_classify( state ); } );
      }

      break;

    case Stage::PreAttention:
      timeit<IntDistributions::KernelPreAttentionForwardTime>( __stats__,
                                                               [&] { model.forward_pre_attention( state ); } );
      break;

    case Stage::Classification:
      timeit<IntDistributions::KernelClassificationForwardTime>( __stats__, [&] { model.forward_classify( state ); } );
      break;

    case Stage::Attention: LOG( FATAL ) << "Invalid stage: " << next_stage; break;
  }
}

template<typename ModelA, typename ModelB>
template<typename CtxType>
void SimpleHybridComputeKernel<ModelA, ModelB>::model_step_forward( StateType& state, std::vector<CtxType>& contexts )
{
  StateType output;
  auto& model = *b_.model;
  const auto next_stage = state.next_stage();

  if ( next_stage == Stage::Attention ) {
    timeit<IntDistributions::KernelAttentionForwardTime>( __stats__,
                                                          [&] { model.forward_attention( state, contexts ); } );
  } else {
    LOG( FATAL ) << "Invalid stage: " << next_stage;
  }
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

    const auto start_time = std::chrono::steady_clock::now();

    for ( size_t iteration = 0; iteration < 2 * N + 2; iteration++ ) {
      sync_point.arrive_and_wait();

      // During even iterations, ModelA processes pre/post-attention for [0] and ModelB does attention for [1]. During
      // odd iterations, ModelA does pre/post-attention for [1] and ModelB does attention for [0].
      constexpr auto active_state_index = ( model_idx == 0 ) ? ( iteration % 2 ) : ( ( iteration + 1 ) % 2 );
      StateType& input_state = active_states_[active_state_index];

      // run the corresponding forward function
      const auto next_stage = input_state.next_stage();
      const bool should_skip = ( model_idx == 1 and next_stage != Stage::Attention )
                               or ( model_idx == 0 and next_stage == Stage::Attention );

      if ( not should_skip ) {
        switch ( model_idx ) {
          case 0: model_step_forward( input_state ); break;
          case 1: model_step_forward( input_state, active_contexts_[active_state_index] ); break;
        }
      }
    }

    if constexpr ( model_idx == 0 ) {
      // We don't need to wait for the other thread, since it has nothing to do in the last iteration.
      active_states_[0].merge( std::move( active_states_[1] ) );

      const auto end_time = std::chrono::steady_clock::now();
      const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );
      __stats__.add_point<IntDistributions::KernelForwardTime>( duration.count() );

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
      auto ctx = b_.context_manager.get_context( incoming_state.context_id( i ) );
      CHECK( ctx ) << "TierRouter has guaranteed context space, but compute kernel doesn't have any";
      incoming_contexts.push_back( std::move( ctx ) );
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

} // namespace glinthawk::compute
