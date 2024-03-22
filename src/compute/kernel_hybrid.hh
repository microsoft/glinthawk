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
#include "monitoring/measurement.hh"
#include "util/eventfd.hh"
#include "util/util.hh"

#include "common.hh"
#include "contextman.hh"

namespace glinthawk::compute {

template<typename ModelA, typename ModelB>
requires std::same_as<typename ModelA::ConfigType, typename ModelB::ConfigType>
class HybridComputeKernel
{
public:
  using ConfigType = typename ModelA::ConfigType;
  using ContextPtrA = std::shared_ptr<typename ModelA::ContextType>;
  using ContextPtrB = std::shared_ptr<typename ModelB::ContextType>;

  static constexpr KernelType Type = KernelType::Hybrid;

  class Concurrency
  {
  private:
    std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )> v_;

  public:
    Concurrency( const size_t pre, const size_t att, const size_t post, const size_t classify )
      : v_ { pre, att, post, classify }
    {
      CHECK_GT( pre + att + post + classify, 0 ) << "At least one stage must be enabled";
    }

    void set( const models::InferenceStage stage, const size_t value ) { v_[util::to_underlying( stage )] = value; }
    size_t get( const models::InferenceStage stage ) const { return v_[util::to_underlying( stage )]; }
  };

public:
  template<typename... Args>
  HybridComputeKernel( const Concurrency& concurrency_a, const Concurrency& concurrency_b, Args&&... args );

  ~HybridComputeKernel();

  void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state );
  bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state );

  EventFD& event_fd() { return event_fd_; }

private:
  using Stage = glinthawk::models::InferenceStage;
  using StateType = glinthawk::models::BatchedInferenceState<ConfigType>;

  // std::priority_queue does not allow for moving elements, so we need to wrap the state in a struct
  // to be able to move it around. The struct keeps the comparison key separate from the state itself, so the state
  // can be moved out without affecting the queue's invariant.
  struct StateQueueItem
  {
    std::pair<size_t, size_t> _comp_key; /* (layer, stage) */
    mutable StateType state;

    StateQueueItem( StateType&& in_state )
      : state( std::move( in_state ) )
      , _comp_key( state.next_layer(), util::to_underlying( state.next_stage() ) )
    {
    }
  };

  struct StateCompOp
  {
    bool operator()( const StateQueueItem& lhs, const StateQueueItem& rhs ) const
    {
      return lhs._comp_key > rhs._comp_key;
    }
  };

  using StatePriorityQueue = std::priority_queue<StateQueueItem, std::deque<StateQueueItem>, StateCompOp>;

  template<typename M>
  requires std::same_as<M, ModelA> || std::same_as<M, ModelB>
  struct ModelData
  {
    ModelData( std::unique_ptr<M>&& in_model, const Concurrency& in_concurrency );

    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;
    const Concurrency concurrency;

    StatePriorityQueue processing {};
    std::mutex mutex {};
    std::condition_variable cv {};
  };

  // ... -> [pre(a|b) -> att(a|b) -> post(a|b)] * n_layers -> classify(a|b)
  ModelData<ModelA> a_;
  ModelData<ModelB> b_;

  // a state with some empty slots
  std::optional<StateType> incomplete_state_ {};

  EventFD event_fd_ {};
  std::atomic<bool> running_ { true };

  // keeping track of splitted states and merge them back when needed
  size_t current_local_state_id_ { 0 };
  std::map<size_t, std::pair<std::optional<StateType>, std::optional<StateType>>> splitted_state_map_ {};
  std::mutex splitted_state_mutex_ {};

  // <context management>
  // keeping track of the populated contexts for the states
  std::map<size_t, std::pair<std::vector<ContextPtrA>, std::vector<ContextPtrB>>> context_map_ {};
  std::mutex context_mutex_ {};

  void release_discarded_contexts( const StateType& state );
  // </context management>

  // <queues>
  // global queues:
  struct GlobalQueue
  {
    StatePriorityQueue queue;
    std::mutex mutex;
    std::condition_variable cv;
  };

  // incoming -> (waiting|{a,b}.processing) -> outgoing
  GlobalQueue incoming_;
  GlobalQueue waiting_;
  GlobalQueue outgoing_;
  // </queues>

  // <threads>
  template<typename M>
  void execution_thread_func( ModelData<M>& model_data );

  void bookkeeping_thread_func();
  void backlog_thread_func();

  std::vector<std::thread> threads_;
  // </threads>
};

template<typename ModelA, typename ModelB>
template<typename M>
HybridComputeKernel<ModelA, ModelB>::ModelData<M>::ModelData( std::unique_ptr<M>&& in_model,
                                                              const Concurrency& in_concurrency )
  : model( std::move( in_model ) )
  , context_manager( model->settings() )
  , concurrency( in_concurrency )
{
}

template<typename ModelA, typename ModelB>
template<typename... Args>
HybridComputeKernel<ModelA, ModelB>::HybridComputeKernel( const Concurrency& concurrency_a,
                                                          const Concurrency& concurrency_b,
                                                          Args&&... args )
  : a_( std::make_unique<ModelA>( std::forward<Args>( args )... ), concurrency_a )
  , b_( std::make_unique<ModelB>( std::forward<Args>( args )... ), concurrency_b )
{
  // check the concurrency settings to be permissible
  CHECK_EQ( a_.concurrency.get( Stage::PreAttention ) + b_.concurrency.get( Stage::PreAttention ),
            a_.concurrency.get( Stage::Attention ) + b_.concurrency.get( Stage::Attention ) );

  CHECK_EQ( a_.concurrency.get( Stage::Attention ) + b_.concurrency.get( Stage::Attention ),
            a_.concurrency.get( Stage::PostAttention ) + b_.concurrency.get( Stage::PostAttention ) );

  // Following is not always true; we need to figure it out before enabling it.
  // CHECK_EQ( a_.concurrency.get( Stage::PostAttention ) + b_.concurrency.get( Stage::PostAttention ),
  //           a_.concurrency.get( Stage::Classification ) + b_.concurrency.get( Stage::Classification ) );

  threads_.emplace_back( &HybridComputeKernel::backlog_thread_func, this );
  threads_.emplace_back( &HybridComputeKernel::bookkeeping_thread_func, this );
  threads_.emplace_back( &HybridComputeKernel::execution_thread_func<ModelA>, this, std::ref( a_ ) );
  threads_.emplace_back( &HybridComputeKernel::execution_thread_func<ModelB>, this, std::ref( b_ ) );
}

template<typename ModelA, typename ModelB>
HybridComputeKernel<ModelA, ModelB>::~HybridComputeKernel()
{
  running_ = false;
  for ( auto& t : threads_ ) {
    t.join();
  }
}

template<typename ModelA, typename ModelB>
void HybridComputeKernel<ModelA, ModelB>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  auto push_to_incoming = [this]( StateType&& s ) {
    s.set_id( current_local_state_id_++ );

    DLOG( INFO ) << "Pushing state to incoming queue: " << s.debug_string( false );

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
      push_to_incoming( std::move( new_state ) );
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
    push_to_incoming( std::move( state ) );
  } else if ( state.free_slots() < state.batch_size() ) {
    // some slots are filled; keep it as an incomplete state
    incomplete_state_ = std::move( state );
  } else {
    // all slots are empty; discard it
  }
}

template<typename ModelA, typename ModelB>
bool HybridComputeKernel<ModelA, ModelB>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename ModelA, typename ModelB>
void HybridComputeKernel<ModelA, ModelB>::release_discarded_contexts( const StateType& state )
{
  for ( size_t i = 0; i < state.discarded_contexts(); i++ ) {
    auto& prompt_id = state.discarded_prompt_id( i );
    a_.context_manager.release_context( prompt_id );
    b_.context_manager.release_context( prompt_id );
  }
}

template<typename ModelA, typename ModelB>
template<typename M>
void HybridComputeKernel<ModelA, ModelB>::execution_thread_func(
  typename HybridComputeKernel<ModelA, ModelB>::template ModelData<M>& model_data )
{
  while ( running_ ) {
    StateType input_state {};
    StateType output_state {};

    DLOG( WARNING ) << "Current status: "
                    << "incoming_size=" << incoming_.queue.size() << ", "
                    << "waiting_size=" << waiting_.queue.size() << ", "
                    << "a_processing_size=" << a_.processing.size() << ", "
                    << "b_processing_size=" << b_.processing.size();

    // get the next state to process
    {
      std::unique_lock lock { model_data.mutex };
      model_data.cv.wait( lock, [&model_data] { return !model_data.processing.empty(); } );
      input_state = std::move( model_data.processing.top().state );
      model_data.processing.pop();
    }

    DLOG( INFO ) << "Popped state from processing: " << input_state.debug_string( false ) << " (by "
                 << ( std::is_same_v<ModelA, M> ? "A" : "B" ) << ")";

    const auto local_id = input_state.id();
    const auto next_stage = input_state.next_stage();
    const auto next_layer = input_state.next_layer();
    const auto is_last_step
      = ( next_stage == Stage::Classification )
        or ( next_stage == Stage::PostAttention and next_layer == model_data.model->settings().end_layer_num
             and next_layer < ConfigType::n_layers - 1 );

    // run the corresponding forward function
    switch ( next_stage ) {
      case Stage::PreAttention:
        output_state = model_data.model->pre_attention_forward( std::move( input_state ) );
        break;

      case Stage::Attention: {
        std::unique_lock lock { context_mutex_ };

        if constexpr ( std::same_as<M, ModelA> ) {
          auto& contexts = context_map_[local_id].first;
          lock.unlock();
          output_state = model_data.model->attention_forward( std::move( input_state ), contexts );
        } else {
          auto& contexts = context_map_[local_id].second;
          lock.unlock();
          output_state = model_data.model->attention_forward( std::move( input_state ), contexts );
        }
      } break;

      case Stage::PostAttention:
        output_state = model_data.model->post_attention_forward( std::move( input_state ) );
        break;

      case Stage::Classification: output_state = model_data.model->classify_forward( std::move( input_state ) ); break;
    }

    std::optional<StateType> merged_state;
    {
      std::lock_guard lock { splitted_state_mutex_ };

      auto& [state_a, state_b] = splitted_state_map_[local_id];

      if constexpr ( std::same_as<M, ModelA> ) {
        state_a.emplace( std::move( output_state ) );
      } else {
        state_b.emplace( std::move( output_state ) );
      }

      if ( state_a.has_value() and state_b.has_value() ) {
        // merge back the states
        if ( state_b->empty() ) {
          merged_state.emplace( std::move( *state_a ) );
        } else if ( state_a->empty() ) {
          merged_state.emplace( std::move( *state_b ) );
        } else {
          state_a->merge( std::move( *state_b ) );
          merged_state.emplace( std::move( *state_a ) );
        }
        splitted_state_map_.erase( local_id );
      }
    }

    if ( merged_state.has_value() ) {
      // was this the last layer and stage served by this kernel?
      if ( is_last_step ) {
        // yes; we need to send it to the outgoing queue
        {
          // remove the contexts from the context map
          std::unique_lock lock { context_mutex_ };
          context_map_.erase( local_id );
        }

        DLOG( INFO ) << "Pushing state to outgoing queue: " << merged_state->debug_string( false );

        {
          std::lock_guard lock { outgoing_.mutex };
          outgoing_.queue.push( std::move( *merged_state ) );
        }

        event_fd_.write_event();
      } else {
        // no; we need to keep processing it
        DLOG( INFO ) << "Pushing state back to incoming queue: " << merged_state->debug_string( false );

        {
          std::lock_guard lock { incoming_.mutex };
          incoming_.queue.push( std::move( *merged_state ) );
        }

        incoming_.cv.notify_one();
      }
    }
  }
}

template<typename ModelA, typename ModelB>
void HybridComputeKernel<ModelA, ModelB>::bookkeeping_thread_func()
{
  while ( running_ ) {
    StateType state;

    {
      std::unique_lock lock { incoming_.mutex };
      incoming_.cv.wait( lock, [this] { return !incoming_.queue.empty(); } );

      state = std::move( incoming_.queue.top().state );
      incoming_.queue.pop();

      DLOG( INFO ) << "Popped state from incoming queue: " << state.debug_string( false );
    }

    // TODO(sadjad): check if this state is even valid for this kernel

    // TODO(sadjad): discard the finished prompts (state.discarded_contexts_)

    // can we, or have we already, allocated the contexts for this state?

    if ( context_map_.find( state.id() ) == context_map_.end() ) {
      if ( a_.context_manager.free() < a_.concurrency.get( Stage::Attention )
           or b_.context_manager.free() < b_.concurrency.get( Stage::Attention ) ) {
        // we don't have enough space for these contexts, this is going to the waiting queue
        DLOG( INFO ) << "Pushing state to waiting queue: " << state.debug_string( false );
        std::unique_lock lock { waiting_.mutex };
        waiting_.queue.push( std::move( state ) );
        continue;
      }

      std::vector<ContextPtrA> contexts_a;
      std::vector<ContextPtrB> contexts_b;

      contexts_a.reserve( a_.concurrency.get( Stage::Attention ) );
      contexts_b.reserve( b_.concurrency.get( Stage::Attention ) );

      for ( size_t i = 0; i < state.batch_size(); i++ ) {
        if ( i < a_.concurrency.get( Stage::Attention ) ) {
          auto ctx = a_.context_manager.get_context( state.prompt_id( i ) );
          if ( not ctx ) {
            LOG( FATAL ) << "Cannot allocate context.";
            break;
          }
          contexts_a.push_back( std::move( ctx ) );
        } else if ( i < a_.concurrency.get( Stage::Attention ) + b_.concurrency.get( Stage::Attention ) ) {
          auto ctx = b_.context_manager.get_context( state.prompt_id( i ) );
          if ( not ctx ) {
            LOG( FATAL ) << "Cannot allocate context.";
            break;
          }
          contexts_b.push_back( std::move( ctx ) );
        } else {
          LOG( FATAL ) << "Invalid state: " << state.debug_string( false );
        }
      }

      {
        std::lock_guard lock { context_mutex_ };
        context_map_[state.id()] = std::make_pair( std::move( contexts_a ), std::move( contexts_b ) );
      }
    }

    const auto next_stage = state.next_stage();
    const auto next_layer = state.next_layer();

    // do we need to split this state?
    if ( a_.concurrency.get( next_stage ) > 0 && b_.concurrency.get( next_stage ) > 0 ) {
      // TODO(sadjad): check the batch size against concurrency settings

      // XXX(sadjad): allow for incomplete states?
      // I don't think incomplete states should happen inside the kernel at all.

      // split the state
      auto [state_a, state_b] = state.split( a_.concurrency.get( next_stage ) );

      CHECK_EQ( state_a.batch_size(), a_.concurrency.get( next_stage ) );
      CHECK_EQ( state_b.batch_size(), b_.concurrency.get( next_stage ) );

      {
        std::lock_guard lock { a_.mutex };
        a_.processing.push( std::move( state_a ) );
      }
      a_.cv.notify_one();

      {
        std::lock_guard lock { b_.mutex };
        b_.processing.push( std::move( state_b ) );
      }
      b_.cv.notify_one();
    } else {
      if ( a_.concurrency.get( next_stage ) == state.batch_size() ) {
        DLOG( INFO ) << "Pushing state to A's processing queue: " << state.debug_string( false );

        {
          std::lock_guard lock { splitted_state_mutex_ };
          splitted_state_map_[state.id()].second.emplace();
        }

        {
          std::lock_guard lock { a_.mutex };
          a_.processing.push( std::move( state ) );
        }
        a_.cv.notify_one();
      } else if ( b_.concurrency.get( next_stage ) == state.batch_size() ) {
        DLOG( INFO ) << "Pushing state to B's processing queue: " << state.debug_string( false );

        {
          std::lock_guard lock { splitted_state_mutex_ };
          splitted_state_map_[state.id()].first.emplace();
        }

        {
          std::lock_guard lock { b_.mutex };
          b_.processing.push( std::move( state ) );
        }
        b_.cv.notify_one();
      } else {
        LOG( FATAL ) << "State batch size and concurrency settings do not match";
      }
    }
  }
}

template<typename ModelA, typename ModelB>
void HybridComputeKernel<ModelA, ModelB>::backlog_thread_func()
{
}

} // namespace glinthawk::compute
