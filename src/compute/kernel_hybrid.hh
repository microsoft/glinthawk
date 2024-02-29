#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "monitoring/measurement.hh"
#include "util/eventfd.hh"
#include "util/util.hh"

namespace glinthawk::compute {

namespace {

template<typename Model>
class PreallocatingContextManager
{
public:
  using StateType = glinthawk::models::BatchedInferenceState<typename Model::ConfigType>;
  using ContextPtr = std::shared_ptr<typename Model::ContextType>;

  PreallocatingContextManager( const typename Model::SettingsType& settings );

  ContextPtr get_context( const PromptID& prompt_id );

  /// @brief Returns the contexts for all the prompts in the given state. Returns an empty optional if context cannot
  /// be allocated for any of the prompts.
  std::optional<std::vector<ContextPtr>> get_contexts( const StateType& state );

  bool release_context( const PromptID& prompt_id );

  size_t free() const;
  size_t allocated() const;
  size_t empty() const;
  size_t total() const;
};

} // anonymous namespace

template<typename ModelA, typename ModelB>
requires std::same_as<typename ModelA::ConfigType, typename ModelB::ConfigType>
class HybridComputeKernel
{
public:
  using ConfigType = typename ModelA::ConfigType;
  using ContextPtrA = std::shared_ptr<typename ModelA::ContextType>;
  using ContextPtrB = std::shared_ptr<typename ModelB::ContextType>;

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
  HybridComputeKernel( std::unique_ptr<ModelA>&& model_a,
                       std::unique_ptr<ModelB>&& model_b,
                       const Concurrency& concurrency_a,
                       const Concurrency& concurrency_b );

  ~HybridComputeKernel();

  void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state );
  bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state );

  EventFD& event_fd() { return event_fd_; }

private:
  using Stage = glinthawk::models::InferenceStage;
  using StateType = glinthawk::models::BatchedInferenceState<ConfigType>;

  struct StateCompOp
  {
    bool operator()( const StateType& lhs, const StateType& rhs ) const
    {
      return util::to_underlying( lhs.next_stage() ) == util::to_underlying( rhs.next_stage() )
               ? ( lhs.next_layer() < rhs.next_layer() )
               : ( util::to_underlying( lhs.next_stage() ) < util::to_underlying( rhs.next_stage() ) );
    }
  };

  using StatePriorityQueue = std::priority_queue<StateType, std::deque<StateType>, StateCompOp>;

  template<typename M>
  requires std::same_as<M, ModelA> || std::same_as<M, ModelB>
  struct ModelData
  {
    ModelData( std::unique_ptr<M>&& in_model, const Concurrency& in_concurrency );

    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;
    const Concurrency concurrency;
    const typename M::SettingsType settings;

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
  , settings( model->settings() )
{
}

template<typename ModelA, typename ModelB>
HybridComputeKernel<ModelA, ModelB>::HybridComputeKernel( std::unique_ptr<ModelA>&& model_a,
                                                          std::unique_ptr<ModelB>&& model_b,
                                                          const Concurrency& concurrency_a,
                                                          const Concurrency& concurrency_b )
  : a_( std::move( model_a ), concurrency_a )
  , b_( std::move( model_b ), concurrency_b )
{
  // check the concurrency settings to be permissible
  CHECK_EQ( a_.concurrency.get( Stage::PreAttention ) + b_.concurrency.get( Stage::PreAttention ),
            a_.concurrency.get( Stage::Attention ) + b_.concurrency.get( Stage::Attention ) );

  CHECK_EQ( a_.concurrency.get( Stage::Attention ) + b_.concurrency.get( Stage::Attention ),
            a_.concurrency.get( Stage::PostAttention ) + b_.concurrency.get( Stage::PostAttention ) );

  CHECK_EQ( a_.concurrency.get( Stage::PostAttention ) + b_.concurrency.get( Stage::PostAttention ),
            a_.concurrency.get( Stage::Classification ) + b_.concurrency.get( Stage::Classification ) );

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
  auto push_to_incoming = [this]( StateType&& state ) {
    state.set_id( current_local_state_id_++ );

    {
      std::lock_guard lock { incoming_.mutex };
      incoming_.queue.push( std::move( state ) );
    }

    incoming_.cv.notify_one();
  };

  // (1) discard the contexts we have to discard
  if ( state.discarded_contexts() ) {
    release_discarded_contexts( state );
  }

  // (2) is this the last layer? if so, we can get rid of the discard list.
  if ( a_.settings().end_layer_num == ConfigType::n_layers - 1 ) {
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
    push_to_incoming( std::move( state ) );
  } else if ( state.free_slots() < state.batch_size() ) {
    incomplete_state_ = std::move( state );
  }
}

template<typename ModelA, typename ModelB>
bool HybridComputeKernel<ModelA, ModelB>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top() );
  outgoing_.pop();
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
    std::reference_wrapper<std::vector<std::shared_ptr<typename M::ContextType>>> contexts;

    {
      std::unique_lock lock { model_data.mutex };
      model_data.cv.wait( lock, [&model_data] { return !model_data.processing.empty(); } );
      input_state = std::move( model_data.processing.top() );
      model_data.processing.pop();
    }

    const auto next_stage = input_state.next_stage();
    const auto next_layer = input_state.next_layer();
    const auto is_last_step
      = ( next_stage == Stage::Classification )
        or ( next_stage == Stage::PostAttention and next_layer == model_data.settings().end_layer_num );

    // we only need the contexts for the attention stage
    if ( next_stage == Stage::Attention ) {
      std::lock_guard lock { context_mutex_ };
      auto& context_pair = context_map_[input_state.id()];
      if constexpr ( std::same_as<M, ModelA> ) {
        contexts = std::ref( context_pair.first );
      } else {
        contexts = std::ref( context_pair.second );
      }
    }

    switch ( next_stage ) {
      case Stage::PreAttention:
        output_state = model_data.model->pre_attention_forward( std::move( input_state ) );
        break;

      case Stage::Attention:
        output_state = model_data.model->attention_forward( std::move( input_state ), contexts.get() );
        break;

      case Stage::PostAttention:
        output_state = model_data.model->post_attention_forward( std::move( input_state ) );
        break;

      case Stage::Classification: output_state = model_data.model->classify_forward( std::move( input_state ) ); break;
    }

    std::optional<StateType> merged_state;
    {
      std::lock_guard lock { splitted_state_mutex_ };
      auto& state_pair = splitted_state_map_[output_state.id()];

      if constexpr ( std::same_as<M, ModelA> ) {
        state_pair.first.emplace( std::move( output_state ) );
      } else {
        state_pair.second.emplace( std::move( output_state ) );
      }

      if ( state_pair.first.has_value() and state_pair.second.has_value() ) {
        // merge the states
        merged_state.emplace( state_pair.first->merge( std::move( *state_pair.second ) ) );
        state_pair.erase( output_state.id() );
      }
    }

    if ( merged_state.has_value() ) {
      // was this the last layer and stage served by this kernel?
      if ( is_last_step ) {
        // yes; we need to send it to the outgoing queue
        {
          std::lock_guard lock { outgoing_.mutex };
          outgoing_.queue.push( std::move( *merged_state ) );
        }

        outgoing_.cv.notify_one();
      } else {
        // no; we need to keep processing it
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
      const auto queue_top = incoming_.queue.top();

      state = std::move( incoming_.queue.top() );
      incoming_.queue.pop();
    }

    // TODO(sadjad): check if this state is even valid for this kernel

    // TODO(sadjad): discard the finished prompts (state.discarded_contexts_)

    // can we, or have we already, allocated the contexts for this state?

    if ( context_map_.find( state.id() ) ) {
      if ( a_.context_manager.free() < a_.concurrency.get( Stage::Attention )
           or b_.context_manager.free() < b_.concurrency.get( Stage::Attention ) ) {
        // we don't have enough space for these contexts, this is going to the waiting queue
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
        } else if ( i < b_.concurrency.get( Stage::Attention ) ) {
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

      {
        std::lock_guard lock { b_.mutex };
        b_.processing.push( std::move( state_b ) );
      }

      a_.cv.notify_one();
      b_.cv.notify_one();
    } else {
      if ( a_.concurrency.get( next_stage ) == state.batch_size() ) {
        std::lock_guard lock { a_.mutex };
        a_.processing.push( std::move( state ) );
        a_.cv.notify_one();
      } else if ( b_.concurrency.get( next_stage ) == state.batch_size() ) {
        std::lock_guard lock { b_.mutex };
        b_.processing.push( std::move( state ) );
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
