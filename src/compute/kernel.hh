#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "models/common/state.hh"
#include "models/types.hh"
#include "monitoring/measurement.hh"
#include "prompt/prompt.hh"
#include "util/eventfd.hh"

#include "common.hh"
#include "contextman.hh"

namespace glinthawk::compute {

template<typename Model>
class BatchedComputeKernel
{
public:
  using ModelConfig = typename Model::ConfigType;
  using BatchedState = glinthawk::models::BatchedInferenceState<ModelConfig>;
  using ContextPtr = std::shared_ptr<typename Model::ContextType>;

  static constexpr KernelType Type = KernelType::Batched;

private:
  std::unique_ptr<Model> model_;
  std::unique_ptr<ContextManager<Model>> context_manager_;

  const bool is_serving_first_layer_ { model_->settings().start_layer_num == 0 };
  const bool is_serving_last_layer_ { model_->settings().end_layer_num == ModelConfig::n_layers - 1 };

  const uint64_t target_conc_size_;
  uint64_t released_ { 0 };

  // a state with some empty slots
  std::optional<BatchedState> incomplete_state_ {};

  std::queue<BatchedState> incoming_ {}, waiting_ {}, outgoing_ {};
  std::queue<std::pair<BatchedState, std::vector<ContextPtr>>> processing_ {};

  std::mutex incoming_mutex_ {}, waiting_mutex_ {}, ctx_mgr_mutex_ {}, processing_mutex_ {}, outgoing_mutex_ {};
  std::condition_variable_any incoming_cv_ {}, waiting_cv_ {}, processing_cv_ {};

  EventFD event_fd_ {};
  Measurement& __stats__ { global_measurement() };

  // <threads>
  void execution_thread_func( std::stop_token stoken );
  void bookkeeping_thread_func( std::stop_token stoken );
  void backlog_thread_func( std::stop_token stoken );

  std::jthread execution_thread_;
  std::jthread bookkeeping_thread_;
  std::jthread backlog_thread_;
  // </threads>

  void push_to_incoming( BatchedState&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      incoming_.push( std::move( state ) );
    }

    incoming_cv_.notify_one();
  }

  void release_context( const PromptID& prompt_id )
  {
    // Release the context
    bool released = false;
    {
      std::lock_guard lock( ctx_mgr_mutex_ );
      released = context_manager_->release( prompt_id );
    }

    // if released, notify waiting prompts
    if ( released ) {
      {
        std::lock_guard lock( waiting_mutex_ );
        released_ += 1;
      }

      waiting_cv_.notify_one();
    }
  }

  std::pair<bool, std::vector<ContextPtr>> assemble_contexts( const BatchedState& state )
  {
    bool all_contexts_assigned = true;

    std::vector<ContextPtr> contexts;
    for ( size_t i = 0; i < state.batch_size(); i++ ) {
      if ( state.active( i ) ) {
        auto context = context_manager_->get_context( state.prompt_id( i ) );
        all_contexts_assigned = all_contexts_assigned && !context->empty();
        contexts.push_back( std::move( context ) );
      } else {
        contexts.push_back( nullptr );
      }
    }

    return { all_contexts_assigned, std::move( contexts ) };
  }

public:
  template<typename... Args>
  BatchedComputeKernel( const uint64_t target_conc_size, Args&&... args )
    : model_( std::make_unique<Model>( std::forward<Args>( args )... ) )
    , context_manager_( std::make_unique<ContextManager<Model>>( model_->settings() ) )
    , target_conc_size_( target_conc_size )
    , execution_thread_( std::bind( &BatchedComputeKernel::execution_thread_func, this, std::placeholders::_1 ) )
    , bookkeeping_thread_( std::bind( &BatchedComputeKernel::bookkeeping_thread_func, this, std::placeholders::_1 ) )
    , backlog_thread_( std::bind( &BatchedComputeKernel::backlog_thread_func, this, std::placeholders::_1 ) )
  {
  }

  void push( BatchedState&& state )
  {
    // (1) discard the contexts we have to discard
    for ( size_t i = 0; i < state.discarded_contexts(); i++ ) {
      auto& prompt_id = state.discarded_prompt_id( i );
      release_context( prompt_id );
    }

    // (2) is this the last layer? if so, we can get rid of the discard list.
    if ( is_serving_last_layer_ ) {
      state.clear_discards();
    }

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

  bool pop( BatchedState& state )
  {
    std::lock_guard lock( outgoing_mutex_ );
    if ( outgoing_.empty() )
      return false;
    state = std::move( outgoing_.front() );
    outgoing_.pop();
    return true;
  }

  EventFD& event_fd() { return event_fd_; }

  ~BatchedComputeKernel() {}
};

template<typename Model>
void BatchedComputeKernel<Model>::execution_thread_func( std::stop_token stoken )
{
  LOG( INFO ) << "BatchedComputeKernel execution thread started.";

  std::pair<BatchedState, std::vector<ContextPtr>> action;

  BatchedState state;
  std::vector<ContextPtr> contexts;

  while ( not stoken.stop_requested() ) {
    {
      std::unique_lock lock( processing_mutex_ );
      if ( not processing_cv_.wait( lock, stoken, [this] { return not processing_.empty(); } ) ) {
        continue; // we were woken up by the stop token
      }

      state = std::move( processing_.front().first );
      contexts = std::move( processing_.front().second );
      processing_.pop();
    }

    const auto start = std::chrono::steady_clock::now();
    model_->forward( state, contexts );
    const auto duration
      = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start );
    __stats__.add_point<IntDistributions::KernelForwardTime>( duration.count() );

    {
      std::lock_guard lock( outgoing_mutex_ );
      outgoing_.emplace( std::move( state ) );
    }

    event_fd_.write_event();
  }

  LOG( INFO ) << "BatchedComputeKernel execution thread exiting.";
}

template<typename Model>
void BatchedComputeKernel<Model>::bookkeeping_thread_func( std::stop_token stoken )
{
  LOG( INFO ) << "BatchedComputeKernel bookkeeping thread started.";

  BatchedState state;
  std::vector<ContextPtr> contexts;
  bool all_contexts_assigned = false;

  while ( not stoken.stop_requested() ) {
    // let's get an action from the incoming_
    {
      std::unique_lock lock( incoming_mutex_ );
      if ( not incoming_cv_.wait( lock, stoken, [this] { return not incoming_.empty(); } ) ) {
        continue;
      }

      state = std::move( incoming_.front() );
      incoming_.pop();
    }

    {
      // let's get the contexts for this state
      std::lock_guard lock( ctx_mgr_mutex_ );
      std::tie( all_contexts_assigned, contexts ) = assemble_contexts( state );
    }

    if ( all_contexts_assigned ) {
      {
        std::lock_guard lock( processing_mutex_ );
        processing_.emplace( std::move( state ), std::move( contexts ) );
      }

      processing_cv_.notify_one();
    } else {
      // we couldn't get all the contexts, let's push it to the waiting_
      {
        std::lock_guard lock( waiting_mutex_ );
        waiting_.emplace( std::move( state ) );

        while ( not incoming_.empty() ) {
          waiting_.emplace( std::move( incoming_.front() ) );
          incoming_.pop();
        }
      }

      waiting_cv_.notify_one();
    }
  }

  LOG( INFO ) << "BatchedComputeKernel bookkeeping thread exiting.";
}

template<typename Model>
void BatchedComputeKernel<Model>::backlog_thread_func( std::stop_token stoken )
{
  LOG( INFO ) << "BatchedComputeKernel backlog thread started.";

  BatchedState state;
  std::vector<ContextPtr> contexts;
  bool all_contexts_assigned = false;

  while ( not stoken.stop_requested() ) {
    // let's get an action from the waiting_
    {
      std::unique_lock lock { waiting_mutex_ };
      if ( not waiting_cv_.wait(
             lock, stoken, [this] { return not waiting_.empty() and released_ >= waiting_.front().batch_size(); } ) ) {
        continue;
      }

      state = std::move( waiting_.front() );
      waiting_.pop();

      released_ -= state.batch_size();
    }

    {
      // let's get the context for this state
      std::lock_guard lock { ctx_mgr_mutex_ };
      std::tie( all_contexts_assigned, contexts ) = assemble_contexts( state );
    }

    if ( all_contexts_assigned ) {
      {
        std::lock_guard lock { processing_mutex_ };
        processing_.emplace( std::move( state ), std::move( contexts ) );
      }

      processing_cv_.notify_one();
    } else {
      {
        std::lock_guard lock { waiting_mutex_ };
        waiting_.emplace( std::move( state ) );
      }
    }
  }

  LOG( INFO ) << "BatchedComputeKernel backlog thread exiting.";
}

} // namespace glinthawk::compute
