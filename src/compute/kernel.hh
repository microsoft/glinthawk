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

#include "models/common/model.hh"
#include "monitoring/measurement.hh"
#include "prompt/prompt.hh"
#include "util/eventfd.hh"

namespace glinthawk::compute {

enum class Platform
{
  CPU, // TODO rename to AMD64
  CUDA
};

using DataType = glinthawk::DataType;

template<typename Model>
class ContextManager
{
private:
  std::unordered_map<glinthawk::PromptID, std::shared_ptr<typename Model::ContextType>> contexts_ {};
  const typename Model::SettingsType settings_ {};

public:
  ContextManager( const typename Model::SettingsType& settings )
    : settings_( settings )
  {
  }

  std::shared_ptr<typename Model::ContextType> get_context( const glinthawk::PromptID& prompt_id,
                                                            bool emplace_empty = false )
  {
    auto it = contexts_.find( prompt_id );
    if ( it != contexts_.end() ) {
      return it->second;
    }

    auto context = std::make_shared<typename Model::ContextType>( settings_ );

    if ( not context.get()->empty() or emplace_empty ) {
      contexts_.emplace( prompt_id, context );
    }

    return context;
  }

  bool release( const glinthawk::PromptID& prompt_id ) { return contexts_.erase( prompt_id ) > 0; }
};

template<typename Model>
class ComputeKernel
{
private:
  std::unique_ptr<Model> model_;
  std::unique_ptr<ContextManager<Model>> context_manager_;

  const uint64_t target_conc_size_;
  uint64_t released_;

  std::queue<glinthawk::models::InferenceState> incoming_ {}, waiting_ {}, outgoing_ {};
  std::queue<std::pair<glinthawk::models::InferenceState, std::shared_ptr<typename Model::ContextType>>> processing_ {};
  std::mutex incoming_mutex_ {}, waiting_mutex_ {}, ctx_mgr_mutex_ {}, processing_mutex_ {}, outgoing_mutex_ {};
  std::condition_variable incoming_cv_ {}, waiting_cv_ {}, processing_cv_ {};

  EventFD event_fd_ {};

  std::atomic<bool> running_ { true };

  void execution_thread_func();
  void bookkeeping_thread_func();
  void backlog_thread_func();

  std::thread execution_thread_;
  std::thread bookkeeping_thread_;
  std::thread backlog_thread_;

  Measurement& __stats__ { global_measurement() };

public:
  ComputeKernel( std::unique_ptr<Model>&& model, const uint64_t target_conc_size )
    : model_( std::move( model ) )
    , context_manager_( std::make_unique<ContextManager<Model>>( model_->settings() ) )
    , target_conc_size_( target_conc_size )
    , released_( 0 )
    , running_( true )
    , execution_thread_( &ComputeKernel::execution_thread_func, this )
    , bookkeeping_thread_( &ComputeKernel::bookkeeping_thread_func, this )
    , backlog_thread_( &ComputeKernel::backlog_thread_func, this )
  {
  }

  void push( glinthawk::models::InferenceState&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      incoming_.push( std::move( state ) );
    }

    incoming_cv_.notify_one();
  }

  void push( std::vector<glinthawk::models::InferenceState>&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      for ( auto& s : state ) {
        incoming_.push( std::move( s ) );
      }
    }

    incoming_cv_.notify_one();
  }

  bool pop( glinthawk::models::InferenceState& state )
  {
    std::lock_guard lock( outgoing_mutex_ );
    if ( outgoing_.empty() )
      return false;
    state = std::move( outgoing_.front() );
    outgoing_.pop();
    return true;
  }

  void push_finished( glinthawk::models::InferenceState&& state )
  {
    // Release the context
    bool released;
    {
      std::lock_guard lock( ctx_mgr_mutex_ );
      released = context_manager_->release( state.prompt_id() );
    }

    // if released, notify waiting prompts
    if ( released ) {
      {
        std::lock_guard lock( waiting_mutex_ );
        released_ += 1;
      }

      waiting_cv_.notify_one();
    }

    // do a "fake" forward: remove self from propagation list and set next worker
    model_->dummy_forward( state );

    if ( state.layer_workers().empty() ) {
      // drop release message as it has fully propagated
      DLOG( INFO ) << "Dropping empty (release) inference state: " << state.to_string();
    } else {
      // propagate the release message to the next worker
      {
        std::lock_guard lock( outgoing_mutex_ );
        outgoing_.emplace( std::move( state ) );
      }

      event_fd_.write_event();
    }
  }

  void check_finished( glinthawk::models::InferenceState& state )
  {
    if ( model_->is_finished( state ) ) {
      state.set_finished();
      // TODO: should we send the finished message back to the coordinator?
    }
  }

  EventFD& event_fd() { return event_fd_; }

  ~ComputeKernel()
  {
    running_ = false;
    execution_thread_.join();
    bookkeeping_thread_.join();
    backlog_thread_.join();
  }
};

template<typename Model>
void ComputeKernel<Model>::execution_thread_func()
{
  LOG( INFO ) << "ComputeKernel execution thread started.";

  std::pair<glinthawk::models::InferenceState, std::shared_ptr<typename Model::ContextType>> action;
  std::vector<glinthawk::models::InferenceState> input_states;
  std::vector<std::shared_ptr<typename Model::ContextType>> contexts;

  while ( running_ ) {
    // TODO: possible move bug shenanigans
    input_states.clear();
    contexts.clear();

    {
      std::unique_lock<std::mutex> lock( processing_mutex_ );
      processing_cv_.wait( lock, [this] { return !( processing_.size() < target_conc_size_ ); } );

      for ( size_t j = 0; j < target_conc_size_; j++ ) {
        action = move( processing_.front() );
        processing_.pop();
        // TODO: possible move bug shenanigans
        input_states.push_back( std::move( action.first ) );
        contexts.push_back( action.second );
      }
    }

    const auto start = std::chrono::steady_clock::now();
    auto results = model_->forward( move( input_states ), contexts );
    const auto duration
      = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start );
    __stats__.add_point<IntDistributions::KernelForwardTime>( duration.count() );

    {
      std::lock_guard lock( outgoing_mutex_ );
      for ( auto& state : results ) {
        // TODO: possible move bug shenanigans
        outgoing_.emplace( std::move( state ) );
      }
    }

    event_fd_.write_event();
  }
}

template<typename Model>
void ComputeKernel<Model>::bookkeeping_thread_func()
{
  LOG( INFO ) << "ComputeKernel bookkeeping thread started.";

  glinthawk::models::InferenceState action;
  std::shared_ptr<typename Model::ContextType> context;

  while ( running_ ) {
    // let's get an action from the incoming_
    {
      std::unique_lock<std::mutex> lock( incoming_mutex_ );
      incoming_cv_.wait( lock, [this] { return !incoming_.empty(); } );
      action = std::move( incoming_.front() );
      incoming_.pop();
    }
    {
      // let's get the context for this action
      std::lock_guard lock( ctx_mgr_mutex_ );
      context = context_manager_->get_context( action.prompt_id() );

      //    if ( not context ) {
      //      LOG( ERROR ) << "Could not get context for prompt_id=" << action.prompt_id().to_string();
      //    }
    }

    if ( context ) {
      {
        std::lock_guard lock( processing_mutex_ );
        processing_.emplace( std::move( action ), context );
      }

      processing_cv_.notify_one();
    } else {
      {
        std::lock_guard lock( waiting_mutex_ );
        waiting_.emplace( std::move( action ) );
      }

      waiting_cv_.notify_one();
    }
  }
}

template<typename Model>
void ComputeKernel<Model>::backlog_thread_func()
{
  LOG( INFO ) << "ComputeKernel backlog thread started.";

  glinthawk::models::InferenceState action;
  std::shared_ptr<typename Model::ContextType> context;

  while ( running_ ) {
    // let's get an action from the incoming_
    {
      std::unique_lock<std::mutex> lock( waiting_mutex_ );
      while ( not( released_ > 0 && !waiting_.empty() ) )
        waiting_cv_.wait( lock );
      action = std::move( waiting_.front() );
      waiting_.pop();
      released_ -= 1;
    }

    {
      // let's get the context for this action
      std::lock_guard lock( ctx_mgr_mutex_ );
      context = context_manager_->get_context( action.prompt_id() );

      //    if ( not context ) {
      //      LOG( ERROR ) << "Could not get context for prompt_id=" << action.prompt_id().to_string();
      //    }
    }

    if ( context ) {
      {
        std::lock_guard lock( processing_mutex_ );
        processing_.emplace( std::move( action ), context );
      }

      processing_cv_.notify_one();
    } else {
      {
        std::lock_guard lock( waiting_mutex_ );
        waiting_.emplace( std::move( action ) );
      }
    }
  }
}

} // namespace glinthawk::compute
