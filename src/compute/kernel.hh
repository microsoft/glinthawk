#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "models/common/model.hh"
#include "util/eventfd.hh"

namespace glinthawk::compute {

template<typename Model>
class ContextManager
{
private:
  std::unordered_map<glinthawk::PromptID, std::shared_ptr<typename Model::ContextType>> contexts_ {};
  const typename Model::ConfigType config_ {};

public:
  ContextManager( const typename Model::ConfigType& config )
    : config_( config )
  {
  }

  std::shared_ptr<typename Model::ContextType> get_context( const glinthawk::PromptID& prompt_id )
  {
    auto it = contexts_.find( prompt_id );
    if ( it != contexts_.end() ) {
      return it->second;
    }

    auto context = std::make_shared<typename Model::ContextType>( config_ );

    if ( context.get()->empty() ) {
      return nullptr;
    } else {
      contexts_.emplace( prompt_id, context );
      return context;
    }
  }

  bool release( const glinthawk::PromptID& prompt_id )
  {
    auto it = contexts_.find( prompt_id );
    if ( it != contexts_.end() ) {
      contexts_.erase( prompt_id );
      return true;
    }
    return false;
  }
};

template<typename Model>
class ComputeKernel
{
private:
  std::unique_ptr<Model> model_;
  ContextManager<Model> context_manager_;

  const uint64_t target_batch_;
  uint64_t released_;

  std::queue<glinthawk::models::InferenceState> incoming_ {}, waiting_ {}, outgoing_ {};
  std::queue<std::pair<glinthawk::models::InferenceState, std::shared_ptr<typename Model::ContextType>>> processing_ {};
  std::mutex incoming_mutex_ {}, waiting_mutex_ {}, ctx_mgr_mutex_ {}, processing_mutex_ {}, outgoing_mutex_ {};
  std::condition_variable incoming_cv_ {}, waiting_cv_ {}, processing_cv_ {}, outgoing_cv_ {};

  EventFD event_fd_ {};

  std::atomic<bool> running_ { true };

  void execution_thread_func();
  void bookkeeping_thread_func();
  void backlog_thread_func();

  std::thread execution_thread_;
  std::thread bookkeeping_thread_;
  std::thread backlog_thread_;

public:
  ComputeKernel( std::unique_ptr<Model>&& model, const uint64_t target_batch )
    : model_( std::move( model ) )
    , context_manager_( model_->config() )
    , target_batch_( target_batch )
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
      released = context_manager_.release( state.prompt_id() );
    }
    // if released, notify waiting prompts
    if ( released ) {
      {
        std::lock_guard lock( waiting_mutex_ );
        released_ += 1;
      }

      waiting_cv_.notify_one();
    }

    // propagate the release message to the next worker, and remove self from propagation list
    state.erase_from_workers( model_ -> config().start_layer_num );
    state.set_next_layer( model_ -> config().end_layer_num + 1 );

    {
      std::lock_guard lock( outgoing_mutex_ );
      outgoing_.emplace( std::move( state ) );
    }

    event_fd_.write_event();
  }

  void check_finished( glinthawk::models::InferenceState& state )
  {
    if ( state.next_layer() == 0 ) {
      if ( state.token() == 2 or state.token_pos() >= model_ -> config().seq_len ) { // EOS or out of length
        state.set_finished();
        // TODO: should we send the finished message back to the coordinator?
      }
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

} // namespace glinthawk::compute
