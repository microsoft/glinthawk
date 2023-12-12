#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
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

enum class DataType
{
  Float16,
  Float32
};

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
      DLOG (INFO) << "(size: " << contexts_.size() << ") Added context for " << prompt_id;
    }

    return context;
  }

  bool release( const glinthawk::PromptID& prompt_id ) {
    bool released = contexts_.erase( prompt_id ) > 0;
    if ( released ) {
      DLOG (INFO) << "(size: " << contexts_.size() << ") Released context for " << prompt_id;
    }
    return released;
  }
};

template<typename Model>
class ComputeKernel
{
private:
  std::unique_ptr<Model> model_;
  ContextManager<Model> context_manager_;

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
    , context_manager_( model_->settings() )
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
