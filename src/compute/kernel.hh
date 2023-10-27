#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "models/common/model.hh"
#include "prompt/prompt.hh"
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
    contexts_.emplace( prompt_id, context );
    return context;
  }
};

template<typename Model>
class ComputeKernel
{
private:
  std::unique_ptr<Model> model_;
  ContextManager<Model> context_manager_;

  prompt::PromptManager prompt_manager_;
  prompt::CompletionManager completion_manager_;

  std::thread execution_thread_;
  std::thread bookkeeping_thread_;

  std::queue<glinthawk::models::InferenceState> incoming_ {}, outgoing_ {};
  std::queue<std::pair<glinthawk::models::InferenceState, std::shared_ptr<typename Model::ContextType>>> processing_ {};
  std::mutex incoming_mutex_ {}, processing_mutex_ {}, outgoing_mutex_ {};
  std::condition_variable incoming_cv_ {}, processing_cv_ {}, outgoing_cv_ {};

  EventFD event_fd_ {};

  void execution_thread_func();
  void bookkeeping_thread_func();

  std::atomic<bool> running_ { true };

public:
  ComputeKernel( std::unique_ptr<Model>&& model, std::shared_ptr<storage::BlobStore> blobstore )
    : model_( std::move( model ) )
    , context_manager_( model_->config() )
    , prompt_manager_( blobstore )
    , completion_manager_( blobstore )
    , execution_thread_( &ComputeKernel::execution_thread_func, this )
    , bookkeeping_thread_( &ComputeKernel::bookkeeping_thread_func, this )
  {
    // NOTE(sadjad) I don't like the idea of the compute kernel doing anything other than compute... but for now,
    // for convenience, we'll just deal the prompts and completions here.
  }

  void push( glinthawk::models::InferenceState&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      incoming_.push( std::move( state ) );
    }

    incoming_cv_.notify_one();
  }

  void pop( glinthawk::models::InferenceState& state )
  {
    std::lock_guard lock( outgoing_mutex_ );
    state = std::move( outgoing_.front() );
    outgoing_.pop();
  }

  EventFD& event_fd() { return event_fd_; }

  ~ComputeKernel()
  {
    running_ = false;
    execution_thread_.join();
    bookkeeping_thread_.join();
  }
};

} // namespace glinthawk::compute
