#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "kernel.hh"
#include "models/common/model.hh"
#include "monitoring/measurement.hh"
#include "prompt/prompt.hh"
#include "util/eventfd.hh"

namespace glinthawk::compute {

template<typename Model>
class ComputeKernelPiped
{
private:
  std::unique_ptr<Model> model_;
  ContextManager<Model> context_manager_;

  const uint64_t target_conc_pre_size_;
  const uint64_t target_conc_att_size_;
  const uint64_t target_conc_post_size_;

  const bool process_pre_;
  const bool process_att_;
  const bool process_post_;

  const uint64_t start_layer_;
  const uint64_t end_layer_;
  const uint64_t n_layers_;

  uint64_t released_;

  std::vector<std::queue<std::pair<glinthawk::models::InferenceState, std::shared_ptr<typename Model::ContextType>>>>
    processing_pre_attention_;
  std::vector<std::queue<glinthawk::models::InferenceState>> processing_post_attention_;

  std::queue<std::pair<glinthawk::models::InferenceState, std::shared_ptr<typename Model::ContextType>>>
    processing_attention_ {};
  std::queue<glinthawk::models::InferenceState> incoming_ {}, waiting_attention_ {}, outgoing_ {};

  std::mutex ctx_mgr_mutex_ {}, outgoing_mutex_ {}, incoming_mutex_ {}, waiting_attention_mutex_ {},
    processing_mutex_ {};

  std::condition_variable incoming_cv_ {}, processing_cv_ {}, waiting_attention_cv_ {};

  EventFD event_fd_ {};

  std::atomic<bool> running_ { true };

  Measurement& __stats__ { global_measurement() };

  void execution_thread_func();
  void bookkeeping_thread_func();
  void backlog_thread_func();

  std::thread execution_thread_;
  std::thread bookkeeping_thread_;
  std::thread backlog_thread_;

public:
  ComputeKernelPiped( std::unique_ptr<Model>&& model,
                      const uint64_t target_conc_pre_size,
                      const uint64_t target_conc_att_size,
                      const uint64_t target_conc_post_size,
                      const uint64_t start_layer,
                      const uint64_t end_layer )
    : model_( std::move( model ) )
    , context_manager_( model_->settings() )
    , target_conc_pre_size_( target_conc_pre_size )
    , target_conc_att_size_( target_conc_att_size )
    , target_conc_post_size_( target_conc_post_size )
    , process_pre_( target_conc_pre_size > 0 )
    , process_att_( target_conc_att_size > 0 )
    , process_post_( target_conc_post_size > 0 )
    , start_layer_( start_layer )
    , end_layer_( end_layer )
    , n_layers_( end_layer_ - start_layer_ + 1 )
    , released_( 0 )
    , processing_pre_attention_( n_layers_ )
    , processing_post_attention_( n_layers_ )
    , running_( true )
    , execution_thread_( &ComputeKernelPiped::execution_thread_func, this )
    , bookkeeping_thread_( &ComputeKernelPiped::bookkeeping_thread_func, this )
    , backlog_thread_( &ComputeKernelPiped::backlog_thread_func, this )
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
        std::lock_guard lock( waiting_attention_mutex_ );
        released_ += 1;
      }

      waiting_attention_cv_.notify_one();
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
//    LOG (INFO) << "got this in check_finished: " << state;
    if ( model_->is_finished( state ) ) {
      state.set_finished();
    }
  }

  EventFD& event_fd() { return event_fd_; }

  ~ComputeKernelPiped()
  {
    running_ = false;
    execution_thread_.join();
    bookkeeping_thread_.join();
    backlog_thread_.join();
  }
};

} // namespace glinthawk::compute
