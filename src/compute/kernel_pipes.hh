#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
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
class ContextManagerPreAllocated
{
private:
  std::deque<std::shared_ptr<typename Model::ContextType>> free_contexts_ {};
  std::unordered_map<glinthawk::PromptID, std::shared_ptr<typename Model::ContextType>> assigned_contexts_ {};
  const typename Model::SettingsType settings_ {};
  size_t empty_contexts { 0 };
  size_t total_contexts { 0 };

public:
  ContextManagerPreAllocated( const typename Model::SettingsType& settings )
    : settings_( settings )
  {
    uint64_t i;
    for ( i = 0; i < settings_.max_context_count; i++ ) {
      auto context = std::make_shared<typename Model::ContextType>( settings_ );

      if ( not context.get()->empty() ) {
        free_contexts_.push_back( context );
      } else {
        break;
      }
    }
    total_contexts = free_contexts_.size();

    LOG( INFO ) << "Allocated " << total() << " contexts";
  }

  size_t free() const { return free_contexts_.size(); }

  size_t allocated() const { return assigned_contexts_.size() - empty_contexts; }

  size_t empty() const { return empty_contexts; }

  size_t total() const { return total_contexts; }

  std::shared_ptr<typename Model::ContextType> get_context( const glinthawk::PromptID& prompt_id,
                                                            const bool emplace_empty = false )
  {
    auto it = assigned_contexts_.find( prompt_id );
    if ( it != assigned_contexts_.end() ) {
      return it->second;
    }

    std::shared_ptr<typename Model::ContextType> context;

    if ( free_contexts_.empty() ) {
      context = std::make_shared<typename Model::ContextType>( settings_, true );

      if ( emplace_empty ) {
        assigned_contexts_.emplace( prompt_id, context );
        empty_contexts += 1;
      }
    } else {
      context = free_contexts_.front();
      free_contexts_.pop_front();
      assigned_contexts_.emplace( prompt_id, context );
      DLOG( INFO ) << "(size: " << allocated() << "/" << total() << ") Added context for " << prompt_id;
    }

    return context;
  }

  bool release( const glinthawk::PromptID& prompt_id )
  {
    auto pair_freed = assigned_contexts_.find( prompt_id );
    // If a context was assigned to this prompt, release it
    // and if it's non-empty, add it back to unallocated contexts
    if ( pair_freed != assigned_contexts_.end() ) {
      if ( not pair_freed->second->empty() ) {
        free_contexts_.push_back( pair_freed->second );
      } else if ( pair_freed->second->empty() ) {
        empty_contexts -= 1;
      }

      assigned_contexts_.erase( pair_freed );

      DLOG( INFO ) << "(size: " << allocated() << "/" << total() << ") Released context for " << prompt_id;
      return true;
    }
    return false;
  }
};

template<typename Model>
class ComputeKernelPiped
{
private:
  using ContextPtr = std::shared_ptr<typename Model::ContextType>;
  using StateContextPair = std::pair<models::InferenceState, ContextPtr>;

  std::unique_ptr<Model> model_;
  ContextManagerPreAllocated<Model> context_manager_;

  const uint64_t target_conc_pre_size_;
  const uint64_t target_conc_att_size_;
  const uint64_t target_conc_post_size_;
  const uint64_t target_conc_cls_size_;

  const bool process_pre_;
  const bool process_att_;
  const bool process_post_;
  const bool process_cls_;

  const uint64_t start_layer_;
  const uint64_t end_layer_;
  const uint64_t n_layers_;

  std::vector<std::queue<StateContextPair>> processing_pre_attention_;
  std::vector<std::queue<models::InferenceState>> processing_post_attention_;

  std::queue<StateContextPair> processing_attention_ {};
  std::queue<models::InferenceState> incoming_ {}, waiting_attention_ {}, outgoing_ {}, processing_classification_ {};

  std::mutex ctx_mgr_mutex_ {}, outgoing_mutex_ {}, incoming_mutex_ {}, waiting_attention_mutex_ {},
    processing_mutex_ {};

  std::condition_variable ctx_mgr_cv_ {}, incoming_cv_ {}, processing_cv_ {}, waiting_attention_cv_ {};

  EventFD event_fd_ {};

  std::atomic<bool> running_ { true };

  Measurement& __stats__ { global_measurement() };

  void execution_thread_func();
  void bookkeeping_thread_func();
  void backlog_thread_func();
  void qmeasure_thread_func();

  std::thread execution_thread_;
  std::thread bookkeeping_thread_;
  std::thread backlog_thread_;
  std::thread qmeasure_thread_;

public:
  ComputeKernelPiped( std::unique_ptr<Model>&& model,
                      const uint64_t target_conc_pre_size,
                      const uint64_t target_conc_att_size,
                      const uint64_t target_conc_post_size,
                      const uint64_t target_conc_cls_size,
                      const uint64_t start_layer,
                      const uint64_t end_layer )
    : model_( std::move( model ) )
    , context_manager_( model_->settings() )
    , target_conc_pre_size_( target_conc_pre_size )
    , target_conc_att_size_( target_conc_att_size )
    , target_conc_post_size_( target_conc_post_size )
    , target_conc_cls_size_( target_conc_cls_size )
    , process_pre_( target_conc_pre_size > 0 )
    , process_att_( target_conc_att_size > 0 )
    , process_post_( target_conc_post_size > 0 )
    , process_cls_( target_conc_cls_size > 0 )
    , start_layer_( start_layer )
    , end_layer_( end_layer )
    , n_layers_( end_layer_ - start_layer_ + 1 )
    , processing_pre_attention_( n_layers_ )
    , processing_post_attention_( n_layers_ )
    , running_( true )
    , execution_thread_( &ComputeKernelPiped::execution_thread_func, this )
    , bookkeeping_thread_( &ComputeKernelPiped::bookkeeping_thread_func, this )
    , backlog_thread_( &ComputeKernelPiped::backlog_thread_func, this )
    , qmeasure_thread_( &ComputeKernelPiped::qmeasure_thread_func, this )
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
    {
      std::lock_guard lock( ctx_mgr_mutex_ );
      context_manager_.release( state.prompt_id() );
      ctx_mgr_cv_.notify_one();
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

  ~ComputeKernelPiped()
  {
    running_ = false;
    execution_thread_.join();
    bookkeeping_thread_.join();
    backlog_thread_.join();
  }
};

template<typename Model>
void ComputeKernelPiped<Model>::execution_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped execution thread started.";

  while ( running_ ) {
    std::vector<models::InferenceState> input_states;
    std::vector<std::shared_ptr<typename Model::ContextType>> contexts;
    models::InferenceState::Stage next_stage;
    uint32_t next_layer_idx;
    {
      // hold lock until one queue has enough data for one batch
      std::unique_lock<std::mutex> lock( processing_mutex_ );
      // TODO: With splitting KV cache to in-context and out-context, will large batch sizes cause deadlocks?
      // TODO: Is there any reason we shouldn't always use the best batch size for any pipe?
      processing_cv_.wait( lock, [this, &next_stage, &next_layer_idx] {
        if ( processing_classification_.size() >= target_conc_cls_size_ and process_cls_ ) {
          next_stage = models::InferenceState::Stage::Classification;
          next_layer_idx = static_cast<uint32_t>( -1 );
          return true;
        }

        for ( int layer_idx = static_cast<int>( n_layers_ - 1 ); layer_idx >= 0; layer_idx-- ) {
          if ( processing_post_attention_[layer_idx].size() >= target_conc_post_size_ and process_post_ ) {
            next_stage = models::InferenceState::Stage::PostAttention;
            next_layer_idx = static_cast<uint32_t>( layer_idx );
            return true;
          }
        }

        if ( processing_attention_.size() >= target_conc_att_size_ and process_att_ ) {
          next_stage = models::InferenceState::Stage::Attention;
          next_layer_idx = static_cast<uint32_t>( -1 );
          return true;
        }

        for ( int layer_idx = static_cast<int>( n_layers_ - 1 ); layer_idx >= 0; layer_idx-- ) {
          if ( processing_pre_attention_[layer_idx].size() >= target_conc_pre_size_ and process_pre_ ) {
            next_stage = models::InferenceState::Stage::PreAttention;
            next_layer_idx = static_cast<uint32_t>( layer_idx );
            return true;
          }
        }
        return false;
      } );

      // find the queue and pop the data to input_states and possibly contexts
      switch ( next_stage ) {
        case models::InferenceState::Stage::PreAttention: {
          for ( size_t j = 0; j < target_conc_pre_size_; j++ ) {
            StateContextPair action = std::move( processing_pre_attention_[next_layer_idx].front() );
            //            LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action.first;
            processing_pre_attention_[next_layer_idx].pop();
            input_states.push_back( std::move( action.first ) );
            contexts.push_back( action.second );
          }
          break;
        }
        case models::InferenceState::Stage::Attention: {
          for ( size_t j = 0; j < target_conc_att_size_; j++ ) {
            StateContextPair action = std::move( processing_attention_.front() );
            //            LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action.first;
            processing_attention_.pop();
            input_states.push_back( std::move( action.first ) );
            contexts.push_back( action.second );
          }
          break;
        }
        case models::InferenceState::Stage::PostAttention: {
          for ( size_t j = 0; j < target_conc_post_size_; j++ ) {
            models::InferenceState action = std::move( processing_post_attention_[next_layer_idx].front() );
            //            LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action;
            processing_post_attention_[next_layer_idx].pop();
            input_states.push_back( std::move( action ) );
          }
          break;
        }
        case models::InferenceState::Stage::Classification: {
          for ( size_t j = 0; j < target_conc_cls_size_; j++ ) {
            models::InferenceState action = std::move( processing_classification_.front() );
            //            LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action;
            processing_classification_.pop();
            input_states.push_back( std::move( action ) );
          }
          break;
        }
        default: LOG( FATAL ) << "Invalid stage";
      }
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<models::InferenceState> results;
    switch ( next_stage ) {
      case models::InferenceState::Stage::PreAttention: {
        results = model_->pre_attention_forward( std::move( input_states ), contexts );
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        __stats__.add_point<IntDistributions::KernelPreAttentionForwardTime>( duration.count() );
        for ( auto& result : results ) {
          result.set_timestamp( end.time_since_epoch().count() );
        }
      } break;
      case models::InferenceState::Stage::Attention: {
        for ( auto& state : input_states ) {
          const auto queueing_time = start.time_since_epoch().count() - state.timestamp();
          __stats__.add_point<IntDistributions::AttentionQueueingTime>( queueing_time );
        }
        results = model_->attention_forward( std::move( input_states ), contexts );
        const auto duration
          = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start );
        __stats__.add_point<IntDistributions::KernelAttentionForwardTime>( duration.count() );
      } break;
      case models::InferenceState::Stage::PostAttention: {
        results = model_->post_attention_forward( std::move( input_states ) );
        const auto duration
          = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start );
        __stats__.add_point<IntDistributions::KernelPostAttentionForwardTime>( duration.count() );
      } break;
      case models::InferenceState::Stage::Classification: {
        results = model_->classify_forward( std::move( input_states ) );
        const auto duration
          = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start );
        __stats__.add_point<IntDistributions::KernelClassificationForwardTime>( duration.count() );
      } break;
      default: LOG( FATAL ) << "Invalid stage";
    }

    std::vector<models::InferenceState> outgoing_states;
    std::vector<StateContextPair> processing_states;
    switch ( next_stage ) {
      case models::InferenceState::Stage::PreAttention:
        // the next stage is attention, so if we hold the context and serve attention, add it directly to processing
        for ( size_t j = 0; j < target_conc_pre_size_; j++ ) {
          if ( process_att_ and !contexts[j].get()->empty() ) {
            processing_states.emplace_back( std::move( results[j] ), contexts[j] );
          } else {
            outgoing_states.emplace_back( std::move( results[j] ) );
          }
        }
        break;
      case models::InferenceState::Stage::Attention:
        // The next stage is post-attention, so if we serve that specific layer, add it directly to processing
        for ( size_t j = 0; j < target_conc_att_size_; j++ ) {
          if ( process_post_ and results[j].next_layer() <= end_layer_ and results[j].next_layer() >= start_layer_ ) {
            processing_states.emplace_back( std::move( results[j] ), contexts[j] );
          } else {
            outgoing_states.emplace_back( std::move( results[j] ) );
          }
        }
        break;
      case models::InferenceState::Stage::PostAttention:
        if ( results[0].next_stage() == models::InferenceState::Stage::PreAttention and process_pre_
             and results[0].next_layer() <= end_layer_ and results[0].next_layer() >= start_layer_ ) {
          // If the next stage is pre-attention, and we serve that layer, get context and add it directly to processing
          std::lock_guard lock( ctx_mgr_mutex_ );
          for ( size_t j = 0; j < target_conc_post_size_; j++ ) {
            processing_states.emplace_back( std::move( results[j] ),
                                            context_manager_.get_context( results[j].prompt_id(), true ) );
          }
        } else if ( results[0].next_stage() == models::InferenceState::Stage::Classification and process_cls_ ) {
          // If next stage is classification, and we serve it, add it directly to processing
          for ( size_t j = 0; j < target_conc_post_size_; j++ ) {
            processing_states.emplace_back( std::move( results[j] ), nullptr );
          }
        } else {
          for ( size_t j = 0; j < target_conc_post_size_; j++ ) {
            outgoing_states.emplace_back( std::move( results[j] ) );
          }
        }

        break;
      case models::InferenceState::Stage::Classification:
        // the next stage is pre-attention in layer 0, so we have to return the state from the kernel path for logging
        for ( size_t j = 0; j < target_conc_cls_size_; j++ ) {
          check_finished( results[j] );
          CHECK_EQ( results[j].next_layer(), 0 );
          outgoing_states.emplace_back( std::move( results[j] ) );
        }
        break;
      default: LOG( FATAL ) << "Invalid stage";
    }

    {
      std::lock_guard lock( outgoing_mutex_ );
      for ( auto& state : outgoing_states ) {
        outgoing_.emplace( std::move( state ) );
      }
    }

    if ( outgoing_states.size() > 0 ) {
      event_fd_.write_event();
    }

    {
      std::lock_guard lock( processing_mutex_ );
      switch ( next_stage ) {
        case models::InferenceState::Stage::PreAttention:
          for ( auto& action : processing_states ) {
            processing_attention_.emplace( std::move( action.first ), action.second );
          }
          break;
        case models::InferenceState::Stage::Attention:
          for ( auto& action : processing_states ) {
            processing_post_attention_[action.first.next_layer() - start_layer_].emplace( std::move( action.first ) );
          }
          break;
        case models::InferenceState::Stage::PostAttention:
          if ( processing_states.size() > 0 ) {
            switch ( processing_states[0].first.next_stage() ) {
              case models::InferenceState::Stage::PreAttention:
                for ( auto& action : processing_states ) {
                  processing_pre_attention_[action.first.next_layer() - start_layer_].emplace(
                    std::move( action.first ), action.second );
                }
                break;
              case models::InferenceState::Stage::Classification:
                for ( auto& action : processing_states ) {
                  processing_classification_.emplace( std::move( action.first ) );
                }
                break;
              default: LOG( FATAL ) << "Invalid stage";
            }
          }
          break;
          // we should not fast-path states after classification, otherwise they won't be logged
        case models::InferenceState::Stage::Classification: break;
        default: LOG( FATAL ) << "Invalid stage";
      }
    }
  }
}

template<typename Model>
void ComputeKernelPiped<Model>::bookkeeping_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped bookkeeping thread started.";

  while ( running_ ) {
    models::InferenceState action;
    // let's get an action from the incoming_
    {
      std::unique_lock<std::mutex> lock( incoming_mutex_ );
      incoming_cv_.wait( lock, [this] { return !incoming_.empty(); } );
      action = std::move( incoming_.front() );
      incoming_.pop();
    }

    //    LOG (INFO) << "got this in incoming: " << action;
    // make sure this action is for our serving layers
    const uint32_t next_layer_index = action.next_layer() - start_layer_;
    CHECK_LT( next_layer_index, n_layers_ )
      << "InferenceState can not be processed in this machine, original next layer was: " << action.next_layer()
      << ", but we host " << start_layer_ << " to " << end_layer_;

    switch ( action.next_stage() ) {
      case models::InferenceState::Stage::PreAttention: {
        // for action in pre-attention stage, get (try to create) context and just push to compute.
        CHECK_EQ( process_pre_, true ) << "This machine does not service the PreAttention pipeline";
        ContextPtr context;
        {
          // let's get the context for this action, but it doesn't matter if it's empty
          std::lock_guard lock( ctx_mgr_mutex_ );
          context = context_manager_.get_context( action.prompt_id(), true );
        }
        {
          std::lock_guard lock( processing_mutex_ );
          processing_pre_attention_[next_layer_index].emplace( std::move( action ), context );
        }

        processing_cv_.notify_one();
        break;
      }

      case models::InferenceState::Stage::Attention: {
        {
          const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
          __stats__.add_point<IntDistributions::IncomingKernelQueueingTime>( current_time - action.timestamp() );
          action.set_timestamp( current_time );
        }

        // for action in attention stage, get non-empty context and just push to compute.
        // if the context doesn't exist, just wait
        CHECK_EQ( process_att_, true ) << "This machine does not service the Attention pipeline";
        ContextPtr context;
        {
          // let's get the context for this action
          std::lock_guard lock( ctx_mgr_mutex_ );
          context = context_manager_.get_context( action.prompt_id(), false );
        }
        if ( not context.get()->empty() ) {
          const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
          __stats__.add_point<IntDistributions::ContextAdmissionTime>( current_time - action.timestamp() );
          action.set_timestamp( current_time );
          {
            std::lock_guard lock( processing_mutex_ );
            processing_attention_.emplace( std::move( action ), context );
          }

          processing_cv_.notify_one();
        } else {
          {
            std::lock_guard lock( waiting_attention_mutex_ );
            waiting_attention_.emplace( std::move( action ) );
          }

          waiting_attention_cv_.notify_one();
        }
        break;
      }

      case models::InferenceState::Stage::PostAttention: {
        // for action in post-attention stage, push to compute without context
        CHECK_EQ( process_post_, true ) << "This machine does not service the PostAttention pipeline";
        {
          std::lock_guard lock( processing_mutex_ );
          processing_post_attention_[next_layer_index].emplace( std::move( action ) );
        }

        processing_cv_.notify_one();
        break;
      }

      case models::InferenceState::Stage::Classification: {
        // for action in classification stage, push to compute without context
        CHECK_EQ( process_cls_, true ) << "This machine does not service the Classification pipeline";
        {
          std::lock_guard lock( processing_mutex_ );
          processing_classification_.emplace( std::move( action ) );
        }

        processing_cv_.notify_one();
        break;
      }
      default: LOG( FATAL ) << "Invalid stage";
    }
  }
}

template<typename Model>
void ComputeKernelPiped<Model>::backlog_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped backlog thread started.";

  while ( running_ ) {
    models::InferenceState action;
    ContextPtr context;

    // let's get an action from the waiting_attention_
    {
      std::unique_lock<std::mutex> lock( waiting_attention_mutex_ );
      waiting_attention_cv_.wait( lock, [this] { return !waiting_attention_.empty(); } );
      action = std::move( waiting_attention_.front() );
      waiting_attention_.pop();
    }

    // let's get a free context from context_manager_
    {
      std::unique_lock<std::mutex> lock( ctx_mgr_mutex_ );
      ctx_mgr_cv_.wait( lock, [this] { return context_manager_.free() > 0; } );
      context = context_manager_.get_context( action.prompt_id(), false );
    }

    CHECK_EQ( context.get()->empty(), false ) << "Context should not be empty";

    const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
    __stats__.add_point<IntDistributions::ContextAdmissionTime>( current_time - action.timestamp() );
    action.set_timestamp( current_time );

    {
      std::lock_guard lock( processing_mutex_ );
      processing_attention_.emplace( std::move( action ), context );
    }

    processing_cv_.notify_one();
  }
}

template<typename Model>
void ComputeKernelPiped<Model>::qmeasure_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped queue measurement thread started.";

  while ( running_ ) {

    {
      std::lock_guard lock( processing_mutex_ );
      __stats__.add_point<IntDistributions::ProcessingClassificationQueue>( processing_classification_.size() );
      __stats__.add_point<IntDistributions::ProcessingAttentionQueue>( processing_attention_.size() );
      for ( uint64_t layer_idx = 0; layer_idx < n_layers_; layer_idx++ ) {
        __stats__.add_point<IntDistributions::ProcessingPreAttentionQueue>(
          processing_pre_attention_[layer_idx].size() );
        __stats__.add_point<IntDistributions::ProcessingPostAttentionQueue>(
          processing_post_attention_[layer_idx].size() );
      }
    }

    {
      std::lock_guard lock( waiting_attention_mutex_ );
      __stats__.add_point<IntDistributions::WaitingQueue>( waiting_attention_.size() );
    }

    {
      std::lock_guard lock( incoming_mutex_ );
      __stats__.add_point<IntDistributions::IncomingQueue>( incoming_.size() );
    }

    {
      std::lock_guard lock( outgoing_mutex_ );
      __stats__.add_point<IntDistributions::OutgoingQueue>( outgoing_.size() );
    }

    {
      std::lock_guard lock( ctx_mgr_mutex_ );
      __stats__.add_point<IntDistributions::AllocatedContexts>( context_manager_.allocated() );
      __stats__.add_point<IntDistributions::FreeContexts>( context_manager_.free() );
      __stats__.add_point<IntDistributions::EmptyContexts>( context_manager_.empty() );
    }

    std::this_thread::sleep_for( std::chrono::seconds { 1 } );
  }
}

} // namespace glinthawk::compute
