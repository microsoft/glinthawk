#include "kernel_pipes.hh"

#include <chrono>
#include <glog/logging.h>

#include "models/llama2/cpu/model.cc"
#ifdef GLINTHAWK_CUDA_ENABLED
#include "models/llama2/cuda/model.cuh"
#endif

using namespace std;
using namespace glinthawk::models;
using namespace glinthawk::compute;
using namespace glinthawk::prompt;

template<typename Model>
void ComputeKernelPiped<Model>::execution_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped execution thread started.";

  while ( running_ ) {
    vector<InferenceState> input_states;
    vector<shared_ptr<typename Model::ContextType>> contexts;
    InferenceState::Stage next_stage;
    uint32_t next_layer_idx;
    {
      // hold lock until one queue has enough data for one batch
      unique_lock<mutex> lock( processing_mutex_ );
      processing_cv_.wait( lock, [this, &next_stage, &next_layer_idx] {
        if ( processing_attention_.size() >= target_conc_att_size_ and process_att_ ) {
          next_stage = InferenceState::Stage::Attention;
          next_layer_idx = -1;
          return true;
        }
        for ( int layer_idx = static_cast<int>( n_layers_ - 1 ); layer_idx >= 0; layer_idx-- ) {
          if ( processing_pre_attention_[layer_idx].size() >= target_conc_pre_size_ and process_pre_ ) {
            next_stage = InferenceState::Stage::PreAttention;
            next_layer_idx = static_cast<uint32_t>( layer_idx );
            return true;
          }
          if ( processing_post_attention_[layer_idx].size() >= target_conc_post_size_ and process_post_ ) {
            next_stage = InferenceState::Stage::PostAttention;
            next_layer_idx = static_cast<uint32_t>( layer_idx );
            return true;
          }
        }
        return false;
      } );

      // TODO: With splitting KV cache to in-context and out-context, will we be memed by batch size?
      // TODO: Right now we reserve GPU memory for all of the layers we host. Is that a good way, if we want to use a
      // TODO: smaller subset of these layers?
      // TODO: Right now the attention cpu machines do not need to know about layers for reserving context. Will they
      // TODO: work that way right now?
      // TODO: any reason we shouldn't always use max batch size?
      // TODO: context state?

      // TODO: fix partial model loading
      // TODO: route is very long, long messages (adds 3x32x11=1056 bytes).
      // TODO: make kv matrix done together so memcpy is together.
      // TODO: either make parallel tokens in one prompt work, or remove the feature altogether (and put protections in
      //       place).

      // find the queue and pop the data to input_states and possibly contexts
      switch ( next_stage ) {
        case InferenceState::Stage::PreAttention: {
          for ( size_t j = 0; j < target_conc_pre_size_; j++ ) {
            pair<InferenceState, shared_ptr<typename Model::ContextType>> action
              = move( processing_pre_attention_[next_layer_idx].front() );
            // LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action.first;
            processing_pre_attention_[next_layer_idx].pop();
            input_states.push_back( move( action.first ) );
            contexts.push_back( action.second );
          }
          break;
        }
        case InferenceState::Stage::Attention: {
          for ( size_t j = 0; j < target_conc_att_size_; j++ ) {
            pair<InferenceState, shared_ptr<typename Model::ContextType>> action
              = move( processing_attention_.front() );
            // LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action.first;
            processing_attention_.pop();
            input_states.push_back( move( action.first ) );
            contexts.push_back( action.second );
          }
          break;
        }
        case InferenceState::Stage::PostAttention: {
          for ( size_t j = 0; j < target_conc_post_size_; j++ ) {
            InferenceState action_post = move( processing_post_attention_[next_layer_idx].front() );
            // LOG_EVERY_N( INFO, 384 ) << "got this in processing: " << action_post;
            processing_post_attention_[next_layer_idx].pop();
            input_states.push_back( move( action_post ) );
          }
          break;
        }
      }
    }

    const auto start = chrono::steady_clock::now();
    std::vector<InferenceState> results;
    switch ( next_stage ) {
      case InferenceState::Stage::PreAttention: {
        results = model_->pre_attention_forward( move( input_states ), contexts );
        const auto duration = chrono::duration_cast<chrono::microseconds>( chrono::steady_clock::now() - start );
        __stats__.add_point<IntDistributions::KernelPreAttentionForwardTime>( duration.count() );
      } break;
      case InferenceState::Stage::Attention: {
        results = model_->attention_forward( move( input_states ), contexts );
        const auto duration = chrono::duration_cast<chrono::microseconds>( chrono::steady_clock::now() - start );
        __stats__.add_point<IntDistributions::KernelAttentionForwardTime>( duration.count() );
      } break;
      case InferenceState::Stage::PostAttention: {
        results = model_->post_attention_forward( move( input_states ) );
        const auto duration = chrono::duration_cast<chrono::microseconds>( chrono::steady_clock::now() - start );
        __stats__.add_point<IntDistributions::KernelPostAttentionForwardTime>( duration.count() );
      } break;
    }

    vector<InferenceState> outgoing_states;
    vector<pair<InferenceState, shared_ptr<typename Model::ContextType>>> processing_states;
    switch ( next_stage ) {
      case InferenceState::Stage::PreAttention:
        // the next stage is attention, so if we hold the context and serve attention, add it directly to processing
        for ( size_t j = 0; j < target_conc_pre_size_; j++ ) {
          if ( process_att_ and !contexts[j].get()->empty() ) {
            processing_states.emplace_back( move( results[j] ), contexts[j] );
          } else {
            outgoing_states.emplace_back( move( results[j] ) );
          }
        }
        break;
      case InferenceState::Stage::Attention:
        // the next stage is post-attention, so if we serve that specific layer, add it directly to processing
        for ( size_t j = 0; j < target_conc_att_size_; j++ ) {
          if ( process_post_ and results[j].next_layer() <= end_layer_ and results[j].next_layer() >= start_layer_ ) {
            processing_states.emplace_back( move( results[j] ), contexts[j] );
          } else {
            outgoing_states.emplace_back( move( results[j] ) );
          }
        }
        break;
      case InferenceState::Stage::PostAttention:
        // the next stage is pre-attention, so if we serve that specific layer, get context and add it directly to
        // processing
        for ( size_t j = 0; j < target_conc_post_size_; j++ ) {
          check_finished( results[j] );
          if ( process_pre_ and results[j].next_layer() <= end_layer_ and results[j].next_layer() >= start_layer_
               and !results[j].finished() and results[j].next_layer() != 0 ) {
            lock_guard lock( ctx_mgr_mutex_ );
            processing_states.emplace_back( move( results[j] ),
                                            context_manager_.get_context( results[j].prompt_id(), true ) );
          } else {
            outgoing_states.emplace_back( move( results[j] ) );
          }
        }
        break;
    }

    {
      lock_guard lock( outgoing_mutex_ );
      for ( auto& state : outgoing_states ) {
        outgoing_.emplace( move( state ) );
      }
    }

    if ( outgoing_states.size() > 0 ) {
      event_fd_.write_event();
    }

    {
      lock_guard lock( processing_mutex_ );
      switch ( next_stage ) {
        case InferenceState::Stage::PreAttention:
          for ( auto& action_loop : processing_states ) {
            processing_attention_.emplace( move( action_loop.first ), action_loop.second );
          }
          break;
        case InferenceState::Stage::Attention:
          for ( auto& action_loop : processing_states ) {
            processing_post_attention_[action_loop.first.next_layer() - start_layer_].emplace(
              move( action_loop.first ) );
          }
          break;
        case InferenceState::Stage::PostAttention:
          for ( auto& action_loop : processing_states ) {
            processing_pre_attention_[action_loop.first.next_layer() - start_layer_].emplace( move( action_loop.first ),
                                                                                              action_loop.second );
          }
          break;
      }
    }
  }
}

template<typename Model>
void ComputeKernelPiped<Model>::bookkeeping_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped bookkeeping thread started.";

  while ( running_ ) {
    InferenceState action;
    // let's get an action from the incoming_
    {
      unique_lock<mutex> lock( incoming_mutex_ );
      incoming_cv_.wait( lock, [this] { return !incoming_.empty(); } );
      action = move( incoming_.front() );
      incoming_.pop();
    }

    //    LOG (INFO) << "got this in incoming: " << action;
    // make sure this action is for our serving layers
    const uint32_t next_layer_index = action.next_layer() - start_layer_;
    CHECK_LT( next_layer_index, n_layers_ )
      << "InferenceState can not be processed in this machine, original next layer was: " << action.next_layer()
      << ", but we host " << start_layer_ << " to " << end_layer_;

    switch ( action.next_stage() ) {
      case InferenceState::Stage::PreAttention: {
        // for action in pre-attention stage, get (try to create) context and just push to compute.
        CHECK_EQ( process_pre_, true ) << "This machine does not service the PreAttention pipeline";
        shared_ptr<typename Model::ContextType> context;
        {
          // let's get the context for this action, but it doesn't matter if it's empty
          lock_guard lock( ctx_mgr_mutex_ );
          context = context_manager_.get_context( action.prompt_id(), true );
        }
        {
          lock_guard lock( processing_mutex_ );
          processing_pre_attention_[next_layer_index].emplace( move( action ), context );
        }

        processing_cv_.notify_one();
        break;
      }

      case InferenceState::Stage::Attention: {
        // for action in attention stage, get non-empty context and just push to compute.
        // if the context doesn't exist, just wait
        CHECK_EQ( process_att_, true ) << "This machine does not service the Attention pipeline";
        shared_ptr<typename Model::ContextType> context;
        {
          // let's get the context for this action
          lock_guard lock( ctx_mgr_mutex_ );
          context = context_manager_.get_context( action.prompt_id(), false );
        }
        if ( not context.get()->empty() ) {
          {
            lock_guard lock( processing_mutex_ );
            processing_attention_.emplace( move( action ), context );
          }

          processing_cv_.notify_one();
        } else {
          {
            lock_guard lock( waiting_attention_mutex_ );
            waiting_attention_.emplace( move( action ) );
          }

          waiting_attention_cv_.notify_one();
        }
        break;
      }

      case InferenceState::Stage::PostAttention: {
        // for action in post-attention stage, push to compute without context
        // if the context doesn't exist, just wait
        CHECK_EQ( process_post_, true ) << "This machine does not service the PostAttention pipeline";
        {
          lock_guard lock( processing_mutex_ );
          processing_post_attention_[next_layer_index].emplace( move( action ) );
        }

        processing_cv_.notify_one();
        break;
      }
    }
  }
}

template<typename Model>
void ComputeKernelPiped<Model>::backlog_thread_func()
{
  LOG( INFO ) << "ComputeKernelPiped backlog thread started.";

  while ( running_ ) {
    InferenceState action;
    shared_ptr<typename Model::ContextType> context;
    // let's get an action from the waiting_attention_
    {
      unique_lock<mutex> lock( waiting_attention_mutex_ );
      waiting_attention_cv_.wait( lock, [this] { return !waiting_attention_.empty(); } );
      action = move( waiting_attention_.front() );
      waiting_attention_.pop();
    }

    // let's get a free context from context_manager_
    {
      unique_lock<mutex> lock( ctx_mgr_mutex_ );
      ctx_mgr_cv_.wait( lock, [this] { return context_manager_.size() > 0; } );
      context = context_manager_.get_context( action.prompt_id(), false );
    }

    CHECK_EQ( context.get()->empty(), false ) << "Context should not be empty";
    {
      lock_guard lock( processing_mutex_ );
      processing_attention_.emplace( move( action ), context );
    }

    processing_cv_.notify_one();
  }
}

namespace glinthawk::compute {

template class ComputeKernelPiped<models::llama2::cpu::Llama2_7B_Chat<float>>;
template class ComputeKernelPiped<models::llama2::cpu::Llama2_13B_Chat<float>>;
template class ComputeKernelPiped<models::llama2::cpu::Llama2_70B_Chat<float>>;
template class ComputeKernelPiped<models::llama2::cpu::Stories_110M<float>>;

template class ComputeKernelPiped<models::llama2::cpu::Llama2_7B_Chat<_Float16>>;
template class ComputeKernelPiped<models::llama2::cpu::Llama2_13B_Chat<_Float16>>;
template class ComputeKernelPiped<models::llama2::cpu::Llama2_70B_Chat<_Float16>>;
template class ComputeKernelPiped<models::llama2::cpu::Stories_110M<_Float16>>;

#ifdef GLINTHAWK_CUDA_ENABLED
template class ComputeKernelPiped<models::llama2::cuda::Llama2_7B_Chat<__half>>;
template class ComputeKernelPiped<models::llama2::cuda::Llama2_13B_Chat<__half>>;
template class ComputeKernelPiped<models::llama2::cuda::Llama2_70B_Chat<__half>>;
template class ComputeKernelPiped<models::llama2::cuda::Stories_110M<__half>>;
#endif

} // namespace glinthawk::compute
