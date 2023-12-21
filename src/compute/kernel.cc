#include "kernel.hh"

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
void ComputeKernel<Model>::execution_thread_func()
{
  LOG( INFO ) << "ComputeKernel execution thread started.";

  pair<InferenceState, shared_ptr<typename Model::ContextType>> action;
  vector<InferenceState> input_states;
  vector<shared_ptr<typename Model::ContextType>> contexts;

  while ( running_ ) {
    input_states.clear();
    contexts.clear();

    {
      unique_lock<mutex> lock( processing_mutex_ );
      processing_cv_.wait( lock, [this] { return !( processing_.size() < target_conc_size_ ); } );

      for ( size_t j = 0; j < target_conc_size_; j++ ) {
        action = move( processing_.front() );
        processing_.pop();
        input_states.push_back( move( action.first ) );
        contexts.push_back( action.second );
      }
    }

    const auto start = chrono::steady_clock::now();
    auto results = model_->forward( move( input_states ), contexts );
    const auto duration = chrono::duration_cast<chrono::microseconds>( chrono::steady_clock::now() - start );
    __stats__.add_point<IntDistributions::KernelForwardTime>( duration.count() );

    {
      lock_guard lock( outgoing_mutex_ );
      for ( auto& state : results ) {
        outgoing_.emplace( move( state ) );
      }
    }

    event_fd_.write_event();
  }
}

template<typename Model>
void ComputeKernel<Model>::bookkeeping_thread_func()
{
  LOG( INFO ) << "ComputeKernel bookkeeping thread started.";

  InferenceState action;
  shared_ptr<typename Model::ContextType> context;

  while ( running_ ) {
    // let's get an action from the incoming_
    {
      unique_lock<mutex> lock( incoming_mutex_ );
      incoming_cv_.wait( lock, [this] { return !incoming_.empty(); } );
      action = move( incoming_.front() );
      incoming_.pop();
    }
    {
      // let's get the context for this action
      lock_guard lock( ctx_mgr_mutex_ );
      context = context_manager_.get_context( action.prompt_id() );
    }

    if ( not context.get()->empty() ) {
      {
        lock_guard lock( processing_mutex_ );
        processing_.emplace( move( action ), context );
      }

      processing_cv_.notify_one();
    } else {
      {
        lock_guard lock( waiting_mutex_ );
        waiting_.emplace( move( action ) );
      }

      waiting_cv_.notify_one();
    }
  }
}

template<typename Model>
void ComputeKernel<Model>::backlog_thread_func()
{
  LOG( INFO ) << "ComputeKernel backlog thread started.";

  InferenceState action;
  shared_ptr<typename Model::ContextType> context;

  while ( running_ ) {
    // let's get an action from the incoming_
    {
      unique_lock<mutex> lock( waiting_mutex_ );
      waiting_cv_.wait( lock, [this] { return released_ > 0 && !waiting_.empty(); } );
      action = move( waiting_.front() );
      waiting_.pop();
      released_ -= 1;
    }

    {
      // let's get the context for this action
      lock_guard lock( ctx_mgr_mutex_ );
      context = context_manager_.get_context( action.prompt_id() );
    }

    if ( not context.get()->empty() ) {
      {
        lock_guard lock( processing_mutex_ );
        processing_.emplace( move( action ), context );
      }

      processing_cv_.notify_one();
    } else {
      {
        lock_guard lock( waiting_mutex_ );
        waiting_.emplace( move( action ) );
      }
    }
  }
}

namespace glinthawk::compute {

template class ComputeKernel<models::llama2::cpu::Llama2_7B_Chat<float>>;
template class ComputeKernel<models::llama2::cpu::Llama2_13B_Chat<float>>;
template class ComputeKernel<models::llama2::cpu::Llama2_70B_Chat<float>>;
template class ComputeKernel<models::llama2::cpu::Stories_110M<float>>;

template class ComputeKernel<models::llama2::cpu::Llama2_7B_Chat<_Float16>>;
template class ComputeKernel<models::llama2::cpu::Llama2_13B_Chat<_Float16>>;
template class ComputeKernel<models::llama2::cpu::Llama2_70B_Chat<_Float16>>;
template class ComputeKernel<models::llama2::cpu::Stories_110M<_Float16>>;

#ifdef GLINTHAWK_CUDA_ENABLED
template class ComputeKernel<models::llama2::cuda::Llama2_7B_Chat<__half>>;
template class ComputeKernel<models::llama2::cuda::Llama2_13B_Chat<__half>>;
template class ComputeKernel<models::llama2::cuda::Llama2_70B_Chat<__half>>;
template class ComputeKernel<models::llama2::cuda::Stories_110M<__half>>;
#endif

} // namespace glinthawk::compute
