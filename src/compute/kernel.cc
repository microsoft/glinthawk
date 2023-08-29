#include "kernel.hh"

#include <glog/logging.h>

#ifdef GLINTHAWK_CUDA_ENABLED
#include "models/llama2/cuda/model.cuh"
#endif

using namespace std;
using namespace glinthawk::models;
using namespace glinthawk::compute;

template<typename Model>
void ComputeKernel<Model>::execution_thread_func()
{
  pair<InferenceState, shared_ptr<typename Model::ContextType>> action;

  while ( running_ ) {
    {
      unique_lock<mutex> lock( processing_mutex_ );
      processing_cv_.wait( lock, [this] { return !processing_.empty(); } );
      action = move( processing_.front() );
      processing_.pop();
    }

    auto result = model_->forward( move( action.first ), *action.second );

    {
      unique_lock<mutex> lock( outgoing_mutex_ );
      outgoing_.emplace( move( result ) );
    }

    event_fd_.write_event();
  }
}

template<typename Model>
void ComputeKernel<Model>::bookkeeping_thread_func()
{
  InferenceState action;

  while ( running_ ) {
    // let's get an action from the incoming_
    {
      unique_lock<mutex> lock( incoming_mutex_ );
      incoming_cv_.wait( lock, [this] { return !incoming_.empty(); } );
      action = move( incoming_.front() );
      incoming_.pop();
    }

    // let's get the context for this action
    auto context = context_manager_.get_context( action.prompt_id() );

    if ( not context ) {
      LOG( ERROR ) << "Could not get context for prompt_id=" << action.prompt_id().to_string();
    }

    {
      unique_lock<mutex> lock( processing_mutex_ );
      processing_.emplace( move( action ), context );
    }

    processing_cv_.notify_one();
  }
}

#ifdef GLINTHAWK_CUDA_ENABLED
template class glinthawk::compute::ComputeKernel<glinthawk::models::llama2::cuda::Llama2<__half>>;
#endif
