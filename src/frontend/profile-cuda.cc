#ifndef GLINTHAWK_CUDA_ENABLED
#error "This file should only be compiled when CUDA is enabled."
#endif

#include <csignal>
#include <filesystem>
#include <iostream>

#include <cuda_fp16.h>
#include <glog/logging.h>

#include "compute/kernel.hh"
#include "models/llama2/cuda/model.cuh"
#include "util/timer.hh"

#define OOF_IMPL
#include "oof/oof.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_dir_path> begin_slice end_slice batch_size" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 5 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  FLAGS_log_year_in_prefix = false;
  FLAGS_timestamp_in_logfile_name = false;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir_path { argv[1] };

    const uint32_t start_slice = atoi( argv[2] );
    const uint32_t end_slice = atoi( argv[3] );
    const uint64_t batch_size = atoi( argv[4] );

    using Llama2 = models::llama2::cuda::Llama2<__half>;

    Llama2 llama { model_dir_path, start_slice, end_slice, batch_size };
    compute::ContextManager<Llama2> context_manager = compute::ContextManager<Llama2>( llama.config() );
    const uint64_t seq_len = llama.config().seq_len;
    const uint64_t dim = llama.config().dim;

    auto input_states = vector<vector<models::InferenceState>>( seq_len );
    auto contexts = vector<vector<shared_ptr<Llama2::ContextType>>>( seq_len );

    for ( size_t i = 0; i < input_states.size(); i++ ) {
      input_states[i] = vector<models::InferenceState>();
      input_states[i].reserve( batch_size );
      contexts[i] = vector<shared_ptr<Llama2::ContextType>>();
      contexts[i].reserve( batch_size );

      for ( uint64_t j = 0; j < batch_size; j++ ) {
        PromptID id;
        util::digest::sha256( to_string( j ), id );

        models::InferenceState state;

        state.set_prompt_id( id );
        state.set_token_pos( i );
        state.set_next_layer( start_slice );
        state.set_temperature( 0.0f );
        if ( start_slice == 0 ) {
          state.set_token( 5 );
        } else {
          models::DataBuffer activations { models::SerializedDataType::Type::Float16,
                                           make_unique<uint8_t[]>( dim * sizeof( __half ) ),
                                           dim };
          state.set_activations( move( activations ) );
        }

        input_states[i].emplace_back( state.serialize() );
        contexts[i].push_back( context_manager.get_context( id ) );
      }
    }

    for ( size_t i = 0; i < input_states.size(); i++ ) {
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      llama.forward( input_states[i], contexts[i] );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
