#include <csignal>
#include <filesystem>
#include <iostream>

#include <cuda_fp16.h>
#include <glog/logging.h>

#include "models/llama2/cuda/model.cuh"
#include "util/timer.hh"

#ifndef GLINTHAWK_CUDA_ENABLED
#error "This file should only be compiled when CUDA is enabled."
#endif

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 ) { cout << "Usage: " << argv0 << " <model_dir_path> <tokenizer_path>" << endl; }

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 6 ) {
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
    const filesystem::path tokenizer_path { argv[2] };

    const int max_batch_size = atoi( argv[3] );
    const int conc_size = atoi( argv[4] );
    const int batch_size = atoi( argv[5] );

    const int seq_len = 1024;

    using Llama2 = models::llama2::cuda::Llama2<__half>;

    models::llama2::Vocabulary vocabulary { tokenizer_path };

    vector<uint32_t> prompt_tokens { 1,   518,  25580, 29962, 25538, 2211,  25562, 363,  7952,
                                     292, 9045, 29891, 29889, 518,   29914, 25580, 29962 };

    // create inference state for all prompt tokens
    vector<models::InferenceState> inference_states;
    for ( size_t i = 0; i < prompt_tokens.size(); i++ ) {
      models::InferenceState state;
      state.set_token( prompt_tokens[i] );
      state.set_token_pos( i );
      state.set_next_layer( 0 );
      state.set_temperature( 0.0f );
      inference_states.emplace_back( state );
    }

    // create context for each layer of this prompt
    vector<unique_ptr<Llama2::ContextType>> contexts; // each layer needs a different context

    size_t i = 0;
    for ( uint32_t token_pos = 0; ; token_pos++ ) {

    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
