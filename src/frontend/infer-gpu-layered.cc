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

  if ( argc != 3 ) {
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

    using Llama2 = models::llama2::cuda::Llama2<__half>;

    models::llama2::Vocabulary vocabulary { tokenizer_path };
    vector<shared_ptr<Llama2::ContextType>> contexts; // each layer needs a different context
    string serialized_state = models::InferenceState {}.serialize();

    while ( true ) {
      // first, let's deserialize the state
      models::InferenceState state { serialized_state };

      // load the model for the next layer
      const auto current_layer = state.next_layer();
      auto llama = Llama2::load( model_dir_path, current_layer, current_layer, 1, 1 );

      if ( contexts.empty() ) {
        contexts.resize( llama->config().n_layers );
      }

      // create context for this layer if it doesn't exist yet
      if ( contexts[current_layer] == nullptr ) {
        contexts[current_layer] = make_shared<Llama2::ContextType>( llama->config() );
      }

      // forward the current token
      state = llama->forward( state, contexts[current_layer] );

      if ( state.token() == 2 /* EOS */ ) {
        break;
      }

      if ( state.next_layer() == 0 ) {
        cout << vocabulary.get_word( state.token() ) << flush;
      }

      serialized_state = state.serialize();
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
