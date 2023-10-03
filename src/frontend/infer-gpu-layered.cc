#include <csignal>
#include <filesystem>
#include <iostream>

#include <cuda_fp16.h>
#include <glog/logging.h>

#include "compute/kernel.hh"
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

    const size_t batch_size = 8;
    vector<string> serialized_states( batch_size );

    for ( size_t i = 0; i < batch_size; i++ ) {
      models::InferenceState state;

      PromptID id;
      util::digest::sha256( to_string( i ), id );

      state.set_prompt_id( id );
      state.set_token( 1 );
      state.set_token_pos( 0 );
      state.set_next_layer( 0 );
      state.set_temperature( 0.0f );

      serialized_states[i] = state.serialize();
    }

    models::llama2::Vocabulary vocabulary { tokenizer_path };

    vector<shared_ptr<compute::ContextManager<Llama2>>> context_managers;

    while ( true ) {
      // first, let's deserialize the state
      vector<models::InferenceState> states;
      vector<shared_ptr<Llama2::ContextType>> contexts;

      for ( const auto& serialized_state : serialized_states ) {
        states.emplace_back( serialized_state );
      }

      // load the model for the next layer
      const auto current_layer = states[0].next_layer();
      auto llama = Llama2::load( model_dir_path, current_layer, current_layer, 1, batch_size );

      if ( context_managers.empty() ) {
        context_managers.resize( llama->config().n_layers );
      }

      if ( context_managers[current_layer] == nullptr ) {
        context_managers[current_layer] = make_shared<compute::ContextManager<Llama2>>( llama->config() );
      }

      // get the contexts for the current layer
      for ( const auto& state : states ) {
        contexts.push_back( context_managers[current_layer]->get_context( state.prompt_id() ) );
      }

      auto output_states = llama->forward( states, contexts );
      serialized_states.clear();

      bool token_generated = false;
      for ( const auto& state : output_states ) {
        if ( state.next_layer() == 0 ) {
          token_generated = true;
          cout << vocabulary.get_word( state.token() ) << '\t';
        }

        serialized_states.push_back( state.serialize() );
      }

      if ( token_generated ) {
        cout << endl;
      }
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
