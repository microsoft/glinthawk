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

    PromptID id;
    util::digest::sha256( "0", id );

    vector<uint32_t> prompt_tokens { 1,   518,  25580, 29962, 25538, 2211,  25562, 363,  7952,
                                     292, 9045, 29891, 29889, 518,   29914, 25580, 29962 };

    vector<string> input_states;

    for ( size_t i = 0; i < prompt_tokens.size(); i++ ) {
      models::InferenceState state;

      state.set_prompt_id( id );
      state.set_token( prompt_tokens[i] );
      state.set_token_pos( i );
      state.set_next_layer( 0 );
      state.set_temperature( 0.0f );

      input_states.push_back( state.serialize() );
    }

    bool prompt_processed = false;

    // since we're loading/executing the model layer by layer, each layer has its own context manager
    vector<shared_ptr<compute::ContextManager<Llama2>>> context_managers;

    while ( true ) {
      // first, let's deserialize the state
      vector<models::InferenceState> states;
      vector<shared_ptr<Llama2::ContextType>> contexts;

      for ( const auto& serialized_state : input_states ) {
        states.emplace_back( serialized_state );
      }

      // load the model for the next layer
      const auto current_layer = states[0].next_layer();
      auto llama = Llama2::load( model_dir_path, current_layer, current_layer, 1, input_states.size() );

      if ( context_managers.empty() ) {
        context_managers.resize( llama->config().n_layers );
      }

      // do we have a context manager for the current layer?
      if ( context_managers[current_layer] == nullptr ) {
        context_managers[current_layer] = make_shared<compute::ContextManager<Llama2>>( llama->config() );
      }

      // get the contexts for the current layer
      for ( const auto& state : states ) {
        contexts.push_back( context_managers[current_layer]->get_context( state.prompt_id() ) );
      }

      auto output_states = llama->forward( states, contexts );
      input_states.clear();

      for ( const auto& state : output_states ) {
        input_states.push_back( state.serialize() );
      }

      if ( not prompt_processed and current_layer == llama->config().n_layers - 1 ) {
        // this is the last layer and we're done processing the prompt.
        prompt_processed = true;

        // (1) let's print the prompt in full
        cout << oof::fg_color( { 0, 255, 0 } ) << oof::underline();
        for ( auto& token : prompt_tokens ) {
          cout << vocabulary.get_word( token );
        }
        cout << oof::reset_formatting();

        // remove all elements in the input states except the last one
        input_states.erase( input_states.begin(), input_states.end() - 1 );
      } else if ( current_layer == llama->config().n_layers - 1 ) {
        // (2) let's print the output
        for ( const auto& state : output_states ) {
          cout << vocabulary.get_word( state.token() ) << flush;
        }
      }
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
