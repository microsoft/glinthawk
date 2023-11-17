#include <csignal>
#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "compute/kernel.hh"
#include "models/llama2/cpu/model.hh"
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
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir_path { argv[1] };
    const filesystem::path tokenizer_path { argv[2] };

    using Llama2 = models::llama2::cpu::Llama2<_Float16>;
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

    constexpr size_t LAYERS_AT_ONCE = 2;
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
      Llama2 llama {
        model_dir_path, current_layer, static_cast<uint32_t>( current_layer + LAYERS_AT_ONCE - 1 ), input_states.size()
      };

      if ( context_managers.empty() ) {
        context_managers.resize( llama.config().n_layers );
      }

      // do we have a context manager for the current layer?
      if ( context_managers[current_layer] == nullptr ) {
        context_managers[current_layer] = make_shared<compute::ContextManager<Llama2>>( llama.config() );
      }

      // get the contexts for the current layer
      for ( const auto& state : states ) {
        contexts.push_back( context_managers[current_layer]->get_context( state.prompt_id() ) );
      }

      auto output_states = llama.forward( move( states ), contexts );
      input_states.clear();

      for ( const auto& state : output_states ) {
        input_states.push_back( state.serialize() );
      }

      const auto next_layer = output_states[0].next_layer();

      if ( not prompt_processed and next_layer == 0 ) {
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
        output_states.erase( output_states.begin(), output_states.end() - 1 );
        for ( const auto& state : output_states ) {
          cout << vocabulary.get_word( state.token() ) << flush;
        }
      } else if ( next_layer == 0 ) {
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
