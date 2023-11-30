#include <csignal>
#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "compute/simple.hh"
#include "models/llama2/cpu/model.hh"
#include "util/timer.hh"

#define OOF_IMPL
#include "oof/oof.hh"

#include "platform_macros.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_dir> <model_name> <tokenizer_path> batch_size temperature prompt_print_id"
       << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 7 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir_path { argv[1] };
    const string model_name { argv[2] };
    const filesystem::path tokenizer_path { argv[3] };
    const size_t batch_size = atoi( argv[4] );
    const float temp = atof( argv[5] );

    PromptID id_print;
    util::digest::sha256( argv[6], id_print );

    models::llama2::Vocabulary vocabulary { tokenizer_path };

    vector<uint32_t> prompt_tokens { 1 };

    vector<models::InferenceState> input_states;

    for ( size_t j = 0; j < batch_size; j++ ) {
      PromptID id;

      if ( j != 0 ) {
        util::digest::sha256( to_string( j ), id );
      } else {
        id = id_print;
      }

      for ( size_t i = 0; i < prompt_tokens.size(); i++ ) {
        models::InferenceState state { DataType::_GLINTHAWK_DTYPE_NAME_ };

        state.set_prompt_id( id );
        state.set_token( prompt_tokens[i] );
        state.set_token_pos( i );
        state.set_next_layer( 0 );
        state.set_temperature( temp );
        models::DataBuffer activations { models::SerializedDataType::Type::Float16,
                                         nullptr,
                                         0 };
        state.set_activations( move( activations ) );

        input_states.emplace_back( state.serialize() );
      }
    }

    compute::SimpleComputeKernel<compute::Platform::_GLINTHAWK_PLATFORM_NAME_,
                                 compute::DataType::_GLINTHAWK_DTYPE_NAME_>
      llama { model_dir_path, model_name, 0, std::numeric_limits<uint32_t>::max(), input_states.size(), false };

    const unsigned int seq_len = llama.max_seq_len();

    bool prompt_processed = false;

    while ( input_states.size() > 0 ) {
      vector<models::InferenceState> output_states;
      {
        GlobalScopeTimer<Timer::Category::TokenGeneration> _;
        output_states = llama.forward( move( input_states ) );
      }

      input_states.clear();

      if ( not prompt_processed ) {
        // we're done processing the prompt.
        prompt_processed = true;

        // (1) let's print the prompt in full
        cout << oof::fg_color( { 0, 255, 0 } ) << oof::underline();
        for ( auto& token : prompt_tokens ) {
          cout << vocabulary.get_word( token );
        }
        cout << oof::reset_formatting();

        // remove all elements in the input states except the last one in each prompt
        for ( auto& state : output_states ) {
          if ( state.token_pos() == prompt_tokens.size() && state.token_pos() < seq_len
               && state.token() != 2 /* EOS */ ) {
            input_states.emplace_back( state.serialize() );
            if ( state.prompt_id() == id_print )
              cout << vocabulary.get_word( state.token() ) << flush;
          }
        }
      } else {
        // (2) let's print the output
        for ( auto& state : output_states ) {
          if ( state.prompt_id() == id_print )
            cout << vocabulary.get_word( state.token() ) << flush;
          if ( state.token_pos() < seq_len && state.token() != 2 /* EOS */ )
            input_states.emplace_back( state.serialize() );
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
