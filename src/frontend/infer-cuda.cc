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

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_dir_path> <tokenizer_path> batch_size temperature prompt_print_id" << endl;
}

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

    const size_t batch_size = atoi( argv[3] );
    const float temp = atof( argv[4] );

    PromptID id_print;
    util::digest::sha256( argv[5], id_print );

    using Llama2 = models::llama2::cuda::Llama2<__half>;
    models::llama2::Vocabulary vocabulary { tokenizer_path };

    vector<uint32_t> prompt_tokens { 1,   518,  25580, 29962, 25538, 2211,  25562, 363,  7952,
                                     292, 9045, 29891, 29889, 518,   29914, 25580, 29962 };

    vector<models::InferenceState> input_states;

    for ( size_t j = 0; j < batch_size; j++ ) {
      PromptID id;
      util::digest::sha256( to_string( j ), id );

      for ( size_t i = 0; i < prompt_tokens.size(); i++ ) {
        models::InferenceState state;

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

    Llama2 llama { model_dir_path, 0, UINT32_MAX, input_states.size() };
    compute::ContextManager<Llama2> context_manager = compute::ContextManager<Llama2>( llama.config() );
    const unsigned int seq_len = llama.config().seq_len;

    bool prompt_processed = false;

    while ( input_states.size() > 0 ) {
      vector<shared_ptr<Llama2::ContextType>> contexts;

      // get the contexts
      for ( const auto& state : input_states ) {
        contexts.push_back( context_manager.get_context( state.prompt_id() ) );
      }
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      auto output_states = llama.forward( move( input_states ), contexts );
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
        for ( const auto& state : output_states ) {
          if ( state.token_pos() == prompt_tokens.size() && state.token_pos() < seq_len
               && state.token() != 2 /* EOS */ ) {
            input_states.emplace_back( state.serialize() );
            if ( state.prompt_id() == id_print )
              cout << vocabulary.get_word( state.token() ) << flush;
          }
        }
      } else {
        // (2) let's print the output
        for ( const auto& state : output_states ) {
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
