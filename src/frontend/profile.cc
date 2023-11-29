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
  cout << "Usage: " << argv0 << " <model_root> <model_name> begin_slice end_slice batch_size" << endl;
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
    const filesystem::path model_dir { argv[1] };
    const string model_name { argv[2] };
    const uint32_t start_slice = atoi( argv[3] );
    const uint32_t end_slice = atoi( argv[4] );
    const uint64_t batch_size = atoi( argv[5] );

    compute::SimpleComputeKernel<compute::Platform::_GLINTHAWK_PLATFORM_NAME_,
                                 compute::DataType::_GLINTHAWK_DTYPE_NAME_>
      llama { model_dir, model_name, start_slice, end_slice, batch_size };

    const uint64_t seq_len = llama.max_seq_len();
    const uint64_t dim = llama.dim();

    auto input_states = vector<vector<models::InferenceState>>( seq_len );

    for ( size_t i = 0; i < input_states.size(); i++ ) {
      input_states[i] = vector<models::InferenceState>();
      input_states[i].reserve( batch_size );

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
          models::DataBuffer activations { models::SerializedDataType::Type::_GLINTHAWK_DTYPE_NAME_, nullptr, 0 };
          state.set_activations( move( activations ) );
        } else {
          models::DataBuffer activations {
            models::SerializedDataType::Type::_GLINTHAWK_DTYPE_NAME_,
            make_unique<uint8_t[]>(
              dim
              * sizeof(
                models::SerializedDataType { models::SerializedDataType::Type::_GLINTHAWK_DTYPE_NAME_ }.size() ) ),
            dim
          };
          state.set_activations( move( activations ) );
        }

        input_states[i].emplace_back( state.serialize() );
      }
    }

    for ( size_t i = 0; i < input_states.size(); i++ ) {
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      llama.forward( move( input_states[i] ) );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
