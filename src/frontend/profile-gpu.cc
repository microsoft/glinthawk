#include <csignal>
#include <filesystem>
#include <iostream>

#include <cuda_fp16.h>
#include <glog/logging.h>

#include "models/llama2/cuda/model.cuh"
#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_dir_path> <tokenizer_path> begin_slice end_slice batch_size" << endl;
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

    const int start_slice = atoi( argv[3] );
    const int end_slice = atoi( argv[4] );
    const int batch_size = atoi( argv[5] );

    const int dim = 4096;
    const int seq_len = 2048;

    auto llama = models::llama2::cuda::Llama2<__half>::load( model_dir_path, start_slice, end_slice, batch_size );
    models::llama2::Vocabulary vocabulary { tokenizer_path };

    auto prompt_tokens_batch = vector<vector<models::InferenceState<__half>>>( seq_len );
    for ( size_t i = 0; i < prompt_tokens_batch.size(); i++ ) {
      prompt_tokens_batch[i] = vector<models::InferenceState<__half>>();
      for ( int j = 0; j < batch_size; j++ ) {
        if ( start_slice == 0 ) {
          prompt_tokens_batch[i].emplace_back( 1,                            // token
                                               i,                            // token_pos
                                               0,                            // next_layer
                                               0.0,                          // temperature
                                               models::DataBuffer<__half> {} // activations
          );
        } else {
          models::DataBuffer<__half> activations { make_unique<__half[]>( dim ), dim };
          prompt_tokens_batch[i].emplace_back( 1,                       // token
                                               i,                       // token_pos
                                               start_slice,             // next_layer
                                               0.0,                     // temperature
                                               std::move( activations ) // activations
          );
        }
      }
    }

    vector<uint32_t> prompt_ids_batch;
    for ( int i = 0; i < batch_size; i++ )
      prompt_ids_batch.push_back( i );

    for ( size_t i = 0; i < prompt_tokens_batch.size(); i++ ) {
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      llama->forward( prompt_tokens_batch[i], prompt_ids_batch );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
