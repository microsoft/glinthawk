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

void usage( const char* argv0 ) { cout << "Usage: " << argv0 << " <model_dir_path> <tokenizer_path>" << endl; }

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 5 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  // TODO:
  // 1. Figure out batch with non-contiguous prompts
  // 2. Profile batches with 1 layer copied 32 times
  // 3. Change 70B model to glint format

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  FLAGS_log_year_in_prefix = false;
  FLAGS_timestamp_in_logfile_name = false;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir_path { argv[1] };
    const filesystem::path tokenizer_path { argv[2] };

    const int max_batch_size = atoi(argv[3]);
    const int batch_size = atoi(argv[4]);

    auto llama = models::llama2::cuda::Llama2<__half>::load( model_dir_path, 0, -1, max_batch_size );
   /////////////////////////////////////////////////////////// profile batching //////////////////////////////////////////////////////////////
    // auto llama = models::llama2::cuda::Llama2<__half>::load( model_dir_path, 0, 0, batch_size );
   /////////////////////////////////////////////////////////// profile batching //////////////////////////////////////////////////////////////
    models::llama2::Vocabulary vocabulary { tokenizer_path };

    vector<uint32_t> prompt_tokens { 1,   518,  25580, 29962, 25538, 2211,  25562, 363,  7952,
                                     292, 9045, 29891, 29889, 518,   29914, 25580, 29962 };

    vector<vector<uint32_t>> prompt_tokens_batch;
    for (size_t i = 0; i < prompt_tokens.size(); i++)
      prompt_tokens_batch.push_back(vector<uint32_t>(batch_size, prompt_tokens[i]));

    vector<uint32_t> prompt_ids_batch;
    for (int i = 0; i < batch_size; i++)
      prompt_ids_batch.push_back((i * max_batch_size) / batch_size);

    size_t i = 0;

    for ( vector<uint32_t> token = prompt_tokens_batch[0] /* BOS */; token[0] != 2 /* EOS */; ) {
      if ( i < prompt_tokens_batch.size() ) {
        token = prompt_tokens_batch[i];
        i++;
      }

      cout << vocabulary.get_word( token[0] ) << flush;
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      token = llama -> forward( token, prompt_ids_batch );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
