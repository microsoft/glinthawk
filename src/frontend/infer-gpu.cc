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

    const int max_batch_size = atoi(argv[3]);
    const int conc_size = atoi(argv[4]);
    const int batch_size = atoi(argv[5]);

    const int seq_len = 1024;

    auto llama = models::llama2::cuda::Llama2<__half>::load( model_dir_path, 0, -1, max_batch_size, conc_size );
    models::llama2::Vocabulary vocabulary { tokenizer_path };

    using Llama2 = models::llama2::cuda::Llama2<__half>;

    vector<vector<uint32_t>> prompt_tokens_batch;
    for (size_t i = 0; i < prompt_tokens.size(); i++)
      prompt_tokens_batch.push_back(vector<uint32_t>(batch_size, prompt_tokens[i]));

    vector<uint32_t> prompt_ids_batch;
    for (int i = 0; i < batch_size; i++)
      prompt_ids_batch.push_back((i * max_batch_size) / batch_size);

    vector<vector<uint32_t>> token_pos_batch;
    for (size_t i = 0; i < seq_len; i++)
      token_pos_batch.push_back(vector<uint32_t>(batch_size, i));

    size_t i = 0;
    for ( vector<uint32_t> token = prompt_tokens_batch[0] /* BOS */; token[0] != 2 /* EOS */ && i < seq_len; i++) {
      if ( i < prompt_tokens_batch.size() ) {
        token = prompt_tokens_batch[i];
      }

      cout << vocabulary.get_word( token[0] ) << flush;
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      token = llama -> forward( token, prompt_ids_batch, token_pos_batch[i] );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
