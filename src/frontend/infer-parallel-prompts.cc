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

  if ( argc != 4 ) {
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

    vector<uint32_t> prompt_tokens { 1,   518,  25580, 29962, 25538, 2211,  25562, 363,  7952,
                                     292, 9045, 29891, 29889, 518,   29914, 25580, 29962 };

    const unsigned int batch_size = atoi(argv[3]);
    const unsigned int max_batch_size = batch_size;
    const unsigned int conc_size = batch_size + prompt_tokens.size() - 1;

    const unsigned int seq_len = 1024;

    auto llama = models::llama2::cuda::Llama2<__half>::load( model_dir_path, 0, -1, max_batch_size, conc_size );
    models::llama2::Vocabulary vocabulary { tokenizer_path };

    vector<vector<uint32_t>> prompt_tokens_batch;
    for (unsigned int i = 0; i < prompt_tokens.size(); i++){
      vector<uint32_t> new_vec;
      if (i == 0){
        new_vec.insert(new_vec.end(), prompt_tokens.begin(), prompt_tokens.end());
      } else {
        new_vec.push_back(-1);
      }
      for (unsigned int j = 0; j < batch_size - 1; j++){
        new_vec.push_back(prompt_tokens[i]);
      }
      prompt_tokens_batch.push_back(new_vec);
    }

    vector<vector<uint32_t>> prompt_ids_batch;
    for (unsigned int i = 0; i < seq_len-prompt_tokens.size(); i++){
      vector<uint32_t> new_vec;
      if (i == 0){
        new_vec = vector<uint32_t>(prompt_tokens.size(), 0);
      } else {
        new_vec.push_back(0);
      }
      for (unsigned int j = 0; j < batch_size - 1; j++){
        new_vec.push_back(j+1);
      }
      prompt_ids_batch.push_back(new_vec);
    }

    vector<vector<uint32_t>> token_pos_batch;
    for (unsigned int i = 0; i < seq_len-prompt_tokens.size(); i++){
      vector<uint32_t> new_vec;
      if (i == 0){
        for (unsigned int j = 0; j < prompt_tokens.size(); j++){
          new_vec.push_back(j);
        }
      } else {
        new_vec.push_back(i+prompt_tokens.size()-1);
      }
      for (unsigned int j = 0; j < batch_size - 1; j++){
        new_vec.push_back(i);
      }
      token_pos_batch.push_back(new_vec);
    }

    vector<vector<uint32_t>> response_tokens(batch_size);

    vector<uint32_t> token;
    for ( size_t i = 0; i < prompt_ids_batch.size(); i++) {
      if ( i < prompt_tokens_batch.size() ) {
        token = prompt_tokens_batch[i];
      }
      for (unsigned int j = 0; j < token.size(); j++){
        response_tokens[prompt_ids_batch[i][j]].push_back(token[j]);
      }
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      token = llama -> forward( token, prompt_ids_batch[i], token_pos_batch[i] );
      if ( i == 0 ) {
        prompt_tokens_batch[1][0] = token[prompt_tokens.size()-1];
      } else if ( i + 1 < prompt_tokens_batch.size()) {
        prompt_tokens_batch[i+1][0] = token[0];
      }
    }

    for ( size_t i = 0; i < response_tokens.size(); i++) {
      cout << "Prompt " << i << ":" << endl;
      for ( uint32_t tok: response_tokens[i] )
        cout << vocabulary.get_word( tok ) << flush;
      cout << endl << endl;
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
