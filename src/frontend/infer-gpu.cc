#include <csignal>
#include <filesystem>
#include <iostream>

#include <glog/logging.h>
#include <cuda_fp16.h>

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
  cout << "Usage: " << argv0 << " <model_dir_path> <tokenizer_path>" << endl;
}

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

    auto llama = models::llama2::cuda::Llama2<__half>::load( model_dir_path );
    models::llama2::Vocabulary vocabulary { tokenizer_path };

    std::vector<int> prompt_tokens = std::vector<int>{ 1, 518, 25580, 29962, 25538, 2211, 25562, 363, 7952, 292, 9045, 29891, 29889, 518, 29914, 25580, 29962 };
    size_t i = 0;

    for ( int token = prompt_tokens[0] /* BOS */; token != 2 /* EOS */; ) {
      if (i < prompt_tokens.size()){
        token = prompt_tokens[i];
        i++;
      }
      cout << vocabulary.get_word( token ) << flush;
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      token = llama.forward( token );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
