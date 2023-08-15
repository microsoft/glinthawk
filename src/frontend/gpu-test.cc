#include <csignal>
#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "cuda/llama2.cuh"
#include "util/timer.hh"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::gpu;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 ) { cout << "Usage: " << argv0 << " <tokenizer_path> <weights_path>" << endl; }

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
    const filesystem::path tokenizer_path { argv[1] };
    const filesystem::path weights_path { argv[2] };

    Llama2 llama { tokenizer_path, weights_path };

    for ( pair<int, string> token { 1, "<s>" } /* BOS */; token.first != 2 /* EOS */; ) {
      cout << token.second << flush;
      GlobalScopeTimer<Timer::Category::TokenGeneration> _;
      token = llama.forward( token.first );
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
