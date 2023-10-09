#ifndef GLINTHAWK_CUDA_ENABLED
#error "This file should only be compiled when CUDA is enabled."
#endif

#include <csignal>
#include <iostream>

#include <cuda_fp16.h>
#include <glog/logging.h>

#include "compute/kernel.hh"
#include "models/llama2/cuda/model.cuh"
#include "worker/worker.hh"

#include "oof/oof.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_dir_path> <tokenizer_path> <start_layer> <end_layer>" << endl;
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

  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
