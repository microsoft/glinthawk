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
  cout << "Usage: " << argv0 << " <model_dir_path>"
       << " <listen_ip> <listen_port>"
       << " <coordinator_ip> <coordinator_port>" << endl;
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
  google::InitGoogleLogging( argv[0] );

  const filesystem::path model_path { argv[1] };
  const string listen_ip { argv[2] };
  const uint16_t listen_port = static_cast<uint16_t>( stoi( argv[3] ) );
  const string coordinator_ip { argv[4] };
  const uint16_t coordinator_port = static_cast<uint16_t>( stoi( argv[5] ) );

  using Llama2 = models::llama2::cuda::Llama2<__half>;

  try {
    net::Address listen_addr { listen_ip, listen_port };
    net::Address coordinator_addr { coordinator_ip, coordinator_port };
    core::Worker<Llama2> worker { listen_addr, coordinator_addr, model_path };

    worker.run();
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
