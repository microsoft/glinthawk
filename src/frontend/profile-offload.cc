#include <csignal>
#include <filesystem>
#include <iostream>

#include "models/common/cuda/ops.cuh"
#include "util/timer.hh"
#include <cuda_fp16.h>
#include <glog/logging.h>

using namespace std;
using namespace glinthawk;
using namespace glinthawk::models::common::cuda;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 ) { cout << "Usage: " << argv0 << " num_tries dim" << endl; }

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
    const size_t num_tries = atoi( argv[1] );
    const size_t dim = atoi( argv[2] );

    uint8_t* cuda_pointer;
    ops::CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>( &cuda_pointer ), dim * num_tries * sizeof( uint8_t ) ) );

    uint8_t* cpu_pointer = static_cast<uint8_t*>( malloc( dim * num_tries * sizeof( uint8_t ) ) );

    for ( size_t i = 0; i < num_tries * dim; i++ )
      cpu_pointer[i] = i;

    for ( size_t i = 0; i < num_tries; i++ ) {
      GlobalScopeTimer<Timer::Category::CopyToGPU> _;
      ops::CHECK_CUDA(
        cudaMemcpy( cuda_pointer + i * dim, cpu_pointer + i * dim, dim * sizeof( uint8_t ), cudaMemcpyHostToDevice ) );
    }

    for ( size_t i = 0; i < num_tries * dim; i++ )
      cpu_pointer[i] = 0;

    for ( size_t i = 0; i < num_tries; i++ ) {
      GlobalScopeTimer<Timer::Category::CopyFromGPU> _;
      ops::CHECK_CUDA(
        cudaMemcpy( cpu_pointer + i * dim, cuda_pointer + i * dim, dim * sizeof( uint8_t ), cudaMemcpyDeviceToHost ) );
    }

    for ( size_t i = 0; i < num_tries * dim; i++ )
      CHECK_EQ( cpu_pointer[i], static_cast<uint8_t>( i ) ) << "Copying was unsuccessful.";

    cudaFree( cuda_pointer );
    free( cpu_pointer );

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
