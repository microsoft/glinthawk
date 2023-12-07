#include <csignal>
#include <filesystem>
#include <iostream>

#include <cuda_fp16.h>
#include <glog/logging.h>

#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

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
  google::InitGoogleLogging( argv[0] );

  try {
    const size_t num_tries = atoi( argv[1] );
    const size_t dim = atoi( argv[2] );

    uint8_t* cuda_pointer;
    cudaMalloc( reinterpret_cast<void**>( &cuda_pointer ), dim * num_tries * sizeof( uint8_t ) );

    uint8_t* cpu_pointer;
    cudaMallocHost( reinterpret_cast<void**>( &cpu_pointer ), dim * num_tries * sizeof( uint8_t ) );

    for ( size_t i = 0; i < num_tries * dim; i++ )
      cpu_pointer[i] = i;

    for ( size_t i = 0; i < num_tries; i++ ) {
      GlobalScopeTimer<Timer::Category::CopyToGPU> _;

      cudaMemcpy( cuda_pointer + i * dim, cpu_pointer + i * dim, dim * sizeof( uint8_t ), cudaMemcpyHostToDevice );
    }

    for ( size_t i = 0; i < num_tries * dim; i++ )
      cpu_pointer[i] = 0;

    for ( size_t i = 0; i < num_tries; i++ ) {
      GlobalScopeTimer<Timer::Category::CopyFromGPU> _;

      cudaMemcpy( cpu_pointer + i * dim, cuda_pointer + i * dim, dim * sizeof( uint8_t ), cudaMemcpyDeviceToHost );
    }

    for ( size_t i = 0; i < num_tries * dim; i++ )
      CHECK_EQ( cpu_pointer[i], static_cast<uint8_t>( i ) ) << "Copying was unsuccessful.";

    cudaStream_t stream_up;
    cudaStreamCreate( &stream_up );
    cudaStream_t stream_down;
    cudaStreamCreate( &stream_down );

    for ( size_t i = 0; i < num_tries; i++ ) {
      GlobalScopeTimer<Timer::Category::ConcurrentCopyGPU> _;
      cudaMemcpyAsync(
        cuda_pointer + i * dim, cpu_pointer + i * dim, dim * sizeof( uint8_t ), cudaMemcpyHostToDevice, stream_up );
      cudaMemcpyAsync( cpu_pointer + ( i + 1 ) % num_tries * dim,
                       cuda_pointer + ( i + 1 ) % num_tries * dim,
                       dim * sizeof( uint8_t ),
                       cudaMemcpyDeviceToHost,
                       stream_down );
      cudaDeviceSynchronize();
    }

    cudaStreamDestroy( stream_up );
    cudaStreamDestroy( stream_down );
    cudaFree( cuda_pointer );
    cudaFreeHost( cpu_pointer );

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
