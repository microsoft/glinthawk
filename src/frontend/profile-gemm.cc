#include <filesystem>
#include <iostream>
#include <tuple>

#include <glog/logging.h>

#include "arch/platform_macros.hh"
#include "profile/gemm_profiler.hh"

using namespace std;
using namespace glinthawk;

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0 << " <dim1> <dim2> <dim3> <duration_s> <output_log>" << endl;
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

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const size_t dim1_ = atoi( argv[1] );
    const size_t dim2_ = atoi( argv[2] );
    const size_t dim3_ = atoi( argv[3] );
    const uint64_t duration_s = atoi( argv[4] );
    const filesystem::path log_path { argv[5] };

    using OperationsType = models::common::_GLINTHAWK_ARCH_NS_::Operations<_GLINTHAWK_DTYPE_>;
    GEMMProfiler<OperationsType, _GLINTHAWK_DTYPE_> profiler_gemm { log_path, dim1_, dim2_, dim3_, duration_s };
    profiler_gemm.run_in_thread();
    profiler_gemm.wait();

  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
