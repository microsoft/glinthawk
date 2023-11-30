#include "cuda.cuh"

#include <glog/logging.h>

using namespace std;
using namespace glinthawk::models::common::cuda;

void CHECK_CUBLAS( const cublasStatus_t err, const source_location location )
{
  if ( err != CUBLAS_STATUS_SUCCESS ) {
    LOG( FATAL ) << "CUBLAS error "s << cublasGetStatusName( err ) << ": " << cublasGetStatusString( err ) << " ("
                 << location.file_name() << ":" << to_string( location.line() ) << ")";
  }
}

void CHECK_CUDA( const cudaError_t err, const source_location location )
{
  if ( err != cudaSuccess ) {
    LOG( FATAL ) << "CUDA error " << string( cudaGetErrorName( err ) ) << ": " << string( cudaGetErrorString( err ) )
                 << " (" << location.file_name() << ":" << to_string( location.line() ) << ")";
  }
}
