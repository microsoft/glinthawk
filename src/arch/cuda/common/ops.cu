#include "ops.cuh"

namespace glinthawk::models::common::cuda {

void CHECK_CUBLAS( const cublasStatus_t err, const std::source_location location )
{
  if ( err != CUBLAS_STATUS_SUCCESS ) {
    LOG( FATAL ) << "CUBLAS error " << cublasGetStatusName( err ) << ": " << cublasGetStatusString( err ) << " ("
                 << location.file_name() << ":" << std::to_string( location.line() ) << ")";
  }
}

void CHECK_CUDA( const cudaError_t err, const std::source_location location )
{
  if ( err != cudaSuccess ) {
    LOG( FATAL ) << "CUDA error " << cudaGetErrorName( err ) << ": " << cudaGetErrorString( err ) << " ("
                 << location.file_name() << ":" << std::to_string( location.line() ) << ")";
  }
}

}
