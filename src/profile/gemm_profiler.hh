#pragma once

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <typeinfo>
#include <vector>

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
#include "arch/amd64/common/ops.hh"
#endif

#if defined( TARGET_PLATFORM_CUDA )
#include "arch/cuda/common/ops.cuh"
#endif

#include "models/common/ops/concept.hh"

namespace glinthawk {

template<typename CommonOperations, typename DType>
requires models::common::OperationsConcept<CommonOperations, DType>
class GEMMProfiler
{
public:
  using Operations = CommonOperations;

private:
  static constexpr size_t REPEATS = 128;

#if defined( TARGET_PLATFORM_AMD64 )
  Operations ops_ {};
#endif

#if defined( TARGET_PLATFORM_CUDA )
  Operations ops_ { 0, 0, false, false, 0 };
#endif

  const size_t dim1_, dim2_, dim3_;

  // in_buffer_:      [( REPEATS * batch ) x dim1]
  // static_buffer_:  [ dim1 x dim2 ]
  // out_buffer_:  [ ( REPEATS * batch ) x dim2 ]
  const size_t in_size_ = dim1_ * dim2_;
  const size_t stat_size_ = dim3_ * dim2_;
  const size_t out_size_ = dim1_ * dim3_;

  typename Operations::DeviceUniquePtr in_buffer_, static_buffer_, out_buffer_;

  const std::chrono::seconds duration_;
  std::thread thread_ {};

  const std::filesystem::path log_path_;
  std::ofstream lout_ { log_path_, std::ios::out | std::ios::trunc };

public:
  GEMMProfiler( const std::filesystem::path& log_path,
                const size_t dim_a,
                const size_t dim_b,
                const size_t dim_c,
                const size_t duration_s )
    : dim1_( dim_a )
    , dim2_( dim_b )
    , dim3_( dim_c )
    , in_buffer_( ops_.device_allocate( REPEATS * in_size_ * sizeof( DType ) ) )
    , static_buffer_( ops_.device_allocate( stat_size_ * sizeof( DType ) ) )
    , out_buffer_( ops_.device_allocate( REPEATS * out_size_ * sizeof( DType ) ) )
    , duration_( duration_s )
    , log_path_( log_path )
  {

    ops_.randomize_device_buffer( in_buffer_.get(), REPEATS * in_size_, -10.0 / sqrtf( dim2_ ), 10.0 / sqrtf( dim2_ ) );

    ops_.randomize_device_buffer( static_buffer_.get(), stat_size_, -10.0 / sqrtf( dim2_ ), 10.0 / sqrtf( dim2_ ) );

    ops_.randomize_device_buffer(
      out_buffer_.get(), REPEATS * out_size_, -10.0 / sqrtf( dim2_ ), 10.0 / sqrtf( dim2_ ) );

    lout_ << "# " << "dim1='" << dim1_ << "', dim2=" << dim2_ << ", dim3=" << dim3_ << ", duration_s=" << duration_s
          << '\n';

    lout_ << "repeat,timestamp_ms,duration_us\n";

    prep_ops();
  }

  void prep_ops()
  {
#if defined( TARGET_PLATFORM_CUDA )
    cudaDeviceSynchronize();
#endif
  }

  void run()
  {
    const auto end_time = std::chrono::steady_clock::now() + duration_;

    const DType* static_w = static_buffer_.get();

    for ( size_t r = 0;; r++ ) {
      const DType* in_w = in_buffer_.get() + ( r % REPEATS ) * in_size_;
      DType* out_w = out_buffer_.get() + ( r % REPEATS ) * out_size_;

      const auto now = std::chrono::system_clock::now();
      const auto start = std::chrono::steady_clock::now();

      prep_ops();
      ops_.untemplated_matmul( out_w, in_w, static_w, dim2_, dim3_, dim1_ );
      prep_ops();

      const auto end = std::chrono::steady_clock::now();
      const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
      lout_ << r << "," << std::chrono::duration_cast<std::chrono::milliseconds>( now.time_since_epoch() ).count()
            << "," << duration << '\n';

      if ( end >= end_time ) {
        break;
      }
    }
  }

  void run_in_thread()
  {
    if ( thread_.joinable() ) {
      LOG( FATAL ) << "GEMMProfiler thread is already running";
    }

    thread_ = std::thread( &GEMMProfiler::run, this );
  }

  void wait()
  {
    if ( thread_.joinable() ) {
      thread_.join();
    }
  }
};

} // namespace glinthawk
