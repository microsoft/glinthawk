#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <list>
#include <memory>

namespace glinthawk::models::common::cuda {

class PhysicalMemoryRegion
{
private:
  CUmemGenericAllocationHandle alloc_handle_ {};
  size_t len_ { 0 };
  size_t padded_len_ { 0 };

  CUdeviceptr mapped_to_ {};
  CUdevice device_ {};

public:
  PhysicalMemoryRegion( CUdevice device, const size_t len )
    : len_( len )
    , device_( device )
  {
    size_t allocation_granularity;

    CUmemAllocationProp alloc_props = {};
    alloc_props.location.id = 0;
    alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    cuMemGetAllocationGranularity( &allocation_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_MINIMUM );
    CHECK( len % allocation_granularity == 0 )
      << "Length must be a multiple of the allocation granularity (" << allocation_granularity << " bytes)";

    CHECK_EQ( CUDA_SUCCESS, cuMemCreate( &alloc_handle_, len_, &alloc_props, 0 ) );
  }

  // disallow copy & move
  PhysicalMemoryRegion( const PhysicalMemoryRegion& ) = delete;
  PhysicalMemoryRegion& operator=( const PhysicalMemoryRegion& ) = delete;
  PhysicalMemoryRegion( PhysicalMemoryRegion&& other ) = delete;
  PhysicalMemoryRegion& operator=( PhysicalMemoryRegion&& other ) = delete;

  void map_to( CUdeviceptr dptr )
  {
    if ( mapped_to_ ) {
      // each physical region can only be mapped to one virtual region at a time
      CHECK_EQ( CUDA_SUCCESS, cuMemUnmap( mapped_to_, len_ ) );
    }

    mapped_to_ = dptr;

    CUmemAccessDesc access_desc = {};
    access_desc.location.id = device_;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CHECK_EQ( CUDA_SUCCESS, cuMemMap( mapped_to_, len_, 0, alloc_handle_, 0 ) );
    CHECK_EQ( CUDA_SUCCESS, cuMemSetAccess( mapped_to_, len_, &access_desc, 1 ) );
  }

  ~PhysicalMemoryRegion()
  {
    if ( mapped_to_ ) {
      CHECK_EQ( CUDA_SUCCESS, cuMemUnmap( mapped_to_, len_ ) );
      mapped_to_ = {};
    }

    if ( len_ ) {
      cuMemRelease( alloc_handle_ );
    }
  }
};

class VirtualMemoryRegion
{
private:
  CUdeviceptr dptr_ {};
  CUdevice device_ {};
  CUcontext context_ {};
  size_t max_len_ { 0 };

  std::list<PhysicalMemoryRegion> physical_regions_ {};

public:
  VirtualMemoryRegion( const size_t len )
    : max_len_( len )
  {
    CHECK_EQ( CUDA_SUCCESS, cuInit( 0 ) );
    CHECK_EQ( CUDA_SUCCESS, cuDevicePrimaryCtxRetain( &context_, 0 ) );
    CHECK_EQ( CUDA_SUCCESS, cuCtxSetCurrent( context_ ) );
    CHECK_EQ( CUDA_SUCCESS, cuCtxGetDevice( &device_ ) );
    CHECK_EQ( CUDA_SUCCESS, cuMemAddressReserve( &dptr_, max_len_, 0, 0, 0 ) );
  }

  ~VirtualMemoryRegion()
  {
    physical_regions_.clear();

    if ( dptr_ ) {
      CHECK_EQ( cuMemAddressFree( dptr_, max_len_ ), CUDA_SUCCESS );
      CHECK_EQ( CUDA_SUCCESS, cuDevicePrimaryCtxRelease( device_ ) );
    }
  }

  // disallow copy
  VirtualMemoryRegion( const VirtualMemoryRegion& ) = delete;
  VirtualMemoryRegion& operator=( const VirtualMemoryRegion& ) = delete;

  VirtualMemoryRegion( VirtualMemoryRegion&& other )
  {
    dptr_ = other.dptr_;
    device_ = other.device_;
    context_ = other.context_;
    max_len_ = other.max_len_;
    physical_regions_ = std::move( other.physical_regions_ );

    other.dptr_ = {};
    other.device_ = {};
    other.context_ = {};
    other.max_len_ = 0;
  }

  VirtualMemoryRegion& operator=( VirtualMemoryRegion&& other )
  {
    if ( this != &other ) {
      dptr_ = other.dptr_;
      device_ = other.device_;
      context_ = other.context_;
      max_len_ = other.max_len_;
      physical_regions_ = std::move( other.physical_regions_ );

      other.dptr_ = {};
      other.device_ = {};
      other.context_ = {};
      other.max_len_ = 0;
    }

    return *this;
  }

  void allocate_span( const void* ptr, const size_t len )
  {
    if ( reinterpret_cast<const uint8_t*>( ptr ) + len > reinterpret_cast<const uint8_t*>( dptr_ ) + max_len_ ) {
      LOG( FATAL ) << "Out of bounds allocation";
    }

    // Let's create a physical region for this span.
    // NOTE: this function does not check for overlaps with existing physical regions.
    auto& region = physical_regions_.emplace_back( device_, len );
    region.map_to( reinterpret_cast<CUdeviceptr>( ptr ) );
  }

  void* ptr() const { return reinterpret_cast<void*>( dptr_ ); }
  size_t max_len() const { return max_len_; }

  void clear() { physical_regions_.clear(); }
};

} // namespace glinthawk::models::common::cuda
