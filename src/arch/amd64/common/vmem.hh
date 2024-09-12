#pragma once

#include <cstring>
#include <fcntl.h>
#include <glog/logging.h>
#include <list>
#include <memory>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>

#include "util/ring_buffer.hh"

namespace glinthawk::models::common::amd64 {

class PhysicalMemoryRegion
{
private:
  size_t len_;
  std::optional<MMap_Region> region_ {};

public:
  PhysicalMemoryRegion( const size_t len )
    : len_( len )
  {
  }

  // disallow copy & move
  PhysicalMemoryRegion( const PhysicalMemoryRegion& ) = delete;
  PhysicalMemoryRegion& operator=( const PhysicalMemoryRegion& ) = delete;
  PhysicalMemoryRegion( PhysicalMemoryRegion&& other ) = delete;
  PhysicalMemoryRegion& operator=( PhysicalMemoryRegion&& other ) = delete;

  void map_to( void* dptr )
  {
    region_.emplace(
      reinterpret_cast<char*>( dptr ), len_, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );
  }

  ~PhysicalMemoryRegion() {}
};

class VirtualMemoryRegion
{
private:
  MMap_Region virtual_region_;
  std::list<PhysicalMemoryRegion> physical_regions_ {};

public:
  VirtualMemoryRegion( const size_t len )
    : virtual_region_( nullptr, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0 )
  {
  }

  ~VirtualMemoryRegion() {}

  // disallow copy
  VirtualMemoryRegion( const VirtualMemoryRegion& ) = delete;
  VirtualMemoryRegion& operator=( const VirtualMemoryRegion& ) = delete;

  // allow move
  VirtualMemoryRegion( VirtualMemoryRegion&& other ) = default;
  VirtualMemoryRegion& operator=( VirtualMemoryRegion&& other ) = default;

  void allocate_span( const void* ptr, const size_t len )
  {
    if ( reinterpret_cast<const uint8_t*>( ptr ) + len > reinterpret_cast<const uint8_t*>( this->ptr() ) + max_len() ) {
      LOG( FATAL ) << "Out of bounds allocation";
    }

    physical_regions_.emplace_back( len );
    physical_regions_.back().map_to( const_cast<void*>( ptr ) );
  }

  void* ptr() const { return virtual_region_.addr(); }
  size_t max_len() const { return virtual_region_.length(); }

  void clear() { physical_regions_.clear(); }
};

} // namespace glinthawk::models::common::amd64
