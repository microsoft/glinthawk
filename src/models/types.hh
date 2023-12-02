#pragma once

#include <memory>

#include "util/digest.hh"

namespace glinthawk {

using PromptID = glinthawk::util::digest::SHA256Hash;
using ModelID = uint32_t;

enum class DataType : uint8_t
{
  Float16 = 0,
  Float32 = 1
};

size_t DataTypeSize( const DataType dtype );

/* note: DataBuffer is always on the host */
struct DataBuffer
{
private:
  std::unique_ptr<uint8_t[]> ptr_ { nullptr };

  /// Length of the buffer in bytes
  uint64_t len_ { 0 };

public:
  DataBuffer() = default;

  DataBuffer( const DataBuffer& ) = delete;
  DataBuffer& operator=( const DataBuffer& ) = delete;

  DataBuffer( DataBuffer&& other )
    : ptr_( std::move( other.ptr_ ) )
    , len_( other.len_ )
  {
    other.len_ = 0;
  }

  DataBuffer& operator=( DataBuffer&& other )
  {
    ptr_ = std::move( other.ptr_ );
    len_ = other.len_;
    other.len_ = 0;
    return *this;
  }

  DataBuffer( const size_t n )
    : ptr_( std::make_unique<uint8_t[]>( n ) )
    , len_( n )
  {
  }

  DataBuffer( std::unique_ptr<uint8_t[]>&& other_ptr, const uint64_t other_len )
    : ptr_( std::move( other_ptr ) )
    , len_( other_len )
  {
  }

  uint64_t len() const { return len_; }
  uint8_t* data() { return ptr_.get(); }
  const uint8_t* data() const { return ptr_.get(); }
};

/*
  Definition of host is where the networking stack is running and device is where the compute kernel is running.
  For example, in the case of a GPU, host is the CPU and device is the GPU.
  For example, in the case of a CPU, host is the CPU and device is the CPU.
 */
enum class CopyType
{
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  HostToHost
};

} // namespace glinthawk
