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

struct DataBuffer
{
private:
  std::unique_ptr<uint8_t[]> ptr_ { nullptr };

  /// Length of the buffer in bytes
  uint64_t len_ { 0 };

public:
  DataBuffer() = default;

  DataBuffer( const size_t n )
    : ptr_( std::make_unique<uint8_t[]>( n ) )
    , len_( n )
  {
  }

  template<typename T>
  DataBuffer( const size_t n, const T* data )
    : ptr_( std::make_unique<uint8_t[]>( n ) )
    , len_( n )
  {
    memcpy( ptr_.get(), data, n );
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

enum class CopyType
{
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  HostToHost
};

} // namespace glinthawk
