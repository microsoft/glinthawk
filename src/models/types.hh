#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <queue>

#include "util/digest.hh"

namespace glinthawk {

using HashID = glinthawk::util::digest::SHA256Hash;
using PromptID = glinthawk::util::digest::SHA256Hash;
using ContextID = glinthawk::util::digest::SHA256Hash;
using RouteID = uint32_t;
using ModelID = uint32_t;

ContextID NULL_CONTEXT {};

enum class DataType : uint8_t
{
  Float16 = 0,
  Float32 = 1
};

namespace models {

enum class InferenceStage : uint8_t
{
  PreAttention,
  Attention,
  PostAttention,
  Classification,
  __COUNT__,
};

} // namespace glinthawk::models

size_t DataTypeSize( const DataType dtype );

class DataBufferPool;

class DataBufferDeleter
{
private:
  DataBufferPool* pool_ { nullptr };
  size_t buffer_len_ { 0 };

public:
  constexpr DataBufferDeleter() = default;
  void operator()( uint8_t* ptr ) const;
  void set_buffer_pool( DataBufferPool* pool, const size_t len );
};

class DataBufferPool
{
private:
  using PtrType = std::unique_ptr<uint8_t[], DataBufferDeleter>;

  mutable std::mutex mutex_ {};
  std::map<size_t, std::queue<PtrType>> unused_buffers_ {};

  size_t reused_count_ { 0 };
  size_t reused_bytes_ { 0 };

public:
  DataBufferPool() = default;
  ~DataBufferPool() { print_stats(); }

  PtrType get( const size_t n );
  void release( uint8_t* ptr, const size_t n );
  void print_stats() const;
};

/* note: DataBuffer is always on the host */
struct DataBuffer
{
private:
  static DataBufferPool pool_;
  std::unique_ptr<uint8_t[], DataBufferDeleter> ptr_ {};

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
    : ptr_( pool_.get( n ) )
    , len_( n )
  {
  }

  DataBuffer( std::unique_ptr<uint8_t[]>&& other_ptr, const uint64_t other_len )
    : ptr_( other_ptr.release(), DataBufferDeleter() )
    , len_( other_len )
  {
    ptr_.get_deleter().set_buffer_pool( &pool_, other_len );
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

std::ostream& operator<<( std::ostream& os, const glinthawk::DataType& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::DataBuffer& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::models::InferenceStage& v );
