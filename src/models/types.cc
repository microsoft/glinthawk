#include "types.hh"

#include <glog/logging.h>

using namespace std;

namespace glinthawk {

namespace {

constexpr bool ENABLE_DATA_BUFFER_POOL = true;
constexpr size_t MIN_BUFFER_SIZE_POOLED = 8 * 1024; // 8 KiB

}

size_t DataTypeSize( const DataType dtype )
{
  switch ( dtype ) {
    case DataType::Float16: return 2;
    case DataType::Float32: return 4;
  }

  throw std::runtime_error( "Unknown DataType" );
}

DataBufferPool DataBuffer::pool_ {};

void DataBufferDeleter::operator()( uint8_t* ptr ) const
{
  if ( !ptr ) {
    return;
  }

  if ( ENABLE_DATA_BUFFER_POOL && pool_ && buffer_len_ >= MIN_BUFFER_SIZE_POOLED ) {
    pool_->release( ptr, buffer_len_ );
  } else {
    delete[] ptr;
  }
}

void DataBufferDeleter::set_buffer_pool( DataBufferPool* pool, const size_t len )
{
  if ( !ENABLE_DATA_BUFFER_POOL ) {
    return;
  }

  this->pool_ = pool;
  this->buffer_len_ = len;
}

DataBufferPool::PtrType DataBufferPool::get( const size_t n )
{
  if ( n == 0 ) {
    return nullptr;
  }

  if ( !ENABLE_DATA_BUFFER_POOL || n < MIN_BUFFER_SIZE_POOLED ) {
    // small buffers are not pooled
    return PtrType { new uint8_t[n], DataBufferDeleter() };
  }

  lock_guard<mutex> lock { mutex_ };
  PtrType result;

  // do we have a buffer already allocated in the pool?
  auto it = unused_buffers_.find( n );
  if ( it == unused_buffers_.end() or it->second.empty() ) {
    result = PtrType { new uint8_t[n], DataBufferDeleter() };
  } else {
    // return the one from the pool
    reused_bytes_ += n;
    reused_count_++;
    result = move( it->second.front() );
    it->second.pop();
  }

  result.get_deleter().set_buffer_pool( this, n );
  return result;
}

void DataBufferPool::release( uint8_t* ptr, const size_t n )
{
  DCHECK_GE( n, MIN_BUFFER_SIZE_POOLED );

  lock_guard<mutex> lock { mutex_ };
  unused_buffers_[n].push( move( PtrType { ptr, DataBufferDeleter() } ) );
}

void DataBufferPool::print_stats() const
{
  lock_guard<mutex> lock { mutex_ };
  size_t total_unused_bytes = 0;
  size_t total_unused_buffers = 0;

  for ( auto& [k, v] : unused_buffers_ ) {
    total_unused_bytes += k * v.size();
    total_unused_buffers += v.size();
  }

  LOG( INFO ) << "DataBufferPool: " << total_unused_buffers << " buffers, " << total_unused_bytes << " bytes";
  LOG( INFO ) << "DataBufferPool: " << reused_count_ << " reused buffers, " << reused_bytes_ << " bytes";
}

} // namespace glinthawk

ostream& operator<<( ostream& os, const glinthawk::DataType& v )
{
  switch ( v ) {
    case glinthawk::DataType::Float16: os << "FP16"; break;
    case glinthawk::DataType::Float32: os << "FP32"; break;
  }
  return os;
}

ostream& operator<<( ostream& os, const glinthawk::DataBuffer& v )
{
  os << "DataBuffer{}.len=" << v.len() << " bytes";
  return os;
}

std::ostream& operator<<( std::ostream& os, const glinthawk::models::InferenceStage& v )
{
  switch ( v ) {
    case glinthawk::models::InferenceStage::PreAttention: os << "Pre"; break;
    case glinthawk::models::InferenceStage::Attention: os << "Att"; break;
    case glinthawk::models::InferenceStage::PostAttention: os << "Post"; break;
    case glinthawk::models::InferenceStage::Classification: os << "Cls"; break;
    default: os << "Unknown"; break;
  }

  return os;
}
