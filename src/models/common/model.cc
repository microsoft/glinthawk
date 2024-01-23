#include "model.hh"

#include <iostream>
#include <sstream>

#include "chrono"
#include <glog/logging.h>

using namespace std;

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

ostream& operator<<( ostream& os, const glinthawk::models::InferenceState::Stage& v )
{
  using namespace glinthawk::models;

  switch ( v ) {
    case InferenceState::Stage::PreAttention: os << "Pre"; break;
    case InferenceState::Stage::Attention: os << "Att"; break;
    case InferenceState::Stage::PostAttention: os << "Post"; break;
    case InferenceState::Stage::Classification: os << "Cls"; break;
  }
  return os;
}

ostream& operator<<( ostream& os, const glinthawk::models::InferenceState& v )
{
  os << v.to_string();
  return os;
}

namespace glinthawk {

size_t DataTypeSize( const DataType dtype )
{
  switch ( dtype ) {
    case DataType::Float16: return 2;
    case DataType::Float32: return 4;
  }

  throw std::runtime_error( "Unknown DataType" );
}

} // namespace glinthawk

namespace glinthawk::models {

namespace {

template<typename FieldType, typename PtrType>
FieldType _get_and_advance( const PtrType*& ptr )
{
  const auto result = reinterpret_cast<const FieldType*>( ptr );
  ptr = reinterpret_cast<const PtrType*>( reinterpret_cast<const uint8_t*>( ptr ) + sizeof( FieldType ) );
  return *result;
}

template<typename FieldType, typename PtrType>
void _put_and_advance( PtrType*& ptr, const FieldType& field )
{
  memcpy( ptr, &field, sizeof( FieldType ) );
  ptr = reinterpret_cast<PtrType*>( reinterpret_cast<uint8_t*>( ptr ) + sizeof( FieldType ) );
}

} // namespace

InferenceState::InferenceState( const string_view serialized )
{
  auto ptr = serialized.data();
  data_ = Data { serialized.data() };
  ptr += sizeof( data_ );

  // These next 4 lines take 10us on average
  const auto len_data = _get_and_advance<uint64_t>( ptr );
  activations_ = DataBuffer { len_data };
  memcpy( activations_.data(), ptr, len_data );
  ptr += len_data;
}

net::Address InferenceState::next_worker() const
{
  auto it = layer_workers_.find( { data_.next_layer_, data_.next_stage_ } );
  CHECK( it != layer_workers_.end() ) << "No worker found for layer " << data_.next_layer_ << ", stage "
                                      << data_.next_stage_;
  return it->second;
}

void InferenceState::loop_till_next_worker( const uint32_t n_layers )
{
  switch ( data_.next_stage_ ) {
    case InferenceState::Stage::PreAttention: data_.next_stage_ = InferenceState::Stage::Attention; break;
    case InferenceState::Stage::Attention: data_.next_stage_ = InferenceState::Stage::PostAttention; break;
    case InferenceState::Stage::PostAttention:
      if ( data_.next_layer_ == n_layers - 1 ) {
        data_.next_stage_ = InferenceState::Stage::Classification;
      } else {
        data_.next_stage_ = InferenceState::Stage::PreAttention;
        data_.next_layer_++;
      }
      break;
    case InferenceState::Stage::Classification:
      data_.next_stage_ = InferenceState::Stage::PreAttention;
      data_.next_layer_ = 0;
      break;
    default: LOG( FATAL ) << "Invalid stage";
  }
}

void InferenceState::erase_from_workers( const uint32_t next_layer, const Stage next_stage )
{
  layer_workers_.erase( { next_layer, next_stage } );
}

string InferenceState::serialize() const
{
  string result;
  result.resize( serialized_size() );

  auto ptr = result.data();
  memcpy( ptr, &data_, sizeof( data_ ) );
  ptr += sizeof( data_ );

  _put_and_advance( ptr, static_cast<uint64_t>( activations_.len() ) );
  memcpy( ptr, activations_.data(), activations_.len() );

  return result;
}

string InferenceState::to_string() const
{
  ostringstream oss;
  oss << "InferenceState(" << "prompt_id=" << data_.prompt_id_.base58digest().substr( 0, 8 ) << ", "
      << "route_id=" << data_.route_id_.base58digest().substr( 0, 8 ) << ", " << "token=" << data_.token_ << ", "
      << "token_pos=" << data_.token_pos_ << ", " << "next_layer=" << data_.next_layer_ << ", "
      << "next_stage=" << data_.next_stage_ << ", " << "prompt_len=" << data_.prompt_length_ << ", "
      << "temperature=" << data_.temperature_ << ", " << "finished=" << data_.finished_ << ", "
      << "dtype=" << data_.dtype_ << ", " << "activations.len=" << activations_ << ", " << "peers={";

  for ( auto& [layer_stage, address] : layer_workers_ ) {
    oss << " (" << layer_stage.first << "-" << layer_stage.second << " -> " << address.to_string() << ")";
  }

  oss << " })";

  return oss.str();
}

size_t InferenceState::serialized_size() const
{
  return sizeof( data_ )       /* base */
         + sizeof( uint64_t )  /* activations_.len */
         + activations_.len(); /* activations_ data */
}

}
