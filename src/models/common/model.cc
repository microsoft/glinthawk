#include "model.hh"

#include <iostream>
#include <sstream>

#include "chrono"
#include <glog/logging.h>
#include "monitoring/measurement.hh"

using namespace std;
namespace {
  glinthawk::Measurement& __stats__ { glinthawk::global_measurement() };
}

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
  const auto t1 = std::chrono::steady_clock::now().time_since_epoch().count();
  auto ptr = serialized.data();

  prompt_id_ = _get_and_advance<decltype( prompt_id_ )>( ptr );
  model_id_ = _get_and_advance<decltype( model_id_ )>( ptr );
  token_ = _get_and_advance<decltype( token_ )>( ptr );
  token_pos_ = _get_and_advance<decltype( token_pos_ )>( ptr );
  next_layer_ = _get_and_advance<decltype( next_layer_ )>( ptr );
  next_stage_ = _get_and_advance<decltype( next_stage_ )>( ptr );
  prompt_length_ = _get_and_advance<decltype( prompt_length_ )>( ptr );
  temperature_ = _get_and_advance<decltype( temperature_ )>( ptr );
  finished_ = _get_and_advance<decltype( finished_ )>( ptr );
  timestamp_ = _get_and_advance<decltype( timestamp_ )>( ptr );
  loop_start_timestamp_ = _get_and_advance<decltype( loop_start_timestamp_ )>( ptr );
  dtype_ = static_cast<DataType>( _get_and_advance<underlying_type_t<DataType>>( ptr ) );

  const auto t2 = std::chrono::steady_clock::now().time_since_epoch().count();

  const auto len_data = _get_and_advance<uint64_t>( ptr );
  activations_ = DataBuffer { len_data };
  memcpy( activations_.data(), ptr, len_data );
  ptr += len_data;

  const auto t3 = std::chrono::steady_clock::now().time_since_epoch().count();

  const auto num_workers = _get_and_advance<uint32_t>( ptr );

  for ( uint32_t i = 0; i < num_workers; i++ ) {
    const auto layer = _get_and_advance<uint32_t>( ptr );
    const auto stage = _get_and_advance<Stage>( ptr );
    const auto ipv4_numeric = _get_and_advance<uint32_t>( ptr );
    const auto port = _get_and_advance<uint16_t>( ptr );
    layer_workers_.emplace( std::make_pair( layer, stage ), net::Address::from_ipv4_numeric( ipv4_numeric, port ) );
  }
  const auto t4 = std::chrono::steady_clock::now().time_since_epoch().count();
  __stats__.add_point<glinthawk::IntDistributions::DeserializeFirst>( t2-t1 );
  __stats__.add_point<glinthawk::IntDistributions::DeserializeSecond>( t3-t2 );
  __stats__.add_point<glinthawk::IntDistributions::DeserializeThird>( t4-t3 );
}

net::Address InferenceState::next_worker() const
{
  auto it = layer_workers_.find( { next_layer_, next_stage_ } );
  CHECK( it != layer_workers_.end() ) << "No worker found for layer " << next_layer_ << ", stage " << next_stage_;
  return it->second;
}

void InferenceState::loop_till_next_worker( const uint32_t n_layers )
{
  while ( layer_workers_.find( { next_layer_, next_stage_ } ) == layer_workers_.end() and layer_workers_.size() > 0 ) {
    switch ( next_stage_ ) {
      case InferenceState::Stage::PreAttention: next_stage_ = InferenceState::Stage::Attention; break;
      case InferenceState::Stage::Attention: next_stage_ = InferenceState::Stage::PostAttention; break;
      case InferenceState::Stage::PostAttention:
        if ( next_layer_ == n_layers - 1 ) {
          next_stage_ = InferenceState::Stage::Classification;
        } else {
          next_stage_ = InferenceState::Stage::PreAttention;
          next_layer_++;
        }
        break;
      case InferenceState::Stage::Classification:
        next_stage_ = InferenceState::Stage::PreAttention;
        next_layer_ = 0;
        break;
      default: LOG( FATAL ) << "Invalid stage";
    }
  }
}

void InferenceState::erase_from_workers( const uint32_t next_layer, const Stage next_stage )
{
  layer_workers_.erase( { next_layer, next_stage } );
}

string InferenceState::serialize() const
{
  const auto t1 = std::chrono::steady_clock::now().time_since_epoch().count();

  string result;
  result.resize( serialized_size() );

  const auto t2 = std::chrono::steady_clock::now().time_since_epoch().count();

  auto ptr = result.data();
  _put_and_advance( ptr, prompt_id_ );
  _put_and_advance( ptr, model_id_ );
  _put_and_advance( ptr, token_ );
  _put_and_advance( ptr, token_pos_ );
  _put_and_advance( ptr, next_layer_ );
  _put_and_advance( ptr, next_stage_ );
  _put_and_advance( ptr, prompt_length_ );
  _put_and_advance( ptr, temperature_ );
  _put_and_advance( ptr, finished_ );

  _put_and_advance( ptr, timestamp_ );
  _put_and_advance( ptr, loop_start_timestamp_ );

  _put_and_advance( ptr, static_cast<underlying_type_t<DataType>>( dtype_ ) );
  _put_and_advance( ptr, static_cast<uint64_t>( activations_.len() ) );
  const auto t3 = std::chrono::steady_clock::now().time_since_epoch().count();

  memcpy( ptr, activations_.data(), activations_.len() );
  ptr += activations_.len();
  const auto t4 = std::chrono::steady_clock::now().time_since_epoch().count();

  _put_and_advance( ptr, static_cast<uint32_t>( layer_workers_.size() ) );

  for ( auto& [layer_stage, address] : layer_workers_ ) {
    _put_and_advance( ptr, layer_stage.first );
    _put_and_advance( ptr, layer_stage.second );
    _put_and_advance( ptr, address.ipv4_numeric() );
    _put_and_advance( ptr, address.port() );
  }
  const auto t5 = std::chrono::steady_clock::now().time_since_epoch().count();
  __stats__.add_point<glinthawk::IntDistributions::SerializeFirst>( t2-t1 );
  __stats__.add_point<glinthawk::IntDistributions::SerializeSecond>( t3-t2 );
  __stats__.add_point<glinthawk::IntDistributions::SerializeThird>( t4-t3 );
  __stats__.add_point<glinthawk::IntDistributions::SerializeFourth>( t5-t4 );

  return result;
}

string InferenceState::to_string() const
{
  ostringstream oss;
  oss << "InferenceState("
      << "prompt_id=" << prompt_id_.base58digest().substr( 0, 8 ) << ", "
      << "token=" << token_ << ", "
      << "token_pos=" << token_pos_ << ", "
      << "next_layer=" << next_layer_ << ", "
      << "next_stage=" << next_stage_ << ", "
      << "prompt_len=" << prompt_length_ << ", "
      << "temperature=" << temperature_ << ", "
      << "finished=" << finished_ << ", "
      << "dtype=" << dtype_ << ", "
      << "activations.len=" << activations_ << ", "
      << "peers={";

  for ( auto& [layer_stage, address] : layer_workers_ ) {
    oss << " (" << layer_stage.first << "-" << layer_stage.second << " -> " << address.to_string() << ")";
  }

  oss << " })";

  return oss.str();
}

size_t InferenceState::serialized_size() const
{
  return sizeof( PromptID )                                                            /* prompt_id_ */
         + sizeof( ModelID )                                                           /* model_id_ */
         + sizeof( token_ )                                                            /* token_ */
         + sizeof( token_pos_ )                                                        /* token_pos_ */
         + sizeof( next_layer_ )                                                       /* next_layer_ */
         + sizeof( next_stage_ )                                                       /* next_stage_ */
         + sizeof( prompt_length_ )                                                    /* prompt_length_ */
         + sizeof( temperature_ )                                                      /* temperature_ */
         + sizeof( finished_ )                                                         /* finished_ */
         + sizeof( timestamp_ ) + sizeof( loop_start_timestamp_ ) + sizeof( DataType ) /* dtype_ */
         + sizeof( uint64_t )                                                          /* activations_.len */
         + activations_.len()                                                          /* activations_ data */
         + sizeof( uint32_t )                                                          /* layer_workers_.size() */
         + layer_workers_.size()
             * ( 2 * sizeof( uint32_t ) + sizeof( Stage ) + sizeof( uint16_t ) ); /* layer_workers_ */
}

}
