#include "model.hh"

#include <sstream>

#include <glog/logging.h>

using namespace std;

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

  prompt_id_ = _get_and_advance<decltype( prompt_id_ )>( ptr );
  model_id_ = _get_and_advance<decltype( model_id_ )>( ptr );
  token_ = _get_and_advance<decltype( token_ )>( ptr );
  token_pos_ = _get_and_advance<decltype( token_pos_ )>( ptr );
  next_layer_ = _get_and_advance<decltype( next_layer_ )>( ptr );
  prompt_length_ = _get_and_advance<decltype( prompt_length_ )>( ptr );
  temperature_ = _get_and_advance<decltype( temperature_ )>( ptr );

  activations_.dtype.dtype
    = static_cast<SerializedDataType::Type>( _get_and_advance<underlying_type_t<SerializedDataType::Type>>( ptr ) );
  activations_.len = _get_and_advance<decltype( activations_.len )>( ptr );
  activations_.ptr = make_unique<uint8_t[]>( activations_.len * activations_.dtype.size() );

  memcpy( activations_.ptr.get(), ptr, activations_.len * activations_.dtype.size() );
  ptr += activations_.len * activations_.dtype.size();

  const auto num_workers = _get_and_advance<uint32_t>( ptr );

  for ( uint32_t i = 0; i < num_workers; i++ ) {
    const auto layer = _get_and_advance<uint32_t>( ptr );
    const auto ipv4_numeric = _get_and_advance<uint32_t>( ptr );
    const auto port = _get_and_advance<uint16_t>( ptr );
    layer_workers_.emplace( layer, net::Address::from_ipv4_numeric( ipv4_numeric, port ) );
  }
}

net::Address InferenceState::next_worker() const
{
  auto it = layer_workers_.find( next_layer_ );
  CHECK( it != layer_workers_.end() ) << "No worker found for layer " << next_layer_;
  return it->second;
}

string InferenceState::serialize() const
{
  string result;
  result.resize( serialized_size() );

  auto ptr = result.data();
  _put_and_advance( ptr, prompt_id_ );
  _put_and_advance( ptr, model_id_ );
  _put_and_advance( ptr, token_ );
  _put_and_advance( ptr, token_pos_ );
  _put_and_advance( ptr, next_layer_ );
  _put_and_advance( ptr, prompt_length_ );
  _put_and_advance( ptr, temperature_ );

  _put_and_advance( ptr, static_cast<underlying_type_t<SerializedDataType::Type>>( activations_.dtype.dtype ) );
  _put_and_advance( ptr, activations_.len );

  memcpy( ptr, activations_.ptr.get(), activations_.len * activations_.dtype.size() );
  ptr += activations_.len * activations_.dtype.size();

  _put_and_advance( ptr, static_cast<uint32_t>( layer_workers_.size() ) );

  for ( auto& [layer, address] : layer_workers_ ) {
    _put_and_advance( ptr, layer );
    _put_and_advance( ptr, address.ipv4_numeric() );
    _put_and_advance( ptr, address.port() );
  }

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
      << "prompt_len=" << prompt_length_ << ", "
      << "temperature=" << temperature_ << ", "
      << "activations.len=" << activations_.len << ", "
      << "peers={";

  for ( auto& [layer, address] : layer_workers_ ) {
    oss << " (" << layer << " -> " << address.to_string() << ")";
  }

  oss << " })";

  return oss.str();
}

size_t InferenceState::serialized_size() const
{
  return sizeof( PromptID )                                                         /* prompt_id_ */
         + sizeof( ModelID )                                                        /* model_id_ */
         + sizeof( token_ )                                                         /* token_ */
         + sizeof( token_pos_ )                                                     /* token_pos_ */
         + sizeof( next_layer_ )                                                    /* next_layer_ */
         + sizeof( prompt_length_ )                                                 /* prompt_length_ */
         + sizeof( temperature_ )                                                   /* temperature_ */
         + sizeof( SerializedDataType::Type )                                       /* activations_.dtype.dtype */
         + sizeof( activations_.len )                                               /* activations_.len */
         + activations_.dtype.size() * activations_.len                             /* activations_ data */
         + sizeof( uint32_t )                                                       /* layer_workers_.size() */
         + layer_workers_.size() * ( 2 * sizeof( uint32_t ) + sizeof( uint16_t ) ); /* layer_workers_ */
}

}
