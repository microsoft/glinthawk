#include "model.hh"

#include <sstream>

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
  temperature_ = _get_and_advance<decltype( temperature_ )>( ptr );

  activations_.dtype.dtype = static_cast<DataType::Type>( _get_and_advance<underlying_type_t<DataType::Type>>( ptr ) );
  activations_.len = _get_and_advance<decltype( activations_.len )>( ptr );
  activations_.ptr = make_unique<uint8_t[]>( activations_.len * activations_.dtype.size() );

  memcpy( activations_.ptr.get(), ptr, activations_.len * activations_.dtype.size() );
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
  _put_and_advance( ptr, temperature_ );

  _put_and_advance( ptr, static_cast<underlying_type_t<DataType::Type>>( activations_.dtype.dtype ) );
  _put_and_advance( ptr, activations_.len );

  memcpy( ptr, activations_.ptr.get(), activations_.len * activations_.dtype.size() );

  return result;
}

string InferenceState::to_string() const
{
  ostringstream oss;
  oss << "InferenceState("
      << "token=" << token_ << ", "
      << "token_pos=" << token_pos_ << ", "
      << "next_layer=" << next_layer_ << ", "
      << "temperature=" << temperature_ << ", "
      << "activations.len=" << activations_.len << ")";
  return oss.str();
}

size_t InferenceState::serialized_size() const
{
  return sizeof( PromptID )                              /* prompt_id_ */
         + sizeof( ModelID )                             /* model_id_ */
         + sizeof( token_ )                              /* token_ */
         + sizeof( token_pos_ )                          /* token_pos_ */
         + sizeof( next_layer_ )                         /* next_layer_ */
         + sizeof( temperature_ )                        /* temperature_ */
         + sizeof( activations_.len )                    /* activations_.len */
         + activations_.dtype.size() * activations_.len; /* activations_ data */
}

}
