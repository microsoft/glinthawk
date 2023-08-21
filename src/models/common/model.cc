#include "model.hh"

#include <sstream>

using namespace std;

namespace glinthawk::models {

namespace {

template<typename FieldType, typename PtrType>
FieldType _get_and_advance( const PtrType*& ptr )
{
  const auto result = reinterpret_cast<const FieldType*>( ptr );
  ptr = reinterpret_cast<uint8_t*>( ptr ) + sizeof( FieldType );
  return *result;
}

template<typename FieldType, typename PtrType>
void _put_and_advance( PtrType*& ptr, const FieldType& field )
{
  memcpy( ptr, &field, sizeof( FieldType ) );
  ptr = reinterpret_cast<uint8_t*>( ptr ) + sizeof( FieldType );
}

} // namespace

template<typename DType>
InferenceState<DType>::InferenceState( const string_view serialized )
{
  const auto ptr = serialized.data();

  prompt_id_ = _get_and_advance<decltype( prompt_id_ )>( ptr );
  model_id_ = _get_and_advance<decltype( model_id_ )>( ptr );
  token_ = _get_and_advance<decltype( token_ )>( ptr );
  token_pos_ = _get_and_advance<decltype( token_pos_ )>( ptr );
  next_layer_ = _get_and_advance<decltype( next_layer_ )>( ptr );
  temperature_ = _get_and_advance<decltype( temperature_ )>( ptr );

  activations_.len = _get_and_advance<decltype( activations_.len )>( ptr );
  activations_.ptr = make_unique<DType[]>( activations_.len );

  memcpy( activations_.ptr.get(),
          _advance_pointer( ptr, activations_.len * sizeof( DType ) ),
          activations_.len * sizeof( DType ) );
}

template<typename DType>
string InferenceState<DType>::serialize() const
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

  _put_and_advance( ptr, activations_.len );
  memcpy( ptr, activations_.ptr.get(), activations_.len * sizeof( DType ) );

  return result;
}

template<typename DType>
string InferenceState<DType>::to_string() const
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

template<typename DType>
size_t InferenceState<DType>::serialized_size() const
{
  return sizeof( PromptID )                    /* prompt_id_ */
         + sizeof( ModelID )                   /* model_id_ */
         + sizeof( token_ )                    /* token_ */
         + sizeof( token_pos_ )                /* token_pos_ */
         + sizeof( next_layer_ )               /* next_layer_ */
         + sizeof( temperature_ )              /* temperature_ */
         + sizeof( activations_.len )          /* activations_.len */
         + sizeof( DType ) * activations_.len; /* activations_ data */
}
}
