#include "inference.hh"

#include <cstring>

using namespace std;
using namespace glinthawk;

InferenceState::InferenceState( const string_view serialized )
{
  const auto ptr = serialized.data();
  token = *reinterpret_cast<const int32_t*>( ptr );
  token_pos = *reinterpret_cast<const int32_t*>( ptr + sizeof( int32_t ) );
  next_layer = *reinterpret_cast<const int32_t*>( ptr + 2 * sizeof( int32_t ) );
  activations.len = *reinterpret_cast<const int32_t*>( ptr + 3 * sizeof( int32_t ) );

  activations.ptr = make_unique<float[]>( activations.len );
  memcpy( activations.ptr.get(), ptr + 4 * sizeof( int32_t ), activations.len * sizeof( float ) );
}

string InferenceState::serialize()
{
  string result;
  result.resize( 3 * sizeof( int32_t ) + sizeof( int32_t ) + activations.len * sizeof( float ) );

  auto ptr = result.data();

  memcpy( ptr, &token, sizeof( int32_t ) );
  memcpy( ptr + sizeof( int32_t ), &token_pos, sizeof( int32_t ) );
  memcpy( ptr + 2 * sizeof( int32_t ), &next_layer, sizeof( int32_t ) );
  memcpy( ptr + 3 * sizeof( int32_t ), &activations.len, sizeof( int32_t ) );
  memcpy( ptr + 4 * sizeof( int32_t ), activations.ptr.get(), activations.len * sizeof( float ) );

  return result;
}
