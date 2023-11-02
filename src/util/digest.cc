#include "digest.hh"

#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <vector>

using namespace std;

namespace glinthawk::util::digest {

// taken from bitcoin/bitcoin/src/base58.cpp

static const char* const ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const int8_t MAP_BASE_58[256] = {
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,
  -1, -1, -1, -1, -1, -1, -1, 9,  10, 11, 12, 13, 14, 15, 16, -1, 17, 18, 19, 20, 21, -1, 22, 23, 24, 25, 26, 27, 28,
  29, 30, 31, 32, -1, -1, -1, -1, -1, -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -1, 44, 45, 46, 47, 48, 49, 50,
  51, 52, 53, 54, 55, 56, 57, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

string encode_base_58( const string_view input )
{
  auto bytes = reinterpret_cast<unsigned const char*>( input.data() );
  const auto len = input.length();
  vector<unsigned char> digits( len * 138 / 100 + 1 );

  size_t digits_len = 1;

  for ( size_t i = 0; i < len; i++ ) {
    unsigned int carry = static_cast<unsigned int>( bytes[i] );

    for ( size_t j = 0; j < digits_len; j++ ) {
      carry += static_cast<unsigned int>( digits[j] ) << 8;
      digits[j] = static_cast<unsigned char>( carry % 58 );
      carry /= 58;
    }

    while ( carry > 0 ) {
      digits[digits_len++] = static_cast<unsigned char>( carry % 58 );
      carry /= 58;
    }
  }

  string result;
  size_t result_len = 0;

  for ( ; result_len < len && bytes[result_len] == 0; result_len++ ) {
    result += '1';
  }

  for ( size_t i = 0; i < digits_len; i++ ) {
    result += ALPHABET[digits[digits_len - 1 - i]];
  }

  return result;
}

template<size_t MaxOutputLength>
bool decode_base_58( const string_view psz, std::array<uint8_t, MaxOutputLength>& vch )
{
  // Skip leading spaces.
  size_t psz_index = 0;
  while ( psz_index < psz.length() && psz[psz_index] == ' ' )
    psz_index++;

  // Skip and count leading '1's.
  size_t zeroes = 0;
  size_t length = 0;

  while ( psz[psz_index] == '1' ) {
    zeroes++;
    if ( zeroes > MaxOutputLength )
      return false;
    psz_index++;
  }

  // Allocate enough space in big-endian base256 representation.
  size_t size = psz.length() * 733 / 1000 + 1; // log(58) / log(256), rounded up.
  std::vector<unsigned char> b256( size );

  // Process the characters.
  static_assert( std::size( MAP_BASE_58 ) == 256, "mapBase58.size() should be 256" ); // guarantee not out of range

  while ( psz_index < psz.length() && psz[psz_index] != ' ' ) {
    // Decode base58 character
    int carry = MAP_BASE_58[static_cast<uint8_t>( psz[psz_index] )];
    if ( carry == -1 ) // Invalid b58 character
      return false;

    size_t i = 0;
    for ( std::vector<unsigned char>::reverse_iterator it = b256.rbegin();
          ( carry != 0 || i < length ) && ( it != b256.rend() );
          ++it, ++i ) {
      carry += 58 * ( *it );
      *it = carry % 256;
      carry /= 256;
    }

    length = i;
    if ( length + zeroes > MaxOutputLength )
      return false;

    psz_index++;
  }

  // Skip trailing spaces.
  while ( psz[psz_index] == ' ' )
    psz_index++;

  if ( psz_index != psz.length() )
    return false;

  // Skip leading zeroes in b256.
  std::vector<unsigned char>::iterator it = b256.begin() + ( size - length );

  // Copy result into output vector.
  size_t i = 0;
  while ( it != b256.end() )
    vch[i++] = *( it++ );

  return true;
}

string SHA256Hash::hexdigest() const
{
  ostringstream result;

  for ( const auto& byte : hash ) {
    result << hex << setfill( '0' ) << setw( 2 ) << static_cast<int>( byte );
  }

  return result.str();
}

SHA256Hash SHA256Hash::from_base58digest( const std::string_view base58digest )
{
  SHA256Hash result;

  if ( not decode_base_58( base58digest, result.hash ) ) {
    throw runtime_error( "Could not decode base58 digest: " + string( base58digest ) );
  }

  return result;
}

string SHA256Hash::base58digest() const
{
  return encode_base_58( { reinterpret_cast<const char*>( hash.data() ), SHA256_DIGEST_LENGTH } );
}

void sha256( const string_view input, SHA256Hash& hash )
{
  SHA256( reinterpret_cast<const unsigned char*>( input.data() ), input.length(), hash.hash.data() );
}

}
