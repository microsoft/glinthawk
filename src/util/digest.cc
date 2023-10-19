#include "digest.hh"

#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <vector>

using namespace std;

namespace glinthawk::util::digest {

// taken from bitcoin/bitcoin
const char* const ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

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

std::string SHA256Hash::hexdigest() const
{
  std::ostringstream result;

  for ( const auto& byte : hash ) {
    result << std::hex << std::setfill( '0' ) << std::setw( 2 ) << static_cast<int>( byte );
  }

  return result.str();
}

std::string SHA256Hash::base58digest() const
{
  return encode_base_58( { reinterpret_cast<const char*>( hash.data() ), SHA256_DIGEST_LENGTH } );
}

void sha256( const std::string_view input, SHA256Hash& hash )
{
  SHA256( reinterpret_cast<const unsigned char*>( input.data() ), input.length(), hash.hash.data() );
}

}
