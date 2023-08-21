#pragma once

#include <cstring>
#include <openssl/sha.h>
#include <string>
#include <string_view>

namespace glinthawk::util::digest {

struct SHA256Hash
{
  uint8_t hash[SHA256_DIGEST_LENGTH];

  SHA256Hash() = default;
  SHA256Hash( const SHA256Hash& other ) { std::memcpy( hash, other.hash, SHA256_DIGEST_LENGTH ); }
  SHA256Hash& operator=( const SHA256Hash& other )
  {
    std::memcpy( hash, other.hash, SHA256_DIGEST_LENGTH );
    return *this;
  }

  SHA256Hash( SHA256Hash&& other ) noexcept = default;
  SHA256Hash& operator=( SHA256Hash&& other ) = default;

  auto operator<=>( const SHA256Hash& other ) const { return std::memcmp( hash, other.hash, SHA256_DIGEST_LENGTH ); }
};

std::string sha256_base58( std::string_view input );
void sha256( const std::string_view input, SHA256Hash& hash );

}
