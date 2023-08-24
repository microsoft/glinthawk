#pragma once

#include <array>
#include <cstring>
#include <functional>
#include <openssl/sha.h>
#include <string>
#include <string_view>

namespace glinthawk::util::digest {

struct SHA256Hash
{
  std::array<uint8_t, SHA256_DIGEST_LENGTH> hash {};

  SHA256Hash() = default;
  SHA256Hash( const SHA256Hash& other ) { std::memcpy( hash.data(), other.hash.data(), SHA256_DIGEST_LENGTH ); }
  SHA256Hash& operator=( const SHA256Hash& other )
  {
    std::memcpy( hash.data(), other.hash.data(), SHA256_DIGEST_LENGTH );
    return *this;
  }

  SHA256Hash( SHA256Hash&& other ) noexcept = default;
  SHA256Hash& operator=( SHA256Hash&& other ) = default;

  auto operator<=>( const SHA256Hash& other ) const
  {
    return std::memcmp( hash.data(), other.hash.data(), SHA256_DIGEST_LENGTH );
  }
};

std::string sha256_base58( std::string_view input );
void sha256( const std::string_view input, SHA256Hash& hash );

}

template<>
struct std::hash<glinthawk::util::digest::SHA256Hash>
{
  std::size_t operator()( const glinthawk::util::digest::SHA256Hash& v ) const noexcept
  {
    std::size_t seed = 0;
    for ( const auto& byte : v.hash ) {
      seed ^= byte + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
    }
    return seed;
  }
};
