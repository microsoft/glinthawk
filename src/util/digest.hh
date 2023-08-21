#pragma once

#include <openssl/sha.h>
#include <string>
#include <string_view>

namespace glinthawk::util::digest {

struct SHA256Hash
{
  uint8_t hash[SHA256_DIGEST_LENGTH];
};

std::string sha256_base58( std::string_view input );
void sha256( const std::string_view input, SHA256Hash& hash );

}
