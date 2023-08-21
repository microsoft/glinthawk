#pragma once

#include <string>
#include <string_view>

namespace glinthawk::util::digest {

std::string sha256_base58( std::string_view input );

}
