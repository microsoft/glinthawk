#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace glinthawk::prompt {

class SerializedPrompt
{
private:
  std::vector<uint32_t> tokens_ {};

public:
  SerializedPrompt( const std::filesystem::path& path );
  uint32_t token( const uint32_t token_pos ) const { return tokens_.at( token_pos ); }
};

}
