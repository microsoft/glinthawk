#include "base.hh"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace glinthawk::models::llama2 {

/* VOCABULARY */

Vocabulary::Vocabulary( const std::filesystem::path& vocabulary_path )
{
  std::ifstream fin { vocabulary_path, std::ios::binary };
  int len = 0;

  CHECK( fin.good() ) << "Failed to open vocabulary file: " << vocabulary_path;

  int i;
  for ( i = 0;; i++ ) {
    if ( not fin.read( reinterpret_cast<char*>( &len ), sizeof( int ) ) ) {
      break;
    }

    CHECK_GT( len, 0 ) << "Vocabulary entry length must be positive.";

    std::string val;
    val.resize( len );
    CHECK( fin.read( val.data(), val.length() ) ) << "Failed to read vocabulary entry.";

    token_to_word_.push_back( val );
    word_to_token_.emplace( val, i );
  }

  LOG( INFO ) << "Loaded vocabulary of size " << i << " from " << vocabulary_path;
}

std::string Vocabulary::get_word( int token ) const
{
  CHECK_GE( token, 0 ) << "Token index must be non-negative.";
  CHECK_LT( token, token_to_word_.size() ) << "Token index out of bounds.";
  return token_to_word_[token];
}

int Vocabulary::get_token( const std::string& word ) const
{
  auto it = word_to_token_.find( word );
  CHECK( it != word_to_token_.end() ) << "Unknown word: " << word;
  return it->second;
}

} // namespace glinthawk::models::llama2
