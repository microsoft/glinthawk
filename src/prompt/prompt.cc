#include "prompt.hh"

#include <endian.h>
#include <fstream>
#include <glog/logging.h>


using namespace std;
using namespace glinthawk::prompt;

SerializedPrompt::SerializedPrompt( const filesystem::path& path )
{
  ifstream fin { path };
  CHECK( fin.good() ) << "Failed to open prompt file " << path;

  uint32_t token_count;
  fin >> token_count;

  token_count = le32toh( token_count );
  tokens_.reserve( token_count );

  for ( uint32_t token, i = 0; i < token_count; ++i ) {
    fin >> token;
    tokens_.push_back( le32toh( token ) );

    if (fin.eof()) {
      break;
    }
  }

  CHECK( tokens_.size() == token_count ) << "Failed to read all tokens from prompt file " << path;
}
