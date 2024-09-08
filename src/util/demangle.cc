#include "demangle.hh"

#include <cstdlib>
#include <cxxabi.h>

std::string glinthawk::util::demangle( const std::string& name, const bool keep_template_args )
{
  int status = 0;
  char* buffer = abi::__cxa_demangle( name.c_str(), nullptr, nullptr, &status );

  if ( status != 0 || buffer == nullptr ) {
    return name;
  }

  std::string result { buffer };
  free( buffer );

  if ( !keep_template_args ) {
    const auto pos = result.find( '<' );
    if ( pos != std::string::npos ) {
      result = result.substr( 0, pos + 1 ) + std::string { ">" };
    }
  }

  return result;
}
