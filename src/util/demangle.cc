#include "demangle.hh"

#include <cstdlib>
#include <cxxabi.h>

std::string glinthawk::util::demangle( const std::string& name )
{
  int status = 0;
  char* buffer = abi::__cxa_demangle( name.c_str(), nullptr, nullptr, &status );

  if ( status != 0 ) {
    return name;
  }

  std::string result { buffer };
  free( buffer );

  return result;
}
