#include "demangle.hh"

#include <cstdlib>
#include <cxxabi.h>

std::string glinthawk::util::demangle( const std::string& name )
{
  int status = 0;

  char* buffer = reinterpret_cast<char*>( malloc( name.size() ) );
  size_t length = name.size();

  std::string demangled = abi::__cxa_demangle( name.c_str(), buffer, &length, &status );

  if ( status != 0 ) {
    return name;
  }

  return std::string { buffer, length };
}
