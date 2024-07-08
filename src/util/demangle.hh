#pragma once

#include <cxxabi.h>
#include <string>

namespace glinthawk::util {

std::string demangle( const std::string& name )
{
  int status = 0;
  std::string demangled = abi::__cxa_demangle( name.c_str(), nullptr, nullptr, &status );

  if ( status != 0 ) {
    return name;
  }

  return demangled;
}

} // namespace glinthawk::util
