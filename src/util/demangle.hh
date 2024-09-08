#pragma once

#include <string>

namespace glinthawk::util {

std::string demangle( const std::string& name, const bool keep_template_args = true );

} // namespace glinthawk::util
