cmake_minimum_required ( VERSION 3.11.0 )
cmake_policy ( SET CMP0135 NEW )

project ( Glinthawk VERSION 1.0.0 LANGUAGES CXX )

include ( etc/cflags.cmake )
include ( etc/openmp.cmake )

find_package ( Glog REQUIRED )
link_libraries ( glog::glog )

include_directories ( src )

file ( GLOB SOURCE
  src/util/*.cc
  src/net/*.cc
  src/nn/*.cc
  src/worker/*.cc
)

add_library ( glinthawk STATIC ${SOURCE} )
link_libraries ( glinthawk )

add_executable ( infer src/frontend/infer.cc )
add_executable ( infer-partial src/frontend/infer-partial.cc )
