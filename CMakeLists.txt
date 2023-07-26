cmake_minimum_required ( VERSION 3.10.0 )

project ( Glinthawk )
include ( etc/cflags.cmake )

find_package ( Glog REQUIRED )

include_directories ( src )

file ( GLOB SOURCE
  src/util/*.cc
  src/llama/*.cc
)

add_library ( libglinthawk STATIC ${SOURCE} )

set ( ALL_LIBS
  libglinthawk
  glog::glog
)

add_executable ( glinthawk src/frontend/glinthawk.cc )
target_link_libraries ( glinthawk ${ALL_LIBS} )
