cmake_minimum_required ( VERSION 3.10.0 )

project ( Glinthawk )

include ( etc/cflags.cmake )

include_directories ( src )

file ( GLOB SOURCE
  src/util/*.cc
)

add_library ( libglinthawk STATIC ${SOURCE} )

set ( ALL_LIBS
  libglinthawk
)

add_executable ( glinthawk src/frontend/glinthawk.cc )
target_link_libraries ( glinthawk ${ALL_LIBS} )
