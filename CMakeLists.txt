cmake_minimum_required ( VERSION 3.10.0 )

project ( Starter )

include ( etc/cflags.cmake )
include ( etc/clang_format.cmake )
include ( etc/clang_tidy.cmake )

include ( FindOpenSSL )

include_directories ( src )

file ( GLOB SOURCE
  src/util/*
  src/net/*
)

add_library ( starter STATIC ${SOURCE} )

set ( ALL_LIBS
  starter
  ${OPENSSL_LIBRARIES}
)

add_executable ( prog src/frontend/prog.cc )
target_link_libraries ( prog ${ALL_LIBS} )
