cmake_minimum_required ( VERSION 3.18.0 )
cmake_policy ( SET CMP0135 NEW )

project ( Glinthawk VERSION 1.0.0 LANGUAGES CXX )

include ( etc/cflags.cmake )
include ( etc/openmp.cmake )
include ( etc/cuda.cmake )

find_package ( OpenSSL REQUIRED )
link_libraries ( OpenSSL::SSL )

find_package ( Glog REQUIRED )
link_libraries ( glog::glog )

set ( Protobuf_USE_STATIC_LIBS ON )
find_package ( Protobuf REQUIRED )
add_subdirectory ( src/message/protobuf )
link_libraries ( glinthawk_proto protobuf::libprotobuf )

include_directories ( src )
include_directories ( SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/src/message/protobuf" )
include_directories ( SYSTEM lib )

file ( GLOB SOURCE_CORE
  src/util/*.cc
  src/net/*.cc
  src/net/http/*.cc
  src/storage/*.cc
  src/storage/local/*.cc
  src/storage/azure/*.cc
  src/monitoring/*.cc
  src/message/*.cc
  src/prompt/*.cc
)

file ( GLOB_RECURSE SOURCE_COMPUTE
  src/compute/*.cc
  src/worker/*.cc
  src/models/*.cc
)

# XXX not ready yet
list ( REMOVE_ITEM SOURCE_COMPUTE "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/kernel_pipes.cc" )

add_library ( glinthawkcore OBJECT ${SOURCE_CORE} )
link_libraries ( glinthawkcore )

add_subdirectory ( targets/amd64 amd64/ )

if ( CUDA_ENABLED )
  add_subdirectory ( targets/cuda cuda/ )
endif ()
