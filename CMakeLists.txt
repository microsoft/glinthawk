cmake_minimum_required ( VERSION 3.11.0 )
cmake_policy ( SET CMP0135 NEW )

project ( Glinthawk VERSION 1.0.0 LANGUAGES CXX CUDA )

include ( etc/cflags.cmake )
include ( etc/openmp.cmake )
include ( etc/cuda.cmake )

find_package ( OpenSSL REQUIRED )
link_libraries ( OpenSSL::SSL )

find_package ( Glog REQUIRED )
link_libraries ( glog::glog )

find_package ( Protobuf REQUIRED )
add_subdirectory ( src/message/protobuf )
link_libraries ( protobuf::libprotobuf )

include_directories ( src )
include_directories ( SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/src/message/protobuf" )
include_directories ( SYSTEM lib )

file ( GLOB SOURCE
  src/util/*.cc
  src/net/*.cc
  src/compute/*.cc
  src/message/*.cc
  src/worker/*.cc
)

file ( GLOB_RECURSE MODELS_SOURCE
  src/models/*.cc
  src/models/*.cu
)

list ( APPEND SOURCE ${MODELS_SOURCE} )

add_library ( glinthawk STATIC ${SOURCE} )
link_libraries ( glinthawk )
link_libraries ( glinthawk_proto )

if ( CUDA_ENABLED )
  # add_executable ( profile-gpu src/frontend/profile-gpu.cc )
  add_executable ( infer-gpu src/frontend/infer-gpu.cc )
  add_executable ( infer-gpu-layered src/frontend/infer-gpu-layered.cc )
  add_executable ( profile-offload src/frontend/profile-offload.cc )
  add_executable ( worker-gpu src/frontend/worker-gpu.cc )
  add_executable ( kickoff src/frontend/kickoff.cc )
endif ()
