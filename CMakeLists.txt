cmake_minimum_required ( VERSION 3.11.0 )
cmake_policy ( SET CMP0135 NEW )

project ( Glinthawk VERSION 1.0.0 LANGUAGES CXX CUDA )

include ( etc/cflags.cmake )
include ( etc/openmp.cmake )
include ( etc/cuda.cmake )

find_package ( GTest CONFIG REQUIRED )
include ( GoogleTest )

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
  src/net/http/*.cc
  src/storage/*.cc
  src/storage/azure/*.cc
  src/compute/*.cc
  src/message/*.cc
  src/worker/*.cc
  src/prompt/*.cc
)

file ( GLOB_RECURSE MODELS_SOURCE
  src/models/*.cc
  src/models/*.cu
)

list ( APPEND SOURCE ${MODELS_SOURCE} )

add_library ( glinthawk STATIC ${SOURCE} )
link_libraries ( glinthawk )
link_libraries ( glinthawk_proto )

add_executable ( infer-layered-cpu src/frontend/infer-layered-cpu.cc )
add_executable ( infer-cpu src/frontend/infer-cpu.cc )
add_executable ( worker-cpu src/frontend/worker-cpu.cc )

# Tests
#add_executable ( test_ops_cpu tests/ops_cpu.cc )
#target_link_libraries ( test_ops_cpu GTest::gtest )
#gtest_discover_tests ( test_ops_cpu )


if ( CUDA_ENABLED )
  add_executable ( profile-cuda src/frontend/profile-cuda.cc )
  add_executable ( infer-cuda src/frontend/infer-cuda.cc )
  add_executable ( infer-layered-cuda src/frontend/infer-layered-cuda.cc )
  add_executable ( profile-offload src/frontend/profile-offload.cc )
  add_executable ( worker-cuda src/frontend/worker-cuda.cc )

  # Tests
  add_executable ( test_ops_cuda tests/ops_cuda.cu )
  target_link_libraries ( test_ops_cuda GTest::gtest )
  gtest_discover_tests ( test_ops_cuda )
endif ()
