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

#set ( Protobuf_USE_STATIC_LIBS ON )
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
  src/storage/local/*.cc
  src/storage/azure/*.cc
  src/compute/*.cc
  src/monitoring/*.cc
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

add_executable ( azure-storage-test src/frontend/azure-storage-test.cc )
add_executable ( telegraf-test src/frontend/telegraf-test.cc )

add_executable ( worker-cpu-fp16 src/frontend/worker.cc )
add_executable ( worker-cpu-fp32 src/frontend/worker.cc )
target_compile_definitions ( worker-cpu-fp16 PRIVATE TARGET_PLATFORM_CPU TARGET_DTYPE_FLOAT16  )
target_compile_definitions ( worker-cpu-fp32 PRIVATE TARGET_PLATFORM_CPU TARGET_DTYPE_FLOAT32 )

add_executable ( profile-cpu-fp16 src/frontend/profile.cc )
add_executable ( profile-cpu-fp32 src/frontend/profile.cc )
target_compile_definitions ( profile-cpu-fp16 PRIVATE TARGET_PLATFORM_CPU TARGET_DTYPE_FLOAT16 )
target_compile_definitions ( profile-cpu-fp32 PRIVATE TARGET_PLATFORM_CPU TARGET_DTYPE_FLOAT32 )

add_executable ( infer-cpu-fp16 src/frontend/infer.cc )
add_executable ( infer-cpu-fp32 src/frontend/infer.cc )
target_compile_definitions ( infer-cpu-fp16 PRIVATE TARGET_PLATFORM_CPU TARGET_DTYPE_FLOAT16 )
target_compile_definitions ( infer-cpu-fp32 PRIVATE TARGET_PLATFORM_CPU TARGET_DTYPE_FLOAT32 )

if ( CUDA_ENABLED )

  add_executable ( profile-offload src/frontend/profile-offload.cc )

  add_executable ( worker-cuda-fp16 src/frontend/worker.cc )
  target_compile_definitions ( worker-cuda-fp16 PRIVATE TARGET_PLATFORM_CUDA TARGET_DTYPE_FLOAT16 )

  add_executable ( profile-cuda-fp16 src/frontend/profile.cc )
  target_compile_definitions ( profile-cuda-fp16 PRIVATE TARGET_PLATFORM_CUDA TARGET_DTYPE_FLOAT16 )

  add_executable ( infer-cuda-fp16 src/frontend/infer.cc )
  target_compile_definitions ( infer-cuda-fp16 PRIVATE TARGET_PLATFORM_CUDA TARGET_DTYPE_FLOAT16 )

endif ()
