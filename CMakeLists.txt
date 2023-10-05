cmake_minimum_required ( VERSION 3.11.0 )
cmake_policy ( SET CMP0135 NEW )

project ( Glinthawk VERSION 1.0.0 LANGUAGES CXX CUDA )

include ( etc/cflags.cmake )
include ( etc/openmp.cmake )
include ( etc/cuda.cmake )

find_package ( Glog REQUIRED )
link_libraries ( glog::glog )

include_directories ( src )

file ( GLOB SOURCE
  src/util/*.cc
  src/net/*.cc
  src/nn/*.cc
  src/worker/*.cc
)

file ( GLOB_RECURSE MODELS_SOURCE
  src/models/*.cc
  src/models/*.cu
)

list ( APPEND SOURCE ${MODELS_SOURCE} )

add_library ( glinthawk STATIC ${SOURCE} )
link_libraries ( glinthawk )

add_executable ( infer src/frontend/infer.cc )
add_executable ( infer-partial src/frontend/infer-partial.cc )

if ( CUDA_ENABLED )
  add_executable ( infer-gpu src/frontend/infer-gpu.cc )
  add_executable ( infer-parallel-prompts src/frontend/infer-parallel-prompts.cc )
  add_executable ( profile-gpu src/frontend/profile-gpu.cc )
  add_executable ( profile-offload src/frontend/profile-offload.cc )
endif ()
