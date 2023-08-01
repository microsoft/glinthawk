cmake_minimum_required ( VERSION 3.11.0 )
cmake_policy ( SET CMP0135 NEW )

project ( Glinthawk )

include ( FetchContent )
include ( etc/cflags.cmake )

# fetch libtorch
FetchContent_Declare (
  libtorch
  URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
)

FetchContent_Populate(libtorch)
message ( STATUS "libtorch is available in " ${libtorch_SOURCE_DIR} )
list ( APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}" )

find_package ( Glog REQUIRED )
find_package ( Torch REQUIRED )
find_package ( OpenMP REQUIRED )

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories ( SYSTEM ${TORCH_INCLUDE_DIRS} )
include_directories ( src )

file ( GLOB SOURCE
  src/util/*.cc
  src/net/*.cc
  src/nn/*.cc
)

add_library ( libglinthawk STATIC ${SOURCE} )

set ( ALL_LIBS
  libglinthawk
  glog::glog
  OpenMP::OpenMP_CXX
)

add_executable ( infer src/frontend/infer.cc )
target_link_libraries ( infer ${ALL_LIBS} )
