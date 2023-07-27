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

include_directories ( src )

file ( GLOB SOURCE
  src/util/*.cc
  src/nn/*.cc
)

add_library ( libglinthawk STATIC ${SOURCE} )

set ( ALL_LIBS
  libglinthawk
  glog::glog
  ${TORCH_LIBRARIES}
)

add_executable ( glinthawk src/frontend/glinthawk.cc )
target_link_libraries ( glinthawk ${ALL_LIBS} )
