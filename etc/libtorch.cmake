include ( FetchContent )

## fetch libtorch
#FetchContent_Declare (
#        libtorch
#        URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
#)

# fetch libtorch
# FetchContent_Declare (
#         libtorch
#         URL https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.1%2Bcu118.zip
# )

# set (CUDNN_ROOT /data/users/pouya/.conda/envs/glint/)
# set (CUDA_HOME /data/users/pouya/.conda/envs/glint/)

# set(CAFFE2_USE_CUDNN 1)
# set(USE_CUDA 1)
# set(USE_CUDNN 1)

# fetch libtorch
FetchContent_Declare (
  libtorch
  URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
  SYSTEM
)

FetchContent_MakeAvailable ( libtorch )
list ( APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}" )
message ( STATUS "libtorch is available in " ${libtorch_SOURCE_DIR} )

find_package ( Torch REQUIRED )
include_directories ( SYSTEM ${TORCH_INCLUDE_DIRS} )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )
