include ( CheckLanguage )
check_language ( CUDA )

cmake_policy ( SET CMP0074 NEW )
find_package ( CUDAToolkit REQUIRED )

if ( CMAKE_CUDA_COMPILER )
  message ( "CUDA compiler found: ${CMAKE_CUDA_COMPILER}" )
  add_definitions ( -DCUDA_ENABLED )
  set ( CMAKE_CUDA_ARCHITECTURES 90 )
  set ( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" )
  set ( CUDA_ENABLED ON )
  link_libraries ( CUDA::cublas )
endif()
