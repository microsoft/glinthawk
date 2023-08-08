include ( CheckLanguage )
check_language ( CUDA )

if ( CMAKE_CUDA_COMPILER )
  message ( "CUDA compiler found: ${CMAKE_CUDA_COMPILER}" )
  add_definitions ( -DCUDA_ENABLED )
  set ( CMAKE_CUDA_ARCHITECTURES native )
endif()
