include ( CheckLanguage )
check_language ( CUDA )

cmake_policy ( SET CMP0074 NEW )
find_package ( CUDAToolkit )

if ( CMAKE_CUDA_COMPILER )
  message ( NOTICE "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}" )
  enable_language( CUDA )

  set ( CMAKE_CUDA_FLAGS "" )
  set ( CMAKE_CUDA_FLAGS_DEBUG "-g" )
  set ( CMAKE_CUDA_FLAGS_RELEASE "-O3" )
  set ( CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g" )

  set ( CMAKE_CUDA_ARCHITECTURES 75 )
  set ( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" )
  set ( CUDA_ENABLED ON )

  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fopenmp>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-O3>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--optimize=3>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-ffast-math>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fsingle-precision-constant>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>" )
  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-O3>" )

  set_property ( GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON )
  set_property ( GLOBAL PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )

endif()
