include ( CheckLanguage )
check_language ( CUDA )

cmake_policy ( SET CMP0074 NEW )
find_package ( CUDAToolkit REQUIRED )

if ( CMAKE_CUDA_COMPILER )
  message ( NOTICE "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}" )

  set ( CMAKE_CUDA_ARCHITECTURES 75 )
  set ( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" )
  set ( CUDA_ENABLED ON )

  add_compile_options ( "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>" )

  set_property ( GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON )
  set_property ( GLOBAL PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )

endif()

function ( CUDA_CONVERT_FLAGS EXISTING_TARGET )
  get_property ( old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS )
  if ( NOT "${old_flags}" STREQUAL "" )
    string ( REPLACE ";" "," CUDA_flags "${old_flags}")
    set_property ( TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
        "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>" )
  endif()
endfunction()
