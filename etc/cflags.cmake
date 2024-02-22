set ( CMAKE_CXX_STANDARD 20 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS OFF )
set ( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if ( MSVC )
  message ( ERROR "MSVC is not supported yet" )
else ()
  if ( NOT DEFINED GCC_TARGET_ARCH )
      set ( GCC_TARGET_ARCH native )
  endif ()

  message ( NOTICE "GCC_TARGET_ARCH: ${GCC_TARGET_ARCH}" )

  set ( CMAKE_CXX_FLAGS "" )
  set ( CMAKE_CXX_FLAGS_DEBUG "-g" )
  set ( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" )
  set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG" )

  list ( APPEND OPTIMIZATION_FLAGS -march=${GCC_TARGET_ARCH} -ffast-math -fsingle-precision-constant )
  list ( APPEND GCC_STRICT_FLAGS -pedantic -pedantic-errors -Werror -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wformat=2 -Weffc++ -Wold-style-cast )

  foreach ( flag IN LISTS OPTIMIZATION_FLAGS  )
      add_compile_options ( "$<$<COMPILE_LANGUAGE:CXX>:${flag}>$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>" )
  endforeach ()

  foreach ( flag IN LISTS GCC_STRICT_FLAGS )
      add_compile_options ( "$<$<COMPILE_LANGUAGE:CXX>:${flag}>" )
  endforeach ()
endif ()
