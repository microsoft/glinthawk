set ( CMAKE_CXX_STANDARD 20 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS OFF )
set ( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if ( MSVC )
    add_compile_options ( /W4 /WX )
else ()
    if ( NOT DEFINED GCC_TARGET_ARCH )
        set ( GCC_TARGET_ARCH native )
    endif ()

    message ( STATUS "GCC_TARGET_ARCH: ${GCC_TARGET_ARCH}" )

    set ( GCC_OPTIMIZATION_FLAGS -march=${GCC_TARGET_ARCH} -O3 -ffast-math -fsingle-precision-constant )
    set ( GCC_STRICT_FLAGS -pedantic -pedantic-errors -Werror -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wformat=2 -Weffc++ -Wold-style-cast )
    add_compile_options ( "$<$<COMPILE_LANGUAGE:CXX>:${GCC_OPTIMIZATION_FLAGS}>" )
    add_compile_options ( "$<$<COMPILE_LANGUAGE:CXX>:${GCC_STRICT_FLAGS}>" )
endif ()
