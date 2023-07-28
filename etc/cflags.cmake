set ( CMAKE_CXX_STANDARD 20 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS OFF )
set ( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if ( MSVC )
    add_compile_options ( /W4 /WX )
else ()
    add_compile_options ( -march=native -O3 )
    add_compile_options ( -pedantic -pedantic-errors -Werror -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wformat=2 -Weffc++ -Wold-style-cast )
endif ()
