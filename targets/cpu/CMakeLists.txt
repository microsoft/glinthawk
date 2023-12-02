add_library ( glinthawkcompute_cpu OBJECT ${SOURCE_COMPUTE} )
target_compile_definitions ( glinthawkcompute_cpu PUBLIC TARGET_PLATFORM_CPU )

set ( __PLATFORM cpu )
include ( ../_targets.cmake )
