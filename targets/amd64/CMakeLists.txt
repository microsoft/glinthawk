add_library ( glinthawkcompute_amd64 OBJECT ${SOURCE_COMPUTE} )
target_compile_definitions ( glinthawkcompute_amd64 PUBLIC TARGET_PLATFORM_AMD64 )

set ( __PLATFORM amd64 )
include ( ../_targets.cmake )
