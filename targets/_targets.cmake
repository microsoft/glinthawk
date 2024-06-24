list ( APPEND GLINTHAWK_TARGETS worker infer ramble )
list ( APPEND GLINTHAWK_DTYPES FLOAT16 FLOAT32 )

foreach ( TARGET ${GLINTHAWK_TARGETS} )
  foreach ( DTYPE ${GLINTHAWK_DTYPES} )

    if ( DTYPE STREQUAL FLOAT16 )
      set ( TYPE_SUFFIX "fp16" )
    elseif ( DTYPE STREQUAL FLOAT32 )
      set ( TYPE_SUFFIX "fp32" )
    endif ()

    if ( "${__PLATFORM}" STREQUAL "cuda" )
      set_source_files_properties ( ../../src/frontend/${TARGET}.cc PROPERTIES LANGUAGE CUDA )
    endif()

    set ( TARGET_FULL_NAME ${TARGET}-${__PLATFORM}-${TYPE_SUFFIX} )
    add_executable ( ${TARGET_FULL_NAME} ../../src/frontend/${TARGET}.cc )
    target_compile_definitions ( ${TARGET_FULL_NAME} PRIVATE TARGET_DTYPE_${DTYPE} )
    target_link_libraries ( ${TARGET_FULL_NAME} PRIVATE glinthawkcompute_${__PLATFORM} )
    set_target_properties ( ${TARGET_FULL_NAME} PROPERTIES OUTPUT_NAME ${TARGET}-${TYPE_SUFFIX} )

  endforeach ()
endforeach ()
