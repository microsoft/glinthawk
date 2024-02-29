file ( GLOB_RECURSE SOURCE_COMPUTE_CUDA ../../src/arch/*.cu )
list ( APPEND SOURCE_COMPUTE_CUDA ${SOURCE_COMPUTE} )

set_source_files_properties( ${SOURCE_COMPUTE_CUDA} PROPERTIES LANGUAGE CUDA)

add_library ( glinthawkcompute_cuda OBJECT ${SOURCE_COMPUTE_CUDA} )
target_compile_definitions ( glinthawkcompute_cuda PUBLIC TARGET_PLATFORM_CUDA )
target_link_libraries ( glinthawkcompute_cuda
  CUDA::cudart CUDA::cuda_driver cudart CUDA::cublas CUDA::cublasLt cublas cublasLt )

# CUDA_CONVERT_FLAGS ( glinthawkcompute_cuda )

list ( APPEND GLINTHAWK_TARGETS profile-stage infer-hybrid )

set ( __PLATFORM cuda )
include ( ../_targets.cmake )
