file ( GLOB_RECURSE SOURCE_COMPUTE_CUDA ../../src/arch/*.cu )
list ( APPEND SOURCE_COMPUTE_CUDA ${SOURCE_COMPUTE} )

set_source_files_properties( ${SOURCE_COMPUTE_CUDA} PROPERTIES LANGUAGE CUDA)

add_library ( glinthawkcompute_cuda OBJECT ${SOURCE_COMPUTE_CUDA} )
target_compile_definitions ( glinthawkcompute_cuda PUBLIC TARGET_PLATFORM_CUDA )
target_link_libraries ( glinthawkcompute_cuda
  CUDA::cudart CUDA::cuda_driver cudart CUDA::cublas CUDA::cublasLt cublas cublasLt )

# CUDA_CONVERT_FLAGS ( glinthawkcompute_cuda )

add_executable ( profile-offload ../../src/frontend/profile-offload.cu )
target_link_libraries ( profile-offload glinthawkcompute_cuda )

# Probably incorrect:
add_executable ( worker_both ../../src/frontend/worker_both.cc )
target_link_libraries ( worker_both glinthawkcompute_cuda )

set ( __PLATFORM cuda )
include ( ../_targets.cmake )
