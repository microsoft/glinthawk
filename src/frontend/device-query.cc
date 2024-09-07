/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// These are helper functions for the SDK samples (string parsing, timers, etc)
#ifndef COMMON_HELPER_STRING_H_
#define COMMON_HELPER_STRING_H_

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cuda_runtime.h>

#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef STRCASECMP
#define STRCASECMP _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#ifndef STRCPY
#define STRCPY( sFilePath, nLength, sPath ) strcpy_s( sFilePath, nLength, sPath )
#endif

#ifndef FOPEN
#define FOPEN( fHandle, filename, mode ) fopen_s( &fHandle, filename, mode )
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL( result ) ( result != 0 )
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#ifndef SPRINTF
#define SPRINTF sprintf_s
#endif
#else // Linux Includes
#include <string.h>
#include <strings.h>

#ifndef STRCASECMP
#define STRCASECMP strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY( sFilePath, nLength, sPath ) strcpy( sFilePath, sPath )
#endif

#ifndef FOPEN
#define FOPEN( fHandle, filename, mode ) ( fHandle = fopen( filename, mode ) )
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL( result ) ( result == NULL )
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#ifndef SPRINTF
#define SPRINTF sprintf
#endif
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// CUDA Utility Helper Functions
inline int stringRemoveDelimiter( char delimiter, const char* string )
{
  int string_start = 0;

  while ( string[string_start] == delimiter ) {
    string_start++;
  }

  if ( string_start >= static_cast<int>( strlen( string ) - 1 ) ) {
    return 0;
  }

  return string_start;
}

inline int getFileExtension( char* filename, char** extension )
{
  int string_length = static_cast<int>( strlen( filename ) );

  while ( filename[string_length--] != '.' ) {
    if ( string_length == 0 )
      break;
  }

  if ( string_length > 0 )
    string_length += 2;

  if ( string_length == 0 )
    *extension = NULL;
  else
    *extension = &filename[string_length];

  return string_length;
}

inline bool checkCmdLineFlag( const int argc, const char** argv, const char* string_ref )
{
  bool bFound = false;

  if ( argc >= 1 ) {
    for ( int i = 1; i < argc; i++ ) {
      int string_start = stringRemoveDelimiter( '-', argv[i] );
      const char* string_argv = &argv[i][string_start];

      const char* equal_pos = strchr( string_argv, '=' );
      int argv_length = static_cast<int>( equal_pos == 0 ? strlen( string_argv ) : equal_pos - string_argv );

      int length = static_cast<int>( strlen( string_ref ) );

      if ( length == argv_length && !STRNCASECMP( string_argv, string_ref, length ) ) {
        bFound = true;
        continue;
      }
    }
  }

  return bFound;
}

// This function wraps the CUDA Driver API into a template function
template<class T>
inline bool getCmdLineArgumentValue( const int argc, const char** argv, const char* string_ref, T* value )
{
  bool bFound = false;

  if ( argc >= 1 ) {
    for ( int i = 1; i < argc; i++ ) {
      int string_start = stringRemoveDelimiter( '-', argv[i] );
      const char* string_argv = &argv[i][string_start];
      int length = static_cast<int>( strlen( string_ref ) );

      if ( !STRNCASECMP( string_argv, string_ref, length ) ) {
        if ( length + 1 <= static_cast<int>( strlen( string_argv ) ) ) {
          int auto_inc = ( string_argv[length] == '=' ) ? 1 : 0;
          *value = (T)atoi( &string_argv[length + auto_inc] );
        }

        bFound = true;
        i = argc;
      }
    }
  }

  return bFound;
}

inline int getCmdLineArgumentInt( const int argc, const char** argv, const char* string_ref )
{
  bool bFound = false;
  int value = -1;

  if ( argc >= 1 ) {
    for ( int i = 1; i < argc; i++ ) {
      int string_start = stringRemoveDelimiter( '-', argv[i] );
      const char* string_argv = &argv[i][string_start];
      int length = static_cast<int>( strlen( string_ref ) );

      if ( !STRNCASECMP( string_argv, string_ref, length ) ) {
        if ( length + 1 <= static_cast<int>( strlen( string_argv ) ) ) {
          int auto_inc = ( string_argv[length] == '=' ) ? 1 : 0;
          value = atoi( &string_argv[length + auto_inc] );
        } else {
          value = 0;
        }

        bFound = true;
        continue;
      }
    }
  }

  if ( bFound ) {
    return value;
  } else {
    return 0;
  }
}

inline float getCmdLineArgumentFloat( const int argc, const char** argv, const char* string_ref )
{
  bool bFound = false;
  float value = -1;

  if ( argc >= 1 ) {
    for ( int i = 1; i < argc; i++ ) {
      int string_start = stringRemoveDelimiter( '-', argv[i] );
      const char* string_argv = &argv[i][string_start];
      int length = static_cast<int>( strlen( string_ref ) );

      if ( !STRNCASECMP( string_argv, string_ref, length ) ) {
        if ( length + 1 <= static_cast<int>( strlen( string_argv ) ) ) {
          int auto_inc = ( string_argv[length] == '=' ) ? 1 : 0;
          value = static_cast<float>( atof( &string_argv[length + auto_inc] ) );
        } else {
          value = 0.f;
        }

        bFound = true;
        continue;
      }
    }
  }

  if ( bFound ) {
    return value;
  } else {
    return 0;
  }
}

inline bool getCmdLineArgumentString( const int argc, const char** argv, const char* string_ref, char** string_retval )
{
  bool bFound = false;

  if ( argc >= 1 ) {
    for ( int i = 1; i < argc; i++ ) {
      int string_start = stringRemoveDelimiter( '-', argv[i] );
      char* string_argv = const_cast<char*>( &argv[i][string_start] );
      int length = static_cast<int>( strlen( string_ref ) );

      if ( !STRNCASECMP( string_argv, string_ref, length ) ) {
        *string_retval = &string_argv[length + 1];
        bFound = true;
        continue;
      }
    }
  }

  if ( !bFound ) {
    *string_retval = NULL;
  }

  return bFound;
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
inline char* sdkFindFilePath( const char* filename, const char* executable_path )
{
  // <executable_name> defines a variable that is replaced with the name of the
  // executable

  // Typical relative search paths to locate needed companion files (e.g. sample
  // input data, or JIT source files) The origin for the relative search may be
  // the .exe file, a .bat file launching an .exe, a browser .exe launching the
  // .exe or .bat, etc
  const char* searchPath[] = {
    "./",      // same dir
    "./data/", // same dir

    "../../../../Samples/<executable_name>/", // up 4 in tree
    "../../../Samples/<executable_name>/",    // up 3 in tree
    "../../Samples/<executable_name>/",       // up 2 in tree

    "../../../../Samples/<executable_name>/data/", // up 4 in tree
    "../../../Samples/<executable_name>/data/",    // up 3 in tree
    "../../Samples/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/0_Introduction/<executable_name>/", // up 4 in tree
    "../../../Samples/0_Introduction/<executable_name>/",    // up 3 in tree
    "../../Samples/0_Introduction/<executable_name>/",       // up 2 in tree

    "../../../../Samples/1_Utilities/<executable_name>/", // up 4 in tree
    "../../../Samples/1_Utilities/<executable_name>/",    // up 3 in tree
    "../../Samples/1_Utilities/<executable_name>/",       // up 2 in tree

    "../../../../Samples/2_Concepts_and_Techniques/<executable_name>/", // up 4 in tree
    "../../../Samples/2_Concepts_and_Techniques/<executable_name>/",    // up 3 in tree
    "../../Samples/2_Concepts_and_Techniques/<executable_name>/",       // up 2 in tree

    "../../../../Samples/3_CUDA_Features/<executable_name>/", // up 4 in tree
    "../../../Samples/3_CUDA_Features/<executable_name>/",    // up 3 in tree
    "../../Samples/3_CUDA_Features/<executable_name>/",       // up 2 in tree

    "../../../../Samples/4_CUDA_Libraries/<executable_name>/", // up 4 in tree
    "../../../Samples/4_CUDA_Libraries/<executable_name>/",    // up 3 in tree
    "../../Samples/4_CUDA_Libraries/<executable_name>/",       // up 2 in tree

    "../../../../Samples/5_Domain_Specific/<executable_name>/", // up 4 in tree
    "../../../Samples/5_Domain_Specific/<executable_name>/",    // up 3 in tree
    "../../Samples/5_Domain_Specific/<executable_name>/",       // up 2 in tree

    "../../../../Samples/6_Performance/<executable_name>/", // up 4 in tree
    "../../../Samples/6_Performance/<executable_name>/",    // up 3 in tree
    "../../Samples/6_Performance/<executable_name>/",       // up 2 in tree

    "../../../../Samples/0_Introduction/<executable_name>/data/", // up 4 in tree
    "../../../Samples/0_Introduction/<executable_name>/data/",    // up 3 in tree
    "../../Samples/0_Introduction/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/1_Utilities/<executable_name>/data/", // up 4 in tree
    "../../../Samples/1_Utilities/<executable_name>/data/",    // up 3 in tree
    "../../Samples/1_Utilities/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/2_Concepts_and_Techniques/<executable_name>/data/", // up 4 in tree
    "../../../Samples/2_Concepts_and_Techniques/<executable_name>/data/",    // up 3 in tree
    "../../Samples/2_Concepts_and_Techniques/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/3_CUDA_Features/<executable_name>/data/", // up 4 in tree
    "../../../Samples/3_CUDA_Features/<executable_name>/data/",    // up 3 in tree
    "../../Samples/3_CUDA_Features/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/4_CUDA_Libraries/<executable_name>/data/", // up 4 in tree
    "../../../Samples/4_CUDA_Libraries/<executable_name>/data/",    // up 3 in tree
    "../../Samples/4_CUDA_Libraries/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/5_Domain_Specific/<executable_name>/data/", // up 4 in tree
    "../../../Samples/5_Domain_Specific/<executable_name>/data/",    // up 3 in tree
    "../../Samples/5_Domain_Specific/<executable_name>/data/",       // up 2 in tree

    "../../../../Samples/6_Performance/<executable_name>/data/", // up 4 in tree
    "../../../Samples/6_Performance/<executable_name>/data/",    // up 3 in tree
    "../../Samples/6_Performance/<executable_name>/data/",       // up 2 in tree

    "../../../../Common/data/", // up 4 in tree
    "../../../Common/data/",    // up 3 in tree
    "../../Common/data/"        // up 2 in tree
  };

  // Extract the executable name
  std::string executable_name;

  if ( executable_path != 0 ) {
    executable_name = std::string( executable_path );

#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
    // Windows path delimiter
    size_t delimiter_pos = executable_name.find_last_of( '\\' );
    executable_name.erase( 0, delimiter_pos + 1 );

    if ( executable_name.rfind( ".exe" ) != std::string::npos ) {
      // we strip .exe, only if the .exe is found
      executable_name.resize( executable_name.size() - 4 );
    }

#else
    // Linux & OSX path delimiter
    size_t delimiter_pos = executable_name.find_last_of( '/' );
    executable_name.erase( 0, delimiter_pos + 1 );
#endif
  }

  // Loop over all search paths and return the first hit
  for ( unsigned int i = 0; i < sizeof( searchPath ) / sizeof( char* ); ++i ) {
    std::string path( searchPath[i] );
    size_t executable_name_pos = path.find( "<executable_name>" );

    // If there is executable_name variable in the searchPath
    // replace it with the value
    if ( executable_name_pos != std::string::npos ) {
      if ( executable_path != 0 ) {
        path.replace( executable_name_pos, strlen( "<executable_name>" ), executable_name );
      } else {
        // Skip this path entry if no executable argument is given
        continue;
      }
    }

#ifdef _DEBUG
    printf( "sdkFindFilePath <%s> in %s\n", filename, path.c_str() );
#endif

    // Test if the file exists
    path.append( filename );
    FILE* fp;
    FOPEN( fp, path.c_str(), "rb" );

    if ( fp != NULL ) {
      fclose( fp );
      // File found
      // returning an allocated array here for backwards compatibility reasons
      char* file_path = reinterpret_cast<char*>( malloc( path.length() + 1 ) );
      STRCPY( file_path, path.length() + 1, path.c_str() );
      return file_path;
    }

    if ( fp ) {
      fclose( fp );
    }
  }

  // File not found
  printf( "\nerror: sdkFindFilePath: file <%s> not found!\n", filename );
  return 0;
}

#endif // COMMON_HELPER_STRING_H_

#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header
// files, please refer the CUDA examples for examples of the needed CUDA
// headers, which may change depending on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char* _cudaGetErrorEnum( cudaError_t error ) { return cudaGetErrorName( error ); }
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char* _cudaGetErrorEnum( CUresult error )
{
  static char unknown[] = "<unknown>";
  const char* ret = NULL;
  cuGetErrorName( error, &ret );
  return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char* _cudaGetErrorEnum( cublasStatus_t error )
{
  switch ( error ) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef _CUFFT_H_
// cuFFT API errors
static const char* _cudaGetErrorEnum( cufftResult error )
{
  switch ( error ) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef CUSPARSEAPI
// cuSPARSE API errors
static const char* _cudaGetErrorEnum( cusparseStatus_t error )
{
  switch ( error ) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
// cuSOLVER API errors
static const char* _cudaGetErrorEnum( cusolverStatus_t error )
{
  switch ( error ) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
#endif

#ifdef CURAND_H_
// cuRAND API errors
static const char* _cudaGetErrorEnum( curandStatus_t error )
{
  switch ( error ) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NVJPEGAPI
// nvJPEG API errors
static const char* _cudaGetErrorEnum( nvjpegStatus_t error )
{
  switch ( error ) {
    case NVJPEG_STATUS_SUCCESS:
      return "NVJPEG_STATUS_SUCCESS";

    case NVJPEG_STATUS_NOT_INITIALIZED:
      return "NVJPEG_STATUS_NOT_INITIALIZED";

    case NVJPEG_STATUS_INVALID_PARAMETER:
      return "NVJPEG_STATUS_INVALID_PARAMETER";

    case NVJPEG_STATUS_BAD_JPEG:
      return "NVJPEG_STATUS_BAD_JPEG";

    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
      return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

    case NVJPEG_STATUS_EXECUTION_FAILED:
      return "NVJPEG_STATUS_EXECUTION_FAILED";

    case NVJPEG_STATUS_ARCH_MISMATCH:
      return "NVJPEG_STATUS_ARCH_MISMATCH";

    case NVJPEG_STATUS_INTERNAL_ERROR:
      return "NVJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
// NPP API errors
static const char* _cudaGetErrorEnum( NppStatus error )
{
  switch ( error ) {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";

    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";

    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
      return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ( ( NPP_VERSION_MAJOR << 12 ) + ( NPP_VERSION_MINOR << 4 ) ) <= 0x5000

    case NPP_BAD_ARG_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFF_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECT_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUAD_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEM_ALLOC_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_INVALID_INPUT:
      return "NPP_INVALID_INPUT";

    case NPP_POINTER_ERROR:
      return "NPP_POINTER_ERROR";

    case NPP_WARNING:
      return "NPP_WARNING";

    case NPP_ODD_ROI_WARNING:
      return "NPP_ODD_ROI_WARNING";
#else

    // These are for CUDA 5.5 or higher
    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFFICIENT_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECTANGLE_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUADRANGLE_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR";

    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";

    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";

    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";

    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";

    case NPP_RESIZE_FACTOR_ERROR:
      return "NPP_RESIZE_FACTOR_ERROR";

    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
      return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

#if ( ( NPP_VERSION_MAJOR << 12 ) + ( NPP_VERSION_MINOR << 4 ) ) <= 0x5000

    case NPP_MEMFREE_ERR:
      return "NPP_MEMFREE_ERR";

    case NPP_MEMSET_ERR:
      return "NPP_MEMSET_ERR";

    case NPP_MEMCPY_ERR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERR:
      return "NPP_MIRROR_FLIP_ERR";
#else

    case NPP_MEMFREE_ERROR:
      return "NPP_MEMFREE_ERROR";

    case NPP_MEMSET_ERROR:
      return "NPP_MEMSET_ERROR";

    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERROR:
      return "NPP_MIRROR_FLIP_ERROR";
#endif

    case NPP_ALIGNMENT_ERROR:
      return "NPP_ALIGNMENT_ERROR";

    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";

    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";

    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";

    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR";

    case NPP_ERROR:
      return "NPP_ERROR";

    case NPP_SUCCESS:
      return "NPP_SUCCESS";

    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

    case NPP_MISALIGNED_DST_ROI_WARNING:
      return "NPP_MISALIGNED_DST_ROI_WARNING";

    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

    case NPP_DOUBLE_SIZE_WARNING:
      return "NPP_DOUBLE_SIZE_WARNING";

    case NPP_WRONG_INTERSECTION_ROI_WARNING:
      return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ( ( NPP_VERSION_MAJOR << 12 ) + ( NPP_VERSION_MINOR << 4 ) ) >= 0x6000
    /* These are 6.0 or higher */
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
      return "NPP_LUT_PALETTE_BITSIZE_ERROR";

    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

    case NPP_QUALITY_INDEX_ERROR:
      return "NPP_QUALITY_INDEX_ERROR";

    case NPP_CHANNEL_ORDER_ERROR:
      return "NPP_CHANNEL_ORDER_ERROR";

    case NPP_ZERO_MASK_VALUE_ERROR:
      return "NPP_ZERO_MASK_VALUE_ERROR";

    case NPP_NUMBER_OF_CHANNELS_ERROR:
      return "NPP_NUMBER_OF_CHANNELS_ERROR";

    case NPP_COI_ERROR:
      return "NPP_COI_ERROR";

    case NPP_DIVISOR_ERROR:
      return "NPP_DIVISOR_ERROR";

    case NPP_CHANNEL_ERROR:
      return "NPP_CHANNEL_ERROR";

    case NPP_STRIDE_ERROR:
      return "NPP_STRIDE_ERROR";

    case NPP_ANCHOR_ERROR:
      return "NPP_ANCHOR_ERROR";

    case NPP_MASK_SIZE_ERROR:
      return "NPP_MASK_SIZE_ERROR";

    case NPP_MOMENT_00_ZERO_ERROR:
      return "NPP_MOMENT_00_ZERO_ERROR";

    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
      return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

    case NPP_THRESHOLD_ERROR:
      return "NPP_THRESHOLD_ERROR";

    case NPP_CONTEXT_MATCH_ERROR:
      return "NPP_CONTEXT_MATCH_ERROR";

    case NPP_FFT_FLAG_ERROR:
      return "NPP_FFT_FLAG_ERROR";

    case NPP_FFT_ORDER_ERROR:
      return "NPP_FFT_ORDER_ERROR";

    case NPP_SCALE_RANGE_ERROR:
      return "NPP_SCALE_RANGE_ERROR";

    case NPP_DATA_TYPE_ERROR:
      return "NPP_DATA_TYPE_ERROR";

    case NPP_OUT_OFF_RANGE_ERROR:
      return "NPP_OUT_OFF_RANGE_ERROR";

    case NPP_DIVIDE_BY_ZERO_ERROR:
      return "NPP_DIVIDE_BY_ZERO_ERROR";

    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";

    case NPP_NO_MEMORY_ERROR:
      return "NPP_NO_MEMORY_ERROR";

    case NPP_ERROR_RESERVED:
      return "NPP_ERROR_RESERVED";

    case NPP_NO_OPERATION_WARNING:
      return "NPP_NO_OPERATION_WARNING";

    case NPP_DIVIDE_BY_ZERO_WARNING:
      return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ( ( NPP_VERSION_MAJOR << 12 ) + ( NPP_VERSION_MINOR << 4 ) ) >= 0x7000
    /* These are 7.0 or higher */
    case NPP_OVERFLOW_ERROR:
      return "NPP_OVERFLOW_ERROR";

    case NPP_CORRUPTED_DATA_ERROR:
      return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}
#endif

template<typename T>
void check( T result, char const* const func, const char* const file, int const line )
{
  if ( result ) {
    fprintf( stderr,
             "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
             file,
             line,
             static_cast<unsigned int>( result ),
             _cudaGetErrorEnum( result ),
             func );
    exit( EXIT_FAILURE );
  }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors( val ) check( ( val ), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError( msg ) __getLastCudaError( msg, __FILE__, __LINE__ )

inline void __getLastCudaError( const char* errorMessage, const char* file, const int line )
{
  cudaError_t err = cudaGetLastError();

  if ( cudaSuccess != err ) {
    fprintf( stderr,
             "%s(%i) : getLastCudaError() CUDA error :"
             " %s : (%d) %s.\n",
             file,
             line,
             errorMessage,
             static_cast<int>( err ),
             cudaGetErrorString( err ) );
    exit( EXIT_FAILURE );
  }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError( msg ) __printLastCudaError( msg, __FILE__, __LINE__ )

inline void __printLastCudaError( const char* errorMessage, const char* file, const int line )
{
  cudaError_t err = cudaGetLastError();

  if ( cudaSuccess != err ) {
    fprintf( stderr,
             "%s(%i) : getLastCudaError() CUDA error :"
             " %s : (%d) %s.\n",
             file,
             line,
             errorMessage,
             static_cast<int>( err ),
             cudaGetErrorString( err ) );
  }
}
#endif

#ifndef MAX
#define MAX( a, b ) ( a > b ? a : b )
#endif

// Float To Int conversion
inline int ftoi( float value )
{
  return ( value >= 0 ? static_cast<int>( value + 0.5 ) : static_cast<int>( value - 0.5 ) );
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores( int major, int minor )
{
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[]
    = { { 0x30, 192 }, { 0x32, 192 }, { 0x35, 192 }, { 0x37, 192 }, { 0x50, 128 }, { 0x52, 128 }, { 0x53, 128 },
        { 0x60, 64 },  { 0x61, 128 }, { 0x62, 128 }, { 0x70, 64 },  { 0x72, 64 },  { 0x75, 64 },  { 0x80, 64 },
        { 0x86, 128 }, { 0x87, 128 }, { 0x89, 128 }, { 0x90, 128 }, { -1, -1 } };

  int index = 0;

  while ( nGpuArchCoresPerSM[index].SM != -1 ) {
    if ( nGpuArchCoresPerSM[index].SM == ( ( major << 4 ) + minor ) ) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf( "MapSMtoCores for SM %d.%d is undefined."
          "  Default to use %d Cores/SM\n",
          major,
          minor,
          nGpuArchCoresPerSM[index - 1].Cores );
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char* _ConvertSMVer2ArchName( int major, int minor )
{
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[]
    = { { 0x30, "Kepler" },  { 0x32, "Kepler" },  { 0x35, "Kepler" }, { 0x37, "Kepler" },       { 0x50, "Maxwell" },
        { 0x52, "Maxwell" }, { 0x53, "Maxwell" }, { 0x60, "Pascal" }, { 0x61, "Pascal" },       { 0x62, "Pascal" },
        { 0x70, "Volta" },   { 0x72, "Xavier" },  { 0x75, "Turing" }, { 0x80, "Ampere" },       { 0x86, "Ampere" },
        { 0x87, "Ampere" },  { 0x89, "Ada" },     { 0x90, "Hopper" }, { -1, "Graphics Device" } };

  int index = 0;

  while ( nGpuArchNameSM[index].SM != -1 ) {
    if ( nGpuArchNameSM[index].SM == ( ( major << 4 ) + minor ) ) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf( "MapSMtoArchName for SM %d.%d is undefined."
          "  Default to use %s\n",
          major,
          minor,
          nGpuArchNameSM[index - 1].name );
  return nGpuArchNameSM[index - 1].name;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit( int devID )
{
  int device_count;
  checkCudaErrors( cudaGetDeviceCount( &device_count ) );

  if ( device_count == 0 ) {
    fprintf( stderr,
             "gpuDeviceInit() CUDA error: "
             "no devices supporting CUDA.\n" );
    exit( EXIT_FAILURE );
  }

  if ( devID < 0 ) {
    devID = 0;
  }

  if ( devID > device_count - 1 ) {
    fprintf( stderr, "\n" );
    fprintf( stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count );
    fprintf( stderr,
             ">> gpuDeviceInit (-device=%d) is not a valid"
             " GPU device. <<\n",
             devID );
    fprintf( stderr, "\n" );
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkCudaErrors( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, devID ) );
  checkCudaErrors( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, devID ) );
  checkCudaErrors( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, devID ) );
  if ( computeMode == cudaComputeModeProhibited ) {
    fprintf( stderr,
             "Error: device is running in <Compute Mode "
             "Prohibited>, no threads can use cudaSetDevice().\n" );
    return -1;
  }

  if ( major < 1 ) {
    fprintf( stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n" );
    exit( EXIT_FAILURE );
  }

  checkCudaErrors( cudaSetDevice( devID ) );
  printf( "gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName( major, minor ) );

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors( cudaGetDeviceCount( &device_count ) );

  if ( device_count == 0 ) {
    fprintf( stderr,
             "gpuGetMaxGflopsDeviceId() CUDA error:"
             " no devices supporting CUDA.\n" );
    exit( EXIT_FAILURE );
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while ( current_device < device_count ) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, current_device ) );
    checkCudaErrors( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, current_device ) );
    checkCudaErrors( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, current_device ) );

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if ( computeMode != cudaComputeModeProhibited ) {
      if ( major == 9999 && minor == 9999 ) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ConvertSMVer2Cores( major, minor );
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors( cudaDeviceGetAttribute( &multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device ) );
      cudaError_t result = cudaDeviceGetAttribute( &clockRate, cudaDevAttrClockRate, current_device );
      if ( result != cudaSuccess ) {
        // If cudaDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
        if ( result == cudaErrorInvalidValue ) {
          clockRate = 1;
        } else {
          fprintf( stderr,
                   "CUDA error at %s:%d code=%d(%s) \n",
                   __FILE__,
                   __LINE__,
                   static_cast<unsigned int>( result ),
                   _cudaGetErrorEnum( result ) );
          exit( EXIT_FAILURE );
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if ( compute_perf > max_compute_perf ) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if ( devices_prohibited == device_count ) {
    fprintf( stderr,
             "gpuGetMaxGflopsDeviceId() CUDA error:"
             " all devices have compute mode prohibited.\n" );
    exit( EXIT_FAILURE );
  }

  return max_perf_device;
}

// Initialization code to find the best CUDA Device
inline int findCudaDevice( int argc, const char** argv )
{
  int devID = 0;

  // If the command-line has a device number specified, use it
  if ( checkCmdLineFlag( argc, argv, "device" ) ) {
    devID = getCmdLineArgumentInt( argc, argv, "device=" );

    if ( devID < 0 ) {
      printf( "Invalid command line parameter\n " );
      exit( EXIT_FAILURE );
    } else {
      devID = gpuDeviceInit( devID );

      if ( devID < 0 ) {
        printf( "exiting...\n" );
        exit( EXIT_FAILURE );
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors( cudaSetDevice( devID ) );
    int major = 0, minor = 0;
    checkCudaErrors( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, devID ) );
    checkCudaErrors( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, devID ) );
    printf( "GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
            devID,
            _ConvertSMVer2ArchName( major, minor ),
            major,
            minor );
  }

  return devID;
}

inline int findIntegratedGPU()
{
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  checkCudaErrors( cudaGetDeviceCount( &device_count ) );

  if ( device_count == 0 ) {
    fprintf( stderr, "CUDA error: no devices supporting CUDA.\n" );
    exit( EXIT_FAILURE );
  }

  // Find the integrated GPU which is compute capable
  while ( current_device < device_count ) {
    int computeMode = -1, integrated = -1;
    checkCudaErrors( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, current_device ) );
    checkCudaErrors( cudaDeviceGetAttribute( &integrated, cudaDevAttrIntegrated, current_device ) );
    // If GPU is integrated and is not running on Compute Mode prohibited,
    // then cuda can map to GLES resource
    if ( integrated && ( computeMode != cudaComputeModeProhibited ) ) {
      checkCudaErrors( cudaSetDevice( current_device ) );

      int major = 0, minor = 0;
      checkCudaErrors( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, current_device ) );
      checkCudaErrors( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, current_device ) );
      printf( "GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
              current_device,
              _ConvertSMVer2ArchName( major, minor ),
              major,
              minor );

      return current_device;
    } else {
      devices_prohibited++;
    }

    current_device++;
  }

  if ( devices_prohibited == device_count ) {
    fprintf( stderr,
             "CUDA error:"
             " No GLES-CUDA Interop capable GPU found.\n" );
    exit( EXIT_FAILURE );
  }

  return -1;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities( int major_version, int minor_version )
{
  int dev;
  int major = 0, minor = 0;

  checkCudaErrors( cudaGetDevice( &dev ) );
  checkCudaErrors( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, dev ) );
  checkCudaErrors( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, dev ) );

  if ( ( major > major_version ) || ( major == major_version && minor >= minor_version ) ) {
    printf(
      "  Device %d: <%16s >, Compute SM %d.%d detected\n", dev, _ConvertSMVer2ArchName( major, minor ), major, minor );
    return true;
  } else {
    printf( "  No GPU device was found that can support "
            "CUDA compute capability %d.%d.\n",
            major_version,
            minor_version );
    return false;
  }
}
#endif

// end of CUDA Helper Functions

#endif // COMMON_HELPER_CUDA_H_

#include <iostream>
#include <memory>
#include <string>

int* pArgc = NULL;
char** pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template<class T>
inline void getCudaAttribute( T* attribute, CUdevice_attribute device_attribute, int device )
{
  CUresult error = cuDeviceGetAttribute( attribute, device_attribute, device );

  if ( CUDA_SUCCESS != error ) {
    fprintf(
      stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n", error, __FILE__, __LINE__ );

    exit( EXIT_FAILURE );
  }
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
  pArgc = &argc;
  pArgv = argv;

  printf( "%s Starting...\n\n", argv[0] );
  printf( " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n" );

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount( &deviceCount );

  if ( error_id != cudaSuccess ) {
    printf( "cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>( error_id ), cudaGetErrorString( error_id ) );
    printf( "Result = FAIL\n" );
    exit( EXIT_FAILURE );
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if ( deviceCount == 0 ) {
    printf( "There are no available device(s) that support CUDA\n" );
  } else {
    printf( "Detected %d CUDA Capable device(s)\n", deviceCount );
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for ( dev = 0; dev < deviceCount; ++dev ) {
    cudaSetDevice( dev );
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, dev );

    printf( "\nDevice %d: \"%s\"\n", dev, deviceProp.name );

    // Console log
    cudaDriverGetVersion( &driverVersion );
    cudaRuntimeGetVersion( &runtimeVersion );
    printf( "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
            driverVersion / 1000,
            ( driverVersion % 100 ) / 10,
            runtimeVersion / 1000,
            ( runtimeVersion % 100 ) / 10 );
    printf( "  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor );

    char msg[256];
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
    sprintf_s( msg,
               sizeof( msg ),
               "  Total amount of global memory:                 %.0f MBytes "
               "(%llu bytes)\n",
               static_cast<float>( deviceProp.totalGlobalMem / 1048576.0f ),
               (unsigned long long)deviceProp.totalGlobalMem );
#else
    snprintf( msg,
              sizeof( msg ),
              "  Total amount of global memory:                 %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>( deviceProp.totalGlobalMem / 1048576.0f ),
              (unsigned long long)deviceProp.totalGlobalMem );
#endif
    printf( "%s", msg );

    printf( "  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
            deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores( deviceProp.major, deviceProp.minor ),
            _ConvertSMVer2Cores( deviceProp.major, deviceProp.minor ) * deviceProp.multiProcessorCount );
    printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
            "GHz)\n",
            deviceProp.clockRate * 1e-3f,
            deviceProp.clockRate * 1e-6f );

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf( "  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f );
    printf( "  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth );

    if ( deviceProp.l2CacheSize ) {
      printf( "  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize );
    }

#else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>( &memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev );
    printf( "  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f );
    int memBusWidth;
    getCudaAttribute<int>( &memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev );
    printf( "  Memory Bus Width:                              %d-bit\n", memBusWidth );
    int L2CacheSize;
    getCudaAttribute<int>( &L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev );

    if ( L2CacheSize ) {
      printf( "  L2 Cache Size:                                 %d bytes\n", L2CacheSize );
    }

#endif

    printf( "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
            "%d), 3D=(%d, %d, %d)\n",
            deviceProp.maxTexture1D,
            deviceProp.maxTexture2D[0],
            deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0],
            deviceProp.maxTexture3D[1],
            deviceProp.maxTexture3D[2] );
    printf( "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
            deviceProp.maxTexture1DLayered[0],
            deviceProp.maxTexture1DLayered[1] );
    printf( "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
            "layers\n",
            deviceProp.maxTexture2DLayered[0],
            deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2] );

    printf( "  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem );
    printf( "  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock );
    printf( "  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor );
    printf( "  Total number of registers available per block: %d\n", deviceProp.regsPerBlock );
    printf( "  Warp size:                                     %d\n", deviceProp.warpSize );
    printf( "  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor );
    printf( "  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock );
    printf( "  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2] );
    printf( "  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2] );
    printf( "  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch );
    printf( "  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment );
    printf( "  Concurrent copy and kernel execution:          %s with %d copy "
            "engine(s)\n",
            ( deviceProp.deviceOverlap ? "Yes" : "No" ),
            deviceProp.asyncEngineCount );
    printf( "  Run time limit on kernels:                     %s\n",
            deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No" );
    printf( "  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No" );
    printf( "  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No" );
    printf( "  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No" );
    printf( "  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled" );
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
    printf( "  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
            deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)" );
#endif
    printf( "  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No" );
    printf( "  Device supports Managed Memory:                %s\n", deviceProp.managedMemory ? "Yes" : "No" );
    printf( "  Device supports Compute Preemption:            %s\n",
            deviceProp.computePreemptionSupported ? "Yes" : "No" );
    printf( "  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No" );
    printf( "  Supports MultiDevice Co-op Kernel Launch:      %s\n",
            deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No" );
    printf( "  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
            deviceProp.pciDomainID,
            deviceProp.pciBusID,
            deviceProp.pciDeviceID );

    const char* sComputeMode[] = { "Default (multiple host threads can use ::cudaSetDevice() with device "
                                   "simultaneously)",
                                   "Exclusive (only one host thread in one process is able to use "
                                   "::cudaSetDevice() with this device)",
                                   "Prohibited (no host thread can use ::cudaSetDevice() with this "
                                   "device)",
                                   "Exclusive Process (many threads in one process is able to use "
                                   "::cudaSetDevice() with this device)",
                                   "Unknown",
                                   NULL };
    printf( "  Compute Mode:\n" );
    printf( "     < %s >\n", sComputeMode[deviceProp.computeMode] );
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if ( deviceCount >= 2 ) {
    cudaDeviceProp prop[64];
    int gpuid[64]; // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;

    for ( int i = 0; i < deviceCount; i++ ) {
      checkCudaErrors( cudaGetDeviceProperties( &prop[i], i ) );

      // Only boards based on Fermi or later can support P2P
      if ( ( prop[i].major >= 2 )
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
           // on Windows (64-bit), the Tesla Compute Cluster driver for windows
           // must be enabled to support this
           && prop[i].tccDriver
#endif
      ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;

    if ( gpu_p2p_count >= 2 ) {
      for ( int i = 0; i < gpu_p2p_count; i++ ) {
        for ( int j = 0; j < gpu_p2p_count; j++ ) {
          if ( gpuid[i] == gpuid[j] ) {
            continue;
          }
          checkCudaErrors( cudaDeviceCanAccessPeer( &can_access_peer, gpuid[i], gpuid[j] ) );
          printf( "> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                  prop[gpuid[i]].name,
                  gpuid[i],
                  prop[gpuid[j]].name,
                  gpuid[j],
                  can_access_peer ? "Yes" : "No" );
        }
      }
    }
  }

  // csv masterlog info
  // *****************************
  // exe and CUDA driver name
  printf( "\n" );
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
  char cTemp[16];

  // driver version
  sProfileString += ", CUDA Driver Version = ";
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
  sprintf_s( cTemp, 10, "%d.%d", driverVersion / 1000, ( driverVersion % 100 ) / 10 );
#else
  snprintf( cTemp, sizeof( cTemp ), "%d.%d", driverVersion / 1000, ( driverVersion % 100 ) / 10 );
#endif
  sProfileString += cTemp;

  // Runtime version
  sProfileString += ", CUDA Runtime Version = ";
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
  sprintf_s( cTemp, 10, "%d.%d", runtimeVersion / 1000, ( runtimeVersion % 100 ) / 10 );
#else
  snprintf( cTemp, sizeof( cTemp ), "%d.%d", runtimeVersion / 1000, ( runtimeVersion % 100 ) / 10 );
#endif
  sProfileString += cTemp;

  // Device count
  sProfileString += ", NumDevs = ";
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
  sprintf_s( cTemp, 10, "%d", deviceCount );
#else
  snprintf( cTemp, sizeof( cTemp ), "%d", deviceCount );
#endif
  sProfileString += cTemp;
  sProfileString += "\n";
  printf( "%s", sProfileString.c_str() );

  printf( "Result = PASS\n" );

  // finish
  exit( EXIT_SUCCESS );
}