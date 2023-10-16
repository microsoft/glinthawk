#pragma once

#include <endian.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#include <google/protobuf/util/json_util.h>

namespace glinthawk::core {

std::string put_field( const bool n );
std::string put_field( const uint64_t n );
std::string put_field( const uint32_t n );
std::string put_field( const uint16_t n );

void put_field( char* message, const bool n, size_t loc );
void put_field( char* message, const uint64_t n, size_t loc );
void put_field( char* message, const uint32_t n, size_t loc );
void put_field( char* message, const uint16_t n, size_t loc );

template<class T>
T get_field( const std::string_view str );

/* avoid implicit conversions */
template<class T>
std::string put_field( T n ) = delete;

template<class T>
std::string put_field( char* message, T n, size_t ) = delete;

namespace protoutil {

template<class ProtobufType>
std::string to_json( const ProtobufType& protobuf, const bool pretty_print = false )
{
  using namespace google::protobuf::util;
  JsonPrintOptions print_options;
  print_options.add_whitespace = pretty_print;
  print_options.always_print_primitive_fields = true;

  std::string ret;
  if ( not MessageToJsonString( protobuf, &ret, print_options ).ok() ) {
    throw std::runtime_error( "cannot convert protobuf to json" );
  }

  return ret;
}

template<class ProtobufType>
void from_json( const std::string& data, ProtobufType& dest )
{
  using namespace google::protobuf::util;

  if ( not JsonStringToMessage( data, &dest ).ok() ) {
    throw std::runtime_error( "cannot convert json to protobuf" );
  }
}

} // namespace protoutil

} // namespace glinthawk::core
