#pragma once

#include <deque>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "message/handler.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"
#include "util/void.hh"

namespace glinthawk::monitoring {

namespace {

template<typename T>
constexpr bool is_string = std::is_same_v<T, std::string> or std::is_same_v<T, const char*> or std::is_same_v<T, char*>
                           or ( std::is_bounded_array_v<T> and std::is_same_v<char, std::remove_extent_t<T>> )
                           or std::is_same_v<T, std::string_view>;
}

class Measurement
{
private:
  std::string name_;
  std::unordered_map<std::string, std::string> tags_ {};
  std::unordered_map<std::string, std::string> fields_ {};

public:
  Measurement( const std::string& name )
    : name_( name )
  {
  }

  Measurement& tag( const std::string& key, const std::string& value )
  {
    tags_[key] = value;
    return *this;
  }

  template<class T>
  Measurement& field( const std::string& key, const T& value )
  {
    static_assert( is_string<T> or std::is_arithmetic_v<T> or std::is_same_v<T, bool>,
                   "Measurement::field(): Unsupported type" );

    if constexpr ( std::is_same_v<T, bool> ) {
      fields_[key] = value ? "true" : "false";
    } else if constexpr ( is_string<T> ) {
      fields_[key] = '"' + value + '"';
    } else if constexpr ( std::is_floating_point_v<T> ) {
      fields_[key] = std::to_string( value );
    } else if constexpr ( std::is_integral_v<T> and std::is_signed_v<T> ) {
      fields_[key] = std::to_string( value ) + "i";
    } else if constexpr ( std::is_integral_v<T> and std::is_unsigned_v<T> ) {
      fields_[key] = std::to_string( value ) + "u";
    }

    return *this;
  }

  std::string to_string() const
  {
    std::string result = name_;
    for ( const auto& [key, value] : tags_ ) {
      if ( value.empty() ) {
        continue;
      }

      result += "," + key + "=" + value;
    }
    result += " ";
    for ( const auto& [key, value] : fields_ ) {
      result += key + "=" + value + ",";
    }
    result.back() = '\n';
    return result;
  }
};

class TelegrafLogger : public glinthawk::MessageHandler<glinthawk::net::UDSSession, Measurement, glinthawk::util::Void>
{
private:
  glinthawk::util::Void incoming_ {};
  std::deque<std::string> outgoing_ {};

  std::string_view unsent_outgoing_measurement_ {};

  void load();

  bool outgoing_empty() const override;
  bool incoming_empty() const override { return true; }
  glinthawk::util::Void& incoming_front() override { return incoming_; }
  void incoming_pop() override { return; }

  void write( RingBuffer& out ) override;
  void read( RingBuffer& in ) override;

  void push_message( Measurement&& ) override {}

public:
  TelegrafLogger( const std::filesystem::path& socket_file );
  ~TelegrafLogger() {}

  void push_measurement( const Measurement& msg );
};

} // namespace glinthawk::monitoring
