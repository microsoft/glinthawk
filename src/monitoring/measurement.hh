#pragma once

#include <string>
#include <type_traits>
#include <unordered_map>

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

  std::unordered_map<std::string, uint64_t> fields_uint_ {};
  std::unordered_map<std::string, int64_t> fields_int_ {};
  std::unordered_map<std::string, double> fields_double_ {};
  std::unordered_map<std::string, float> fields_float_ {};
  std::unordered_map<std::string, bool> fields_bool_ {};
  std::unordered_map<std::string, std::string> fields_string_ {};

public:
  Measurement( const std::string& name )
    : name_( name )
  {
  }

  void tag( const std::string& key, const std::string& value ) { tags_[key] = value; }

  /// @brief Set a field by key
  template<class T>
  void field( const std::string& key, const T& value )
  {
    static_assert( is_string<T> or std::is_arithmetic_v<T> or std::is_same_v<T, bool>,
                   "Measurement::field(): Unsupported type" );

    if constexpr ( std::is_same_v<T, bool> ) {
      fields_bool_[key] = value;
    } else if constexpr ( is_string<T> ) {
      fields_string_[key] = value;
    } else if constexpr ( std::is_same_v<T, double> ) {
      fields_double_[key] = value;
    } else if constexpr ( std::is_same_v<T, float> ) {
      fields_double_[key] = value;
    } else if constexpr ( std::is_integral_v<T> and std::is_signed_v<T> ) {
      fields_int_[key] = value;
    } else if constexpr ( std::is_integral_v<T> and std::is_unsigned_v<T> ) {
      fields_uint_[key] = value;
    }
  }

  /// @brief Get a reference to value of a field by key
  template<class T>
  T& field( const std::string& key )
  {
    static_assert( is_string<T> or std::is_arithmetic_v<T> or std::is_same_v<T, bool>,
                   "Measurement::field(): Unsupported type" );

    if constexpr ( std::is_same_v<T, bool> ) {
      return fields_bool_[key];
    } else if constexpr ( is_string<T> ) {
      return fields_string_[key];
    } else if constexpr ( std::is_same_v<T, double> ) {
      return fields_double_[key];
    } else if constexpr ( std::is_same_v<T, float> ) {
      return fields_double_[key];
    } else if constexpr ( std::is_integral_v<T> and std::is_signed_v<T> ) {
      return fields_int_[key];
    } else if constexpr ( std::is_integral_v<T> and std::is_unsigned_v<T> ) {
      return fields_uint_[key];
    }
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

    for ( const auto& [key, value] : fields_uint_ ) {
      result += key + "=" + std::to_string( value ) + "u,";
    }

    for ( const auto& [key, value] : fields_int_ ) {
      result += key + "=" + std::to_string( value ) + "i,";
    }

    for ( const auto& [key, value] : fields_double_ ) {
      result += key + "=" + std::to_string( value ) + ",";
    }

    for ( const auto& [key, value] : fields_float_ ) {
      result += key + "=" + std::to_string( value ) + ",";
    }

    for ( const auto& [key, value] : fields_bool_ ) {
      result += key + "=" + ( value ? "true," : "false," );
    }

    for ( const auto& [key, value] : fields_string_ ) {
      result += key + "=\"" + value + "\",";
    }

    result.back() = '\n';
    return result;
  }
};

} // namespace glinthawk::monitoring
