#include <cassert>
#include <stdexcept>

#include "http_message.hh"

using namespace std;
using namespace glinthawk::net;

/* methods called by an external parser */
void HTTPMessage::set_first_line( const string_view str )
{
  assert( state_ == FIRST_LINE_PENDING );
  first_line_ = str;
  state_ = HEADERS_PENDING;
}

void HTTPMessage::add_header( const string_view str )
{
  assert( state_ == HEADERS_PENDING );
  headers_.emplace_back( str );
}

void HTTPMessage::add_header( const HTTPHeader& header )
{
  assert( state_ == HEADERS_PENDING );
  headers_.push_back( header );
}

void HTTPMessage::done_with_headers()
{
  assert( state_ == HEADERS_PENDING );
  state_ = BODY_PENDING;

  calculate_expected_body_size();
}

void HTTPMessage::set_expected_body_size( const bool is_known, const size_t value )
{
  assert( state_ == BODY_PENDING );

  expected_body_size_ = make_pair( is_known, value );
}

size_t HTTPMessage::read_in_body( const string_view str )
{
  assert( state_ == BODY_PENDING );

  if ( body_size_is_known() ) {
    /* body size known in advance */

    if ( body_.empty() ) {
      body_.reserve( expected_body_size() );
    }

    assert( body_.size() <= expected_body_size() );
    const size_t amount_to_append = min( expected_body_size() - body_.size(), str.size() );

    body_.append( str.substr( 0, amount_to_append ) );
    if ( body_.size() == expected_body_size() ) {
      state_ = COMPLETE;
    }

    return amount_to_append;
  } else {
    /* body size not known in advance */
    return read_in_complex_body( str );
  }
}

void HTTPMessage::eof()
{
  switch ( state() ) {
    case FIRST_LINE_PENDING:
      return;
    case HEADERS_PENDING:
      throw runtime_error( "HTTPMessage: EOF received in middle of headers" );
    case BODY_PENDING:
      if ( eof_in_body() ) {
        state_ = COMPLETE;
      }
      break;
    case COMPLETE:
      assert( false ); /* nobody should be calling methods on a complete message */
      return;
  }
}

bool HTTPMessage::body_size_is_known() const
{
  assert( state_ > HEADERS_PENDING );
  return expected_body_size_.first;
}

size_t HTTPMessage::expected_body_size() const
{
  assert( body_size_is_known() );
  return expected_body_size_.second;
}

/* locale-insensitive ASCII conversion */
static char http_to_lower( char c )
{
  const char diff = 'A' - 'a';
  if ( c >= 'A' and c <= 'Z' ) {
    c -= diff;
  }
  return c;
}

static string_view strip_initial_whitespace( const string_view str )
{
  size_t first_nonspace = str.find_first_not_of( " " );
  if ( first_nonspace == std::string::npos ) {
    return "";
  } else {
    return str.substr( first_nonspace );
  }
}

/* check if two strings are equivalent per HTTP 1.1 comparison
 * (case-insensitive) */
bool HTTPMessage::equivalent_strings( const string_view a, const string_view b )
{
  const string_view new_a = strip_initial_whitespace( a ), new_b = strip_initial_whitespace( b );

  if ( new_a.size() != new_b.size() ) {
    return false;
  }

  for ( auto it_a = new_a.begin(), it_b = new_b.begin(); it_a < new_a.end(); it_a++, it_b++ ) {
    if ( http_to_lower( *it_a ) != http_to_lower( *it_b ) ) {
      return false;
    }
  }

  return true;
}

bool HTTPMessage::has_header( const string_view header_name ) const
{
  for ( const auto& header : headers_ ) {
    /* canonicalize header name per RFC 2616 section 2.1 */
    if ( equivalent_strings( header.key(), header_name ) ) {
      return true;
    }
  }

  return false;
}

const string_view HTTPMessage::get_header_value( const string_view header_name ) const
{
  for ( const auto& header : headers_ ) {
    /* canonicalize header name per RFC 2616 section 2.1 */
    if ( equivalent_strings( header.key(), header_name ) ) {
      return header.value();
    }
  }

  throw runtime_error( "HTTPMessage header not found: " + string( header_name ) );
}

/* serialize the first line and headers */
void HTTPMessage::serialize_headers( std::string& output ) const
{
  assert( state_ == COMPLETE );

  /* start with first line */
  output = first_line_ + CRLF;

  /* iterate through headers and add "key: value\r\n" to request */
  for ( const auto& header : headers_ ) {
    output.append( header.str() + CRLF );
  }

  /* blank line between headers and body */
  output.append( CRLF );
}

/* convenience constructor */
HTTPMessage::HTTPMessage( string&& first_line, vector<HTTPHeader>&& headers, string&& body )
  : first_line_( std::move( first_line ) )
  , headers_( std::move( headers ) )
  , body_( std::move( body ) )
  , state_( COMPLETE )
{
}
