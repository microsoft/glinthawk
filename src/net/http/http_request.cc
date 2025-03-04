#include <cassert>
#include <stdexcept>
#include <string>

#include "http_request.hh"
#include "util/convert.hh"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::net;

void HTTPRequest::calculate_expected_body_size()
{
  assert( state_ == BODY_PENDING );
  if ( first_line_.substr( 0, 4 ) == "GET " or first_line_.substr( 0, 5 ) == "HEAD " ) {
    set_expected_body_size( true, 0 );
  } else if ( first_line_.substr( 0, 5 ) == "POST " or first_line_.substr( 0, 4 ) == "PUT " ) {
    if ( !has_header( "Content-Length" ) ) {
      throw runtime_error( "HTTPRequest: does not support chunked requests" );
    }

    set_expected_body_size( true, util::to_uint64( get_header_value( "Content-Length" ) ) );
  } else {
    throw runtime_error( "Cannot handle HTTP method: " + first_line_ );
  }
}

size_t HTTPRequest::read_in_complex_body( const string_view )
{
  /* we don't support complex bodies */
  throw runtime_error( "HTTPRequest: does not support chunked requests" );
}

bool HTTPRequest::eof_in_body() const { throw runtime_error( "HTTPRequest: got EOF in middle of body" ); }

bool HTTPRequest::is_head() const
{
  assert( state_ > FIRST_LINE_PENDING );
  /* RFC 2616 5.1.1 says "The method is case-sensitive." */
  return first_line_.substr( 0, 5 ) == "HEAD ";
}
