/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */

#include "uri.hh"

#include "split.hh"

#include <regex>
#include <stdexcept>

using namespace std;
using namespace glinthawk::util;

ParsedURI::ParsedURI( const std::string& uri )
{
  const static regex uri_regex {
    R"RAWSTR((([A-Za-z0-9]+)://)?(([^:\n\r]+):([^@\n\r]+)@)?(([^?:/\n\r]+):?(\d*))?/?([^?\n\r]+)?\??([^#\n\r]*)?#?([^\n\r]*))RAWSTR",
    regex::optimize
  };

  smatch uri_match_result;

  if ( regex_match( uri, uri_match_result, uri_regex ) ) {
    protocol = uri_match_result[2];
    username = uri_match_result[4];
    password = uri_match_result[5];
    host = uri_match_result[7];
    path = uri_match_result[9];

    if ( uri_match_result[8].length() ) {
      port = stoul( uri_match_result[8] );
    }

    if ( uri_match_result[10].length() ) {
      const string query_string = uri_match_result[10];
      vector<string_view> tokens;

      split( query_string, '&', tokens );
      for ( const string_view& token : tokens ) {
        if ( token.length() == 0 )
          continue;

        string::size_type eq_pos = token.find( '=' );
        if ( eq_pos != string::npos ) {
          options[string { token.substr( 0, eq_pos ) }] = token.substr( eq_pos + 1 );
        } else {
          options[string { token }] = {};
        }
      }
    }
  } else {
    throw runtime_error( "Malformed URI" );
  }
}
