#include "blobstore.hh"

#include <functional>
#include <future>
#include <span>
#include <sstream>
#include <string_view>
#include <vector>

#include "message/handler.hh"
#include "net/address.hh"
#include "net/http/http_request.hh"
#include "net/http/http_response.hh"
#include "net/http/http_response_parser.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/simple_string_span.hh"

using namespace std;
using namespace glinthawk::net;
using namespace glinthawk::storage::azure;

using glinthawk::storage::OpResult;

constexpr size_t BUFFER_SIZE = 65536;

BlobStore::BlobStore( const string& container_uri, const string& sas_token )
  : container_uri_( container_uri )
  , sas_token_( sas_token )
{
}

HTTPRequest BlobStore::make_request( const Op operation, const std::string& key, std::string&& payload )
{
  string method;
  ostringstream first_line;
  vector<HTTPHeader> header { { "Host", container_uri_.host } };

  if ( not payload.empty() ) {
    header.emplace_back( "Content-Length", ::to_string( payload.size() ) );
  }

  switch ( operation ) {
    case Op::Get:
      method = "GET";
      break;

    case Op::Put:
      method = "PUT";
      header.emplace_back( "x-ms-blob-type", "BlockBlob" );
      break;

    case Op::Remove:
      method = "DELETE";
      break;
  }

  first_line << method << " /" << container_uri_.path << "/" << key << "?" << sas_token_ << " HTTP/1.1";
  return { move( first_line.str() ), move( header ), move( payload ) };
}

TCPSocket make_connection( const Address& endpoint )
{
  TCPSocket socket;
  socket.set_blocking( true );
  socket.connect( endpoint );
  return socket;
}

pair<OpResult, string> process_GET_response( HTTPResponse&& response )
{
  if ( response.status_code() == "200" ) {
    return make_pair( OpResult::OK, move( response.body() ) );
  } else if ( response.status_code() == "404" ) {
    return make_pair( OpResult::NotFound, ""s );
  } else {
    return make_pair( OpResult::Error, ""s );
  }
}

OpResult process_PUT_response( HTTPResponse&& response )
{
  if ( response.status_code() == "201" ) {
    return OpResult::OK;
  } else if ( response.status_code() == "404" ) {
    return OpResult::NotFound;
  } else {
    return OpResult::Error;
  }
}

OpResult process_DELETE_response( HTTPResponse&& response )
{
  if ( response.status_code() == "202" ) {
    return OpResult::OK;
  } else if ( response.status_code() == "404" ) {
    return OpResult::NotFound;
  } else {
    return OpResult::Error;
  }
}

pair<OpResult, string> BlobStore::get( const string& key ) { return get( vector { key } ).front(); }
OpResult BlobStore::put( const string& k, const string& v ) { return put( vector { make_pair( k, v ) } ).front(); }
OpResult BlobStore::remove( const string& key ) { return remove( vector { key } ).front(); }

template<BlobStore::Op operation, typename InputType, typename ResultType>
void BlobStore::worker_thread( const size_t thread_num,
                               const vector<InputType>& requests,
                               std::vector<ResultType>& responses,
                               const Address& endpoint )
{
  CHECK( requests.size() > 0 ) << "Requests must be non-empty";
  CHECK( requests.size() == responses.size() ) << "Requests and responses must be the same size and preallocated";

  const auto requests_per_thread = requests.size() / MAX_THREADS + ( requests.size() % MAX_THREADS != 0 );

  char read_buffer[BUFFER_SIZE];
  simple_string_span read_buffer_span { read_buffer, BUFFER_SIZE };
  SSLContext ssl_context;

  const size_t first_item_index = thread_num * requests_per_thread;
  const size_t last_item_index = min( ( thread_num + 1 ) * requests_per_thread, requests.size() );

  for ( size_t i = first_item_index; i < last_item_index; i += MAX_REQUESTS_PER_CONNECTION ) {
    SimpleSSLSession ssl_session { ssl_context.make_SSL_handle(), make_connection( endpoint ) };
    HTTPResponseParser response_parser;

    size_t request_count = 0;
    size_t response_count = 0;

    for ( size_t j = i; j < min( last_item_index, i + MAX_REQUESTS_PER_CONNECTION ); j++ ) {
      string headers;
      HTTPRequest request;

      if constexpr ( operation == Op::Get || operation == Op::Remove ) {
        request = make_request( operation, requests[j] );
      } else if constexpr ( operation == Op::Put ) {
        request = make_request( Op::Put, requests[j].first, string { requests[j].second } );
      } else {
        []<bool flag = false>() { static_assert( flag, "Invalid operation" ); }
        ();
      }

      request.serialize_headers( headers );

      string_view headers_sv { headers };
      while ( not headers_sv.empty() ) {
        // TODO(sadjad): check for errors
        const auto len = ssl_session.write( headers_sv );
        headers_sv.remove_prefix( len );
      }

      string_view payload_sv { request.body() };
      while ( not payload_sv.empty() ) {
        // TODO(sadjad): check for errors
        const auto len = ssl_session.write( payload_sv );
        payload_sv.remove_prefix( len );
      }

      response_parser.new_request_arrived( request );
      request_count++;
    }

    while ( response_count < request_count ) {
      // TODO(sadjad): check for errors
      const auto len = ssl_session.read( read_buffer_span );
      if ( len == 0 ) {
        break;
      }

      response_parser.parse( read_buffer_span.substr( 0, len ) );

      while ( not response_parser.empty() ) {
        auto response = move( response_parser.front() );
        response_parser.pop();

        if constexpr ( operation == Op::Get ) {
          responses[i + response_count] = process_GET_response( move( response ) );
        } else if constexpr ( operation == Op::Put ) {
          responses[i + response_count] = process_PUT_response( move( response ) );
        } else if constexpr ( operation == Op::Remove ) {
          responses[i + response_count] = process_DELETE_response( move( response ) );
        } else {
          []<bool flag = false>() { static_assert( flag, "Invalid operation" ); }
          ();
        }

        response_count++;
      }
    }

    if ( response_count < request_count ) {
      throw runtime_error( "azure::BlobStore::get failed" );
    }
  }
}

vector<pair<OpResult, string>> BlobStore::get( const vector<string>& keys )
{
  vector<pair<OpResult, string>> results;
  results.resize( keys.size() );

  const Address endpoint { container_uri_.host, "https" };

  vector<future<void>> futures;
  futures.reserve( MAX_THREADS );

  for ( size_t thread_idx = 0; thread_idx < min( MAX_THREADS, keys.size() ); thread_idx++ ) {
    futures.emplace_back( async(
      launch::async,
      [&]( const size_t tid ) {
        worker_thread<Op::Get, string, pair<OpResult, string>>( tid, keys, results, endpoint );
      },
      thread_idx ) );
  }

  for ( auto& f : futures ) {
    f.wait();
  }

  return results;
}

vector<OpResult> BlobStore::put( const vector<pair<string, string>>& keys_and_values )
{
  vector<OpResult> results;
  results.resize( keys_and_values.size() );

  const Address endpoint { container_uri_.host, "https" };

  vector<future<void>> futures;
  futures.reserve( MAX_THREADS );

  for ( size_t thread_idx = 0; thread_idx < min( MAX_THREADS, keys_and_values.size() ); thread_idx++ ) {
    futures.emplace_back( async(
      launch::async,
      [&]( const size_t tid ) {
        worker_thread<Op::Put, pair<string, string>, OpResult>( tid, keys_and_values, results, endpoint );
      },
      thread_idx ) );
  }

  for ( auto& f : futures ) {
    f.wait();
  }

  return results;
}

vector<OpResult> BlobStore::remove( const vector<string>& keys )
{
  vector<OpResult> results;
  results.resize( keys.size() );

  const Address endpoint { container_uri_.host, "https" };

  vector<future<void>> futures;
  futures.reserve( MAX_THREADS );

  for ( size_t thread_idx = 0; thread_idx < min( MAX_THREADS, keys.size() ); thread_idx++ ) {
    futures.emplace_back( async(
      launch::async,
      [&]( const size_t tid ) { worker_thread<Op::Remove, string, OpResult>( tid, keys, results, endpoint ); },
      thread_idx ) );
  }

  for ( auto& f : futures ) {
    f.wait();
  }

  return results;
}

string BlobStore::to_string() const { return "azure://"s + container_uri_.host + "/" + container_uri_.path; }
