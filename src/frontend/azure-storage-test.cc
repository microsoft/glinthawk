#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "storage/azure/blobstore.hh"

using namespace std;
using namespace glinthawk::storage;

int main( int argc, char* argv[] )
{
  if ( argc != 4 ) {
    cerr << "Usage: " << argv[0] << " <container-uri> <sas_token> <N>" << endl;
    return EXIT_FAILURE;
  }

  const string container_uri { argv[1] };
  const string sas_token { argv[2] };
  const size_t N { stoull( argv[3] ) };

  azure::BlobStore blob_store { container_uri, sas_token };

  vector<pair<string, string>> put_requests;
  put_requests.reserve( N );
  for ( size_t i = 1; i <= N; i++ ) {
    put_requests.emplace_back( "object_"s + to_string( i ) + ".txt"s, "#"s + to_string( i ) );
  }

  const auto put_responses = blob_store.put( put_requests );
  if ( any_of( put_responses.begin(), put_responses.end(), []( const auto& r ) { return r != OpResult::OK; } ) ) {
    cerr << "Error putting objects." << endl;
    return 1;
  }

  vector<string> get_requests;
  get_requests.reserve( N );
  for ( size_t i = 1; i <= N; i++ ) {
    get_requests.emplace_back( "object_"s + to_string( i ) + ".txt"s );
  }

  const auto get_responses = blob_store.get( get_requests );
  if ( any_of( get_responses.begin(), get_responses.end(), []( const auto& r ) { return r.first != OpResult::OK; } ) ) {
    cerr << "Error getting objects." << endl;
    return 1;
  }

  const auto remove_responses = blob_store.remove( get_requests );
  if ( any_of( remove_responses.begin(), remove_responses.end(), []( const auto& r ) { return r != OpResult::OK; } ) ) {
    cerr << "Error removing objects." << endl;
    return 1;
  }

  return 0;
}
