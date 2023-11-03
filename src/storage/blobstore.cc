#include "blobstore.hh"

#include <iostream>

#include "azure/blobstore.hh"
#include "local/blobstore.hh"
#include "util/uri.hh"

using namespace std;
using namespace glinthawk::storage;

unique_ptr<BlobStore> BlobStore::create( const string& uri )
{
  util::ParsedURI parsed_uri( uri );

  if ( parsed_uri.protocol == "file" ) {
    return make_unique<local::BlobStore>( "/"s + parsed_uri.path );
  } else if ( parsed_uri.protocol == "azure" ) {
    return {};
  }

  return {};
}
