#include "blobstore.hh"

#include <fstream>
#include <stdexcept>
#include <streambuf>
#include <string>

using namespace std;
using namespace glinthawk::storage::local;

using glinthawk::storage::OpResult;

BlobStore::BlobStore( const filesystem::path& root )
  : root_( root )
{
  if ( root_.is_relative() ) {
    throw invalid_argument( "BlobStore root must be an absolute path" );
  }

  if ( not filesystem::exists( root_ ) ) {
    filesystem::create_directories( root_ );
  }
}

[[nodiscard]] bool check_key( const string& key )
{
  if ( key.empty() ) {
    return false;
  }

  if ( key.find( "../" ) != string::npos ) {
    return false;
  }

  if ( key.front() == '/' or key.back() == '/' ) {
    return false;
  }

  return true;
}

pair<OpResult, string> BlobStore::get( const string& key )
{
  if ( not check_key( key ) ) {
    return make_pair( OpResult::InvalidKey, ""s );
  }

  const filesystem::path path = root_ / key;

  ifstream fin { path };
  if ( !fin.good() ) {
    return make_pair( OpResult::NotFound, ""s );
  }

  string contents { istreambuf_iterator<char>( fin ), istreambuf_iterator<char>() };
  return { OpResult::OK, std::move( contents ) };
}

OpResult BlobStore::put( const string& key, const string& value )
{
  if ( not check_key( key ) ) {
    return OpResult::InvalidKey;
  }

  const filesystem::path path = root_ / key;

  if ( not filesystem::exists( path.parent_path() ) ) {
    filesystem::create_directories( path.parent_path() );
  }

  ofstream fout { path, ios::binary };
  if ( !fout.good() ) {
    return OpResult::Error;
  }

  fout.write( value.data(), value.size() );
  fout.close();

  return OpResult::OK;
}

OpResult BlobStore::remove( const string& key )
{
  if ( not check_key( key ) ) {
    return OpResult::InvalidKey;
  }

  const filesystem::path path = root_ / key;

  if ( not filesystem::exists( path ) or not filesystem::is_regular_file( path ) ) {
    return OpResult::NotFound;
  }

  filesystem::remove( path );
  return OpResult::OK;
}

vector<pair<OpResult, string>> BlobStore::get( const vector<string>& keys )
{
  vector<pair<OpResult, string>> results {};

  for ( const auto& key : keys ) {
    auto [result, value] = get( key );
    results.emplace_back( result, std::move( value ) );
  }

  return results;
}

vector<OpResult> BlobStore::put( const vector<pair<string, string>>& kvs )
{
  vector<OpResult> results {};

  for ( const auto& [key, value] : kvs ) {
    results.emplace_back( put( key, value ) );
  }

  return results;
}

vector<OpResult> BlobStore::remove( const vector<string>& keys )
{
  vector<OpResult> results {};

  for ( const auto& key : keys ) {
    results.emplace_back( remove( key ) );
  }

  return results;
}

string BlobStore::to_string() const { return "file://"s + root_.string(); }
