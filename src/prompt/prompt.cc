#include "prompt.hh"

#include <endian.h>
#include <fstream>
#include <glog/logging.h>

using namespace std;
using namespace glinthawk::prompt;

namespace {

template<typename FieldType, typename PtrType>
FieldType _get_and_advance( const PtrType*& ptr )
{
  const auto result = reinterpret_cast<const FieldType*>( ptr );
  ptr = reinterpret_cast<const PtrType*>( reinterpret_cast<const uint8_t*>( ptr ) + sizeof( FieldType ) );
  return *result;
}

} // namespace

Prompt::Prompt( const string_view serialized_prompt )
{
  if ( serialized_prompt.length() < sizeof( uint32_t ) ) {
    LOG( FATAL ) << "Incomplete prompt";
  }

  auto ptr = serialized_prompt.data();
  const auto token_count = le32toh( _get_and_advance<uint32_t>( ptr ) );

  if ( serialized_prompt.length() < sizeof( uint32_t ) + token_count * sizeof( uint32_t ) ) {
    LOG( FATAL ) << "Incomplete prompt";
  }

  tokens_.reserve( token_count );

  for ( uint32_t i = 0; i < token_count; ++i ) {
    tokens_.push_back( le32toh( _get_and_advance<uint32_t>( ptr ) ) );
  }
}

PromptManager::PromptManager( unique_ptr<storage::BlobStore>&& blobstore )
  : blobstore_( move( blobstore ) )
{
}

shared_ptr<Prompt> PromptManager::prompt( const PromptID& prompt_id )
{
  if ( prompts_.count( prompt_id ) == 0 ) {
    LOG( FATAL ) << "Prompt " << prompt_id.base58digest() << " not loaded";
  }

  return prompts_.at( prompt_id );
}

void PromptManager::preload_prompts( const vector<PromptID>& prompt_ids )
{
  vector<string> keys;
  for ( const auto& prompt_id : prompt_ids ) {
    keys.push_back( "raw/"s + prompt_id.base58digest() + ".ghp" );
  }

  auto results = blobstore_->get( keys );
  CHECK( results.size() == prompt_ids.size() ) << "Failed to load all prompts";

  for ( size_t i = 0; i < results.size(); i++ ) {
    if ( results[i].first == storage::OpResult::OK ) {
      prompts_[prompt_ids[i]] = make_shared<Prompt>( results[i].second );
    } else {
      LOG( FATAL ) << "Failed to load prompt " << prompt_ids[i].base58digest();
    }
  }
}
