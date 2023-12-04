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

PromptManager::PromptManager( shared_ptr<storage::BlobStore> blobstore )
  : blobstore_( move( blobstore ) )
{
}

const Prompt& PromptManager::get( const PromptID& prompt_id )
{
  if ( prompts_.count( prompt_id ) == 0 ) {
    // XXX avoid this
    LOG( WARNING ) << "Prompt " << prompt_id.base58digest() << " not loaded; fetching...";
    fetch( { prompt_id } );
  }

  return prompts_.at( prompt_id );
}

void PromptManager::fetch( const vector<PromptID>& prompt_ids )
{
  vector<string> keys;
  for ( const auto& prompt_id : prompt_ids ) {
    keys.push_back( "processed/"s + prompt_id.base58digest() + ".ghp" );
  }

  auto results = blobstore_->get( keys );
  CHECK( results.size() == prompt_ids.size() ) << "Failed to load all prompts";

  for ( size_t i = 0; i < results.size(); i++ ) {
    if ( results[i].first == storage::OpResult::OK ) {
      prompts_.emplace( prompt_ids[i], results[i].second );
    } else {
      LOG( FATAL ) << "Failed to load prompt " << prompt_ids[i].base58digest();
    }
  }
}

string Completion::serialize() const
{
  CHECK( is_terminated_ ) << "Cannot serialize an incomplete completion";

  string serialized_completion;
  serialized_completion.reserve( sizeof( uint32_t ) + tokens_.size() * sizeof( uint32_t ) );

  const auto token_count = htole32( tokens_.size() );
  serialized_completion.append( reinterpret_cast<const char*>( &token_count ), sizeof( token_count ) );

  for ( const auto& token : tokens_ ) {
    const auto le_token = htole32( token );
    serialized_completion.append( reinterpret_cast<const char*>( &le_token ), sizeof( le_token ) );
  }

  return serialized_completion;
}

CompletionManager::CompletionManager( shared_ptr<storage::BlobStore> blobstore )
  : blobstore_( move( blobstore ) )
{
}

Completion& CompletionManager::get( const PromptID& prompt_id ) { return completions_[prompt_id]; }

void CompletionManager::commit()
{
  if ( terminated_completions_.empty() ) {
    return;
  }

  vector<pair<string, string>> key_values;

  {
    std::lock_guard<std::mutex> lock { terminated_mutex_ };

    for ( auto& [id, completion] : terminated_completions_ ) {
      if ( completion.is_terminated() ) {
        key_values.emplace_back( "completed/"s + id.base58digest() + ".ghc", completion.serialize() );
      } else {
        LOG( ERROR ) << "Completion " << id.base58digest() << " is not terminated.";
      }
    }

    terminated_completions_.clear();
  }

  auto results = blobstore_->put( key_values );
  CHECK( results.size() == key_values.size() ) << "Failed to commit completions";

  size_t failed = 0;
  for ( size_t i = 0; i < results.size(); i++ ) {
    if ( results[i] != storage::OpResult::OK ) {
      LOG( WARNING ) << "Failed to commit completion " << key_values[i].first;
      failed++;
    }
  }

  LOG( INFO ) << "Committed " << ( key_values.size() - failed ) << " completions";
}
