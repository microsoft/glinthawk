#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "models/types.hh"
#include "storage/blobstore.hh"

namespace glinthawk {

/// @brief A dialogue is a sequence of prompts and completions.
class Dialogue
{
  friend class DialogueManager;

protected:
  const std::vector<uint32_t> prompt_tokens_ {};
  std::vector<uint32_t> completion_tokens_ {};

  bool terminated_ { false };
  void terminate() { terminated_ = true; }

public:
  template<typename... Args>
  Dialogue( Args&&... args )
    : prompt_tokens_( std::forward<Args>( args )... )
  {
  }

  std::size_t length() const { return prompt_tokens_.size() + completion_tokens_.size(); }

  uint32_t token( const uint32_t pos ) const;
  void push( const uint32_t token ) { completion_tokens_.push_back( token ); }
};

class DialogueManager
{
private:
  std::shared_ptr<storage::BlobStore> blobstore_;
  std::unordered_map<PromptID, Dialogue> dialogues_ {};
  std::unordered_map<PromptID, Dialogue> terminated_dialogues_ {};

public:
  DialogueManager( std::shared_ptr<storage::BlobStore> blobstore )
    : blobstore_( std::move( blobstore ) )
  {
  }

  Dialogue& get( const PromptID& id ) { return dialogues_.at( id ); }
  void terminate( const PromptID& id );
};

uint32_t Dialogue::token( const uint32_t pos ) const
{
  if ( pos < prompt_tokens_.size() ) {
    return prompt_tokens_.at( pos );
  }

  return completion_tokens_.at( pos - prompt_tokens_.size() );
}

void DialogueManager::terminate( const PromptID& id )
{
  auto it = dialogues_.find( id );
  if ( it == dialogues_.end() ) {
    return;
  }

  it->second.terminate();
  terminated_dialogues_.emplace( id, std::move( it->second ) );
  dialogues_.erase( it );
}

} // namespace glinthawk
