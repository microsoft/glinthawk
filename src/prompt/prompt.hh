#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "models/common/model.hh"
#include "storage/blobstore.hh"

namespace glinthawk::prompt {

class Prompt
{
private:
  std::vector<uint32_t> tokens_ {};

public:
  Prompt( const std::string_view serialized_prompt );
  uint32_t token( const uint32_t token_pos ) const { return tokens_.at( token_pos ); }
};

class PromptManager
{
private:
  std::unique_ptr<storage::BlobStore> blobstore_ {};
  std::unordered_map<PromptID, std::shared_ptr<Prompt>> prompts_ {};

public:
  PromptManager( std::unique_ptr<storage::BlobStore>&& blobstore );

  std::shared_ptr<Prompt> prompt( const PromptID& prompt_id );
  void preload_prompts( const std::vector<PromptID>& prompt_ids );
};

}
