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
protected:
  std::vector<uint32_t> tokens_ {};

  Prompt() = default;

public:
  Prompt( const std::string_view serialized_prompt );
  uint32_t token( const uint32_t token_pos ) const { return tokens_.at( token_pos ); }
};

class Completion : public Prompt
{
private:
  bool is_terminated_ { false };

public:
  Completion() {}

  void add_token( const uint32_t token ) { tokens_.push_back( token ); }
  std::string serialize();
  size_t length() { return tokens_.size(); }

  void terminate() { is_terminated_ = true; }
  bool is_terminated() { return is_terminated_; }
};

class PromptManager
{
private:
  std::unique_ptr<storage::BlobStore> blobstore_ {};
  std::unordered_map<PromptID, std::shared_ptr<Prompt>> prompts_ {};

public:
  PromptManager( std::unique_ptr<storage::BlobStore>&& blobstore );

  std::shared_ptr<Prompt> get( const PromptID& prompt_id );
  void preload( const std::vector<PromptID>& prompt_ids );
};

class CompletionManager
{
private:
  std::unique_ptr<storage::BlobStore> blobstore_ {};
  std::unordered_map<PromptID, std::shared_ptr<Completion>> completions_ {};

public:
  CompletionManager( std::unique_ptr<storage::BlobStore>&& blobstore );

  std::shared_ptr<Completion> get( const PromptID& prompt_id );

  void commit(); // upload all terminated completions to blobstore
};

} // namespace glinthawk::prompt
