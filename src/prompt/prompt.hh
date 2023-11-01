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
  uint32_t token_count() const { return tokens_.size(); }
};

class Completion : public Prompt
{
private:
  bool is_terminated_ { false };

public:
  Completion() {}

  void add_token( const uint32_t token ) { tokens_.push_back( token ); }
  size_t token_count() { return tokens_.size(); }
  std::string serialize() const;

  void terminate() { is_terminated_ = true; }
  bool is_terminated() const { return is_terminated_; }
};

class PromptManager
{
private:
  std::shared_ptr<storage::BlobStore> blobstore_ {};
  std::unordered_map<PromptID, Prompt> prompts_ {};

public:
  PromptManager( std::shared_ptr<storage::BlobStore> blobstore );

  const Prompt& get( const PromptID& prompt_id );

  /// @brief Fetch the given prompts from the blobstore
  void fetch( const std::vector<PromptID>& prompt_ids );
};

class CompletionManager
{
private:
  std::shared_ptr<storage::BlobStore> blobstore_ {};
  std::unordered_map<PromptID, Completion> completions_ {};

public:
  CompletionManager( std::shared_ptr<storage::BlobStore> blobstore );

  Completion& get( const PromptID& prompt_id );

  /// @brief Upload the completed terminated completions to the blobstore
  void commit();
};

} // namespace glinthawk::prompt
