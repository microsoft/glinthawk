#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "models/types.hh"
#include "storage/blobstore.hh"

#include "glinthawk.pb.h"

namespace glinthawk::prompt {

class TokenSequence
{
public:
  TokenSequence() = default;

  TokenSequence( const std::vector<uint32_t>&& tokens )
    : tokens_( std::move( tokens ) )
  {
  }

  uint32_t at( const uint32_t token_pos ) const { return tokens_.at( token_pos ); }
  uint32_t count() const { return tokens_.size(); }
  void append( const uint32_t token ) { tokens_.push_back( token ); }
  const std::vector<uint32_t>& tokens() const { return tokens_; }

private:
  std::vector<uint32_t> tokens_ {};
};

class Prompt
{
public:
  Prompt( const PromptID& id,
          const uint8_t temperature,
          const size_t max_completion_length,
          std::vector<uint32_t>&& prompt_tokens )
    : id_( id )
    , temperature_( temperature )
    , max_completion_length_( max_completion_length )
    , prompt_tokens_( std::move( prompt_tokens ) )
  {
  }

  static Prompt from_json( const std::string_view json );
  std::string to_json() const;

  static Prompt from_protobuf( const protobuf::Prompt& message );
  protobuf::Prompt to_protobuf() const;

  PromptID id() const { return id_; }
  float temperature() const { return temperature_ / 255.0; }
  size_t max_completion_length() const { return max_completion_length_; }
  const TokenSequence& prompt() const { return prompt_tokens_; }
  TokenSequence& completion() { return completion_tokens_; }

private:
  PromptID id_ {};
  uint8_t temperature_ { 0 };
  size_t max_completion_length_ { 0 };
  TokenSequence prompt_tokens_ {};
  TokenSequence completion_tokens_ {};
};

class PromptStore
{
public:
  PromptStore() = default;
  ~PromptStore();

  void add( const PromptID& id, Prompt&& prompt );
  Prompt& get( const PromptID& id ) { return prompts_.at( id ); }
  void complete( const PromptID& id );

  size_t prompt_count() const { return prompts_.size(); }
  size_t completed_count() const { return completed_prompts_.size(); }

  protobuf::PushCompletions completed_to_protobuf();
  void cleanup_completed();

private:
  std::unordered_map<PromptID, Prompt> prompts_ {};
  std::unordered_map<PromptID, Prompt> completed_prompts_ {};
};

} // namespace glinthawk::prompt
