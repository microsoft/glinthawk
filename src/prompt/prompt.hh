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
#include "monitoring/measurement.hh"
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

  struct TimingInfo
  {
    using Timestamp = std::optional<Measurement::Clock::time_point>;

    struct TimePerToken
    {
      uint64_t count { 0 };
      uint64_t min { std::numeric_limits<uint64_t>::max() };
      uint64_t max { 0 };
      uint64_t sum { 0 };
      uint64_t sum_of_squares { 0 };

      void add_point()
      {
        if ( not last_token_time_.has_value() ) {
          last_token_time_ = Measurement::Clock::now();
          return;
        }

        const auto now = Measurement::Clock::now();

        const auto v = std::chrono::duration_cast<std::chrono::microseconds>( now - *last_token_time_ ).count();

        if ( v < 0 ) {
          throw std::runtime_error( "my time machine worked" );
        }

        const auto value = static_cast<uint64_t>( v );

        count++;
        min = std::min( min, value );
        max = std::max( max, value );
        sum += value;
        sum_of_squares += value * value;

        last_token_time_ = now;
      }

    private:
      Timestamp last_token_time_ {};
    } token_input_time, token_output_time;

    Timestamp assigned {};            // When the prompt was assigned to a worker
    Timestamp prompt_started {};      // When the first token of the prompt started being processed
    Timestamp completion_started {};  // When the first token of the completion started being processed (kinda TTFT)
    Timestamp completion_finished {}; // When the last token of the completion finished being processed

    void set_assigned() { assigned = Measurement::Clock::now(); }
    void set_prompt_started() { prompt_started = Measurement::Clock::now(); }
    void set_completion_started() { completion_started = Measurement::Clock::now(); }
    void set_completion_finished() { completion_finished = Measurement::Clock::now(); }
  };

  static Prompt from_json( const std::string& json );
  std::string to_json() const;

  static Prompt from_protobuf( const protobuf::Prompt& message );
  protobuf::Prompt to_protobuf() const;

  PromptID id() const { return id_; }
  float temperature() const { return temperature_ / 255.0; }
  size_t max_completion_length() const { return max_completion_length_; }
  const TokenSequence& prompt() const { return prompt_tokens_; }
  TokenSequence& completion() { return completion_tokens_; }

  TimingInfo& timing_info() { return timing_info_; }

  static std::string csv_header();
  std::string to_csv() const;

private:
  PromptID id_ {};
  uint8_t temperature_ { 0 };
  size_t max_completion_length_ { 0 };
  TokenSequence prompt_tokens_ {};
  TokenSequence completion_tokens_ {};

  TimingInfo timing_info_ {};
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
