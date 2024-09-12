#include "prompt.hh"

#include <chrono>
#include <endian.h>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <sstream>

#include "util/digest.hh"

#include "glinthawk.pb.h"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::prompt;

Prompt Prompt::from_protobuf( const protobuf::Prompt& message )
{
  return { util::digest::SHA256Hash::from_base58digest( message.id() ),
           static_cast<uint8_t>( message.temperature() ),
           message.max_tokens(),
           vector<uint32_t> { message.prompt().begin(), message.prompt().end() } };
}

protobuf::Prompt Prompt::to_protobuf() const
{
  auto& prompt_tokens = prompt_tokens_.tokens();
  auto& completion_tokens = completion_tokens_.tokens();

  protobuf::Prompt pb_prompt;
  pb_prompt.set_id( id_.base58digest() );
  pb_prompt.set_temperature( temperature_ );
  *pb_prompt.mutable_prompt() = { prompt_tokens.begin(), prompt_tokens.end() };
  *pb_prompt.mutable_completion() = { completion_tokens.begin(), completion_tokens.end() };

  return pb_prompt;
}

Prompt Prompt::from_json( const string& json )
{
  protobuf::Prompt pb_prompt;
  CHECK( google::protobuf::util::JsonStringToMessage( json, &pb_prompt, {} ).ok() ) << "Failed to parse JSON.";
  return from_protobuf( pb_prompt );
}

string Prompt::to_json() const
{
  string json;
  CHECK( google::protobuf::util::MessageToJsonString( to_protobuf(), &json, {} ).ok() )
    << "Failed to serialize to JSON.";
  return json;
}

std::string Prompt::csv_header()
{
  return "id,temperature,max_completion_length,prompt_tokens,completion_tokens,assigned,prompt_started,completion_"
         "started,completion_finished,tpot_count,tpot_min,tpot_max,tpot_sum,tpot_sum_of_squares,tpit_count,tpit_min,"
         "tpit_max,tpit_sum,tpit_sum_of_squares";
}

std::string Prompt::to_csv() const
{
  ostringstream oss;

  auto get_time = []( const optional<Measurement::Clock::time_point>& time ) -> string {
    return time.has_value()
             ? to_string( chrono::duration_cast<chrono::microseconds>( time->time_since_epoch() ).count() )
             : "";
  };

  oss << id_.base58digest() << "," << static_cast<int32_t>( temperature_ ) << "," << max_completion_length_ << ","
      << prompt_tokens_.count() << "," << completion_tokens_.count() << "," << get_time( timing_info_.assigned ) << ","
      << get_time( timing_info_.prompt_started ) << "," << get_time( timing_info_.completion_started ) << ","
      << get_time( timing_info_.completion_finished ) << "," << timing_info_.token_output_time.count << ","
      << timing_info_.token_output_time.min << "," << timing_info_.token_output_time.max << ","
      << timing_info_.token_output_time.sum << "," << timing_info_.token_output_time.sum_of_squares << ","
      << timing_info_.token_input_time.count << "," << timing_info_.token_input_time.min << ","
      << timing_info_.token_input_time.max << "," << timing_info_.token_input_time.sum << ","
      << timing_info_.token_input_time.sum_of_squares;

  return oss.str();
}

PromptStore::~PromptStore()
{
  if ( !completed_prompts_.empty() ) {
    LOG( ERROR ) << "PromptStore destroyed with uncommitted completions";
  }
}

void PromptStore::add( const PromptID& id, Prompt&& prompt ) { prompts_.emplace( id, std::move( prompt ) ); }

void PromptStore::complete( const PromptID& id )
{
  auto it = prompts_.find( id );
  if ( it == prompts_.end() ) {
    LOG( ERROR ) << "Prompt not found: " << id;
    return;
  }

  completed_prompts_.emplace( id, std::move( it->second ) );
  prompts_.erase( it );
}

void PromptStore::cleanup_completed() { completed_prompts_.clear(); }

protobuf::PushCompletions PromptStore::completed_to_protobuf()
{
  protobuf::PushCompletions message;
  for ( auto& [id, prompt] : completed_prompts_ ) {
    *message.add_completions() = prompt.to_protobuf();
  }

  return message;
}
