#include "prompt.hh"

#include <endian.h>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#include "util/digest.hh"

#include "glinthawk.pb.h"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::prompt;

Prompt Prompt::from_protobuf( const protobuf::Prompt& message )
{
  return { util::digest::SHA256Hash::from_base58digest( message.id() ),
           static_cast<uint8_t>( message.temperature() ),
           1024,
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

Prompt Prompt::from_json( const string_view json )
{
  protobuf::Prompt pb_prompt;
  CHECK( google::protobuf::util::JsonStringToMessage( json, &pb_prompt ).ok() ) << "Failed to parse JSON.";
  return from_protobuf( pb_prompt );
}

string Prompt::to_json() const
{
  string json;
  CHECK( google::protobuf::util::MessageToJsonString( to_protobuf(), &json ).ok() ) << "Failed to serialize to JSON.";
  return json;
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
