#include "prompt.hh"

#include <endian.h>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#include "util/digest.hh"

#include "glinthawk.pb.h"

using namespace std;
using namespace glinthawk::prompt;

Prompt Prompt::from_json( const string_view json )
{
  protobuf::Prompt pb_prompt;
  CHECK( google::protobuf::util::JsonStringToMessage( json, &pb_prompt ).ok() ) << "Failed to parse JSON.";

  return { util::digest::SHA256Hash::from_base58digest( pb_prompt.id() ),
           static_cast<uint8_t>( pb_prompt.temperature() ),
           1024,
           vector<uint32_t> { pb_prompt.prompt().begin(), pb_prompt.prompt().end() } };
}

string Prompt::to_json() const
{
  auto& prompt_tokens = prompt_tokens_.tokens();
  auto& completion_tokens = completion_tokens_.tokens();

  protobuf::Prompt pb_prompt;

  pb_prompt.set_id( id_.base58digest() );
  pb_prompt.set_temperature( temperature_ );
  *pb_prompt.mutable_prompt() = { prompt_tokens.begin(), prompt_tokens.end() };
  *pb_prompt.mutable_completion() = { completion_tokens.begin(), completion_tokens.end() };

  string json;
  CHECK( google::protobuf::util::MessageToJsonString( pb_prompt, &json ).ok() ) << "Failed to serialize to JSON.";

  return json;
}

PromptStore::~PromptStore()
{
  if ( !completed_prompts_.empty() ) {
    LOG( ERROR ) << "PromptStore destroyed with uncommitted completions";
  }
}

void PromptStore::add( const PromptID& id, const Prompt& prompt ) { prompts_.emplace( id, prompt ); }

void PromptStore::terminate( const PromptID& id )
{
  auto it = prompts_.find( id );
  if ( it == prompts_.end() ) {
    LOG( ERROR ) << "Prompt not found: " << id;
    return;
  }

  completed_prompts_.emplace( id, std::move( it->second ) );
  prompts_.erase( it );
}

void PromptStore::commit() { return; }
