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
  google::protobuf::util::JsonStringToMessage( json, &pb_prompt );
  TokenSequence tokens { vector<uint32_t> { pb_prompt.prompt().begin(), pb_prompt.prompt().end() } };
  return { util::digest::SHA256Hash::from_base58digest( pb_prompt.id() ),
           static_cast<uint8_t>( pb_prompt.temperature() ),
           1024,
           tokens };
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
