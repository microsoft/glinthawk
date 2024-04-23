#include "prompt.hh"

#include <endian.h>
#include <fstream>
#include <glog/logging.h>

using namespace std;
using namespace glinthawk::prompt;

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
