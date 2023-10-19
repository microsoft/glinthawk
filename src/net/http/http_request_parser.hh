#pragma once

#include "http_message_sequence.hh"
#include "http_request.hh"

namespace glinthawk::net {

class HTTPRequestParser : public HTTPMessageSequence<HTTPRequest>
{
private:
  void initialize_new_message() override {}
};

} // namespace glinthawk::net
