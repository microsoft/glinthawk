#pragma once

#include <queue>
#include <string>
#include <string_view>
#include <vector>

#include "http_request.hh"
#include "http_response_parser.hh"
#include "message/handler.hh"
#include "util/ring_buffer.hh"

namespace glinthawk::net {

template<class SessionType>
class HTTPClient : public MessageHandler<SessionType, HTTPRequest, HTTPResponse>
{
private:
  std::queue<HTTPRequest> requests_ {};
  HTTPResponseParser responses_ {};

  std::string current_request_headers_ {};
  std::string_view current_request_unsent_headers_ {};
  std::string_view current_request_unsent_body_ {};

  void load();

  bool outgoing_empty() override;
  bool incoming_empty() const override { return responses_.empty(); }
  HTTPResponse& incoming_front() override { return responses_.front(); }
  void incoming_pop() override { responses_.pop(); }

  void write( RingBuffer& out ) override;
  void read( RingBuffer& in ) override;

public:
  using MessageHandler<SessionType, HTTPRequest, HTTPResponse>::MessageHandler;

  void push_message( HTTPRequest&& req ) override;
  void push_request( HTTPRequest&& req ) { push_message( std::move( req ) ); }
};

} // namespace glinthawk::net
