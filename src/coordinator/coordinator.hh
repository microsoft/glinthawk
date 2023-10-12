#pragma once

#include <list>

#include "message/handler.hh"
#include "message/message.hh"
#include "net/address.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/eventloop.hh"

namespace glinthawk::core {

template<typename Model>
class Coordinator
{
private:
  class Peer
  {
  public:
    core::MessageHandler<net::TCPSession> message_handler {};

    Peer( net::TCPSocket&& socket )
      : message_handler( std::move( socket ) )
    {
    }
  };

private:
  EventLoop event_loop_ {};
  net::Address listen_address_;
  net::TCPSocket listen_socket_ {};
  std::list<Peer> peers_ {};

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

  void setup_peer( std::list<Peer>::iterator peer_it );

public:
  Coordinator( const net::Address& listen_address );
  void run();
};

} // namespace glinthawk::core
