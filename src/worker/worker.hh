#pragma once

#include <list>
#include <map>

#include "compute/kernel.hh"
#include "message/handler.hh"
#include "message/message.hh"
#include "net/address.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/eventloop.hh"

namespace glinthawk::core {

template<typename Model>
class Worker
{
private:
  class Peer
  {
  public:
    enum class State
    {
      Connecting,
      Connected,
      Disconnected,
    };

    net::Address address;
    core::MessageHandler<net::TCPSession> message_handler;
    State state { State::Connecting };

    Peer( const net::Address& addr, net::TCPSocket&& socket )
      : address( addr )
      , message_handler( net::TCPSession { std::move( socket ) } )
    {
    }
  };

private:
  EventLoop event_loop_ {};

  std::map<net::Address, Peer> peers_ {};

  net::Address listen_address_;
  net::TCPSocket listen_socket_ {};

  compute::ComputeKernel<Model> compute_kernel_;

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

public:
  Worker( const net::Address& address, Model&& model );

  void run();
};

} // namespace glinthawk::core
