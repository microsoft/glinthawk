#pragma once

#include <list>

#include "net/address.hh"
#include "net/message.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/eventloop.hh"

namespace glinthawk {

class Worker
{
private:
  EventLoop event_loop_ {};

  TCPSocket listen_socket_ {};
  std::list<TCPSocket> peers_ {};

  void add_new_peer( TCPSocket&& new_peer_socket );

public:
  Worker( const Address& worker_address );
  ~Worker() = default;

  void run();
};

} // namespace glinthawk
