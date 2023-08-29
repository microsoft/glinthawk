#pragma once

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
  EventLoop event_loop_ {};

  net::Address listen_address_;
  net::TCPSocket listen_socket_ {};

  compute::ComputeKernel<Model> compute_kernel_;

public:
  Worker( const net::Address& address, Model&& model );

  void run();
};

} // namespace glinthawk::core
