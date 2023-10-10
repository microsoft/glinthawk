#pragma once

#include <list>
#include <map>
#include <memory>
#include <optional>

#include "compute/kernel.hh"
#include "message/handler.hh"
#include "message/message.hh"
#include "models/llama2/base.hh"
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
    net::Address address;
    std::vector<models::InferenceState> outgoing_states {};
    core::MessageHandler<net::TCPSession> message_handler {};

    Peer( const net::Address& addr, net::TCPSocket&& socket )
      : address( addr )
      , message_handler( std::move( socket ) )
    {
    }
  };

private:
  EventLoop event_loop_ {};
  net::Address listen_address_;
  net::TCPSocket listen_socket_ {};
  std::map<net::Address, Peer> peers_ {};

  compute::ComputeKernel<Model> compute_kernel_;
  std::optional<typename Model::TokenizerType> tokenizer_;

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

  void setup_peer( std::map<net::Address, Peer>::iterator peer_it );

public:
  Worker( const net::Address& address,
          std::unique_ptr<Model>&& model,
          std::optional<typename Model::TokenizerType>&& tokenizer = std::nullopt );

  void run();
};

} // namespace glinthawk::core
