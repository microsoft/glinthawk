#include "worker.hh"

#include <glog/logging.h>

#ifdef GLINTHAWK_CUDA_ENABLED
#include "models/llama2/cuda/model.cuh"
#endif

using namespace std;
using namespace glinthawk;
using namespace glinthawk::core;
using namespace glinthawk::net;

template<typename Model>
Worker<Model>::Worker( const Address& address, std::unique_ptr<Model>&& model )
  : listen_address_( address )
  , compute_kernel_( move( model ) )
{
  listen_socket_.set_reuseaddr();
  listen_socket_.bind( listen_address_ );
  listen_socket_.set_blocking( false );
  listen_socket_.listen();

  LOG( INFO ) << "Listening on " << listen_address_.to_string();

  // handle fd failures gracefully
  event_loop_.set_fd_failure_callback( [] {} );

  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    [this] {
      TCPSocket socket = listen_socket_.accept();
      auto addr = socket.peer_address();

      LOG( INFO ) << "Accepted connection from " << addr.to_string();

      auto [peer_it, peer_new]
        = peers_.emplace( piecewise_construct, forward_as_tuple( addr ), forward_as_tuple( addr, move( socket ) ) );

      CHECK( peer_new ) << "A peer with this address already exists.";

      peer_it->second.message_handler.install_rules(
        this->event_loop_,
        this->rule_categories_,
        []( Message&& msg ) {
          LOG( INFO ) << "Incoming message: " << msg.info();
          return true;
        },
        [] { LOG( INFO ) << "Connection to peer closed."; } );
    },
    [] { return true; },
    [] { LOG( INFO ) << "STOPPED LISTENING"; } );
}

template<typename Model>
void Worker<Model>::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
  }
}

namespace glinthawk::core {

#ifdef GLINTHAWK_CUDA_ENABLED
template class Worker<models::llama2::cuda::Llama2<__half>>;
#endif

} // namespace glinthawk::core
