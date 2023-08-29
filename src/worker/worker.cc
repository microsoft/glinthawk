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
Worker<Model>::Worker( const Address& address, Model&& model )
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
      LOG( INFO ) << "GOT A CONNECTION FROM " << socket.peer_address().to_string();
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
