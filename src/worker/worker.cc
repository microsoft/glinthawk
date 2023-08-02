#include "worker.hh"

using namespace std;
using namespace glinthawk;

Worker::Worker( const Address& worker_address )
{
  listen_socket_.bind( worker_address );
  listen_socket_.set_blocking( false );
  listen_socket_.listen();

  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    [this] {
      auto new_peer_socket = listen_socket_.accept();
      LOG( INFO ) << "Accepted new peer connection from " << new_peer_socket.peer_address().to_string();
      add_new_peer( move( new_peer_socket ) );
    },
    [] { return true; } );
}

void Worker::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
    // Do nothing.
  }
}
