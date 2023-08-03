#include "worker.hh"

using namespace std;
using namespace glinthawk;

Worker::Worker( const Address& this_address, const Address& next_address )
  : this_address_ { this_address }
  , next_address_ { next_address }
{
  /* Setting up the listen socket and callback */
  listen_socket_.bind( this_address_ );
  listen_socket_.set_blocking( false );
  listen_socket_.listen();

  event_loop_.set_fd_failure_callback( [] { LOG( WARNING ) << "Connection failed."; } );

  /* Listening for incoming connections */
  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    [this] {
      auto new_peer_socket = listen_socket_.accept();
      LOG( INFO ) << "Accepted new peer connection from " << new_peer_socket.peer_address().to_string();
      incoming_message_handler_ = make_unique<InferenceStateMessageHandler>( move( new_peer_socket ) );
      incoming_message_handler_->install_rules(
        event_loop_,
        rule_categories_,
        [this]( InferenceState&& ) {
          LOG( INFO ) << "Got an inference state";
          return true;
        },
        [this] {
          LOG( INFO ) << "Peer closed connection";
          incoming_message_handler_.reset();
        } );
    },
    [this] { return not incoming_message_handler_; } );

  /* Periodically try to connect to the next peer */
  event_loop_.add_rule(
    "Reconnect to next",
    Direction::In,
    reconnect_timer_fd_,
    [this] {
      reconnect_timer_fd_.read_event();
      reconnect_to_next();
    },
    [this] { return not outgoing_message_handler_; } );
}

void Worker::reconnect_to_next()
{
  TCPSocket next_socket {};
  next_socket.set_blocking( false );
  next_socket.connect( next_address_ );
  outgoing_message_handler_ = make_unique<InferenceStateMessageHandler>( move( next_socket ) );

  outgoing_message_handler_->install_rules(
    event_loop_,
    rule_categories_,
    [this]( InferenceState&& ) {
      LOG( INFO ) << "Sent an inference state";
      return true;
    },
    [this] {
      LOG( INFO ) << "Next peer closed connection";
      outgoing_message_handler_.reset();
    },
    [] { LOG( ERROR ) << "Connection error."; } );
}

void Worker::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
    // Do nothing.
  }
}
