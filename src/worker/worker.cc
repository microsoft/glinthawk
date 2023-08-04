#include "worker.hh"

#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

Worker::Worker( const Address& this_address,
                const Address& next_address,
                std::unique_ptr<Model>&& model,
                const Type type )
  : this_address_ { this_address }
  , next_address_ { next_address }
  , model_( move( model ) )
  , type_( type )
{
  /* Setting up the listen socket and callback */
  listen_socket_.set_reuseaddr();
  listen_socket_.bind( this_address_ );
  listen_socket_.set_blocking( false );
  listen_socket_.listen();

  event_loop_.set_fd_failure_callback( [] {} );

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
        [this]( InferenceState&& state ) {
          InferenceResult result;

          {
            GlobalScopeTimer<Timer::Category::PartialInference> _;
            result = model_->forward( state );
          }

          if ( result.word ) {
            cout << *result.word << flush;
          }

          LOG( INFO ) << state.to_string() << " => " << result.inference_state.to_string();

          if ( result.inference_state.token == -1 ) {
            cerr << "\n\n"
                 << "End of sequence reached." << endl;
            return false;
          }

          if ( outgoing_message_handler_ ) {
            outgoing_message_handler_->push_message( move( result.inference_state ) );
          } else {
            LOG( WARNING ) << "No outgoing connection to send result to";
          }

          return true;
        },
        [this] {
          LOG( WARNING ) << "Peer closed connection";
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
  next_socket.set_reuseaddr();
  next_socket.set_blocking( false );
  next_socket.connect( next_address_ );
  outgoing_message_handler_ = make_unique<InferenceStateMessageHandler>( move( next_socket ) );

  outgoing_message_handler_->install_rules(
    event_loop_,
    rule_categories_,
    [this]( InferenceState&& ) -> bool { throw runtime_error { "Received message on outgoing connection" }; },
    [this] { outgoing_message_handler_.reset(); },
    [] { LOG( ERROR ) << "Connection error."; } );

  // Kick off the computation
  if ( type_ == Type::Last ) {
    InferenceState initial_state {};
    outgoing_message_handler_->push_message( move( initial_state ) );
  }
}

void Worker::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
    // Do nothing.
  }
}
