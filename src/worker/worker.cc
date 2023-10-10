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
void Worker<Model>::setup_peer( std::map<net::Address, Peer>::iterator peer_it )
{
  peer_it->second.message_handler.install_rules(
    this->event_loop_,
    this->rule_categories_,
    [this]( Message&& msg ) {
      LOG( INFO ) << "Incoming message: " << msg.info();

      switch ( msg.opcode() ) {
        case Message::OpCode::InferenceState: {
          auto state = models::InferenceState( msg.payload() );
          LOG( INFO ) << "Inference state: " << state.to_string();
          this->compute_kernel_.push( move( state ) );
          break;
        }

        default: {
          LOG( WARNING ) << "Message not handled." << endl;
          break;
        }
      }

      return true;
    },
    [] { LOG( INFO ) << "Connection to peer closed."; } );

  event_loop_.add_rule(
    "Outgoing message",
    [this, peer_it] {
      for ( auto& state : peer_it->second.outgoing_states ) {
        LOG( INFO ) << "Sending state to " << peer_it->first.to_string() << ": " << state.to_string();

        if ( state.next_layer() == 0 ) {
          if ( tokenizer_.has_value() ) {
            cout << tokenizer_->get_word( state.token() ) << flush;
          }

          // EOS
          if ( state.token() == 2 ) {
            continue;
          }
        }
        peer_it->second.message_handler.push_message( Message( Message::OpCode::InferenceState, state.serialize() ) );
      }

      peer_it->second.outgoing_states.clear();
    },
    [peer_it] { return not peer_it->second.outgoing_states.empty(); } );
}

template<typename Model>
Worker<Model>::Worker( const Address& address,
                       std::unique_ptr<Model>&& model,
                       std::optional<typename Model::TokenizerType>&& tokenizer )
  : listen_address_( address )
  , compute_kernel_( move( model ) )
  , tokenizer_( move( tokenizer ) )
{
  listen_socket_.set_reuseaddr();
  listen_socket_.bind( listen_address_ );
  listen_socket_.set_blocking( false );
  listen_socket_.listen();

  LOG( INFO ) << "Listening on " << listen_address_.to_string();

  // handle fd failures gracefully
  event_loop_.set_fd_failure_callback( [] { LOG( ERROR ) << "FD failure callback called."; } );

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
      setup_peer( peer_it );
    },
    [] { return true; },
    [] { LOG( ERROR ) << "Worker stopped listening."; } );

  event_loop_.add_rule(
    "Compute Kernel",
    Direction::In,
    compute_kernel_.event_fd(),
    [this] {
      this->compute_kernel_.event_fd().read_event();

      models::InferenceState state;
      this->compute_kernel_.pop( state );
      LOG( INFO ) << "Got state from compute kernel: " << state.to_string();

      const auto& next_worker = state.next_worker();
      auto peer_it = peers_.find( next_worker );
      bool peer_new = false;

      // are we connected to this?
      if ( peer_it == peers_.end() ) {
        TCPSocket socket;
        socket.set_blocking( false );
        socket.connect( next_worker );

        tie( peer_it, peer_new ) = peers_.emplace(
          piecewise_construct, forward_as_tuple( next_worker ), forward_as_tuple( next_worker, move( socket ) ) );

        setup_peer( peer_it );
      }

      peer_it->second.outgoing_states.push_back( move( state ) );
    },
    [] { return true; } );
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
