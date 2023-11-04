#include "worker.hh"

#include <glog/logging.h>

#include "message/util.hh"
#include "models/common/model.hh"
#include "util/digest.hh"

#include "glinthawk.pb.h"

#include "models/llama2/cpu/model.cc"
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
  peer_it->second.message_handler.install_rules( this->event_loop_,
                                                 this->rule_categories_,
                                                 bind( &Worker<Model>::handle_peer_message, this, placeholders::_1 ),
                                                 [] { LOG( INFO ) << "Connection to peer closed."; } );

  event_loop_.add_rule(
    "Outgoing message",
    [this, peer_it] {
      for ( auto& state : peer_it->second.outgoing_states ) {
        LOG( INFO ) << "Sending state to " << peer_it->first.to_string() << ": " << state.to_string();
        peer_it->second.message_handler.push_message( Message( Message::OpCode::InferenceState, state.serialize() ) );
      }

      peer_it->second.outgoing_states.clear();
    },
    [peer_it] { return not peer_it->second.outgoing_states.empty(); } );
}

template<typename Model>
void Worker<Model>::setup_blobstore( const string& blobstore_uri )
{
  auto blobstore = storage::BlobStore::create( blobstore_uri );
  CHECK( blobstore ) << "Could not create blobstore: " << blobstore_uri;

  blobstore_ = move( blobstore );
  prompt_manager_ = make_unique<prompt::PromptManager>( blobstore_ );
  completion_manager_ = make_unique<prompt::CompletionManager>( blobstore_ );
  LOG( INFO ) << "Blobstore setup complete: " << blobstore_->to_string();
}

template<typename Model>
void Worker<Model>::setup_compute_kernel( const filesystem::path& model_root,
                                          const int start_layer,
                                          const int end_layer,
                                          const int concurrency_size )
{
  CHECK_LE( start_layer, end_layer ) << "start_layer must be less than or equal to end_layer";

  compute_kernel_ = make_unique<compute::ComputeKernel<Model>>(
    make_unique<Model>( model_root, start_layer, end_layer, concurrency_size ), concurrency_size );

  event_loop_.add_rule( "Compute Kernel",
                        Direction::In,
                        compute_kernel_->event_fd(),
                        bind( &Worker<Model>::handle_compute_kernel_event, this ),
                        [this] { return this->compute_kernel_ != nullptr; } );
}

template<typename Model>
Worker<Model>::Worker( const Address& worker_address,
                       const Address& coordinator_address,
                       const std::filesystem::path& model_root )
  : listen_address_( worker_address )
  , listen_socket_( [this]() -> TCPSocket {
    TCPSocket socket;
    socket.set_reuseaddr();
    socket.bind( this->listen_address_ );
    socket.set_blocking( false );
    socket.listen();
    LOG( INFO ) << "Listening on " << this->listen_address_.to_string();
    return socket;
  }() )
  , coordinator_address_( coordinator_address )
  , coordinator_( coordinator_address,
                  [this]() -> TCPSocket {
                    TCPSocket socket;
                    socket.set_blocking( false );
                    socket.connect( this->coordinator_address_ );
                    LOG( INFO ) << "Connecting to coordinator at " << this->coordinator_address_.to_string();
                    return socket;
                  }() )
  , model_root_( model_root )
{
  // handle fd failures gracefully
  event_loop_.set_fd_failure_callback( [] { LOG( ERROR ) << "FD failure callback called."; } );

  coordinator_.message_handler.install_rules(
    this->event_loop_,
    this->rule_categories_,
    bind( &Worker<Model>::handle_coordinator_message, this, placeholders::_1 ),
    [] { LOG( FATAL ) << "Connection to coordinator closed."; },
    [] { LOG( FATAL ) << "Exception in coordinator message handler."; } );

  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    bind( &Worker<Model>::listen_callback, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Worker stopped listening."; } );

  // Send "HEY" to coordinator
  Message hey_message { Message::OpCode::Hey, this->listen_address_.to_string() };
  coordinator_.message_handler.push_message( move( hey_message ) );
}

template<typename Model>
void Worker<Model>::listen_callback()
{
  TCPSocket socket = listen_socket_.accept();
  auto addr = socket.peer_address();
  LOG( INFO ) << "Accepted connection from " << addr.to_string();

  auto [peer_it, peer_new]
    = peers_.emplace( piecewise_construct, forward_as_tuple( addr ), forward_as_tuple( addr, move( socket ) ) );

  CHECK( peer_new ) << "A peer with this address already exists.";
  setup_peer( peer_it );
}

template<typename Model>
bool Worker<Model>::handle_coordinator_message( core::Message&& msg )
{

  LOG( INFO ) << "(Coordinator) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InitializeWorker: {
      protobuf::InitializeWorker proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Initializing worker with params=" << proto.ShortDebugString();

      // TODO(sadjad): eventually allow for loading different models
      // const auto& model_name = proto.model_name();

      setup_compute_kernel( model_root_, proto.start_layer(), proto.end_layer(), proto.concurrency_size() );
      setup_blobstore( proto.blobstore_uri() );

      LOG( INFO ) << "Worker initialized.";
      break;
    }

    case Message::OpCode::InferenceState: {
      // got an inference state from the coordinator
      auto state = models::InferenceState( msg.payload() );
      LOG( INFO ) << "Inference state: " << state.to_string();
      this->compute_kernel_->push( move( state ) );
      break;
    }

    case Message::OpCode::SetRoute: {
      protobuf::SetRoute proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Setting route: " << proto.ShortDebugString();

      for ( int i = 0; i < proto.layer_to_address_size(); i++ ) {
        const auto& route = proto.layer_to_address( i );
        current_route_.emplace( route.layer_num(), Address { route.ip(), static_cast<uint16_t>( route.port() ) } );
      }

      LOG( INFO ) << "Route set; will be used for future prompts.";

      break;
    }

    case Message::OpCode::ProcessPrompts: {
      if ( not this->prompt_manager_ ) {
        // XXX(sadjad): should do something better here
        LOG( ERROR ) << "Got prompts, but prompt manager not initialized.";
        break;
      }

      protobuf::ProcessPrompts proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Got prompts from the coordinator: " << proto.prompt_ids_size() << " prompt(s)";

      vector<PromptID> prompt_ids;
      for ( int i = 0; i < proto.prompt_ids_size(); i++ ) {
        prompt_ids.push_back( PromptID::from_base58digest( proto.prompt_ids( i ) ) );
      }

      this->prompt_manager_->fetch( prompt_ids );
      LOG( INFO ) << "Fetched prompts from blobstore.";

      break;
    }

    default: {
      LOG( WARNING ) << "[Coordinator] Message not handled." << endl;
      break;
    }
  }

  return true;
}

template<typename Model>
void Worker<Model>::handle_compute_kernel_event()
{
  this->compute_kernel_->event_fd().read_event();

  models::InferenceState state;
  while ( this->compute_kernel_->pop( state ) ) {
    LOG( INFO ) << "Got state from compute kernel: " << state.to_string();

    // little hack to test pull queue on one GPU without running out of memory.
    // if (state.next_layer() == 10)
    //   state.set_next_layer( 22 );

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
  }
}

template<typename Model>
bool Worker<Model>::handle_peer_message( core::Message&& msg )
{
  LOG( INFO ) << "(Peer) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InferenceState: {
      auto state = models::InferenceState( msg.payload() );
      LOG( INFO ) << "Inference state: " << state.to_string();

      if ( state.next_layer() == 0 ) {
        // We are the first layer, so send the result back to coordinator
        coordinator_.message_handler.push_message( Message( Message::OpCode::InferenceState, state.serialize() ) );
      }

      this->compute_kernel_->check_finished( state );

      if ( state.finished() ) {
        this->compute_kernel_->push_finished( move( state ) );
      } else {
        this->compute_kernel_->push( move( state ) );
      }
      break;
    }

    default: {
      LOG( WARNING ) << "[Peer] Message not handled." << endl;
      break;
    }
  }

  return true;
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

template class Worker<models::llama2::cpu::Llama2<_Float16>>;
template class Worker<models::llama2::cpu::Llama2<float>>;

} // namespace glinthawk::core
