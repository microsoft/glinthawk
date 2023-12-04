#include "worker_pipes.hh"

#include <chrono>
#include <filesystem>
#include <random>

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

using glinthawk::models::InferenceState;

template<typename Model>
void WorkerPiped<Model>::setup_stats_handler()
{
  /* let's see if telegraph is listening */
  error_code err;
  const filesystem::path telegraf_socket { "/tmp/telegraf.sock" };
  if ( filesystem::is_socket( telegraf_socket, err ) ) {
    LOG( INFO ) << "Telegraf socket found at " << telegraf_socket.string();
    telegraf_logger_ = make_unique<monitoring::TelegrafLogger>( telegraf_socket );
  } else {
    LOG( WARNING ) << "Telegraf socket not found at " << telegraf_socket.string();
    return;
  }

  telegraf_logger_->install_rules(
    event_loop_, telegraf_rule_categories_, []( auto&& ) { return true; }, [] {} );

  event_loop_.add_rule(
    "Stats timer",
    Direction::In,
    stats_timer_,
    bind( &WorkerPiped<Model>::handle_stats, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Stats timer stopped."; } );
}

template<typename Model>
void WorkerPiped<Model>::setup_peer( std::map<net::Address, Peer>::iterator peer_it )
{
  peer_it->second.message_handler.install_rules( this->event_loop_,
                                                 this->rule_categories_,
                                                 bind( &WorkerPiped<Model>::handle_peer_message, this, placeholders::_1 ),
                                                 [] { LOG( INFO ) << "Connection to peer closed."; } );

  event_loop_.add_rule(
    "Outgoing message",
    [this, peer_it] {
      for ( auto& state : peer_it->second.outgoing_states ) {
        DLOG( INFO ) << "Sending state to " << peer_it->first.to_string() << ": " << state.to_string();
        auto state_ser = state.serialize();
        peer_it->second.message_handler.push_message( Message( Message::OpCode::InferenceState, move( state_ser ) ) );
      }

      peer_it->second.outgoing_states.clear();
    },
    [peer_it] { return not peer_it->second.outgoing_states.empty(); } );
}

template<typename Model>
void WorkerPiped<Model>::setup_blobstore( const string& blobstore_uri )
{
  auto blobstore = storage::BlobStore::create( blobstore_uri );
  CHECK( blobstore ) << "Could not create blobstore: " << blobstore_uri;

  blobstore_ = move( blobstore );
  prompt_manager_ = make_unique<prompt::PromptManager>( blobstore_ );
  completion_manager_ = make_unique<prompt::CompletionManager>( blobstore_ );
  LOG( INFO ) << "Blobstore setup complete: " << blobstore_->to_string();
}

template<typename Model>
void WorkerPiped<Model>::setup_compute_kernel( const filesystem::path& model_root,
                                               const int start_layer,
                                               const int end_layer,
                                               const int concurrency_size_pre_attention,
                                               const int concurrency_size_attention,
                                               const int concurrency_size_post_attention )
{
  CHECK_LE( start_layer, end_layer ) << "start_layer must be less than or equal to end_layer";

  compute_kernel_ = make_unique<compute::ComputeKernelPiped<Model>>(
    make_unique<Model>( model_root, start_layer, end_layer, concurrency_size ),
    concurrency_size_pre_attention,
    concurrency_size_attention,
    concurrency_size_post_attention,
    start_layer,
    end_layer );

  event_loop_.add_rule( "Compute Kernel",
                        Direction::In,
                        compute_kernel_->event_fd(),
                        bind( &WorkerPiped<Model>::handle_compute_kernel_event, this ),
                        [this] { return this->compute_kernel_ != nullptr; } );
}

template<typename Model>
WorkerPiped<Model>::WorkerPiped( const Address& worker_address,
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
    bind( &WorkerPiped<Model>::handle_coordinator_message, this, placeholders::_1 ),
    [] { LOG( FATAL ) << "Connection to coordinator closed."; },
    [] { LOG( FATAL ) << "Exception in coordinator message handler."; } );

  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    bind( &WorkerPiped<Model>::listen_callback, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Worker stopped listening."; } );

  // Send "HEY" to coordinator
  Message hey_message { Message::OpCode::Hey, this->listen_address_.to_string() };
  coordinator_.message_handler.push_message( move( hey_message ) );

  setup_stats_handler();
}

template<typename Model>
void WorkerPiped<Model>::listen_callback()
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
bool WorkerPiped<Model>::handle_coordinator_message( core::Message&& msg )
{

  LOG( INFO ) << "(Coordinator) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InitializeWorker: {
      protobuf::InitializeWorker proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Initializing worker with params=" << proto.ShortDebugString();

      // TODO(sadjad): eventually allow for loading different models
      // const auto& model_name = proto.model_name();

      setup_compute_kernel( model_root_,
                            proto.start_layer(),
                            proto.end_layer(),
                            proto.concurrency_size(),
                            proto.concurrency_pre_att_size(),
                            proto.concurrency_att_size(),
                            proto.concurrency_post_att_size() );
      setup_blobstore( proto.blobstore_uri() );

      LOG( INFO ) << "Worker initialized.";
      break;
    }

    case Message::OpCode::InferenceState: {
      // got an inference state from the coordinator
      auto state = models::InferenceState( msg.payload() );
      LOG( ERROR ) << "Got inference state from coordinator; this behavior is not supported.";
      break;
    }

    case Message::OpCode::SetRoute: {
      protobuf::SetRoute proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Setting route: " << proto.ShortDebugString();

      current_route_.clear();

      for ( int i = 0; i < proto.layer_to_address_size(); i++ ) {
        const auto& route = proto.layer_to_address( i );
        current_route_.emplace( route.layer_num(), Address { route.ip(), static_cast<uint16_t>( route.port() ) } );
      }

      LOG( INFO ) << "Route set; will be used for future prompts.";

      break;
    }

    case Message::OpCode::PushDummyPrompts: {
      // create some random inference states and feed them into the system
      const size_t prompt_count = stoull( msg.payload() );
      CHECK_LE( prompt_count, 2048 ) << "Too many dummy prompts requested.";

      if ( current_route_.empty() ) {
        LOG( ERROR ) << "No route set; cannot push dummy prompts.";
        break;
      }

      vector<InferenceState> states {};

      // generating random temperatures
      random_device rd {};
      mt19937 temp_gen { rd() };
      uniform_real_distribution<float> temp_dist( 0.0f, 1.0f );

      for ( size_t i = 0; i < prompt_count; i++ ) {
        PromptID prompt_id;
        util::digest::sha256( { reinterpret_cast<const char*>( &i ), sizeof( i ) }, prompt_id );

        // generate a random number between 0 and 1

        InferenceState state {};
        state.set_prompt_id( prompt_id );
        state.set_token( 1 /* BOS */ );
        state.set_token_pos( 0 );
        state.set_next_layer( 0 );
        state.set_next_stage( InferenceState::Stage::PreAttention );
        state.set_prompt_length( 1 );
        state.set_temperature( temp_dist( temp_gen ) );
        state.set_layer_workers( current_route_ );

        states.push_back( move( state ) );
      }

      this->compute_kernel_->push( move( states ) );
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

      for ( auto& pid : prompt_ids ) {
        prompt_queue_.enqueue( pid );
      }

      // make sure the prompt preparation thread is running
      if ( not prompt_preparation_thread_.joinable() ) {
        prompt_preparation_thread_ = thread( bind( &WorkerPiped<Model>::prompt_preparation_thread_func, this ) );
      }

      // also run the completion commiting thread
      if ( not completion_commit_thread_.joinable() ) {
        completion_commit_thread_ = thread( bind( &WorkerPiped<Model>::completion_commit_thread_func, this ) );
      }

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
void WorkerPiped<Model>::handle_compute_kernel_event()
{
  this->compute_kernel_->event_fd().read_event();

  models::InferenceState state;
  while ( this->compute_kernel_->pop( state ) ) {
    __stats__.increment<Counters::StatesProcessed>();

    DLOG( INFO ) << "Got state from compute kernel: " << state.to_string();

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
bool WorkerPiped<Model>::handle_peer_message( core::Message&& msg )
{
  DLOG( INFO ) << "(Peer) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InferenceState: {
      __stats__.increment<Counters::StatesReceived>();

      auto state = models::InferenceState( msg.payload() );
      DLOG( INFO ) << "Inference state: " << state.to_string();

      this->compute_kernel_->check_finished( state );

      if ( state.next_layer() == 0 ) {
        // We are the first layer: if this inference state contains a generated token, we should save it.
        // otherwise, we load the next token from the prompt.

        if ( state.token_pos() > 0 ) {
          __stats__.increment<Counters::TokensProcessed>();
        }

        if ( state.token_pos() >= state.prompt_length() ) {
          __stats__.increment<Counters::TokensGenerated>();

          /* we're done processing the prompt */
          auto& completion = this->completion_manager_->get( state.prompt_id() );
          completion.add_token( state.token() );

          if ( state.finished() ) {
            __stats__.increment<Counters::PromptsCompleted>();
            __stats__.add_point<IntDistributions::PromptLength>( state.token_pos() );
            completion.terminate();
          }
        } else {
          /* we're still processing the prompt, load the next token */
          auto& prompt = this->prompt_manager_->get( state.prompt_id() );
          state.set_token( prompt.token( state.token_pos() ) );
        }
      }

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
void WorkerPiped<Model>::handle_stats()
{
  stats_timer_.read_event();

  telegraf_logger_->push_measurement( __stats__ );
  __stats__.zero_out();
}

template<typename Model>
void WorkerPiped<Model>::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {}
}

template<typename Model>
void WorkerPiped<Model>::prompt_preparation_thread_func()
{
  while ( running_ ) {
    vector<InferenceState> states {};
    PromptID prompt_id;

    while ( prompt_queue_.try_dequeue( prompt_id ) && states.size() < 1'000 ) {
      DLOG( INFO ) << "Preparing the state for the prompt: " << prompt_id.base58digest();

      auto prompt = prompt_manager_->get( prompt_id );

      InferenceState state {};
      state.set_prompt_id( prompt_id );
      state.set_token( prompt.token( 0 ) );
      state.set_token_pos( 0 );
      state.set_next_layer( 0 );
      state.set_next_stage( InferenceState::Stage::PreAttention );
      state.set_prompt_length( prompt.token_count() );
      state.set_temperature( 0.0f );
      state.set_layer_workers( current_route_ );
      states.push_back( move( state ) );
    }

    if ( not states.empty() ) {
      LOG( INFO ) << "Pushing states to compute kernel: " << states.size();
      __stats__.increment<Counters::PromptsStarted>( states.size() );
      this->compute_kernel_->push( move( states ) );
    }

    this_thread::sleep_for( chrono::seconds { 1 } );
  }
}

template<typename Model>
void WorkerPiped<Model>::completion_commit_thread_func()
{
  while ( running_ ) {
    completion_manager_->commit();

    // XXX(sadjad): make this configurable
    this_thread::sleep_for( chrono::seconds { 5 } );
  }
}

template<typename Model>
WorkerPiped<Model>::~WorkerPiped()
{
  LOG( INFO ) << "Worker shutting down.";
  running_ = false;

  if ( prompt_preparation_thread_.joinable() ) {
    prompt_preparation_thread_.join();
  }

  if ( completion_commit_thread_.joinable() ) {
    completion_commit_thread_.join();
  }
}

namespace glinthawk::core {

template class WorkerPiped<models::llama2::cpu::Llama2_7B_Chat<_Float16>>;
template class WorkerPiped<models::llama2::cpu::Llama2_13B_Chat<_Float16>>;
template class WorkerPiped<models::llama2::cpu::Llama2_70B_Chat<_Float16>>;
template class WorkerPiped<models::llama2::cpu::Stories_110M<_Float16>>;

template class WorkerPiped<models::llama2::cpu::Llama2_7B_Chat<float>>;
template class WorkerPiped<models::llama2::cpu::Llama2_13B_Chat<float>>;
template class WorkerPiped<models::llama2::cpu::Llama2_70B_Chat<float>>;
template class WorkerPiped<models::llama2::cpu::Stories_110M<float>>;

#ifdef GLINTHAWK_CUDA_ENABLED
template class WorkerPiped<models::llama2::cuda::Llama2_7B_Chat<__half>>;
template class WorkerPiped<models::llama2::cuda::Llama2_13B_Chat<__half>>;
template class WorkerPiped<models::llama2::cuda::Llama2_70B_Chat<__half>>;
template class WorkerPiped<models::llama2::cuda::Stories_110M<__half>>;
#endif

} // namespace glinthawk::core
