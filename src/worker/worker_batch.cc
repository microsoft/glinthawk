#include "worker_batch.hh"

#include <chrono>
#include <filesystem>
#include <random>

#include <glog/logging.h>

#include "compute/kernel_batch.hh"
#include "message/util.hh"
#include "models/common/model.hh"
#include "util/digest.hh"

#include "models/llama2/model.hh"

#include "glinthawk.pb.h"

using namespace std;
using namespace std::chrono;
using namespace glinthawk;
using namespace glinthawk::core;
using namespace glinthawk::net;

using glinthawk::models::BatchedInferenceState;
using glinthawk::models::InferenceState;

namespace {

template<typename DType>
DataType get_datatype()
{
  if constexpr ( std::is_same_v<DType, float> ) {
    return DataType::Float32;
  }
#if defined( TARGET_PLATFORM_AMD64 )
  else if constexpr ( std::is_same_v<DType, _Float16> ) {
    return DataType::Float16;
  }
#elif defined( TARGET_PLATFORM_CUDA )
  else if constexpr ( std::is_same_v<DType, __half> ) {
    return DataType::Float16;
  }
#endif
}

}

template<typename Model>
void BatchedWorker<Model>::setup_stats_handler()
{
  /* let's see if telegraph is listening */
  error_code err;
  const filesystem::path telegraf_socket { "/tmp/telegraf.sock" };
  if ( filesystem::is_socket( telegraf_socket, err ) ) {
    LOG( INFO ) << "Telegraf socket found at " << telegraf_socket.string();
    telegraf_logger_ = make_unique<monitoring::TelegrafLogger>( telegraf_socket );
    telegraf_logger_->install_rules( event_loop_, telegraf_rule_categories_, []( auto&& ) { return true; }, [] {} );
  } else {
    LOG( WARNING ) << "Telegraf socket not found at " << telegraf_socket.string();
  }

  event_loop_.add_rule(
    "Stats timer",
    Direction::In,
    stats_timer_,
    bind( &BatchedWorker<Model>::handle_stats, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Stats timer stopped."; } );
}

template<typename Model>
void BatchedWorker<Model>::setup_peer( std::map<net::Address, Peer>::iterator peer_it )
{
  peer_it->second.message_handler.install_rules(
    this->event_loop_,
    this->rule_categories_,
    bind( &BatchedWorker<Model>::handle_peer_message, this, placeholders::_1 ),
    [] { LOG( INFO ) << "Connection to peer closed."; } );

  event_loop_.add_rule(
    "Outgoing message",
    [this, peer_it] {
      for ( auto& state : peer_it->second.outgoing_states ) {
        auto state_ser = state.serialize();
        peer_it->second.message_handler.push_message(
          Message( Message::OpCode::BatchedInferenceState, move( state_ser ) ) );
      }

      peer_it->second.outgoing_states.clear();
    },
    [peer_it] { return not peer_it->second.outgoing_states.empty(); } );
}

template<typename Model>
void BatchedWorker<Model>::setup_blobstore( const string& blobstore_uri )
{
  auto blobstore = storage::BlobStore::create( blobstore_uri );
  CHECK( blobstore ) << "Could not create blobstore: " << blobstore_uri;

  blobstore_ = move( blobstore );
  prompt_manager_ = make_unique<prompt::PromptManager>( blobstore_ );
  completion_manager_ = make_unique<prompt::CompletionManager>( blobstore_ );
  LOG( INFO ) << "Blobstore setup complete: " << blobstore_->to_string();
}

template<typename Model>
void BatchedWorker<Model>::setup_compute_kernel( const filesystem::path& model_root,
                                                 const int start_layer,
                                                 const int end_layer,
                                                 const int concurrency_size_pre_attention,
                                                 const int concurrency_size_attention,
                                                 const int concurrency_size_post_attention,
                                                 const int concurrency_size_classification,
                                                 const int max_context_count,
                                                 const bool randomize )
{
  CHECK_LE( start_layer, end_layer ) << "start_layer must be less than or equal to end_layer";

  const int max_concurrency_size = std::max( { concurrency_size_pre_attention,
                                               concurrency_size_attention,
                                               concurrency_size_post_attention,
                                               concurrency_size_classification } );

  compute_kernel_ = make_unique<compute::BatchedComputeKernel<Model>>(
    make_unique<Model>( model_root, start_layer, end_layer, max_concurrency_size, max_context_count, randomize ),
    max_concurrency_size );

  event_loop_.add_rule( "Compute Kernel",
                        Direction::In,
                        compute_kernel_->event_fd(),
                        bind( &BatchedWorker<Model>::handle_compute_kernel_event, this ),
                        [this] { return this->compute_kernel_ != nullptr; } );
}

template<typename Model>
BatchedWorker<Model>::BatchedWorker( const Address& worker_address,
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
    bind( &BatchedWorker<Model>::handle_coordinator_message, this, placeholders::_1 ),
    [] { LOG( FATAL ) << "Connection to coordinator closed."; },
    [] { LOG( FATAL ) << "Exception in coordinator message handler."; } );

  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    bind( &BatchedWorker<Model>::listen_callback, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Worker stopped listening."; } );

  // Send "HEY" to coordinator
  protobuf::Hey hey_proto;
  hey_proto.set_ip( this->listen_address_.ip() );
  hey_proto.set_port( this->listen_address_.port() );
#if defined( TARGET_PLATFORM_AMD64 )
  hey_proto.set_platform( protobuf::Hey::AMD64 );
#elif defined( TARGET_PLATFORM_CUDA )
  hey_proto.set_platform( protobuf::Hey::CUDA );
#endif
  coordinator_.message_handler.push_message( { Message::OpCode::Hey, hey_proto.SerializeAsString() } );

  setup_stats_handler();
}

template<typename Model>
void BatchedWorker<Model>::listen_callback()
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
bool BatchedWorker<Model>::handle_coordinator_message( core::Message&& msg )
{
  LOG( INFO ) << "(Coordinator) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InitializeWorker: {
      protobuf::InitializeWorker proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Initializing worker with params=" << proto.ShortDebugString();

      // TODO(sadjad): eventually allow for loading different models
      // const auto& model_name = proto.model_name();

      __stats__.tag( "start_layer", to_string( proto.start_layer() ) );
      __stats__.tag( "end_layer", to_string( proto.end_layer() ) );
#if defined( TARGET_PLATFORM_AMD64 )
      __stats__.tag( "platform", "amd64" );
#elif defined( TARGET_PLATFORM_CUDA )
      __stats__.tag( "platform", "cuda" );
#endif

      setup_compute_kernel( model_root_,
                            proto.start_layer(),
                            proto.end_layer(),
                            proto.concurrency_pre_att_size(),
                            proto.concurrency_att_size(),
                            proto.concurrency_post_att_size(),
                            proto.concurrency_cls_size(),
                            proto.max_context_count(),
                            proto.randomize() );

      setup_blobstore( proto.blobstore_uri() );

      LOG( INFO ) << "Worker initialized.";
      break;
    }

    case Message::OpCode::InferenceState:
    case Message::OpCode::BatchedInferenceState: {
      // got an inference state from the coordinator
      auto state = models::BatchedInferenceState<typename Model::ConfigType>( msg.payload() );
      LOG( ERROR ) << "Got inference state from coordinator; this behavior is not supported.";
      break;
    }

    case Message::OpCode::SetRoute: {
      protobuf::SetRoute proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Setting route: " << proto.ShortDebugString();

      RouteMap new_route {};
      for ( int i = 0; i < proto.layer_to_address_size(); i++ ) {
        const auto& route = proto.layer_to_address( i );
        InferenceState::Stage next_stage;
        switch ( route.stage() ) {
          case protobuf::SetRoute::LayerToAddress::PreAttention:
            next_stage = InferenceState::Stage::PreAttention;
            break;
          case protobuf::SetRoute::LayerToAddress::Attention: next_stage = InferenceState::Stage::Attention; break;
          case protobuf::SetRoute::LayerToAddress::PostAttention:
            next_stage = InferenceState::Stage::PostAttention;
            break;
          case protobuf::SetRoute::LayerToAddress::Classification:
            next_stage = InferenceState::Stage::Classification;
            break;
          default: throw std::runtime_error( "invalid stage" );
        }
        new_route.emplace( std::make_pair( route.layer_num(), next_stage ),
                           Address { route.ip(), static_cast<uint16_t>( route.port() ) } );
      }

      route_set_.emplace( proto.route_id(), new_route );

      LOG( INFO ) << "Route set; will be used for future prompts.";
      break;
    }

    case Message::OpCode::PushDummyPrompts: {
      // create some random inference states and feed them into the system
      protobuf::PushDummyPrompts proto;
      proto.ParseFromString( msg.payload() );

      const uint32_t prompt_count = proto.count();
      const uint32_t batch_size = proto.batch_size();

      if ( prompt_count == 0 or prompt_count > ( 1 << 16 ) ) {
        LOG( ERROR ) << "Invalid number of dummy prompts requested: " << prompt_count;
        break;
      }

      auto it = route_set_.find( RouteID {} );
      if ( it == route_set_.end() ) {
        LOG( FATAL ) << "No dummy route set; cannot push dummy prompts.";
        break;
      }

      RouteMap dummy_route = it->second;
      vector<BatchedInferenceState<typename Model::ConfigType>> states {};

      // prompt id is sha256( current_time and dummy_prompt_current_id_ )
      auto generate_next_prompt_id = [this]() -> PromptID {
        PromptID prompt_id;
        char prompt_id_buf[sizeof( uint64_t ) * 2];
        const uint64_t current_time = duration_cast<nanoseconds>( system_clock::now().time_since_epoch() ).count();

        memcpy( prompt_id_buf, &current_time, sizeof( uint64_t ) );
        memcpy( prompt_id_buf + sizeof( uint64_t ), &( this->dummy_prompt_current_id_ ), sizeof( uint64_t ) );

        util::digest::sha256( { prompt_id_buf, sizeof( prompt_id_buf ) }, prompt_id );

        this->dummy_prompt_current_id_++;
        return prompt_id;
      };

      // generating random temperatures
      random_device rd {};
      mt19937 temp_gen { rd() };
      uniform_real_distribution<float> temp_dist( 0.0f, 1.0f );

      const uint32_t batch_count = ( prompt_count + ( batch_size - 1 ) ) / batch_size;

      for ( size_t i = 0; i < batch_count; i++ ) {
        BatchedInferenceState<typename Model::ConfigType> state {
          batch_size, get_datatype<typename Model::ModelDataType>(), RouteID {}, ModelID {}, false, false, false
        };

        for ( size_t j = 0; j < batch_size; j++ ) {
          const auto idx = i * batch_size + j;

          if ( idx >= prompt_count ) {
            break;
          }

          state.set_prompt( idx, generate_next_prompt_id(), 1 /* TOKEN_BOS */, 0, temp_dist( temp_gen ), 1 );
        }

        this->compute_kernel_->push( move( state ) );
      }

      // also run the completion commiting thread
      if ( not completion_commit_thread_.joinable() ) {
        completion_commit_thread_ = thread( bind( &BatchedWorker<Model>::completion_commit_thread_func, this ) );
      }

      break;
    }

    case Message::OpCode::ProcessPrompts: {
      // TODO: this path is broken since we are not setting any route ids for prompts
      LOG( FATAL ) << "FIX ME FIRST!";

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
        prompt_preparation_thread_ = thread( bind( &BatchedWorker<Model>::prompt_preparation_thread_func, this ) );
      }

      // also run the completion commiting thread
      if ( not completion_commit_thread_.joinable() ) {
        completion_commit_thread_ = thread( bind( &BatchedWorker<Model>::completion_commit_thread_func, this ) );
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
void BatchedWorker<Model>::handle_compute_kernel_event()
{
  this->compute_kernel_->event_fd().read_event();

  BatchedInferenceState<typename Model::ConfigType> state;

  while ( this->compute_kernel_->pop( state ) ) {
    __stats__.increment<Counters::StatesProcessed>( state.batch_size() );

    const auto next_worker = find_next_worker( route_set_.at( state.route_id() ), state );
    auto peer_it = peers_.find( next_worker );

    // are we connected to this?
    if ( peer_it == peers_.end() ) {
      TCPSocket socket;
      socket.set_blocking( false );
      socket.connect( next_worker );

      tie( peer_it, ignore ) = peers_.emplace(
        piecewise_construct, forward_as_tuple( next_worker ), forward_as_tuple( next_worker, move( socket ) ) );

      setup_peer( peer_it );
    }

    peer_it->second.outgoing_states.push_back( move( state ) );
  }
}

template<typename Model>
bool BatchedWorker<Model>::handle_peer_message( core::Message&& msg )
{
  DLOG( INFO ) << "(Peer) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::BatchedInferenceState: {
      __stats__.increment<Counters::StatesReceived>();

      BatchedState state { msg.payload() };

      // LOG( INFO ) << state.debug_string( true );

      if ( route_set_.find( state.route_id() ) == route_set_.end() ) {
        LOG( FATAL ) << "No route with id=" << state.route_id() << " in route set.";
      }

      if ( state.next_layer() == 0 and state.next_stage() == BatchedState::Stage::PreAttention ) {
        /* first worker in the chain */

        for ( size_t i = 0; i < state.batch_size(); i++ ) {
          __stats__.increment<Counters::TokensGenerated>();

          if ( state.finished( i ) ) {
            auto& completion = this->completion_manager_->get( state.prompt_id( i ) );
            completion.add_token( state.token( i ) );
            __stats__.increment<Counters::PromptsCompleted>();
            __stats__.add_point<IntDistributions::PromptLength>( state.token_pos( i ) );
            completion_manager_->terminate( state.prompt_id( i ) );

            state.set_discarded( i );
          }
        }
      }

      this->compute_kernel_->push( move( state ) );
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
void BatchedWorker<Model>::handle_stats()
{
  stats_timer_.read_event();
  if ( telegraf_logger_ != nullptr ) {
    telegraf_logger_->push_measurement( __stats__ );
  }

  if ( const auto completed_prompts = __stats__.get<Counters::PromptsCompleted>(); completed_prompts > 0 ) {
    Message msg { Message::OpCode::PromptCompleted, to_string( completed_prompts ) };
    this->coordinator_.message_handler.push_message( move( msg ) );
  }

  __stats__.zero_out();
}

template<typename Model>
void BatchedWorker<Model>::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {}
}

template<typename Model>
void BatchedWorker<Model>::prompt_preparation_thread_func()
{
}

template<typename Model>
void BatchedWorker<Model>::completion_commit_thread_func()
{
}

template<typename Model>
BatchedWorker<Model>::~BatchedWorker()
{
  LOG( INFO ) << "BatchedWorker shutting down.";
  running_ = false;

  if ( prompt_preparation_thread_.joinable() ) {
    prompt_preparation_thread_.join();
  }

  if ( completion_commit_thread_.joinable() ) {
    completion_commit_thread_.join();
  }
}

namespace glinthawk::core {

#if defined( TARGET_PLATFORM_AMD64 )
namespace models = glinthawk::models::llama2::amd64;
#elif defined( TARGET_PLATFORM_CUDA )
namespace models = glinthawk::models::llama2::cuda;
#endif

template class BatchedWorker<models::Llama2_7B_Chat<glinthawk::float16_t>>;
template class BatchedWorker<models::Llama2_13B_Chat<glinthawk::float16_t>>;
template class BatchedWorker<models::Llama2_70B_Chat<glinthawk::float16_t>>;
template class BatchedWorker<models::Stories_110M<glinthawk::float16_t>>;
template class BatchedWorker<models::Llama2_7B_Chat<glinthawk::float32_t>>;
template class BatchedWorker<models::Llama2_13B_Chat<glinthawk::float32_t>>;
template class BatchedWorker<models::Llama2_70B_Chat<glinthawk::float32_t>>;
template class BatchedWorker<models::Stories_110M<glinthawk::float32_t>>;

} // namespace glinthawk::core
