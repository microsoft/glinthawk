#include "worker.hh"

#include <chrono>
#include <filesystem>
#include <random>

#include <glog/logging.h>

#include "message/util.hh"
#include "models/common/model.hh"
#include "util/digest.hh"

#include "models/llama2/model.hh"

#include "glinthawk.pb.h"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::core;
using namespace glinthawk::net;

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
void Worker<Model>::setup_stats_handler()
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
    bind( &Worker<Model>::handle_stats, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Stats timer stopped."; } );
}

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
        DLOG( INFO ) << "Sending state to " << peer_it->first.to_string() << ": " << state.to_string();

        const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
        if ( state.next_stage() == InferenceState::Stage::PreAttention and not state.finished() ) {
          __stats__.add_point<IntDistributions::PreWorker2SerializeTime>( current_time - state.timestamp() );
          if ( state.batch_last() ) {
            __stats__.add_point<IntDistributions::PreWorker2SerializeTimeBatch>( current_time
                                                                                 - state.batch_timestamp() );
          }
        } else if ( state.next_stage() == InferenceState::Stage::Attention and not state.finished() ) {
          __stats__.add_point<IntDistributions::AttWorker2SerializeTime>( current_time - state.timestamp() );
          if ( state.batch_last() ) {
            __stats__.add_point<IntDistributions::AttWorker2SerializeTimeBatch>( current_time
                                                                                 - state.batch_timestamp() );
          }
        } else if ( state.next_stage() == InferenceState::Stage::PostAttention and not state.finished() ) {
          __stats__.add_point<IntDistributions::PostWorker2SerializeTime>( current_time - state.timestamp() );
          if ( state.batch_last() ) {
            __stats__.add_point<IntDistributions::PostWorker2SerializeTimeBatch>( current_time
                                                                                  - state.batch_timestamp() );
          }
        } else if ( state.next_stage() == InferenceState::Stage::Classification and not state.finished() ) {
          __stats__.add_point<IntDistributions::ClsWorker2SerializeTime>( current_time - state.timestamp() );
          if ( state.batch_last() ) {
            __stats__.add_point<IntDistributions::ClsWorker2SerializeTimeBatch>( current_time
                                                                                 - state.batch_timestamp() );
          }
        }
        state.set_timestamp( current_time );
        state.set_time_in_node( state.time_in_node() + current_time );

        auto state_ser = state.serialize();
        peer_it->second.message_handler.push_message( Message( Message::OpCode::InferenceState, move( state_ser ) ) );
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

  //  compute_kernel_ = make_unique<compute::ComputeKernel<Model>>(
  //    make_unique<Model>( model_root, start_layer, end_layer, max_concurrency_size, max_context_count, randomize ),
  //    max_concurrency_size );
  compute_kernel_ = make_unique<compute::ComputeKernelPiped<Model>>(
    make_unique<Model>( model_root, start_layer, end_layer, max_concurrency_size, max_context_count, randomize ),
    concurrency_size_pre_attention,
    concurrency_size_attention,
    concurrency_size_post_attention,
    concurrency_size_classification,
    start_layer,
    end_layer );

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
#if defined( TARGET_PLATFORM_AMD64 )
  Message hey_message { Message::OpCode::HeyCPU, this->listen_address_.to_string() };
#elif defined( TARGET_PLATFORM_CUDA )
  Message hey_message { Message::OpCode::HeyGPU, this->listen_address_.to_string() };
#else
#error "Unknown target platform"
#endif

  coordinator_.message_handler.push_message( move( hey_message ) );
  setup_stats_handler();
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

      __stats__.tag( "start_layer", to_string( proto.start_layer() ) );
      __stats__.tag( "end_layer", to_string( proto.end_layer() ) );
#ifdef TARGET_PLATFORM_AMD64
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

      RouteID route_id_dummy;
      util::digest::sha256( proto.route_id(), route_id_dummy );
      route_set_.emplace( route_id_dummy, new_route );

      LOG( INFO ) << "Route set; will be used for future prompts.";

      break;
    }

    case Message::OpCode::PushDummyPrompts: {
      // create some random inference states and feed them into the system
      const size_t prompt_count = stoull( msg.payload() );
      RouteID route_id_dummy;
      util::digest::sha256( "dummy_path", route_id_dummy );

      if ( prompt_count == 0 or prompt_count > ( 1 << 16 ) ) {
        LOG( ERROR ) << "Invalid number of dummy prompts requested: " << prompt_count;
        break;
      }

      auto it = route_set_.find( route_id_dummy );
      if ( it == route_set_.end() ) {
        LOG( FATAL ) << "No dummy route set; cannot push dummy prompts.";
        break;
      }
      RouteMap dummy_route = it->second;

      vector<InferenceState> states {};

      // generating random temperatures
      random_device rd {};
      mt19937 temp_gen { rd() };
      uniform_real_distribution<float> temp_dist( 0.0f, 1.0f );

      // get current time as uint64_t
      uint64_t current_time
        = chrono::duration_cast<chrono::nanoseconds>( chrono::system_clock::now().time_since_epoch() ).count();

      char prompt_id_buf[sizeof( uint64_t ) * 2];
      memcpy( prompt_id_buf, &current_time, sizeof( uint64_t ) );

      for ( size_t i = 0; i < prompt_count; i++ ) {
        PromptID prompt_id;
        memcpy( prompt_id_buf + sizeof( uint64_t ), &dummy_prompt_current_id_, sizeof( dummy_prompt_current_id_ ) );
        util::digest::sha256( { prompt_id_buf, sizeof( prompt_id_buf ) }, prompt_id );

        dummy_prompt_current_id_++;

        // generate a random number between 0 and 1 for temprature
        InferenceState state { get_datatype<typename Model::ModelDataType>() };
        state.set_prompt_id( prompt_id );
        state.set_route_id( route_id_dummy );
        state.set_token( 1 /* BOS */ );
        state.set_token_pos( 0 );
        state.set_next_layer( 0 );
        state.set_next_stage( InferenceState::Stage::PreAttention );
        state.set_prompt_length( 1 );
        state.set_temperature( temp_dist( temp_gen ) );
        state.set_loop_start_timestamp( std::chrono::steady_clock::now().time_since_epoch().count() );
        state.set_time_in_node( -std::chrono::steady_clock::now().time_since_epoch().count() );
        state.set_layer_workers( dummy_route );

        states.push_back( move( state ) );
      }

      this->compute_kernel_->push( move( states ) );

      // also run the completion commiting thread
      if ( not completion_commit_thread_.joinable() ) {
        completion_commit_thread_ = thread( bind( &Worker<Model>::completion_commit_thread_func, this ) );
      }

      break;
    }

    case Message::OpCode::ProcessPrompts: {
      // TODO: this path is broken since we are not setting any route ids for prompts
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
        prompt_preparation_thread_ = thread( bind( &Worker<Model>::prompt_preparation_thread_func, this ) );
      }

      // also run the completion commiting thread
      if ( not completion_commit_thread_.joinable() ) {
        completion_commit_thread_ = thread( bind( &Worker<Model>::completion_commit_thread_func, this ) );
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
void Worker<Model>::handle_compute_kernel_event()
{
  this->compute_kernel_->event_fd().read_event();

  models::InferenceState state;
  while ( this->compute_kernel_->pop( state ) ) {
    __stats__.increment<Counters::StatesProcessed>();

    const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
    if ( state.next_stage() == InferenceState::Stage::PreAttention and not state.finished() ) {
      __stats__.add_point<IntDistributions::PreInference2WorkerTime>( current_time - state.timestamp() );
      if ( state.batch_last() ) {
        __stats__.add_point<IntDistributions::PreInference2WorkerTimeBatch>( current_time - state.batch_timestamp() );
      }
    } else if ( state.next_stage() == InferenceState::Stage::Attention and not state.finished() ) {
      __stats__.add_point<IntDistributions::AttInference2WorkerTime>( current_time - state.timestamp() );
      if ( state.batch_last() ) {
        __stats__.add_point<IntDistributions::AttInference2WorkerTimeBatch>( current_time - state.batch_timestamp() );
      }
    } else if ( state.next_stage() == InferenceState::Stage::PostAttention and not state.finished() ) {
      __stats__.add_point<IntDistributions::PostInference2WorkerTime>( current_time - state.timestamp() );
      if ( state.batch_last() ) {
        __stats__.add_point<IntDistributions::PostInference2WorkerTimeBatch>( current_time - state.batch_timestamp() );
      }
    } else if ( state.next_stage() == InferenceState::Stage::Classification and not state.finished() ) {
      __stats__.add_point<IntDistributions::ClsInference2WorkerTime>( current_time - state.timestamp() );
      if ( state.batch_last() ) {
        __stats__.add_point<IntDistributions::ClsInference2WorkerTimeBatch>( current_time - state.batch_timestamp() );
      }
    }
    state.set_timestamp( current_time );

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
bool Worker<Model>::handle_peer_message( core::Message&& msg )
{
  DLOG( INFO ) << "(Peer) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InferenceState: {
      __stats__.increment<Counters::StatesReceived>();

      auto state = models::InferenceState( msg.payload() );
      auto it = route_set_.find( state.route_id() );
      if ( it == route_set_.end() )
        LOG( FATAL ) << "No route id " << state.route_id().base58digest().substr( 0, 8 )
                     << " in route set for prompt: " << state;
      state.set_layer_workers( it->second );
      DLOG( INFO ) << "Inference state: " << state.to_string();

      const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
      if ( state.next_stage() == InferenceState::Stage::Attention and not state.finished() ) {
        __stats__.add_point<IntDistributions::PreSerialize2AttWorkerTime>( current_time - state.timestamp() );
        if ( state.batch_last() ) {
          __stats__.add_point<IntDistributions::PreSerialize2AttWorkerTimeBatch>( current_time
                                                                                  - state.batch_timestamp() );
        }
      } else if ( state.next_stage() == InferenceState::Stage::PostAttention and not state.finished() ) {
        __stats__.add_point<IntDistributions::AttSerialize2PostWorkerTime>( current_time - state.timestamp() );
        if ( state.batch_last() ) {
          __stats__.add_point<IntDistributions::AttSerialize2PostWorkerTimeBatch>( current_time
                                                                                   - state.batch_timestamp() );
        }
      }
      state.set_timestamp( current_time );

      this->compute_kernel_->check_finished( state );

      if ( state.next_layer() == 0 and state.next_stage() == models::InferenceState::Stage::PreAttention ) {

        // Only log prompt latency after context has been fully allocated
        const auto current_time_prompt = std::chrono::steady_clock::now().time_since_epoch().count();
        if ( state.token_pos() > 1 and not state.finished() ) {
          __stats__.add_point<IntDistributions::PromptLatency>( current_time_prompt - state.loop_start_timestamp() );
          __stats__.add_point<IntDistributions::InNodeLatency>( state.time_in_node() );
          __stats__.add_point<IntDistributions::InNetLatency>( current_time_prompt - state.loop_start_timestamp()
                                                               - state.time_in_node() );
        }
        state.set_loop_start_timestamp( current_time_prompt );
        state.set_time_in_node( 0 );

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
            completion_manager_->terminate( state.prompt_id() );
          }
        } else {
          /* we're still processing the prompt, load the next token */
          auto& prompt = this->prompt_manager_->get( state.prompt_id() );
          state.set_token( prompt.token( state.token_pos() ) );
        }
      }

      state.set_time_in_node( state.time_in_node() - current_time );

      if ( state.finished() ) {
        this->compute_kernel_->push_finished( move( state ) );
      } else {
        if ( msg_counter_ != 0 ) {
          if ( state.next_stage() == InferenceState::Stage::PreAttention ) {
            __stats__.add_point<IntDistributions::PreSerialize2AttWorkerVarTime>( current_time - past_msg_time_ );
          } else if ( state.next_stage() == InferenceState::Stage::Attention ) {
            __stats__.add_point<IntDistributions::AttSerialize2PostWorkerVarTime>( current_time - past_msg_time_ );
          } else if ( state.next_stage() == InferenceState::Stage::PostAttention ) {
            __stats__.add_point<IntDistributions::PostSerialize2ClsWorkerVarTime>( current_time - past_msg_time_ );
          } else if ( state.next_stage() == InferenceState::Stage::Classification ) {
            __stats__.add_point<IntDistributions::ClsSerialize2PreWorkerVarTime>( current_time - past_msg_time_ );
          }
        }
        msg_counter_ = ( msg_counter_ + 1 ) % 24;
        past_msg_time_ = current_time;
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
void Worker<Model>::handle_stats()
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
void Worker<Model>::run()
{
  while ( event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {}
}

template<typename Model>
void Worker<Model>::prompt_preparation_thread_func()
{
  while ( running_ ) {
    vector<InferenceState> states {};
    PromptID prompt_id;

    while ( prompt_queue_.try_dequeue( prompt_id ) && states.size() < 1'000 ) {
      DLOG( INFO ) << "Preparing the state for the prompt: " << prompt_id.base58digest();

      auto prompt = prompt_manager_->get( prompt_id );

      // TODO: this is broken now since there is nothing to assign route_id to the prompt
      InferenceState state { get_datatype<typename Model::ModelDataType>() };
      state.set_prompt_id( prompt_id );
      state.set_token( prompt.token( 0 ) );
      state.set_token_pos( 0 );
      state.set_next_layer( 0 );
      state.set_next_stage( InferenceState::Stage::PreAttention );
      state.set_prompt_length( prompt.token_count() );
      state.set_temperature( 0.0f );
      state.set_loop_start_timestamp( std::chrono::steady_clock::now().time_since_epoch().count() );
      state.set_time_in_node( -std::chrono::steady_clock::now().time_since_epoch().count() );
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
void Worker<Model>::completion_commit_thread_func()
{
  while ( running_ ) {
    completion_manager_->commit();

    // XXX(sadjad): make this configurable
    this_thread::sleep_for( chrono::seconds { 5 } );
  }
}

template<typename Model>
Worker<Model>::~Worker()
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

#if defined( TARGET_PLATFORM_AMD64 )
namespace models = glinthawk::models::llama2::amd64;
#elif defined( TARGET_PLATFORM_CUDA )
namespace models = glinthawk::models::llama2::cuda;
#endif

template class Worker<models::Llama2_7B_Chat<glinthawk::float16_t>>;
template class Worker<models::Llama2_13B_Chat<glinthawk::float16_t>>;
template class Worker<models::Llama2_70B_Chat<glinthawk::float16_t>>;
template class Worker<models::Stories_110M<glinthawk::float16_t>>;
template class Worker<models::Llama2_7B_Chat<glinthawk::float32_t>>;
template class Worker<models::Llama2_13B_Chat<glinthawk::float32_t>>;
template class Worker<models::Llama2_70B_Chat<glinthawk::float32_t>>;
template class Worker<models::Stories_110M<glinthawk::float32_t>>;

} // namespace glinthawk::core
