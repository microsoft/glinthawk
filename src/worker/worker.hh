#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <random>

#include "compute/kernel.hh"
#include "compute/kernel_hybrid.hh"
#include "message/handler.hh"
#include "message/message.hh"
#include "message/util.hh"
#include "models/llama2/base.hh"
#include "models/llama2/model.hh"
#include "models/types.hh"
#include "monitoring/telegraf.hh"
#include "net/address.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "prompt/prompt.hh"
#include "util/digest.hh"
#include "util/eventloop.hh"
#include "util/timerfd.hh"

#include "models/llama2/variants.hh"

#include "glinthawk.pb.h"

namespace glinthawk::core {

namespace {

// XXX(sadjad): this is not ideal. We should unify the way we describe the datatypes across the codebase.
template<typename DType>
constexpr DataType get_datatype()
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

} // anonymous namespace

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
class BatchedWorker
{
private:
  class Peer
  {
  public:
    net::Address address;
    std::vector<models::BatchedInferenceState<ModelConfig>> outgoing_states {};
    core::MessageHandler<net::TCPSession> message_handler;

    Peer( const net::Address& addr, net::TCPSocket&& socket )
      : address( addr )
      , message_handler( std::move( socket ) )
    {
    }
  };

private:
  using BatchedState = glinthawk::models::BatchedInferenceState<ModelConfig>;
  using RouteMap = std::map<std::pair<uint32_t, typename models::InferenceStage>, net::Address>;

  bool running_ { true };
  EventLoop event_loop_ {};

  net::Address listen_address_;
  net::Address coordinator_address_;
  net::TCPSocket listen_socket_;
  Peer coordinator_;
  std::map<net::Address, Peer> peers_ {};

  std::filesystem::path model_root_;
  std::unique_ptr<ComputeKernel> compute_kernel_ { nullptr };

  std::unordered_map<RouteID, RouteMap> route_set_ {};
  glinthawk::prompt::PromptStore prompt_store_ {};

  std::queue<PromptID> prompt_queue_ {};

  TimerFD completion_commit_timer_ { std::chrono::seconds { 5 } };

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

  monitoring::TelegrafLogger::RuleCategories telegraf_rule_categories_ {
    .session = event_loop_.add_category( "Telegraf session" ),
    .endpoint_read = event_loop_.add_category( "Telegraf endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Telegraf endpoint write" ),
    .response = event_loop_.add_category( "Telegraf response" ),
  };

  uint32_t concurrency_size_pre_attention_ { 0 };

  Measurement& __stats__ { global_measurement() };
  std::unique_ptr<monitoring::TelegrafLogger> telegraf_logger_ { nullptr };
  TimerFD stats_timer_ { std::chrono::seconds { 5 } };
  uint64_t dummy_prompt_current_id_ { 0 };

  void setup_peer( std::map<net::Address, Peer>::iterator peer_it );
  void setup_compute_kernel( const std::filesystem::path& model_root,
                             const uint32_t start_layer,
                             const uint32_t end_layer,
                             const uint32_t concurrency_size_pre_attention,
                             const uint32_t concurrency_size_attention,
                             const uint32_t concurrency_size_post_attention,
                             const uint32_t concurrency_size_classification,
                             const uint32_t max_context_count,
                             const bool randomize );
  void setup_stats_handler();

  void listen_callback();
  void handle_compute_kernel_event();
  bool handle_coordinator_message( core::Message&& msg );
  bool handle_peer_message( core::Message&& msg );
  void handle_stats();
  void handle_completions( const bool reset_timer );

  net::Address find_next_worker( const RouteMap& route, const BatchedState& state )
  {
    auto it = route.find( { state.next_layer(), state.next_stage() } );
    CHECK( it != route.end() ) << "No worker found for layer " << state.next_layer() << ", stage "
                               << state.next_stage();
    return it->second;
  }

public:
  /// \brief Construct a new Worker object
  ///
  /// \param worker_address The address of the worker
  /// \param coordinator_address The address of the coordinator
  /// \param model_root The root directory of the model
  BatchedWorker( const net::Address& worker_address,
                 const net::Address& coordinator_address,
                 const std::filesystem::path& model_root );

  ~BatchedWorker();

  void run();
};

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::setup_stats_handler()
{
  /* let's see if telegraph is listening */
  std::error_code err;
  const std::filesystem::path telegraf_socket { "/tmp/telegraf.sock" };
  if ( std::filesystem::is_socket( telegraf_socket, err ) ) {
    LOG( INFO ) << "Telegraf socket found at " << telegraf_socket.string();
    telegraf_logger_ = std::make_unique<monitoring::TelegrafLogger>( telegraf_socket );
    telegraf_logger_->install_rules( event_loop_, telegraf_rule_categories_, []( auto&& ) { return true; }, [] {} );
  } else {
    LOG( WARNING ) << "Telegraf socket not found at " << telegraf_socket.string() << "; stats are not being logged.";
  }

  event_loop_.add_rule(
    "Stats timer",
    Direction::In,
    stats_timer_,
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_stats, this ),
    [] { return true; },
    [] { LOG( ERROR ) << "Stats timer stopped."; } );
}

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::setup_peer( std::map<net::Address, Peer>::iterator peer_it )
{
  peer_it->second.message_handler.install_rules(
    this->event_loop_,
    this->rule_categories_,
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_peer_message, this, std::placeholders::_1 ),
    [] { LOG( INFO ) << "Connection to peer closed."; } );

  event_loop_.add_rule(
    "Outgoing message",
    [this, peer_it] {
      for ( auto& state : peer_it->second.outgoing_states ) {
        auto state_ser = state.serialize();
        peer_it->second.message_handler.push_message(
          core::Message( core::Message::OpCode::BatchedInferenceState, std::move( state_ser ) ) );
      }

      peer_it->second.outgoing_states.clear();
    },
    [peer_it] { return not peer_it->second.outgoing_states.empty(); } );
}

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::setup_compute_kernel( const std::filesystem::path& model_root,
                                                                      const uint32_t start_layer,
                                                                      const uint32_t end_layer,
                                                                      const uint32_t concurrency_size_pre_attention,
                                                                      const uint32_t concurrency_size_attention,
                                                                      const uint32_t concurrency_size_post_attention,
                                                                      const uint32_t concurrency_size_classification,
                                                                      const uint32_t max_context_count,
                                                                      const bool randomize )
{
  CHECK_LE( start_layer, end_layer ) << "start_layer must be less than or equal to end_layer";

  const int max_concurrency_size = std::max( { concurrency_size_pre_attention,
                                               concurrency_size_attention,
                                               concurrency_size_post_attention,
                                               concurrency_size_classification } );

  concurrency_size_pre_attention_ = concurrency_size_pre_attention;

  if constexpr ( ComputeKernel::Type == compute::KernelType::Batched ) {
    compute_kernel_ = std::make_unique<ComputeKernel>(
      max_concurrency_size, model_root, start_layer, end_layer, max_concurrency_size, max_context_count, randomize );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::SimplePiped ) {
    compute_kernel_
      = std::make_unique<ComputeKernel>( typename ComputeKernel::Concurrency { concurrency_size_pre_attention,
                                                                               concurrency_size_attention,
                                                                               concurrency_size_post_attention,
                                                                               concurrency_size_classification },
                                         model_root,
                                         start_layer,
                                         end_layer,
                                         max_concurrency_size,
                                         max_context_count,
                                         randomize );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::Hybrid ) {
    compute_kernel_ = std::make_unique<ComputeKernel>(
      typename ComputeKernel::Concurrency {
        concurrency_size_pre_attention, 0, concurrency_size_post_attention, concurrency_size_classification },
      typename ComputeKernel::Concurrency { 0, concurrency_size_attention, 0, 0 },
      model_root,
      start_layer,
      end_layer,
      max_concurrency_size,
      max_context_count,
      randomize );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::SimpleHybrid ) {
    compute_kernel_ = std::make_unique<ComputeKernel>( concurrency_size_attention,
                                                       model_root,
                                                       start_layer,
                                                       end_layer,
                                                       max_concurrency_size,
                                                       max_context_count,
                                                       randomize );

  } else {
    LOG( FATAL ) << "Invalid ComputeKernel type.";
  }

  event_loop_.add_rule( "Compute Kernel",
                        Direction::In,
                        compute_kernel_->event_fd(),
                        std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_compute_kernel_event, this ),
                        [this] { return this->compute_kernel_ != nullptr; } );

  event_loop_.add_rule(
    "Commit completions",
    Direction::In,
    completion_commit_timer_,
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_completions, this, true ),
    [this] { return prompt_store_.completed_count() > 0; },
    [] { LOG( ERROR ) << "Completion commit timer stopped."; } );
}

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::handle_completions( const bool reset_timer )
{
  // commit all completions
  if ( prompt_store_.completed_count() > 0 ) {
    if ( reset_timer ) {
      completion_commit_timer_.read_event();
    }

    const auto completed_count = prompt_store_.completed_count();
    const auto proto = prompt_store_.completed_to_protobuf();
    prompt_store_.cleanup_completed();
    coordinator_.message_handler.push_message( { Message::OpCode::PushCompletions, proto.SerializeAsString() } );
    LOG( INFO ) << "Pushed " << completed_count << " completions to coordinator.";
  }
}

template<typename ModelConfig, typename ComputeKernel>
BatchedWorker<ModelConfig, ComputeKernel>::BatchedWorker( const net::Address& worker_address,
                                                          const net::Address& coordinator_address,
                                                          const std::filesystem::path& model_root )
  : listen_address_( worker_address )
  , coordinator_address_( coordinator_address )
  , listen_socket_( [this]() -> net::TCPSocket {
    net::TCPSocket socket;
    socket.set_reuseaddr();
    socket.bind( this->listen_address_ );
    socket.set_blocking( false );
    socket.listen();
    LOG( INFO ) << "Listening on " << this->listen_address_.to_string();
    return socket;
  }() )
  , coordinator_( coordinator_address,
                  [this]() -> net::TCPSocket {
                    net::TCPSocket socket;
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
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_coordinator_message, this, std::placeholders::_1 ),
    [this] {
      running_ = false;
      LOG( WARNING ) << "The connection to coordinator closed.";
    },
    [] { LOG( FATAL ) << "Exception in coordinator message handler."; } );

  event_loop_.add_rule(
    "Worker listen",
    Direction::In,
    listen_socket_,
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::listen_callback, this ),
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

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::listen_callback()
{
  net::TCPSocket socket = listen_socket_.accept();
  auto addr = socket.peer_address();
  LOG( INFO ) << "Accepted connection from " << addr.to_string();

  auto [peer_it, peer_new] = peers_.emplace(
    std::piecewise_construct, std::forward_as_tuple( addr ), std::forward_as_tuple( addr, std::move( socket ) ) );

  CHECK( peer_new ) << "A peer with this address already exists.";
  setup_peer( peer_it );
}

template<typename ModelConfig, typename ComputeKernel>
bool BatchedWorker<ModelConfig, ComputeKernel>::handle_coordinator_message( core::Message&& msg )
{
  LOG( INFO ) << "(Coordinator) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::InitializeWorker: {
      protobuf::InitializeWorker proto;
      proto.ParseFromString( msg.payload() );
      LOG( INFO ) << "Initializing worker with params=" << proto.ShortDebugString();

      // TODO(sadjad): eventually allow for loading different models
      // const auto& model_name = proto.model_name();

      __stats__.tag( "start_layer", std::to_string( proto.start_layer() ) );
      __stats__.tag( "end_layer", std::to_string( proto.end_layer() ) );
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

      LOG( INFO ) << "Worker initialized.";
      break;
    }

    case Message::OpCode::Bye: {
      LOG( INFO ) << "Received Bye message; shutting down.";

      // things to do when shutting down:
      // (1) stop the compute kernel right away
      LOG( INFO ) << "Stopping compute kernel...";
      this->compute_kernel_ = nullptr;

      // (2) commit all finished completions
      handle_completions( false );

      // (3) send a Bye back to the coordinator
      this->coordinator_.message_handler.push_message( { Message::OpCode::Bye, "" } );

      // (4) wait for the coordinator to close the connection, otherwise exit in 10 seconds
      event_loop_.add_rule(
        "Shutdown timer",
        Direction::In,
        TimerFD( std::chrono::seconds { 10 } ),
        [this] {
          LOG( WARNING ) << "Shutdown timer expired; exiting.";
          running_ = false;
        },
        [] { return true; },
        [] { LOG( ERROR ) << "Shutdown timer stopped."; } );

      return false;
    }

    case Message::OpCode::BatchedInferenceState: {
      // got an inference state from the coordinator
      auto state = models::BatchedInferenceState<ModelConfig>( msg.payload() );
      LOG( ERROR ) << "Got inference state from coordinator; this behavior is not supported.";
      break;
    }

    case Message::OpCode::SetRoute: {
      protobuf::SetRoute proto;
      proto.ParseFromString( msg.payload() );

      std::ostringstream route_str;

      RouteMap new_route {};

      for ( int i = 0; i < proto.layer_to_address_size(); i++ ) {
        const auto& route = proto.layer_to_address( i );
        models::InferenceStage next_stage;

        switch ( route.stage() ) {
          case protobuf::SetRoute::LayerToAddress::PreAttention:
            next_stage = models::InferenceStage::PreAttention;
            break;
          case protobuf::SetRoute::LayerToAddress::Attention: next_stage = models::InferenceStage::Attention; break;
          case protobuf::SetRoute::LayerToAddress::PostAttention:
            next_stage = models::InferenceStage::PostAttention;
            break;
          case protobuf::SetRoute::LayerToAddress::Classification:
            next_stage = models::InferenceStage::Classification;
            break;
          default: throw std::runtime_error( "invalid stage" );
        }

        route_str << route.layer_num() << "[" << next_stage << "] -> " << route.ip() << ":" << route.port() << "; ";

        new_route.emplace( std::make_pair( route.layer_num(), next_stage ),
                           net::Address { route.ip(), static_cast<uint16_t>( route.port() ) } );
      }

      route_set_.emplace( proto.route_id(), new_route );

      LOG( INFO ) << "Route set: " << route_str.str();
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

      if ( route_set_.find( RouteID {} ) == route_set_.end() ) {
        LOG( FATAL ) << "No dummy route set; cannot push dummy prompts.";
        break;
      }

      std::vector<models::BatchedInferenceState<ModelConfig>> states {};

      // prompt id is sha256( current_time || dummy_prompt_current_id_ )
      auto generate_next_prompt_id = [this]() -> PromptID {
        char prompt_id_buf[2 * sizeof( uint64_t )];
        const uint64_t current_time
          = std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::system_clock::now().time_since_epoch() )
              .count();

        memcpy( prompt_id_buf, &current_time, sizeof( uint64_t ) );
        memcpy( prompt_id_buf + sizeof( uint64_t ), &( this->dummy_prompt_current_id_ ), sizeof( uint64_t ) );

        PromptID prompt_id;
        util::digest::sha256( { prompt_id_buf, sizeof( prompt_id_buf ) }, prompt_id );

        this->dummy_prompt_current_id_++;
        return prompt_id;
      };

      // generating random temperatures
      std::random_device rd {};
      std::mt19937 temp_gen { rd() };
      std::uniform_real_distribution<float> temp_dist { 0.0f, 1.0f };

      const uint32_t batch_count = ( prompt_count + ( batch_size - 1 ) ) / batch_size;

      for ( size_t i = 0; i < batch_count; i++ ) {
        // FIXME hardcoded float16
        models::BatchedInferenceState<ModelConfig> state {
          batch_size, DataType::Float16, RouteID {}, ModelID {}, false, false, false
        };

        DLOG( INFO ) << "Pushing dummy prompts: " << i * batch_size << " to " << ( ( i + 1 ) * batch_size - 1 );

        for ( size_t j = 0; j < batch_size; j++ ) {
          const auto idx = i * batch_size + j;

          if ( idx >= prompt_count ) {
            break;
          }

          state.set_prompt( j, generate_next_prompt_id(), 1 /* TOKEN_BOS */, 0, temp_dist( temp_gen ), 1, -1, -1 );
        }

        DLOG( INFO ) << "Generated state: " << state.debug_string( true );
        this->compute_kernel_->push( std::move( state ) );
      }

      break;
    }

    case Message::OpCode::PushPrompts: {
      protobuf::PushPrompts proto;
      proto.ParseFromString( msg.payload() );

      for ( auto& prompt : proto.prompts() ) {
        auto prompt_obj = prompt::Prompt::from_protobuf( prompt );
        prompt_queue_.push( prompt_obj.id() );
        prompt_store_.add( prompt_obj.id(), std::move( prompt_obj ) );
      }

      size_t added_prompt_count = 0;
      while ( prompt_queue_.size() >= concurrency_size_pre_attention_ ) {
        BatchedState state {
          concurrency_size_pre_attention_, DataType::Float16, RouteID {}, ModelID {}, false, false, false
        };

        for ( size_t i = 0; i < concurrency_size_pre_attention_; i++ ) {
          PromptID prompt_id = prompt_queue_.front();
          prompt_queue_.pop();
          auto& prompt = prompt_store_.get( prompt_id );
          state.set_prompt(
            i, prompt_id, prompt.prompt().at( 0 ), 0, prompt.temperature(), prompt.prompt().count(), -1, -1 );
        }

        this->compute_kernel_->push( std::move( state ) );
        added_prompt_count += concurrency_size_pre_attention_;
      }

      if ( added_prompt_count > 0 ) {
        LOG( INFO ) << "Added " << added_prompt_count << " prompts to the compute kernel.";
      }
    } break;

    default: {
      LOG( WARNING ) << "[Coordinator] Message not handled.";
      break;
    }
  }

  return true;
}

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::handle_compute_kernel_event()
{
  this->compute_kernel_->event_fd().read_event();

  models::BatchedInferenceState<ModelConfig> state;

  while ( this->compute_kernel_->pop( state ) ) {
    __stats__.increment<Counters::StatesProcessed>( state.batch_size() );

    const auto next_worker = find_next_worker( route_set_.at( state.route_id() ), state );
    auto peer_it = peers_.find( next_worker );

    // are we connected to this?
    if ( peer_it == peers_.end() ) {
      net::TCPSocket socket;
      socket.set_blocking( false );
      socket.connect( next_worker );

      std::tie( peer_it, std::ignore ) = peers_.emplace( std::piecewise_construct,
                                                         std::forward_as_tuple( next_worker ),
                                                         std::forward_as_tuple( next_worker, std::move( socket ) ) );

      setup_peer( peer_it );
    }

    peer_it->second.outgoing_states.push_back( std::move( state ) );
  }
}

template<typename ModelConfig, typename ComputeKernel>
bool BatchedWorker<ModelConfig, ComputeKernel>::handle_peer_message( core::Message&& msg )
{
  DLOG( INFO ) << "(Peer) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::BatchedInferenceState: {
      __stats__.increment<Counters::StatesReceived>();

      BatchedState state { msg.payload() };

      DLOG( INFO ) << state.debug_string( true );

      if ( route_set_.find( state.route_id() ) == route_set_.end() ) {
        LOG( FATAL ) << "No route with id=" << state.route_id() << " in route set.";
      }

      if ( state.next_layer() == 0 and state.next_stage() == models::InferenceStage::PreAttention ) {
        /* first worker in the chain */
        for ( size_t i = 0; i < state.batch_size(); i++ ) {
          const auto& prompt_id = state.prompt_id( i );
          auto& prompt = prompt_store_.get( prompt_id );

          // Have we finished processing the prompt?
          if ( state.token_pos( i ) >= state.prompt_length( i ) ) {
            // prompt processing has already finished, and this is a generated token
            __stats__.increment<Counters::TokensGenerated>();
            prompt.completion().append( state.token( i ) );
          } else {
            __stats__.increment<Counters::TokensProcessed>();
            // we are still processing the prompt tokens; the next token comes directly from the prompt
            const auto next_token = prompt.prompt().at( state.token_pos( i ) );
            state.set_token( i, next_token );
          }

          if ( state.finished( i ) ) {
            prompt_store_.complete( prompt_id );

            // XXX(sadjad): this is actually the length of the prompt+completion; will adjust later.
            __stats__.add_point<IntDistributions::PromptLength>( state.token_pos( i ) );
            __stats__.increment<Counters::PromptsCompleted>();

            state.discard( i );

            // let's replace this with the next prompt, if one is available
            // TODO(sadjad): remove the "incomplete state" functionality from the kernels
            if ( not prompt_queue_.empty() ) {
              auto next_prompt_id = prompt_queue_.front();
              prompt_queue_.pop();
              auto& next_prompt = prompt_store_.get( next_prompt_id );
              state.set_prompt( i,
                                next_prompt_id,
                                next_prompt.prompt().at( 0 ),
                                0,
                                next_prompt.temperature(),
                                next_prompt.prompt().count(),
                                state.get_tier_2_routing_group(),
                                state.get_tier_1_routing_group() );
            }
          }
        }
      }

      this->compute_kernel_->push( std::move( state ) );
      break;
    }

    default: {
      LOG( WARNING ) << "[Peer] Message not handled.";
      break;
    }
  }

  return true;
}

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::handle_stats()
{
  stats_timer_.read_event();
  if ( telegraf_logger_ != nullptr ) {
    telegraf_logger_->push_measurement( __stats__ );
  }

  // TODO(sadjad): allow pluggable stats handlers

  __stats__.zero_out();
}

template<typename ModelConfig, typename ComputeKernel>
void BatchedWorker<ModelConfig, ComputeKernel>::run()
{
  while ( event_loop_.wait_next_event( 1'000 ) != EventLoop::Result::Exit ) {
    if ( not running_ ) {
      return;
    }
  }
}

template<typename ModelConfig, typename ComputeKernel>
BatchedWorker<ModelConfig, ComputeKernel>::~BatchedWorker()
{
  LOG( INFO ) << "BatchedWorker shutting down...";
}

} // namespace glinthawk::core
