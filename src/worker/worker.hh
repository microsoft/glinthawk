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
#include "compute/kernel_hybrid_simple.hh"
#include "compute/kernel_piped.hh"
#include "compute/routerman.hh"
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

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
class BatchedWorker
{
protected:
  class Peer
  {
  public:
    net::Address address;
    std::vector<models::BatchedInferenceState<ModelConfig>> outgoing_states {};
    std::unique_ptr<core::MessageHandler<net::TCPSession>> message_handler;

    Peer( const net::Address& addr, net::TCPSocket&& socket, const std::chrono::milliseconds& induced_delay )
      : address( addr )
      , message_handler(
          induced_delay.count()
            ? std::make_unique<core::DelayedMessageHandler<net::TCPSession>>( std::move( socket ), induced_delay )
            : std::make_unique<core::MessageHandler<net::TCPSession>>( std::move( socket ) ) )
    {
    }
  };

protected:
  using BatchedState = glinthawk::models::BatchedInferenceState<ModelConfig>;
  using RouteMap = std::map<std::tuple<uint32_t, typename models::InferenceStage, uint8_t, uint8_t>, net::Address>;
  using HostingTable = typename std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>,
                                           ModelConfig::n_layers>;

  bool running_ { true };
  EventLoop event_loop_ {};

  net::Address listen_address_;
  net::Address coordinator_address_;
  net::TCPSocket listen_socket_;
  Peer coordinator_;
  std::map<net::Address, Peer> peers_ {};

  std::filesystem::path model_root_;
  std::unique_ptr<compute::TierRouter<ComputeKernel, ModelConfig>> tier_router_ { nullptr };
  ContextID next_context_id_ { NULL_CONTEXT + 1 };
  bool first_parent_ { false };
  std::optional<int8_t> tier_ {};
  std::optional<uint8_t> rank_ {};
  BatchedWorker<ModelConfig, ComputeKernel>::HostingTable slice_hosting_table_ {};

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

  uint32_t monolith_concurrency_size_ { 0 };

  Measurement& __stats__ { global_measurement() };
  std::unique_ptr<monitoring::TelegrafLogger> telegraf_logger_ { nullptr };
  TimerFD stats_timer_ { std::chrono::seconds { 5 } };
  uint64_t dummy_hash_current_id_ { 0 };

  /* XXX(sadjad): HACKY WAY TO COLLECT THE STATS LOCALLY WITHOUT TELEGRAF */
  const bool collect_local_stats_ { getenv( "_GLINTHAWK_LOCAL_STATS_FILE_" ) != nullptr };
  std::ofstream fout_local_stats_ {};

  const bool collect_prompt_info_ { getenv( "_GLINTHAWK_PROMPT_INFO_FILE_" ) != nullptr };
  std::ofstream fout_prompt_info_ {};

  /* XXX(sadjad): temporary */
  // all the outgoing states are delayed by this amount
  std::chrono::milliseconds induced_delay_ { getenv( "_GLINTHAWK_INDUCED_DELAY_" )
                                               ? std::stoul( getenv( "_GLINTHAWK_INDUCED_DELAY_" ) )
                                               : 0ul };

  void setup_peer( std::map<net::Address, Peer>::iterator peer_it );
  void setup_tier_router_and_compute_kernel( const std::filesystem::path& model_root,
                                             const HostingTable slice_hosting_table,
                                             const HostingTable node_hosting_table,
                                             compute::SliceConcurrency concurrency_,
                                             std::vector<size_t> max_context_counts_,
                                             const int8_t tier,
                                             const uint8_t rank,
                                             const bool randomize );
  void setup_stats_handler();

  void listen_callback();
  virtual void handle_tier_router_event();
  bool handle_coordinator_message( core::Message&& msg );
  void handle_batch_inference_state( BatchedState&& state );
  bool handle_peer_message( core::Message&& msg );
  void handle_stats();
  void handle_completions( const bool reset_timer );

  net::Address find_next_worker( const RouteMap& route, const BatchedState& state )
  {
    DCHECK( state.is_sharded() ) << "Monoliths should never be sent across nodes";
    auto it = route.find( { state.next_layer(), state.next_stage(), state.next_tier(), state.next_rank() } );
    DCHECK( it != route.end() ) << "No worker found for layer " << state.next_layer() << ", stage "
                                << state.next_stage() << ", tier " << state.next_tier() << ", rank "
                                << state.next_rank();
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

  virtual ~BatchedWorker();

  void run();
};

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
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

  if ( collect_local_stats_ ) {
    fout_local_stats_.open( getenv( "_GLINTHAWK_LOCAL_STATS_FILE_" ) );
    CHECK( fout_local_stats_.is_open() ) << "Failed to open local stats file.";

    fout_local_stats_ << "# "; // XXX some information about the run
    fout_local_stats_ << __stats__.csv_header() << '\n';
    fout_local_stats_ << __stats__.to_csv() << '\n';
  }

  if ( collect_prompt_info_ ) {
    fout_prompt_info_.open( getenv( "_GLINTHAWK_PROMPT_INFO_FILE_" ) );
    CHECK( fout_prompt_info_.is_open() ) << "Failed to open prompt info file.";

    fout_prompt_info_ << "# "; // XXX some information about the run
    fout_prompt_info_ << glinthawk::prompt::Prompt::csv_header() << '\n';
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
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::setup_peer( std::map<net::Address, Peer>::iterator peer_it )
{
  const std::string addr = peer_it->first.to_string();
  peer_it->second.message_handler->install_rules(
    this->event_loop_,
    this->rule_categories_,
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_peer_message, this, std::placeholders::_1 ),
    [addr] { LOG( INFO ) << "Connection to peer " << addr << " closed."; } );

  event_loop_.add_rule(
    "Outgoing message",
    [this, peer_it] {
      for ( auto& state : peer_it->second.outgoing_states ) {
        auto state_ser = state.serialize();
        peer_it->second.message_handler->push_message(
          core::Message( core::Message::OpCode::BatchedInferenceState, std::move( state_ser ) ) );
      }

      peer_it->second.outgoing_states.clear();
    },
    [peer_it] { return not peer_it->second.outgoing_states.empty(); } );
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::setup_tier_router_and_compute_kernel(
  const std::filesystem::path& model_root,
  const HostingTable slice_hosting_table,
  const HostingTable node_hosting_table,
  compute::SliceConcurrency concurrency,
  std::vector<size_t> max_context_counts,
  const int8_t tier,
  const uint8_t rank,
  const bool randomize )
{
  tier_ = tier;
  rank_ = rank;
  slice_hosting_table_ = slice_hosting_table;
  monolith_concurrency_size_ = concurrency.full_batch();
  first_parent_
    = slice_hosting_table[0][util::to_underlying( models::InferenceStage::PreAttention )] and tier == 0 and rank == 0;
  compute::NodeConcurrency kernel_concurrency = concurrency.node_concurrency( tier );
  const size_t kernel_max_context_count = max_context_counts[tier];
  const size_t kernel_max_concurrency_size = kernel_concurrency.max();

  std::unique_ptr<ComputeKernel> kernel_;

  if constexpr ( ComputeKernel::Type == compute::KernelType::Batched ) {
    kernel_ = std::make_unique<ComputeKernel>(
      model_root, node_hosting_table, kernel_max_concurrency_size, kernel_max_context_count, randomize );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::SimplePiped ) {
    kernel_ = std::make_unique<ComputeKernel>( kernel_concurrency,
                                               model_root,
                                               node_hosting_table,
                                               kernel_max_concurrency_size,
                                               kernel_max_context_count,
                                               randomize );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::Hybrid ) {
    kernel_ = std::make_unique<ComputeKernel>(
      compute::NodeConcurrency { kernel_concurrency.get( models::InferenceStage::PreAttention ),
                                 0,
                                 kernel_concurrency.get( models::InferenceStage::PostAttention ),
                                 kernel_concurrency.get( models::InferenceStage::Classification ) },
      compute::NodeConcurrency { 0, kernel_concurrency.get( models::InferenceStage::Attention ), 0, 0 },
      model_root,
      node_hosting_table,
      kernel_max_concurrency_size,
      kernel_max_context_count,
      randomize );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::SimpleHybrid ) {
    // TODO(pouya): shouldn't the kernel_max_concurrency_size passed to the model also be halved?
    kernel_ = std::make_unique<ComputeKernel>( kernel_max_concurrency_size,
                                               model_root,
                                               node_hosting_table,
                                               kernel_max_concurrency_size,
                                               kernel_max_context_count,
                                               randomize );

  } else {
    LOG( FATAL ) << "Invalid ComputeKernel type.";
  }

  if ( tier == 0 and rank == 0 ) {
    tier_router_ = std::make_unique<compute::ParentTierRouter<ComputeKernel, ModelConfig>>(
      std::move( kernel_ ), concurrency, max_context_counts, slice_hosting_table );
  } else {
    tier_router_ = std::make_unique<compute::ChildTierRouter<ComputeKernel, ModelConfig>>( std::move( kernel_ ) );
  }

  event_loop_.add_rule( "Tier Router",
                        Direction::In,
                        tier_router_->event_fd(),
                        std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_tier_router_event, this ),
                        [this] { return this->tier_router_ != nullptr; } );

  event_loop_.add_rule(
    "Commit completions",
    Direction::In,
    completion_commit_timer_,
    std::bind( &BatchedWorker<ModelConfig, ComputeKernel>::handle_completions, this, true ),
    [this] { return prompt_store_.completed_count() > 0; },
    [] { LOG( ERROR ) << "Completion commit timer stopped."; } );
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
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
    coordinator_.message_handler->push_message( { Message::OpCode::PushCompletions, proto.SerializeAsString() } );
    LOG( INFO ) << "Pushed " << completed_count << " completions to coordinator.";
  }
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
           && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
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
  , coordinator_(
      coordinator_address,
      [this]() -> net::TCPSocket {
        net::TCPSocket socket;
        socket.set_blocking( false );
        socket.connect( this->coordinator_address_ );
        LOG( INFO ) << "Connecting to coordinator at " << this->coordinator_address_.to_string();
        return socket;
      }(),
      induced_delay_ )
  , model_root_( model_root )
{
  // handle fd failures gracefully
  event_loop_.set_fd_failure_callback( [] { LOG( ERROR ) << "FD failure callback called."; } );

  coordinator_.message_handler->install_rules(
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
  if constexpr ( ComputeKernel::Type == compute::KernelType::Batched ) {
    hey_proto.set_kernel( protobuf::Hey::Batched );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::Hybrid ) {
    hey_proto.set_kernel( protobuf::Hey::Hybrid );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::SimpleHybrid ) {
    hey_proto.set_kernel( protobuf::Hey::SimpleHybrid );
  } else if constexpr ( ComputeKernel::Type == compute::KernelType::SimplePiped ) {
    hey_proto.set_kernel( protobuf::Hey::SimplePiped );
  } else {
    throw std::logic_error( "No such kernel type" );
  }
  coordinator_.message_handler->push_message( { Message::OpCode::Hey, hey_proto.SerializeAsString() } );

  setup_stats_handler();
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::listen_callback()
{
  net::TCPSocket socket = listen_socket_.accept();
  auto addr = socket.peer_address();
  LOG( INFO ) << "Accepted connection from " << addr.to_string();

  auto [peer_it, peer_new] = peers_.emplace( std::piecewise_construct,
                                             std::forward_as_tuple( addr ),
                                             std::forward_as_tuple( addr, std::move( socket ), induced_delay_ ) );

  CHECK( peer_new ) << "A peer with this address already exists.";
  setup_peer( peer_it );
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
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

      __stats__.tag( "slice", std::to_string( proto.slice_index() ) );
      __stats__.tag( "tier", std::to_string( proto.tier() ) );
      __stats__.tag( "rank", std::to_string( proto.rank() ) );
#if defined( TARGET_PLATFORM_AMD64 )
      __stats__.tag( "platform", "amd64" );
#elif defined( TARGET_PLATFORM_CUDA )
      __stats__.tag( "platform", "cuda" );
#endif

      std::vector<uint8_t> n_tiers;
      std::vector<std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )>> stage_concurrencies;
      std::vector<size_t> max_contexts;

      for ( int i = 0; i < proto.tier_concurrency_s_size(); i++ ) {
        const auto& tier_concurrency = proto.tier_concurrency_s( i );
        n_tiers.push_back( static_cast<uint8_t>( tier_concurrency.ranks() ) );
        stage_concurrencies.push_back( { tier_concurrency.concurrency_pre_att_size(),
                                         tier_concurrency.concurrency_att_size(),
                                         tier_concurrency.concurrency_post_att_size(),
                                         tier_concurrency.concurrency_cls_size() } );
        max_contexts.push_back( tier_concurrency.max_context_count() );
      }

      CHECK_EQ( proto.slice_hosting_table().size(),
                ModelConfig::n_layers * util::to_underlying( models::InferenceStage::__COUNT__ ) )
        << "Slice hosting table is not the correct size";
      CHECK_EQ( proto.node_hosting_table().size(),
                ModelConfig::n_layers * util::to_underlying( models::InferenceStage::__COUNT__ ) )
        << "Node hosting table is not the correct size";

      HostingTable slice_hosting_table, node_hosting_table;

      for ( size_t i = 0; i < ModelConfig::n_layers; i++ ) {
        for ( int j = 0; j < util::to_underlying( models::InferenceStage::__COUNT__ ); j++ ) {
          slice_hosting_table[i][j]
            = proto.slice_hosting_table( i * util::to_underlying( models::InferenceStage::__COUNT__ ) + j );
          node_hosting_table[i][j]
            = proto.node_hosting_table( i * util::to_underlying( models::InferenceStage::__COUNT__ ) + j );
        }
      }

      setup_tier_router_and_compute_kernel( model_root_,
                                            slice_hosting_table,
                                            node_hosting_table,
                                            { n_tiers, stage_concurrencies },
                                            max_contexts,
                                            static_cast<int8_t>( proto.tier() ),
                                            static_cast<uint8_t>( proto.rank() ),
                                            proto.randomize() );

      this->coordinator_.message_handler->push_message( { Message::OpCode::AckInitialize, "" } );
      LOG( INFO ) << "Worker initialized.";
      break;
    }

    case Message::OpCode::Bye: {
      LOG( INFO ) << "Received Bye message; shutting down.";

      // things to do when shutting down:
      // (1) stop the tier router right away
      LOG( INFO ) << "Stopping tier router...";
      this->tier_router_ = nullptr;

      // (2) commit all finished completions
      handle_completions( false );

      // (3) send a Bye back to the coordinator
      this->coordinator_.message_handler->push_message( { Message::OpCode::Bye, "" } );

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
      LOG( INFO ) << "Stopping tier router...";

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

        route_str << "<L" << route.layer_num() << ", T" << static_cast<size_t>( route.tier() ) << ", R"
                  << static_cast<size_t>( route.rank() ) << ">[" << next_stage << "]"
                  << " -> " << route.ip() << ":" << route.port() << "; ";

        new_route.emplace(
          std::make_tuple(
            route.layer_num(), next_stage, static_cast<int8_t>( route.tier() ), static_cast<uint8_t>( route.rank() ) ),
          net::Address { route.ip(), static_cast<uint16_t>( route.port() ) } );
      }

      route_set_.emplace( proto.route_id(), new_route );

      protobuf::AckRoute ack_proto;
      ack_proto.set_route_id( proto.route_id() );
      coordinator_.message_handler->push_message( { Message::OpCode::AckRoute, ack_proto.SerializeAsString() } );

      LOG( INFO ) << "Route set: " << route_str.str();
      break;
    }

    case Message::OpCode::PushDummyPrompts: {
      // create some random inference states and feed them into the system
      protobuf::PushDummyPrompts proto;
      proto.ParseFromString( msg.payload() );

      const uint32_t prompt_count = proto.count();

      if ( prompt_count == 0 or prompt_count > ( 1 << 16 ) ) {
        LOG( ERROR ) << "Invalid number of dummy prompts requested: " << prompt_count;
        break;
      }

      if ( route_set_.find( RouteID {} ) == route_set_.end() ) {
        LOG( FATAL ) << "No dummy route set; cannot push dummy prompts.";
        break;
      }

      // hash id is sha256( current_time || dummy_hash_current_id_ )
      auto generate_next_hash_id = [this]() -> HashID {
        char hash_id_buf[2 * sizeof( uint64_t )];
        const uint64_t current_time
          = std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::system_clock::now().time_since_epoch() )
              .count();

        memcpy( hash_id_buf, &current_time, sizeof( uint64_t ) );
        memcpy( hash_id_buf + sizeof( uint64_t ), &( this->dummy_hash_current_id_ ), sizeof( uint64_t ) );

        HashID hash_id;
        util::digest::sha256( { hash_id_buf, sizeof( hash_id_buf ) }, hash_id );

        this->dummy_hash_current_id_++;
        return hash_id;
      };

      // generating random temperatures
      std::random_device rd {};
      std::mt19937 temp_gen { rd() };
      std::uniform_int_distribution<uint8_t> temp_dist { 0, 255 };

      for ( size_t i = 0; i < prompt_count; i++ ) {
        prompt::Prompt new_prompt { generate_next_hash_id(), temp_dist( temp_gen ), 1, { ModelConfig::token_bos } };
        new_prompt.timing_info().set_assigned();

        prompt_queue_.push( new_prompt.id() );
        prompt_store_.add( new_prompt.id(), std::move( new_prompt ) );
      }

      // TODO(pouya): fix the copy paste
      // TODO: this will break if length of contexts is not a multiple of monolith concurrency size
      size_t added_prompt_count = 0;
      while ( prompt_queue_.size() >= monolith_concurrency_size_ and tier_router_ != nullptr
              and tier_router_->is_context_available() ) {
        BatchedState state { monolith_concurrency_size_, DataType::Float16, RouteID {}, ModelID {} };

        for ( size_t i = 0; i < monolith_concurrency_size_; i++ ) {
          PromptID prompt_id = prompt_queue_.front();
          prompt_queue_.pop();
          auto& prompt = prompt_store_.get( prompt_id );
          prompt.timing_info().set_prompt_started();

          state.set_prompt( i,
                            prompt_id,
                            next_context_id_,
                            prompt.prompt().at( 0 ),
                            0,
                            prompt.temperature(),
                            prompt.prompt().count(),
                            0,
                            -1,
                            0 );

          // TODO: either use uint32_t directly instead of ContextID, or require some add-ability concept.
          next_context_id_++;
        }

        this->tier_router_->push( std::move( state ) );
        added_prompt_count += monolith_concurrency_size_;
      }

      if ( added_prompt_count > 0 ) {
        LOG( INFO ) << "Added " << added_prompt_count << " prompts to the tier router.";
      }

      break;
    }

    case Message::OpCode::PushPrompts: {
      protobuf::PushPrompts proto;
      proto.ParseFromString( msg.payload() );

      for ( auto& prompt : proto.prompts() ) {
        auto prompt_obj = prompt::Prompt::from_protobuf( prompt );
        prompt_obj.timing_info().set_assigned();

        prompt_queue_.push( prompt_obj.id() );
        prompt_store_.add( prompt_obj.id(), std::move( prompt_obj ) );
      }

      size_t added_prompt_count = 0;
      while ( prompt_queue_.size() >= monolith_concurrency_size_ and tier_router_ != nullptr
              and tier_router_->is_context_available() ) {
        BatchedState state { monolith_concurrency_size_, DataType::Float16, RouteID {}, ModelID {} };

        for ( size_t i = 0; i < monolith_concurrency_size_; i++ ) {
          PromptID prompt_id = prompt_queue_.front();
          prompt_queue_.pop();
          auto& prompt = prompt_store_.get( prompt_id );
          prompt.timing_info().set_prompt_started();

          state.set_prompt( i,
                            prompt_id,
                            next_context_id_,
                            prompt.prompt().at( 0 ),
                            0,
                            prompt.temperature(),
                            prompt.prompt().count(),
                            prompt.max_completion_length(),
                            -1,
                            0 );
          next_context_id_++;
        }

        this->tier_router_->push( std::move( state ) );
        added_prompt_count += monolith_concurrency_size_;
      }

      if ( added_prompt_count > 0 ) {
        LOG( INFO ) << "Added " << added_prompt_count << " prompts to the tier router.";
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
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::handle_tier_router_event()
{
  this->tier_router_->event_fd().read_event();

  models::BatchedInferenceState<ModelConfig> state;

  while ( this->tier_router_->pop( state ) ) {
    __stats__.increment<Counters::StatesProcessed>( state.batch_size() );

    const auto next_worker = find_next_worker( route_set_.at( state.route_id() ), state );
    auto peer_it = peers_.find( next_worker );

    // are we connected to this?
    if ( peer_it == peers_.end() ) {
      LOG( INFO ) << "Connecting to peer at " << next_worker.to_string();
      net::TCPSocket socket;
      socket.connect( next_worker );
      socket.set_blocking( false );

      std::tie( peer_it, std::ignore )
        = peers_.emplace( std::piecewise_construct,
                          std::forward_as_tuple( next_worker ),
                          std::forward_as_tuple( next_worker, std::move( socket ), induced_delay_ ) );

      setup_peer( peer_it );
    }

    peer_it->second.outgoing_states.push_back( std::move( state ) );
  }
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::handle_batch_inference_state( BatchedState&& state )
{
  DLOG( INFO ) << state.debug_string( true );

  if ( route_set_.find( state.route_id() ) == route_set_.end() ) {
    LOG( FATAL ) << "No route with id=" << state.route_id() << " in route set.";
  }

  if ( state.next_layer() == 0 and state.next_stage() == models::InferenceStage::PreAttention and first_parent_ ) {
    /* first worker in the chain */
    for ( size_t i = 0; i < state.batch_size(); i++ ) {
      if ( state.active( i ) ) {
        const auto& prompt_id = state.prompt_id( i );
        auto& prompt = prompt_store_.get( prompt_id );

        // Have we finished processing the prompt?
        if ( state.token_pos( i ) == state.prompt_length( i ) ) {
          prompt.timing_info().set_completion_started(); // TTFT
        }

        if ( state.token_pos( i ) >= state.prompt_length( i ) ) {
          // prompt processing has already finished, and this is a generated token
          prompt.timing_info().token_output_time.add_point();

          __stats__.increment<Counters::TokensGenerated>();
          prompt.completion().append( state.token( i ) );

        } else {
          __stats__.increment<Counters::TokensProcessed>();

          prompt.timing_info().token_input_time.add_point();
          // we are still processing the prompt tokens; the next token comes directly from the prompt
          const auto next_token = prompt.prompt().at( state.token_pos( i ) );
          state.set_token( i, next_token );
        }

        if ( state.finished( i ) ) {
          prompt.timing_info().set_completion_finished();

          if ( collect_prompt_info_ ) {
            fout_prompt_info_ << prompt.to_csv() << '\n';
          }

          prompt_store_.complete( prompt_id );

          // XXX(sadjad): this is actually the length of the prompt+completion; will adjust later.
          __stats__.add_point<IntDistributions::PromptLength>( state.token_pos( i ) );
          __stats__.increment<Counters::PromptsCompleted>();

          state.discard( i );
        }
      }

      // let's replace this with the next prompt, if one is available
      if ( not state.active( i ) and not prompt_queue_.empty() ) {
        auto next_prompt_id = prompt_queue_.front();
        prompt_queue_.pop();
        auto& next_prompt = prompt_store_.get( next_prompt_id );
        next_prompt.timing_info().set_prompt_started();

        state.set_prompt( i,
                          next_prompt_id,
                          state.context_id( i ),
                          next_prompt.prompt().at( 0 ),
                          0,
                          next_prompt.temperature(),
                          next_prompt.prompt().count(),
                          next_prompt.max_completion_length(),
                          state.kv_tier( i ),
                          state.kv_rank( i ) );
      }
    }
  }
  this->tier_router_->push( std::move( state ) );
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
bool BatchedWorker<ModelConfig, ComputeKernel>::handle_peer_message( core::Message&& msg )
{
  DLOG( INFO ) << "(Peer) Incoming message: " << msg.info();

  switch ( msg.opcode() ) {
    case Message::OpCode::BatchedInferenceState: {
      __stats__.increment<Counters::StatesReceived>();
      BatchedState state { msg.payload() };
      handle_batch_inference_state( std::move( state ) );
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
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::handle_stats()
{
  stats_timer_.read_event();
  if ( telegraf_logger_ != nullptr ) {
    telegraf_logger_->push_measurement( __stats__ );
  }

  if ( collect_local_stats_ ) {
    fout_local_stats_ << __stats__.to_csv() << '\n';
  }

  // TODO(sadjad): allow pluggable stats handlers

  __stats__.zero_out();
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void BatchedWorker<ModelConfig, ComputeKernel>::run()
{
  while ( event_loop_.wait_next_event( induced_delay_.count() ? std::max( 1l, induced_delay_.count() / 4 ) : 1'000 )
          != EventLoop::Result::Exit ) {
    if ( not running_ ) {
      return;
    }
  }

  LOG( INFO ) << "Worker event loop thread exiting.";
}

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
BatchedWorker<ModelConfig, ComputeKernel>::~BatchedWorker()
{
  LOG( INFO ) << "BatchedWorker shutting down...";
}

} // namespace glinthawk::core
