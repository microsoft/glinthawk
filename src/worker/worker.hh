#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <thread>

#include "compute/kernel.hh"
#include "models/types.hh"
#include "message/handler.hh"
#include "message/message.hh"
#include "models/llama2/base.hh"
#include "monitoring/telegraf.hh"
#include "net/address.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "prompt/prompt.hh"
#include "storage/blobstore.hh"
#include "util/eventloop.hh"
#include "util/timerfd.hh"

#include "concurrentqueue/blockingconcurrentqueue.h"

namespace glinthawk::core {

template<typename Model>
class BatchedWorker
{
private:
  class Peer
  {
  public:
    net::Address address;
    std::vector<models::BatchedInferenceState<typename Model::ConfigType>> outgoing_states {};
    core::MessageHandler<net::TCPSession> message_handler;

    Peer( const net::Address& addr, net::TCPSocket&& socket )
      : address( addr )
      , message_handler( std::move( socket ) )
    {
    }
  };

private:
  using BatchedState = glinthawk::models::BatchedInferenceState<typename Model::ConfigType>;
  using RouteMap = std::map<std::pair<uint32_t, typename models::InferenceStage>, net::Address>;

  std::atomic_bool running_ { true };

  EventLoop event_loop_ {};
  net::Address listen_address_;
  net::TCPSocket listen_socket_;

  net::Address coordinator_address_;
  Peer coordinator_;

  uint64_t msg_counter_ { 0 };
  uint64_t past_msg_time_ {};

  std::map<net::Address, Peer> peers_ {};
  std::filesystem::path model_root_;
  std::unique_ptr<compute::BatchedComputeKernel<Model>> compute_kernel_ { nullptr };

  std::shared_ptr<glinthawk::storage::BlobStore> blobstore_ { nullptr };
  std::unique_ptr<glinthawk::prompt::PromptManager> prompt_manager_ { nullptr };
  std::unique_ptr<glinthawk::prompt::CompletionManager> completion_manager_ { nullptr };

  std::unordered_map<RouteID, RouteMap> route_set_ {};

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

  moodycamel::BlockingConcurrentQueue<glinthawk::PromptID> prompt_queue_ {};
  std::thread prompt_preparation_thread_ {};
  std::thread completion_commit_thread_ {};

  monitoring::TelegrafLogger::RuleCategories telegraf_rule_categories_ {
    .session = event_loop_.add_category( "Telegraf session" ),
    .endpoint_read = event_loop_.add_category( "Telegraf endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Telegraf endpoint write" ),
    .response = event_loop_.add_category( "Telegraf response" ),
  };

  Measurement& __stats__ { global_measurement() };
  std::unique_ptr<monitoring::TelegrafLogger> telegraf_logger_ { nullptr };
  TimerFD stats_timer_ { std::chrono::seconds { 5 } };
  uint64_t dummy_prompt_current_id_ { 0 };

  void setup_peer( std::map<net::Address, Peer>::iterator peer_it );
  void setup_blobstore( const std::string& blobstore_uri );
  void setup_compute_kernel( const std::filesystem::path& model_root,
                             const int start_layer,
                             const int end_layer,
                             const int concurrency_size_pre_attention,
                             const int concurrency_size_attention,
                             const int concurrency_size_post_attention,
                             const int concurrency_size_classification,
                             const int max_context_count,
                             const bool randomize );
  void setup_stats_handler();

  void listen_callback();
  void handle_compute_kernel_event();
  bool handle_coordinator_message( core::Message&& msg );
  bool handle_peer_message( core::Message&& msg );
  void handle_stats();

  void prompt_preparation_thread_func();
  void completion_commit_thread_func();

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

} // namespace glinthawk::core
