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
#include "message/handler.hh"
#include "message/message.hh"
#include "models/llama2/base.hh"
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
class Worker
{
private:
  class Peer
  {
  public:
    net::Address address;
    std::vector<models::InferenceState> outgoing_states {};
    core::MessageHandler<net::TCPSession> message_handler {};

    Peer( const net::Address& addr, net::TCPSocket&& socket )
      : address( addr )
      , message_handler( std::move( socket ) )
    {
    }
  };

  class Stats
  {
  public:
    uint64_t states_received { 0 };
    uint64_t states_sent { 0 };
    uint64_t states_processed { 0 };

    uint64_t tokens_processed { 0 };
    uint64_t tokens_generated { 0 };
    uint64_t prompts_completed { 0 };

    std::chrono::steady_clock::time_point last_stats_time { std::chrono::steady_clock::now() };
  };

private:
  std::atomic_bool running_ { true };

  EventLoop event_loop_ {};
  net::Address listen_address_;
  net::TCPSocket listen_socket_;

  net::Address coordinator_address_;
  Peer coordinator_;

  std::map<net::Address, Peer> peers_ {};
  std::filesystem::path model_root_;
  std::unique_ptr<compute::ComputeKernel<Model>> compute_kernel_ { nullptr };

  std::shared_ptr<glinthawk::storage::BlobStore> blobstore_ { nullptr };
  std::unique_ptr<glinthawk::prompt::PromptManager> prompt_manager_ { nullptr };
  std::unique_ptr<glinthawk::prompt::CompletionManager> completion_manager_ { nullptr };

  std::map<uint32_t, net::Address> current_route_ {};

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

  moodycamel::BlockingConcurrentQueue<glinthawk::PromptID> prompt_queue_ {};
  std::thread prompt_preparation_thread_ {};
  std::thread completion_commit_thread_ {};

  TimerFD stats_timer_ { std::chrono::seconds { 1 } };
  Stats worker_stats_ {};

  void setup_peer( std::map<net::Address, Peer>::iterator peer_it );
  void setup_blobstore( const std::string& blobstore_uri );
  void setup_compute_kernel( const std::filesystem::path& model_root,
                             const int start_layer,
                             const int end_layer,
                             const int concurrency_size );

  void listen_callback();
  void handle_compute_kernel_event();
  bool handle_coordinator_message( core::Message&& msg );
  bool handle_peer_message( core::Message&& msg );
  void handle_stats();

  void prompt_preparation_thread_func();
  void completion_commit_thread_func();

public:
  /// \brief Construct a new Worker object
  ///
  /// \param worker_address The address of the worker
  /// \param coordinator_address The address of the coordinator
  /// \param model_root The root directory of the model
  Worker( const net::Address& worker_address,
          const net::Address& coordinator_address,
          const std::filesystem::path& model_root );

  ~Worker();

  void run();
};

} // namespace glinthawk::core
