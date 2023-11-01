#pragma once

#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>

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

private:
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

  core::MessageHandler<net::TCPSession>::RuleCategories rule_categories_ {
    .session = event_loop_.add_category( "Worker session" ),
    .endpoint_read = event_loop_.add_category( "Worker endpoint read" ),
    .endpoint_write = event_loop_.add_category( "Worker endpoint write" ),
    .response = event_loop_.add_category( "Worker response" ),
  };

  void setup_peer( std::map<net::Address, Peer>::iterator peer_it );
  void setup_compute_kernel( const std::filesystem::path& model_root,
                             const int start_layer,
                             const int end_layer,
                             const int concurrency_size );

public:
  /// \brief Construct a new Worker object
  ///
  /// \param worker_address The address of the worker
  /// \param coordinator_address The address of the coordinator
  /// \param model_root The root directory of the model
  Worker( const net::Address& worker_address,
          const net::Address& coordinator_address,
          const std::filesystem::path& model_root );

  void run();
};

} // namespace glinthawk::core
