#pragma once

#include <chrono>
#include <list>

#include "queue.hh"

#include "net/address.hh"
#include "net/message.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "nn/inference.hh"
#include "util/eventloop.hh"
#include "util/timerfd.hh"

namespace glinthawk {

class Worker
{
public:
  enum class Type
  {
    First,
    Mid,
    Last
  };

private:
  const Address this_address_;
  const Address next_address_;
  EventLoop event_loop_ {};

  std::unique_ptr<Model> model_ {};
  const Type type_;

  InferenceStateMessageHandler::RuleCategories rule_categories_ {
    event_loop_.add_category( "TCP Session" ),
    event_loop_.add_category( "Message read" ),
    event_loop_.add_category( "Message write" ),
    event_loop_.add_category( "Message response" ),
  };

  TCPSocket listen_socket_ {};

  std::unique_ptr<InferenceStateMessageHandler> incoming_message_handler_ {};
  std::unique_ptr<InferenceStateMessageHandler> outgoing_message_handler_ {};

  TimerFD reconnect_timer_fd_ { std::chrono::seconds( 1 ) }; // retry connection to next every second

  void reconnect_to_next();

public:
  Worker( const Address& this_address, const Address& next_address, std::unique_ptr<Model>&& model, const Type type );
  ~Worker() = default;

  void run();
};

} // namespace glinthawk
