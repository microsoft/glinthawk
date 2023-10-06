#pragma once

#include <functional>
#include <optional>
#include <stdexcept>

#include "util/eventloop.hh"
#include "util/ring_buffer.hh"

namespace glinthawk {

template<class SessionType, class OutgoingMessage, class IncomingMessage>
class MessageHandler
{
public:
  struct RuleCategories
  {
    size_t session;
    size_t endpoint_read;
    size_t endpoint_write;
    size_t response;
  };

protected:
  SessionType session_;
  std::vector<EventLoop::RuleHandle> installed_rules_ {};

  virtual bool outgoing_empty() const = 0;
  virtual bool incoming_empty() const = 0;
  virtual IncomingMessage& incoming_front() = 0;
  virtual void incoming_pop() = 0;

  virtual void write( RingBuffer& out ) = 0;
  virtual void read( RingBuffer& in ) = 0;

public:
  MessageHandler( SessionType&& session );
  virtual ~MessageHandler() { uninstall_rules(); }

  virtual void push_message( OutgoingMessage&& req ) = 0;

  SessionType& session() { return session_; }

  void install_rules( EventLoop& loop,
                      const RuleCategories& rule_categories,
                      const std::function<bool( IncomingMessage&& )>& incoming_callback,
                      const std::function<void( void )>& close_callback,
                      const std::optional<std::function<void()>>& exception_handler = {} );

  void uninstall_rules();
};

} // namespace glinthawk
