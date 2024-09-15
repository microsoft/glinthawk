#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "measurement.hh"

#include "message/handler.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"
#include "util/void.hh"

namespace glinthawk::monitoring {

class TelegrafLogger : public glinthawk::MessageHandler<glinthawk::net::UDSSession, Measurement, glinthawk::util::Void>
{
private:
  glinthawk::util::Void incoming_ {};
  std::deque<std::string> outgoing_ {};

  std::string_view unsent_outgoing_measurement_ {};

  void load();

  bool outgoing_empty() override;
  bool incoming_empty() const override { return true; }
  glinthawk::util::Void& incoming_front() override { return incoming_; }
  void incoming_pop() override { return; }

  void write( RingBuffer& out ) override;
  void read( RingBuffer& in ) override;

  void push_message( Measurement&& ) override {}

public:
  TelegrafLogger( const std::filesystem::path& socket_file );
  ~TelegrafLogger() {}

  void push_measurement( Measurement& msg );
};

} // namespace glinthawk::monitoring
