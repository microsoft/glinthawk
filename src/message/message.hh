#pragma once

#include <optional>
#include <queue>
#include <string>
#include <string_view>

#include "handler.hh"
#include "net/session.hh"

namespace glinthawk::core {

class Message
{
public:
  enum class OpCode : uint8_t
  {
    Hey = 0x1,
    Ping,
    Bye,

    InitializeWorker,
    InferenceState,

    COUNT
  };

  static constexpr char const* OPCODE_NAMES[static_cast<int>( OpCode::COUNT )] = {
    "",
    "Hey",
    "Ping",
    "Bye",

    "InitializeWorker",
    "InferenceState",
  };

  constexpr static size_t HEADER_LENGTH = 5;

private:
  uint32_t payload_length_ { 0 };
  OpCode opcode_ { OpCode::Hey };
  std::string payload_ {};

public:
  Message( const std::string_view& header, std::string&& payload );
  Message( const OpCode opcode, std::string&& payload );

  uint32_t payload_length() const { return payload_length_; }
  OpCode opcode() const { return opcode_; }
  const std::string& payload() const { return payload_; }

  void serialize_header( std::string& output );

  size_t total_length() const { return HEADER_LENGTH + payload_length(); }
  static uint32_t expected_payload_length( const std::string_view header );

  std::string info() const;
};

class MessageParser
{
private:
  std::optional<size_t> expected_payload_length_ { std::nullopt };

  std::string incomplete_header_ {};
  std::string incomplete_payload_ {};

  std::queue<Message> completed_messages_ {};

  void complete_message();

public:
  size_t parse( const std::string_view buf );

  bool empty() const { return completed_messages_.empty(); }
  Message& front() { return completed_messages_.front(); }
  void pop() { completed_messages_.pop(); }

  size_t size() const { return completed_messages_.size(); }
};

template<class SessionType>
class MessageHandler : public glinthawk::MessageHandler<SessionType, Message, Message>
{
private:
  std::queue<Message> outgoing_ {};
  MessageParser incoming_ {};

  std::string current_outgoing_header_ {};
  std::string_view current_outgoing_unsent_header_ {};
  std::string_view current_outgoing_unsent_payload_ {};

  void load();

  bool outgoing_empty() const override;
  bool incoming_empty() const override { return incoming_.empty(); }
  Message& incoming_front() override { return incoming_.front(); }
  void incoming_pop() override { incoming_.pop(); }

  void write( RingBuffer& out ) override;
  void read( RingBuffer& in ) override;

public:
  using glinthawk::MessageHandler<SessionType, Message, Message>::MessageHandler;

  ~MessageHandler() {}

  void push_message( Message&& msg ) override;
};

} // namespace glinthawk::core
