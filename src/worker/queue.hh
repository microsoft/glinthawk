#pragma once

#include <queue>
#include <string_view>

#include "net/message.hh"
#include "net/session.hh"
#include "nn/inference.hh"

namespace glinthawk {

class InferenceStateMessageHandler : public MessageHandler<TCPSession, InferenceState, InferenceState>
{
private:
  std::queue<InferenceState> outgoing_states_ {};
  std::queue<InferenceState> incoming_states_ {};

  std::string pending_outgoing_data_ {};
  std::string pending_incoming_data_ {};

  std::string_view pending_outgoing_data_view_ {};

  void load();

  bool outgoing_empty() const override { return outgoing_states_.empty() and pending_outgoing_data_view_.empty(); }
  bool incoming_empty() const override { return incoming_states_.empty(); }
  InferenceState& incoming_front() override { return incoming_states_.front(); }
  void incoming_pop() override { incoming_states_.pop(); }

  void write( RingBuffer& out ) override;
  void read( RingBuffer& in ) override;

public:
  InferenceStateMessageHandler( TCPSession&& session );
  ~InferenceStateMessageHandler() = default;

  void push_message( InferenceState&& state ) override;
};

} // namespace glinthawk
