#pragma once

#include <type_traits>

#include "socket.hh"
#include "util/ring_buffer.hh"
#include "util/simple_string_span.hh"

namespace glinthawk::net {

template<class T, class Enable = void>
class SessionBase;

/* base for TCPSession */
template<class T>
class SessionBase<T, std::enable_if_t<std::is_same<T, TCPSocket>::value>>
{
protected:
  TCPSocket socket_;

public:
  SessionBase( TCPSocket&& socket );
};

/// @brief A session is a connection between two peers.
/// @tparam T
template<class T>
class Session : public SessionBase<T>
{
private:
  static constexpr size_t STORAGE_SIZE = 65536;

  bool incoming_stream_terminated_ { false };

  RingBuffer outbound_plaintext_ { STORAGE_SIZE };
  RingBuffer inbound_plaintext_ { STORAGE_SIZE };

public:
  using SessionBase<T>::SessionBase;

  TCPSocket& socket() { return this->socket_; }

  void do_read();
  void do_write();

  bool want_read() const;
  bool want_write() const;

  RingBuffer& outbound_plaintext() { return outbound_plaintext_; }
  RingBuffer& inbound_plaintext() { return inbound_plaintext_; }

  bool incoming_stream_terminated() const { return incoming_stream_terminated_; }

  // disallow copying
  Session( const Session& ) = delete;
  Session& operator=( const Session& ) = delete;

  // allow moving
  Session( Session&& ) = default;
  Session& operator=( Session&& ) = default;
};

using TCPSession = Session<TCPSocket>;

}
