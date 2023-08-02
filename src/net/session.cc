#include "session.hh"

using namespace std;
using namespace glinthawk;

template<>
SessionBase<TCPSocket>::SessionBase( TCPSocket&& socket )
  : socket_( move( socket ) )
{
}

template<>
bool Session<TCPSocket>::want_read() const
{
  return ( not inbound_plaintext_.writable_region().empty() ) and ( not incoming_stream_terminated_ );
}

template<>
bool Session<TCPSocket>::want_write() const
{
  return not outbound_plaintext_.readable_region().empty();
}

template<>
void Session<TCPSocket>::do_read()
{
  simple_string_span target = inbound_plaintext_.writable_region();
  const auto byte_count = socket_.read( target );

  if ( byte_count == 0 ) {
    incoming_stream_terminated_ = true;
    return;
  }

  if ( byte_count > 0 ) {
    inbound_plaintext_.push( byte_count );
    return;
  }
}

template<>
void Session<TCPSocket>::do_write()
{
  const string_view source = outbound_plaintext_.readable_region();
  const auto bytes_written = socket_.write( source );

  if ( bytes_written > 0 ) {
    outbound_plaintext_.pop( bytes_written );
  }
}
