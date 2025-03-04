#include "socket.hh"

#include "util/exception.hh"

#include <cstddef>
#include <stdexcept>
#include <sys/un.h>
#include <unistd.h>

using namespace std;
using namespace glinthawk::net;

// default constructor for socket of (subclassed) domain and type
//! \param[in] domain is as described in [socket(7)](\ref man7::socket),
//! probably `AF_INET` or `AF_UNIX` \param[in] type is as described in
//! [socket(7)](\ref man7::socket)
Socket::Socket( const int domain, const int type )
  : FileDescriptor( CHECK_SYSCALL( "socket", socket( domain, type, 0 ) ) )
{
}

// construct from file descriptor
//! \param[in] fd is the FileDescriptor from which to construct
//! \param[in] domain is `fd`'s domain; throws std::runtime_error if wrong value
//! is supplied \param[in] type is `fd`'s type; throws std::runtime_error if
//! wrong value is supplied
Socket::Socket( FileDescriptor&& fd, const int domain, const int type )
  : FileDescriptor( std::move( fd ) )
{
  int actual_value;
  socklen_t len;

  // verify domain
  len = getsockopt( SOL_SOCKET, SO_DOMAIN, actual_value );
  if ( ( len != sizeof( actual_value ) ) or ( actual_value != domain ) ) {
    throw runtime_error( "socket domain mismatch" );
  }

  // verify type
  len = getsockopt( SOL_SOCKET, SO_TYPE, actual_value );
  if ( ( len != sizeof( actual_value ) ) or ( actual_value != type ) ) {
    throw runtime_error( "socket type mismatch" );
  }
}

// get the local or peer address the socket is connected to
//! \param[in] name_of_function is the function to call (string passed to
//! CheckSystemCall()) \param[in] function is a pointer to the function \returns
//! the requested Address
Address Socket::get_address( const string& name_of_function,
                             const function<int( int, sockaddr*, socklen_t* )>& function ) const
{
  Address::Raw address;
  socklen_t size = sizeof( address );

  CHECK_SYSCALL( name_of_function, function( fd_num(), address, &size ) );

  return { address, size };
}

//! \returns the local Address of the socket
Address Socket::local_address() const { return get_address( "getsockname", getsockname ); }

//! \returns the socket's peer's Address
Address Socket::peer_address() const { return get_address( "getpeername", getpeername ); }

// bind socket to a specified local address (usually to listen/accept)
//! \param[in] address is a local Address to bind
void Socket::bind( const Address& address ) { CHECK_SYSCALL( "bind", ::bind( fd_num(), address, address.size() ) ); }

// connect socket to a specified peer address
//! \param[in] address is the peer's Address
void Socket::connect( const Address& address )
{
  const int ret = ::connect( fd_num(), address, address.size() );
  register_read();

  if ( ret < 0 ) {
    if ( not is_blocking() and ( errno == EAGAIN or errno == EINPROGRESS ) ) {
      return;
    } else {
      throw unix_error( "connect" );
    }
  }
}

// shut down a socket in the specified way
//! \param[in] how can be `SHUT_RD`, `SHUT_WR`, or `SHUT_RDWR`; see
//! [shutdown(2)](\ref man2::shutdown)
void Socket::shutdown( const int how )
{
  CHECK_SYSCALL( "shutdown", ::shutdown( fd_num(), how ) );
  switch ( how ) {
    case SHUT_RD:
      register_read();
      break;
    case SHUT_WR:
      register_write();
      break;
    case SHUT_RDWR:
      register_read();
      register_write();
      break;
    default:
      throw runtime_error( "Socket::shutdown() called with invalid `how`" );
  }
}

//! \note If `mtu` is too small to hold the received datagram, this method
//! throws a std::runtime_error
void UDPSocket::recv( received_datagram& datagram, const size_t mtu )
{
  // receive source address and payload
  Address::Raw datagram_source_address;
  datagram.payload.resize( mtu );

  socklen_t fromlen = sizeof( datagram_source_address );

  const ssize_t recv_len = CHECK_SYSCALL(
    "recvfrom",
    ::recvfrom(
      fd_num(), datagram.payload.data(), datagram.payload.size(), MSG_TRUNC, datagram_source_address, &fromlen ) );

  if ( recv_len > ssize_t( mtu ) ) {
    throw runtime_error( "recvfrom (oversized datagram)" );
  }

  register_read();
  datagram.source_address = { datagram_source_address, fromlen };
  datagram.payload.resize( recv_len );
}

UDPSocket::received_datagram UDPSocket::recv( const size_t mtu )
{
  received_datagram ret { { nullptr, 0 }, "" };
  recv( ret, mtu );
  return ret;
}

void UDPSocket::sendto( const Address& destination, const string_view payload )
{
  CHECK_SYSCALL( "sendto", ::sendto( fd_num(), payload.data(), payload.length(), 0, destination, destination.size() ) );
  register_write();
}

void UDPSocket::send( const string_view payload )
{
  CHECK_SYSCALL( "send", ::send( fd_num(), payload.data(), payload.length(), 0 ) );
  register_write();
}

// mark the socket as listening for incoming connections
//! \param[in] backlog is the number of waiting connections to queue (see
//! [listen(2)](\ref man2::listen))
void TCPSocket::listen( const int backlog ) { CHECK_SYSCALL( "listen", ::listen( fd_num(), backlog ) ); }

// accept a new incoming connection
//! \returns a new TCPSocket connected to the peer.
//! \note This function blocks until a new connection is available
TCPSocket TCPSocket::accept()
{
  register_read();
  return TCPSocket( FileDescriptor( CHECK_SYSCALL( "accept", ::accept( fd_num(), nullptr, nullptr ) ) ) );
}

// get socket option
template<typename option_type>
socklen_t Socket::getsockopt( const int level, const int option, option_type& option_value ) const
{
  socklen_t optlen = sizeof( option_value );
  CHECK_SYSCALL( "getsockopt", ::getsockopt( fd_num(), level, option, &option_value, &optlen ) );
  return optlen;
}

// set socket option
//! \param[in] level The protocol level at which the argument resides
//! \param[in] option A single option to set
//! \param[in] option_value The value to set
//! \details See [setsockopt(2)](\ref man2::setsockopt) for details.
template<typename option_type>
void Socket::setsockopt( const int level, const int option, const option_type& option_value )
{
  CHECK_SYSCALL( "setsockopt", ::setsockopt( fd_num(), level, option, &option_value, sizeof( option_value ) ) );
}

// allow local address to be reused sooner, at the cost of some robustness
//! \note Using `SO_REUSEADDR` may reduce the robustness of your application
void Socket::set_reuseaddr() { setsockopt( SOL_SOCKET, SO_REUSEADDR, int( true ) ); }

void Socket::throw_if_error() const
{
  int socket_error = 0;
  const socklen_t len = getsockopt( SOL_SOCKET, SO_ERROR, socket_error );
  if ( len != sizeof( socket_error ) ) {
    throw runtime_error( "unexpected length from getsockopt: " + to_string( len ) );
  }

  if ( socket_error ) {
    throw unix_error( "socket error", socket_error );
  }
}

void UnixDomainSocketStream::bind( const std::filesystem::path& path )
{
  sockaddr_un addr;
  memset( &addr, 0, sizeof( addr ) );
  addr.sun_family = AF_UNIX;
  strncpy( addr.sun_path, path.c_str(), sizeof( addr.sun_path ) - 1 );
  CHECK_SYSCALL( "bind", ::bind( fd_num(), reinterpret_cast<sockaddr*>( &addr ), sizeof( addr ) ) );
}

void UnixDomainSocketStream::connect( const std::filesystem::path& path )
{
  sockaddr_un addr;
  memset( &addr, 0, sizeof( addr ) );
  addr.sun_family = AF_UNIX;
  strncpy( addr.sun_path, path.c_str(), sizeof( addr.sun_path ) - 1 );
  CHECK_SYSCALL( "connect", ::connect( fd_num(), reinterpret_cast<sockaddr*>( &addr ), sizeof( addr ) ) );
}

template void Socket::setsockopt( const int level, const int option, const timeval& option_value );
template void Socket::setsockopt( const int level, const int option, const int& option_value );
