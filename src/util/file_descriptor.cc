#include "file_descriptor.hh"

#include <algorithm>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/uio.h>
#include <system_error>
#include <unistd.h>

using namespace std;
using namespace glinthawk;

//! \param[in] fd is the file descriptor number returned by [open(2)](\ref
//! man2::open) or similar
FileDescriptor::FDWrapper::FDWrapper( const int fd )
  : _fd( fd )
{
  if ( fd < 0 ) {
    throw runtime_error( "invalid fd number:" + to_string( fd ) );
  }

  const int flags = CheckSystemCall( "fcntl", fcntl( fd, F_GETFL ) );
  _non_blocking = flags & O_NONBLOCK;
}

void FileDescriptor::FDWrapper::close()
{
  CheckSystemCall( "close", ::close( _fd ) );
  _eof = _closed = true;
}

FileDescriptor::FDWrapper::~FDWrapper()
{
  try {
    if ( _closed ) {
      return;
    }
    close();
  } catch ( const exception& e ) {
    // don't throw an exception from the destructor
    cerr << "Exception destructing FDWrapper: " << e.what() << endl;
  }
}

//! \param[in] fd is the file descriptor number returned by [open(2)](\ref
//! man2::open) or similar
FileDescriptor::FileDescriptor( const int fd )
  : _internal_fd( make_shared<FDWrapper>( fd ) )
{
}

//! Private constructor used by duplicate()
FileDescriptor::FileDescriptor( shared_ptr<FDWrapper> other_shared_ptr )
  : _internal_fd( std::move( other_shared_ptr ) )
{
}

//! \returns a copy of this FileDescriptor
FileDescriptor FileDescriptor::duplicate() const { return FileDescriptor( _internal_fd ); }

//! \param[out] str is the string to be read
size_t FileDescriptor::read( simple_string_span buffer )
{
  if ( buffer.empty() ) {
    throw runtime_error( "FileDescriptor::read: no space to read" );
  }

  const ssize_t bytes_read = ::read( fd_num(), buffer.mutable_data(), buffer.size() );
  if ( bytes_read < 0 ) {
    if ( _internal_fd->_non_blocking and ( errno == EAGAIN or errno == EINPROGRESS ) ) {
      return 0;
    } else {
      throw std::system_error( errno, std::generic_category(), "read" );
    }
  }

  register_read();

  if ( bytes_read == 0 ) {
    _internal_fd->_eof = true;
  }

  if ( bytes_read > static_cast<ssize_t>( buffer.size() ) ) {
    throw runtime_error( "read() read more than requested" );
  }

  return bytes_read;
}

size_t FileDescriptor::write( const string_view buffer )
{
  const ssize_t bytes_written = CheckSystemCall( "write", ::write( fd_num(), buffer.data(), buffer.size() ) );
  register_write();

  if ( bytes_written == 0 and buffer.size() != 0 ) {
    throw runtime_error( "write returned 0 given non-empty input buffer" );
  }

  if ( bytes_written > ssize_t( buffer.size() ) ) {
    throw runtime_error( "write wrote more than length of input buffer" );
  }

  return bytes_written;
}

void FileDescriptor::set_blocking( const bool blocking )
{
  int flags = CheckSystemCall( "fcntl", fcntl( fd_num(), F_GETFL ) );
  if ( blocking ) {
    flags ^= ( flags & O_NONBLOCK );
  } else {
    flags |= O_NONBLOCK;
  }

  CheckSystemCall( "fcntl", fcntl( fd_num(), F_SETFL, flags ) );

  _internal_fd->_non_blocking = not blocking;
}

int FileDescriptor::FDWrapper::CheckSystemCall( const string_view s_attempt, const int return_value ) const
{
  if ( return_value >= 0 ) {
    return return_value;
  }

  if ( _non_blocking and ( errno == EAGAIN or errno == EINPROGRESS ) ) {
    return 0;
  }

  throw system_error( errno, generic_category(), s_attempt.data() );
}
