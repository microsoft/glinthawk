#include "message.hh"

#include <endian.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "util.hh"
#include "util/ring_buffer.hh"

using namespace std;
using namespace glinthawk::core;

constexpr char const* Message::OPCODE_NAMES[static_cast<int>( Message::OpCode::COUNT )];

Message::Message( const string_view& header, string&& payload )
  : payload_( move( payload ) )
{
  if ( header.length() != HEADER_LENGTH ) {
    throw out_of_range( "incomplete header" );
  }

  payload_length_ = get_field<uint32_t>( header.substr( 8 ) );
  opcode_ = static_cast<OpCode>( header[12] );
}

Message::Message( const OpCode opcode, string&& payload )
  : payload_length_( payload.length() )
  , opcode_( opcode )
  , payload_( move( payload ) )
{
}

uint32_t Message::expected_payload_length( const string_view header )
{
  return ( header.length() < HEADER_LENGTH ) ? 0 : get_field<uint32_t>( header.substr( 8, 4 ) );
}

void Message::serialize_header( std::string& output )
{
  output = put_field( payload_length_ ) + static_cast<char>( static_cast<int>( opcode_ ) );
}

string Message::info() const
{
  ostringstream oss;
  oss << "[msg:" << Message::OPCODE_NAMES[static_cast<int>( opcode() )] << ",len=" << payload_length() << "]";

  return oss.str();
}

void MessageParser::complete_message()
{
  completed_messages_.emplace( incomplete_header_, move( incomplete_payload_ ) );

  expected_payload_length_.reset();
  incomplete_header_.clear();
  incomplete_payload_.clear();
}

size_t MessageParser::parse( string_view buf )
{
  size_t consumed_bytes = buf.length();

  while ( not buf.empty() ) {
    if ( not expected_payload_length_.has_value() ) {
      const auto remaining_length = min( buf.length(), Message::HEADER_LENGTH - incomplete_header_.length() );

      incomplete_header_.append( buf.substr( 0, remaining_length ) );
      buf.remove_prefix( remaining_length );

      if ( incomplete_header_.length() == Message::HEADER_LENGTH ) {
        expected_payload_length_ = Message::expected_payload_length( incomplete_header_ );

        if ( *expected_payload_length_ == 0 ) {
          complete_message();
        }
      }
    } else {
      const auto remaining_length = min( buf.length(), *expected_payload_length_ - incomplete_payload_.length() );

      incomplete_payload_.append( buf.substr( 0, remaining_length ) );
      buf.remove_prefix( remaining_length );

      if ( incomplete_payload_.length() == *expected_payload_length_ ) {
        complete_message();
      }
    }
  }

  return consumed_bytes;
}

template<class SessionType>
void MessageHandler<SessionType>::load()
{
  if ( ( not current_outgoing_unsent_header_.empty() ) or ( not current_outgoing_unsent_payload_.empty() )
       or ( outgoing_.empty() ) ) {
    throw std::runtime_error( "MessageHandler cannot load new request" );
  }

  outgoing_.front().serialize_header( current_outgoing_header_ );
  current_outgoing_unsent_header_ = current_outgoing_header_;
  current_outgoing_unsent_payload_ = outgoing_.front().payload();
}

template<class SessionType>
void MessageHandler<SessionType>::push_message( Message&& message )
{
  outgoing_.push( move( message ) );

  if ( current_outgoing_unsent_header_.empty() and current_outgoing_unsent_payload_.empty() ) {
    load();
  }
}

template<class SessionType>
bool MessageHandler<SessionType>::outgoing_empty() const
{
  return current_outgoing_unsent_header_.empty() and current_outgoing_unsent_payload_.empty() and outgoing_.empty();
}

template<class SessionType>
void MessageHandler<SessionType>::read( RingBuffer& in )
{
  in.pop( incoming_.parse( in.readable_region() ) );
}

template<class SessionType>
void MessageHandler<SessionType>::write( RingBuffer& out )
{
  if ( outgoing_empty() ) {
    throw std::runtime_error( "MessageHandler::write(): Client has no more outgoing messages" );
  }

  if ( not current_outgoing_unsent_header_.empty() ) {
    current_outgoing_unsent_header_.remove_prefix( out.write( current_outgoing_unsent_header_ ) );
  } else if ( not current_outgoing_unsent_payload_.empty() ) {
    current_outgoing_unsent_payload_.remove_prefix( out.write( current_outgoing_unsent_payload_ ) );
  } else {
    outgoing_.pop();

    if ( not outgoing_.empty() ) {
      load();
    }
  }
}

template class glinthawk::core::MessageHandler<glinthawk::net::TCPSession>;
