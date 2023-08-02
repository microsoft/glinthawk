#include "queue.hh"

using namespace std;
using namespace glinthawk;

void InferenceStateMessageHandler::load()
{
  // do we already have an outgoing message loaded?
  if ( not pending_outgoing_data_view_.empty() ) {
    return;
  }

  // no outgoing message loaded, do we have something in the queue?
  if ( outgoing_states_.empty() ) {
    return;
  }

  // we have something in the queue, load it
  auto& state = outgoing_states_.front();
  pending_outgoing_data_ = state.serialize();
  pending_outgoing_data_view_ = pending_outgoing_data_;
  outgoing_states_.pop();
}

size_t get_expected_length( const string_view data )
{
  if ( data.length() < sizeof( int32_t ) * 4 ) {
    return sizeof( int32_t ) * 4;
  } else {
    return sizeof( int32_t ) * 4 + *reinterpret_cast<const int32_t*>( data.data() + sizeof( int32_t ) * 3 );
  }
}

void InferenceStateMessageHandler::read( RingBuffer& in )
{
  pending_incoming_data_ += in.readable_region();
  in.pop( in.readable_region().length() );

  for ( auto expected_length = get_expected_length( pending_incoming_data_ );
        expected_length <= pending_outgoing_data_.length();
        expected_length = get_expected_length( pending_incoming_data_ ) ) {
    incoming_states_.emplace( pending_incoming_data_.substr( 0, expected_length ) );
    pending_incoming_data_.erase( 0, expected_length );
  }
}

void InferenceStateMessageHandler::write( RingBuffer& out )
{
  if ( outgoing_empty() ) {
    LOG( ERROR ) << "No more outgoing messages to send.";
  }

  if ( pending_outgoing_data_view_.empty() ) {
    auto& state = outgoing_states_.front();
    pending_outgoing_data_ = state.serialize();
    pending_outgoing_data_view_ = pending_outgoing_data_;
    outgoing_states_.pop();
  }

  size_t bytes_written = out.write( pending_outgoing_data_view_ );
  pending_outgoing_data_view_.remove_prefix( bytes_written );
}
