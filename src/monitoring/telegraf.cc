#include "telegraf.hh"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::monitoring;

TelegrafLogger::TelegrafLogger( const filesystem::path& socket_file )
  : MessageHandler( [&socket_file]() -> net::UDSSession {
    net::UnixDomainSocketStream socket;
    socket.connect( socket_file );
    socket.set_blocking( false );
    return net::UDSSession { std::move( socket ) };
  }() )
{
}

void TelegrafLogger::load()
{
  if ( ( not unsent_outgoing_measurement_.empty() ) or ( outgoing_.empty() ) ) {
    throw runtime_error( "TelegrafLogger cannot load new request" );
  }

  unsent_outgoing_measurement_ = outgoing_.front();
}

bool TelegrafLogger::outgoing_empty() { return unsent_outgoing_measurement_.empty() and outgoing_.empty(); }

void TelegrafLogger::write( RingBuffer& out )
{
  if ( outgoing_empty() ) {
    throw runtime_error( "TelegrafLogger::write(): Client has no more outgoing messages" );
  }

  unsent_outgoing_measurement_.remove_prefix( out.write( unsent_outgoing_measurement_ ) );

  if ( unsent_outgoing_measurement_.empty() ) {
    outgoing_.pop_front();

    if ( not outgoing_.empty() ) {
      load();
    }
  }
}
void TelegrafLogger::read( RingBuffer& in )
{
  // discard incoming data
  in.pop( in.readable_region().size() );
}

void TelegrafLogger::push_measurement( Measurement& msg )
{
  outgoing_.push_back( msg.to_string() );

  if ( unsent_outgoing_measurement_.empty() ) {
    load();
  }
}
