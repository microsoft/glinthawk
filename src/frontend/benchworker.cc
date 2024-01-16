#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <iostream>
#include <vector>

#include "arch/float.hh"
#include "message/handler.hh"
#include "message/message.hh"
#include "models/common/model.hh"
#include "models/types.hh"
#include "net/address.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "util/eventloop.hh"
#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

void usage( const char* name ) { cout << "Usage: " << name << " <id=(0|1)> <ip> <port>" << endl; }

void send_fake_message( auto& message_handler )
{
  models::InferenceState fake_state { DataType::Float32 };
  fake_state.set_activations( { 4096 * sizeof( glinthawk::float32_t ) } );
  fake_state.set_timestamp( chrono::steady_clock::now().time_since_epoch().count() );
  core::Message fake_message { core::Message::OpCode::InferenceState, fake_state.serialize() };
  message_handler.push_message( move( fake_message ) );
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 4 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  const int id = stoi( argv[1] );
  const string ip { argv[2] };
  const uint16_t port = static_cast<uint16_t>( stoi( argv[3] ) );

  if ( id != 0 and id != 1 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  using SocketType = net::TCPSocket;
  using ClockType = chrono::steady_clock;

  net::Address address { ip, port };

  SocketType socket {};
  socket.set_reuseaddr();
  socket.set_blocking( true );

  if ( id == 0 ) { // sender
    socket.bind( address );
    socket.listen();
    socket = socket.accept();
  } else { // receiver
    socket.connect( address );
  }

  socket.set_blocking( false );

  net::Session<SocketType> session { move( socket ) };
  core::MessageHandler<net::Session<SocketType>> message_handler { move( session ) };

  EventLoop loop;
  decltype( message_handler )::RuleCategories rule_categories {
    .session = loop.add_category( "Worker session" ),
    .endpoint_read = loop.add_category( "Worker endpoint read" ),
    .endpoint_write = loop.add_category( "Worker endpoint write" ),
    .response = loop.add_category( "Worker response" ),
  };

  const size_t MESSAGE_COUNT = 1000;

  std::vector<int64_t> time_deltas;
  size_t received_messages = 0;

  message_handler.install_rules(
    loop,
    rule_categories,
    [&]( core::Message&& message ) {
      auto state = models::InferenceState { message.payload() };
      const auto delta = ClockType::now().time_since_epoch().count() - state.timestamp();
      time_deltas.push_back( delta );
      received_messages++;
      LOG( INFO ) << "Received message #" << received_messages << ": " << message.info();

      send_fake_message( message_handler );
      return true;
    },
    []() { LOG( INFO ) << "Connection closed"; } );

  if ( id == 0 ) {
    send_fake_message( message_handler );
  }

  while ( loop.wait_next_event( 1000 ) != EventLoop::Result::Exit ) {
    if ( received_messages >= MESSAGE_COUNT ) {
      LOG( INFO ) << "Received all messages";
      break;
    }
  }

  int64_t sum = 0;
  int64_t sum_of_squares = 0;

  for ( const auto& delta : time_deltas ) {
    sum += delta;
    sum_of_squares += delta * delta;
  }

  const auto mean = sum / time_deltas.size();
  const auto variance = sum_of_squares / time_deltas.size() - mean * mean;
  const auto stddev = sqrt( variance );

  LOG( INFO ) << "Mean: " << mean << "ns, stddev: " << stddev << "ns";

  return EXIT_SUCCESS;
}
