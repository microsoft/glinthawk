#include <iostream>
#include <map>
#include <vector>

#include "message/message.hh"
#include "models/common/model.hh"
#include "net/address.hh"
#include "net/socket.hh"
#include "util/digest.hh"

using namespace std;
using namespace glinthawk;

int main()
{
  PromptID prompt_id;
  util::digest::sha256( "0", prompt_id );

  models::InferenceState state {};
  state.set_prompt_id( prompt_id );
  state.set_model_id( 0 );
  state.set_token( 1 );
  state.set_token_pos( 0 );
  state.set_next_layer( 0 );
  state.set_temperature( 0.0f );
  state.set_layer_workers( { { 0, net::Address { "127.0.0.1", static_cast<uint16_t>( 12000 ) } },
                             { 6, net::Address { "127.0.0.1", static_cast<uint16_t>( 12006 ) } } } );

  cerr << "State created: " << state.to_string() << endl;

  auto serialized_state = state.serialize();
  core::Message message { core::Message::OpCode::InferenceState, move( serialized_state ) };

  string message_header;
  message.serialize_header( message_header );

  string_view message_header_sv { message_header };
  string_view message_payload_sv { message.payload() };

  // let's connect to the first one
  net::TCPSocket socket;
  socket.set_blocking( true );
  socket.set_reuseaddr();

  socket.connect( state.next_worker() );

  while ( not message_header_sv.empty() ) {
    message_header_sv.remove_prefix( socket.write( message_header_sv ) );
  }

  while ( not message_payload_sv.empty() ) {
    message_payload_sv.remove_prefix( socket.write( message_payload_sv ) );
  }

  return 0;
}
