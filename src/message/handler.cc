#include "handler.hh"

#include "models/common/model.hh"
#include "net/http/http_request.hh"
#include "net/http/http_response.hh"
#include "net/session.hh"
#include "util/void.hh"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::net;

template<class SessionType, class OutgoingMessage, class IncomingMessage>
MessageHandler<SessionType, OutgoingMessage, IncomingMessage>::MessageHandler( SessionType&& session )
  : session_( move( session ) )
{
}

template<class SessionType, class OutgoingMessage, class IncomingMessage>
void MessageHandler<SessionType, OutgoingMessage, IncomingMessage>::uninstall_rules()
{
  for ( auto& rule : installed_rules_ ) {
    rule.cancel();
  }

  installed_rules_.clear();
}

template<class SessionType, class OutgoingMessage, class IncomingMessage>
void MessageHandler<SessionType, OutgoingMessage, IncomingMessage>::install_rules(
  EventLoop& loop,
  const RuleCategories& rule_categories,
  const function<bool( IncomingMessage&& )>& incoming_callback,
  const function<void( void )>& close_callback,
  const optional<function<void()>>& exception_handler )
{
  if ( not installed_rules_.empty() ) {
    throw runtime_error( "install_rules: already installed" );
  }

  using CallbackT = function<void( void )>;

  CallbackT socket_read_handler = [this] { session_.do_read(); };
  CallbackT socket_write_handler = [this] { session_.do_write(); };

  CallbackT endpoint_read_handler = [this] { read( session_.inbound_plaintext() ); };

  CallbackT endpoint_write_handler = [this] {
    do {
      write( session_.outbound_plaintext() );
    } while ( ( not session_.outbound_plaintext().writable_region().empty() ) and ( not outgoing_empty() ) );
  };

  if ( exception_handler ) {
    auto handler = *exception_handler;

    socket_read_handler = [this, h = handler] {
      try {
        session_.do_read();
      } catch ( exception& ) {
        h();
      }
    };

    socket_write_handler = [this, h = handler] {
      try {
        session_.do_write();
      } catch ( exception& ) {
        h();
      }
    };

    endpoint_read_handler = [this, h = handler] {
      try {
        read( session_.inbound_plaintext() );
      } catch ( exception& ) {
        h();
      }
    };

    endpoint_write_handler = [this, h = handler] {
      try {
        do {
          write( session_.outbound_plaintext() );
        } while ( ( not session_.outbound_plaintext().writable_region().empty() ) and ( not outgoing_empty() ) );
      } catch ( exception& ) {
        h();
      }
    };
  }

  installed_rules_.push_back( loop.add_rule(
    rule_categories.session,
    session_.socket(),
    socket_read_handler,
    [&] { return session_.want_read(); },
    socket_write_handler,
    [&] { return session_.want_write(); },
    close_callback ) );

  installed_rules_.push_back( loop.add_rule( rule_categories.endpoint_write, endpoint_write_handler, [&] {
    return ( not session_.outbound_plaintext().writable_region().empty() ) and ( not outgoing_empty() );
  } ) );

  installed_rules_.push_back( loop.add_rule( rule_categories.endpoint_read, endpoint_read_handler, [&] {
    return not session_.inbound_plaintext().readable_region().empty();
  } ) );

  installed_rules_.push_back( loop.add_rule(
    rule_categories.response,
    [this, incoming_callback] {
      while ( not incoming_empty() ) {
        auto& response = incoming_front();

        if ( incoming_callback( move( response ) ) ) {
          incoming_pop();
        } else {
          // user doesn't want to continue processing messages
          while ( not incoming_empty() ) {
            incoming_pop();
          }

          return;
        }
      }
    },
    [&] { return not incoming_empty(); } ) );
}

namespace glinthawk {

namespace core {
class Message;
}

class Measurement;

template class MessageHandler<TCPSession, models::InferenceState, models::InferenceState>;
template class MessageHandler<TCPSession, net::HTTPRequest, net::HTTPResponse>;
template class MessageHandler<TCPSession, core::Message, core::Message>;
template class MessageHandler<UDSSession, Measurement, util::Void>;

} // namespace glinthawk
