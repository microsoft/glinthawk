#include <iostream>

#include "monitoring/telegraf.hh"
#include "util/eventloop.hh"

using namespace std;
using namespace glinthawk;

using namespace glinthawk::monitoring;

int main()
{
  Measurement m { "test" };
  m.tag( "tag1", "value1" );
  m.tag( "tag2", "value2" );
  m.field( "field1", 1.5f );
  m.field( "field2", 5.4 );
  m.field( "field3", -1500 );
  m.field( "field4", true );
  m.field( "field5", "string" );
  m.field( "field6", 1500u );

  EventLoop loop;

  TelegrafLogger logger { "/tmp/telegraf.sock" };

  TelegrafLogger::RuleCategories rule_categories {
    .session = loop.add_category( "Worker session" ),
    .endpoint_read = loop.add_category( "Worker endpoint read" ),
    .endpoint_write = loop.add_category( "Worker endpoint write" ),
    .response = loop.add_category( "Worker response" ),
  };

  logger.install_rules(
    loop, rule_categories, []( auto&& ) { return true; }, [] {} );

  for ( int i = 0; i < 10; i++ ) {
    logger.push_measurement( m );
  }

  while ( loop.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
  }
}
