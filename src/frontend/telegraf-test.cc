#include <iostream>

#include "monitoring/telegraf.hh"
#include "util/eventloop.hh"

using namespace std;
using namespace glinthawk;

using namespace glinthawk::monitoring;

int main()
{
  Measurement m { "test" };
  m.tag( "tag1", "value1" ).tag( "tag2", "value2" ).field( "field1", 1.5f ).field( "field2", 5.5 );

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
