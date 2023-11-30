#include <iostream>

#include "monitoring/telegraf.hh"
#include "util/eventloop.hh"

using namespace std;
using namespace glinthawk;

using namespace glinthawk::monitoring;

int main()
{
  auto m = global_measurement();
  m.tag( "sender", "telegraf-test" );
  m.tag( "user", "sadjad" );

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
    m.increment<Counters::PromptsCompleted>();
    m.increment<Counters::PromptsStarted>();

    m.add_point<IntDistributions::KernelForwardTime>( 100 );
    m.add_point<IntDistributions::KernelForwardTime>( 200 );
  }

  while ( loop.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
  }
}
