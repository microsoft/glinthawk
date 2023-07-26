#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "llama/llama2.hh"
#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

void usage( const char* argv0 ) { cout << "Usage: " << argv0 << " <weights_path>" << endl; }

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 2 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );


  try {
    const filesystem::path weights_path { argv[1] };
    Llama2 llama { weights_path };
    cerr << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
