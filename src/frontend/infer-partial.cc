#include <csignal>
#include <filesystem>
#include <iostream>
#include <list>
#include <vector>

#include <glog/logging.h>

#include "net/address.hh"
#include "net/socket.hh"
#include "nn/llama2.hh"
#include "util/eventloop.hh"
#include "util/timer.hh"
#include "worker/worker.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0
       << " <tokenizer_path> <weights_path> <start_layer> <end_layer> <listen_port> <next_host> <next_port>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 8 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  FLAGS_log_year_in_prefix = false;
  FLAGS_timestamp_in_logfile_name = false;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path tokenizer_path { argv[1] };
    const filesystem::path weights_path { argv[2] };

    const int32_t start_layer { stoi( argv[3] ) };
    const int32_t end_layer { stoi( argv[4] ) };

    const uint16_t listen_port { static_cast<uint16_t>( stoi( argv[5] ) ) };
    const string next_host { argv[6] };
    [[maybe_unused]] const uint16_t next_port { static_cast<uint16_t>( stoi( argv[7] ) ) };

    auto llama = make_unique<Llama2>( tokenizer_path, weights_path, start_layer, end_layer );

    Worker::Type worker_type;
    if ( start_layer == 0 ) {
      worker_type = Worker::Type::First;
    } else if ( static_cast<size_t>( end_layer ) + 1 == llama->num_layers() ) {
      worker_type = Worker::Type::Last;
    } else {
      worker_type = Worker::Type::Mid;
    }

    Worker worker { { "0", listen_port }, { next_host, next_port }, move( llama ), worker_type };
    worker.run();

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
