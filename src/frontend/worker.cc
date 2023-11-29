#include <csignal>
#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "compute/kernel.hh"
#include "models/llama2/cpu/model.hh"
#include "worker/worker.hh"

#define OOF_IMPL
#include "oof/oof.hh"

#include "platform_macros.hh"

#ifdef GLINTHAWK_CUDA_ENABLED
#include "models/llama2/cuda/model.cuh"
#endif

using namespace std;
using namespace glinthawk;
using namespace glinthawk::models::llama2;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0 << " <model_dir_path> <model_name>" << " <listen_ip> <listen_port>"
       << " <coordinator_ip> <coordinator_port>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 7 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  const filesystem::path model_path { argv[1] };
  const string model_name { argv[2] };
  const string listen_ip { argv[3] };
  const uint16_t listen_port = static_cast<uint16_t>( stoi( argv[4] ) );
  const string coordinator_ip { argv[5] };
  const uint16_t coordinator_port = static_cast<uint16_t>( stoi( argv[6] ) );

  try {
    net::Address listen_addr { listen_ip, listen_port };
    net::Address coordinator_addr { coordinator_ip, coordinator_port };

#define CREATE_AND_RUN_WORKER( MODEL_NAME, CLASS_NAME )                                                                \
  if ( model_name == MODEL_NAME ) {                                                                                    \
    core::Worker<_GLINTHAWK_ARCH_NS_::CLASS_NAME<_GLINTHAWK_DTYPE_>> worker { listen_addr,                             \
                                                                              coordinator_addr,                        \
                                                                              model_path };                            \
    worker.run();                                                                                                      \
  }

    // clang-format off
    CREATE_AND_RUN_WORKER( "llama2-7b-chat", Llama2_7B_Chat )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", Llama2_13B_Chat )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", Llama2_70B_Chat )
    else CREATE_AND_RUN_WORKER( "stories-100m", Stories_110M )
    else LOG( FATAL ) << "Unknown model name: " << model_name;
    // clang-format on

#undef CREATE_AND_RUN_WORKER
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
