#include <csignal>
#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "compute/kernel_hybrid.hh"
#include "compute/kernel_hybrid_simple.hh"
#include "models/llama2/model.hh"
#include "worker/worker.hh"

#define OOF_IMPL
#include "oof/oof.hh"

#include "arch/platform_macros.hh"

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
  cerr << "Usage: " << argv0 << " <model_dir_path> <model_name> <kernel_name>" << " <listen_ip> <listen_port>"
       << " <coordinator_ip> <coordinator_port>" << endl;
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
  google::InitGoogleLogging( argv[0] );

  const filesystem::path model_path { argv[1] };
  const string model_name { argv[2] };
  const string kernel_name { argv[3] };
  const string listen_ip { argv[4] };
  const uint16_t listen_port = static_cast<uint16_t>( stoi( argv[5] ) );
  const string coordinator_ip { argv[6] };
  const uint16_t coordinator_port = static_cast<uint16_t>( stoi( argv[7] ) );

  try {
    net::Address listen_addr { listen_ip, listen_port };
    net::Address coordinator_addr { coordinator_ip, coordinator_port };

#define CREATE_AND_RUN_WORKER( MODEL_NAME, MODEL_CLASS_NAME, KERNEL_NAME, KERNEL_CLASS_NAME )                          \
  if ( model_name == MODEL_NAME and kernel_name == KERNEL_NAME ) {                                                     \
    core::BatchedWorker<                                                                                               \
      models::llama2::configs::MODEL_CLASS_NAME,                                                                       \
      KERNEL_CLASS_NAME<cuda::MODEL_CLASS_NAME<_GLINTHAWK_DTYPE_>, amd64::MODEL_CLASS_NAME<glinthawk::float32_t>>>     \
      worker { listen_addr, coordinator_addr, model_path };                                                            \
    worker.run();                                                                                                      \
  }

    // clang-format off
    CREATE_AND_RUN_WORKER( "llama2-7b-chat", Llama2_7B_Chat, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", Llama2_13B_Chat, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", Llama2_70B_Chat, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "stories-110m", Stories_110M, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-7b-chat", Llama2_7B_Chat, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", Llama2_13B_Chat, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", Llama2_70B_Chat, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "stories-110m", Stories_110M, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else LOG( FATAL ) << "Unknown model name: " << model_name;
    // clang-format on

#undef CREATE_AND_RUN_WORKER
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }
  LOG( INFO ) << "Worker is finished, exiting...";

  return EXIT_SUCCESS;
}
