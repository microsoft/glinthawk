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
  cerr << "Usage: " << argv0 << " <model_dir_path> <model_name> <kernel_name> (paged|static) <listen_ip> <listen_port>"
       << " <coordinator_ip> <coordinator_port>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 9 ) {
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
  const string context_name { argv[4] };
  const string listen_ip { argv[5] };
  const uint16_t listen_port = static_cast<uint16_t>( stoi( argv[6] ) );
  const string coordinator_ip { argv[7] };
  const uint16_t coordinator_port = static_cast<uint16_t>( stoi( argv[8] ) );

  try {
    net::Address listen_addr { listen_ip, listen_port };
    net::Address coordinator_addr { coordinator_ip, coordinator_port };

#define CREATE_AND_RUN_WORKER( MODEL_NAME, CONTEXT_NAME, MODEL_CLASS_NAME, MODEL_CTX_NAME, KERNEL_NAME, KERNEL_CLASS_NAME )            \
  if ( model_name == MODEL_NAME and kernel_name == KERNEL_NAME and context_name == CONTEXT_NAME ) {                    \
    core::BatchedWorker<                                                                                               \
      models::llama2::configs::MODEL_CLASS_NAME,                                                                       \
      KERNEL_CLASS_NAME<cuda::MODEL_CTX_NAME<_GLINTHAWK_DTYPE_>, amd64::MODEL_CTX_NAME<glinthawk::float32_t>>>     \
      worker { listen_addr, coordinator_addr, model_path };                                                            \
    worker.run();                                                                                                      \
  }

    // clang-format off
    CREATE_AND_RUN_WORKER( "llama2-7b-chat", "static", Llama2_7B_Chat, Llama2_7B_Chat_Static, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", "static", Llama2_13B_Chat, Llama2_13B_Chat_Static, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", "static", Llama2_70B_Chat, Llama2_70B_Chat_Static, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-8b", "static", Llama3_8B, Llama3_8B_Static, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-405b", "static", Llama3_405B, Llama3_405B_Static, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "stories-110m", "static", Stories_110M, Stories_110M_Static, "hybrid", compute::HybridComputeKernel )

    else CREATE_AND_RUN_WORKER( "llama2-7b-chat", "static", Llama2_7B_Chat, Llama2_7B_Chat_Static, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", "static", Llama2_13B_Chat, Llama2_13B_Chat_Static, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", "static", Llama2_70B_Chat, Llama2_70B_Chat_Static, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-8b", "static", Llama3_8B, Llama3_8B_Static, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-405b", "static", Llama3_405B, Llama3_405B_Static, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "stories-110m", "static", Stories_110M, Stories_110M_Static, "simple_hybrid", compute::SimpleHybridComputeKernel )

    else CREATE_AND_RUN_WORKER( "llama2-7b-chat", "paged", Llama2_7B_Chat, Llama2_7B_Chat_Paged, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", "paged", Llama2_13B_Chat, Llama2_13B_Chat_Paged, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", "paged", Llama2_70B_Chat, Llama2_70B_Chat_Paged, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-8b", "paged", Llama3_8B, Llama3_8B_Paged, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-405b", "paged", Llama3_405B, Llama3_405B_Paged, "hybrid", compute::HybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "stories-110m", "paged", Stories_110M, Stories_110M_Paged, "hybrid", compute::HybridComputeKernel )

    else CREATE_AND_RUN_WORKER( "llama2-7b-chat", "paged", Llama2_7B_Chat, Llama2_7B_Chat_Paged, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-13b-chat", "paged", Llama2_13B_Chat, Llama2_13B_Chat_Paged, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama2-70b-chat", "paged", Llama2_70B_Chat, Llama2_70B_Chat_Paged, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-8b", "paged", Llama3_8B, Llama3_8B_Paged, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "llama3-405b", "paged", Llama3_405B, Llama3_405B_Paged, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else CREATE_AND_RUN_WORKER( "stories-110m", "paged", Stories_110M, Stories_110M_Paged, "simple_hybrid", compute::SimpleHybridComputeKernel )
    else LOG( FATAL ) << "Unknown model name: " << model_name << ", kernel name: " << kernel_name << ", or context name: " << context_name;
    // clang-format on

#undef CREATE_AND_RUN_WORKER
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }
  LOG( INFO ) << "Worker is finished, exiting...";

  return EXIT_SUCCESS;
}
