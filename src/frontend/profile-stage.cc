#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <tuple>

#include <glog/logging.h>

#include "arch/platform_macros.hh"
#include "models/llama2/model.hh"
#include "profile/profiler.hh"
#include "util/random.hh"

using namespace std;
using namespace glinthawk;

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0
       << " <model_root> <stage=(pre|att|post|cls)> <batch_size> <token_pos> <duration_s> <output_log>" << endl;
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

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir { argv[1] };
    const std::string stage_str { argv[2] };
    const uint32_t batch_size = atoi( argv[3] );
    const uint64_t token_pos = atoi( argv[4] );
    const uint64_t duration_s = atoi( argv[5] );
    const filesystem::path log_path { argv[6] };

    using ModelType = models::llama2::_GLINTHAWK_ARCH_NS_::Llama2_70B_Chat<_GLINTHAWK_DTYPE_>;

    models::InferenceStage stage;
    if ( stage_str == "pre" ) {
      stage = models::InferenceStage::PreAttention;
    } else if ( stage_str == "att" ) {
      stage = models::InferenceStage::Attention;
    } else if ( stage_str == "post" ) {
      stage = models::InferenceStage::PostAttention;
    } else if ( stage_str == "cls" ) {
      stage = models::InferenceStage::Classification;
    } else {
      cerr << "Unknown stage: " << stage_str << endl;
      return EXIT_FAILURE;
    }

    Profiler<ModelType> profiler_cuda { log_path, model_dir, stage, batch_size, token_pos, duration_s, false };

    profiler_cuda.run_in_thread();
    profiler_cuda.wait();
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
