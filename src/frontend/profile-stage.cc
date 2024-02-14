#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <tuple>

#include <glog/logging.h>

#include "models/llama2/model.hh"
#include "profile/profiler.hh"
#include "util/random.hh"

using namespace std;
using namespace glinthawk;

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0 << " <model_root> <batch_size> <token_pos> <duration_s> <log_root>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 6 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir { argv[1] };
    const uint64_t batch_size = atoi( argv[2] );
    const uint64_t token_pos = atoi( argv[3] );
    const uint64_t duration_s = atoi( argv[4] );
    const filesystem::path log_root { argv[5] };

    using Model_CUDA_FP16 = models::llama2::cuda::Llama2_70B_Chat<glinthawk::float16_t>;
    using Model_AMD64_FP32 = models::llama2::amd64::Llama2_70B_Chat<glinthawk::float32_t>;

    Profiler<Model_CUDA_FP16> profiler_cuda { log_root / "cuda_fp16_post.log",
                                              model_dir,
                                              models::InferenceState::Stage::PostAttention,
                                              batch_size,
                                              token_pos,
                                              duration_s,
                                              false };

    Profiler<Model_AMD64_FP32> profiler_amd64 { log_root / "amd64_fp32_att.log",
                                                model_dir,
                                                models::InferenceState::Stage::Attention,
                                                batch_size,
                                                token_pos,
                                                duration_s,
                                                false };

    profiler_cuda.run_in_thread();
    profiler_amd64.run_in_thread();

    profiler_cuda.wait();
    profiler_amd64.wait();
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
