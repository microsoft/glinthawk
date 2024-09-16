#include <filesystem>
#include <iostream>
#include <tuple>

#include <glog/logging.h>

#include "arch/platform_macros.hh"
#include "models/llama2/model.hh"
#include "profile/profiler.hh"

using namespace std;
using namespace glinthawk;

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0
       << " <model_root> <model_name> (paged|static) <stage=(all|all_no_cls|pre|att|post|cls)> <batch_size> "
          "<token_pos> <duration_s> "
          "<output_log>"
       << endl;
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

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir { argv[1] };
    const std::string model_name { argv[2] };
    const std::string context_name { argv[3] };
    const std::string stage_str { argv[4] };
    const uint32_t batch_size = atoi( argv[5] );
    const uint64_t token_pos = atoi( argv[6] );
    const uint64_t duration_s = atoi( argv[7] );
    const filesystem::path log_path { argv[8] };

    models::InferenceStage stage;
    if ( stage_str == "all" ) {
      stage = models::InferenceStage::__ALL__;
    } else if ( stage_str == "all_no_cls" ) {
      stage = models::InferenceStage::__ALL_NO_CLS__;
    } else if ( stage_str == "pre" ) {
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

#define CREATE_PROFILER_AND_RUN( MODEL_NAME, CONTEXT_NAME, MODEL_CLASS_NAME )                                          \
  if ( model_name == MODEL_NAME and context_name == CONTEXT_NAME ) {                                                   \
    using ModelType = models::llama2::_GLINTHAWK_ARCH_NS_::MODEL_CLASS_NAME<_GLINTHAWK_DTYPE_>;                        \
    CHECK_LT( token_pos, ModelType::ConfigType::seq_len ) << "Token position out of range";                            \
    Profiler<ModelType> profiler_cuda { log_path, model_dir, stage, batch_size, token_pos, duration_s, false };        \
    profiler_cuda.run_in_thread();                                                                                     \
    profiler_cuda.wait();                                                                                              \
  }
    // clang-format off
    CREATE_PROFILER_AND_RUN( "llama2-7b-chat", "static", Llama2_7B_Chat_Static )
    else CREATE_PROFILER_AND_RUN( "llama2-13b-chat", "static", Llama2_13B_Chat_Static )
    else CREATE_PROFILER_AND_RUN( "llama2-70b-chat", "static", Llama2_70B_Chat_Static )
    else CREATE_PROFILER_AND_RUN( "llama3-8b", "static", Llama3_8B_Static )
    else CREATE_PROFILER_AND_RUN( "llama3-405b", "static", Llama3_405B_Static )
    else CREATE_PROFILER_AND_RUN( "stories-110m", "static", Stories_110M_Static )
    else CREATE_PROFILER_AND_RUN( "llama2-7b-chat", "paged", Llama2_7B_Chat_Paged )
    else CREATE_PROFILER_AND_RUN( "llama2-13b-chat", "paged", Llama2_13B_Chat_Paged )
    else CREATE_PROFILER_AND_RUN( "llama2-70b-chat", "paged", Llama2_70B_Chat_Paged )
    else CREATE_PROFILER_AND_RUN( "llama3-8b", "paged", Llama3_8B_Paged )
    else CREATE_PROFILER_AND_RUN( "llama3-405b", "paged", Llama3_405B_Paged )
    else CREATE_PROFILER_AND_RUN( "stories-110m", "paged", Stories_110M_Paged )
    else LOG( FATAL ) << "Unknown model name: " << model_name << ", or context name: " << context_name;
    // clang-format on
#undef CREATE_PROFILER_AND_RUN

  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
