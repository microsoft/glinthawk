#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <tuple>

#include <glog/logging.h>

#include "models/llama2/model.hh"

#include "util/random.hh"
#include "util/timer.hh"

#define OOF_IMPL
#include "oof/oof.hh"

#include "platform_macros.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_root> <stage=(pre|att|post)> <batch_size> <repeats>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 5 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir { argv[1] };
    // const string model_name { argv[2] }; // XXX let's fix the model to Llama2_70B_Chat for now
    const string stage { argv[2] };
    const uint64_t batch_size = atoi( argv[3] );
    const uint64_t repeats = atoi( argv[4] );
    const int start_layer = 1;
    const int end_layer = 1;

    using ModelType = models::llama2::_GLINTHAWK_ARCH_NS_::Llama2_70B_Chat<_GLINTHAWK_DTYPE_>;
    using ContextType = ModelType::ContextType;
    using ConfigType = ModelType::ConfigType;

    ModelType model { model_dir, start_layer, end_layer, batch_size, batch_size, true /* random params */ };

#if defined( TARGET_PLATFORM_AMD64 )
    models::common::_GLINTHAWK_ARCH_NS_::Operations<_GLINTHAWK_DTYPE_> ops;
#elif defined( TARGET_PLATFORM_CUDA )
    models::common::_GLINTHAWK_ARCH_NS_::Operations<_GLINTHAWK_DTYPE_> ops { batch_size };
#endif

    for ( size_t r = 0; r < repeats; r++ ) {
      vector<shared_ptr<ContextType>> contexts( batch_size );
      vector<models::InferenceState> states( batch_size );

      LOG_EVERY_N( INFO, 100 ) << "Preparing states and contexts for repeat " << r << "...";
      for ( size_t i = 0; i < batch_size; i++ ) {
        {
          GlobalScopeTimer<Timer::Category::MemoryAllocationDevice> _ {};
          contexts[i] = make_shared<ContextType>( model.settings() );
        }
        {
          GlobalScopeTimer<Timer::Category::MemoryInitializationDevice> _ {};
          memset( reinterpret_cast<uint8_t*>( contexts[i]->layer( start_layer ).token( 0 ).key() ),
                  1,
                  contexts[i]->max_size( model.settings().n_layers_loaded() ) );
        }

        ops.randomize_device_buffer( contexts[i]->layer( start_layer ).token( 0 ).key(),
                                     contexts[i]->max_size( model.settings().n_layers_loaded() )
                                       / sizeof( _GLINTHAWK_DTYPE_ ),
                                     -10.0 / sqrtf( ConfigType::dim ),
                                     10.0 / sqrtf( ConfigType::dim ) );

        DataBuffer state_buffer;
        {
          GlobalScopeTimer<Timer::Category::MemoryAllocationHost> _ {};
          state_buffer = DataBuffer { ( 2 * ConfigType::dim + 2 * ConfigType::kv_dim ) * sizeof( _GLINTHAWK_DTYPE_ ) };
        }
        {
          GlobalScopeTimer<Timer::Category::MemoryInitializationHost> _ {};
          memset( state_buffer.data(), 1, state_buffer.len() );
        }

        util::randomize_buffer( reinterpret_cast<_GLINTHAWK_DTYPE_*>( state_buffer.data() ),
                                state_buffer.len() / sizeof( _GLINTHAWK_DTYPE_ ),
                                -10.0 / sqrtf( ConfigType::dim ),
                                10.0 / sqrtf( ConfigType::dim ) );

        states[i] = { DataType::_GLINTHAWK_DTYPE_NAME_ };
        states[i].set_token_pos( ConfigType::seq_len - 1 );
        states[i].set_activations( move( state_buffer ) );

        states[i].set_next_layer( start_layer );
        states[i].set_next_stage( stage == "pre"   ? models::InferenceState::Stage::PreAttention
                                  : stage == "att" ? models::InferenceState::Stage::Attention
                                                   : models::InferenceState::Stage::PostAttention );

        string state_serialized;
        {
          GlobalScopeTimer<Timer::Category::Serializing> _ {};
          state_serialized = states[i].serialize();
        }
        {
          GlobalScopeTimer<Timer::Category::Deserializing> _ {};
          auto state = models::InferenceState { state_serialized };
        }
      }
      LOG_EVERY_N( INFO, 100 ) << "Preparing states and contexts for repeat " << r << "... done.";

      GlobalScopeTimer<Timer::Category::PartialInference> _ {};

      if ( stage == "pre" ) {
        std::ignore = model.pre_attention_forward( std::move( states ), contexts );
      } else if ( stage == "att" ) {
        std::ignore = model.attention_forward( std::move( states ), contexts );
      } else if ( stage == "post" ) {
        std::ignore = model.post_attention_forward( std::move( states ) );
      } else {
        usage( argv[0] );
        return EXIT_FAILURE;
      }
    }

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
