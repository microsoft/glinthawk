#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>

#include "models/common/state.hh"
#include "models/llama2/model.hh"
#include "prompt/prompt.hh"
#include "util/timer.hh"

#include "arch/platform_macros.hh"

#define OOF_IMPL
#include "oof/oof.hh"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::models;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( 0 );
}

template<class Model>
class Rambler
{
private:
  using StateType = BatchedInferenceState<typename Model::ConfigType>;

  const uint32_t batch_size_;

  Model model_;
  llama2::Vocabulary vocabulary_;

  vector<shared_ptr<typename Model::ContextType>> contexts_ {};
  StateType state_ {};

  bool breakdown_stages_;

  void create_initial_state()
  {
    state_ = { batch_size_, DataType::_GLINTHAWK_DTYPE_NAME_, {}, {} };
    state_.set_next_layer( 0 );
    state_.set_next_stage( InferenceStage::PreAttention );

    for ( size_t i = 0; i < batch_size_; ++i ) {
      state_.set_prompt( i, {}, i, Model::ConfigType::token_bos, 0, 1.0, 1, 0, 0, 0 );
    }
  }

  float current_temp() const
  {
    constexpr float T0 = 0.9f;
    constexpr float T1 = 0.1f;
    constexpr size_t N = 5;
    // i.e., first N tokens will be sampled with temperature T0, then T1

    const auto n = state_.token_pos( 0 );

    return ( n <= N ) ? ( ( T1 - T0 ) / sqrt( N * 1.f ) * sqrt( n * 1.f ) + T0 ) : T1;
  }

public:
  Rambler( const filesystem::path& model_path,
           const filesystem::path& tokenizer_path,
           const uint32_t batch_size,
           const bool breakdown_stages )
    : batch_size_( batch_size )
    , model_( [&]() -> Model {
      std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>,
                 Model::ConfigType::n_layers>
        hosting_table;
      for ( size_t i = 0; i < Model::ConfigType::n_layers; i++ ) {
        for ( size_t j = 0; j < util::to_underlying( models::InferenceStage::__COUNT__ ); j++ ) {
          if ( i < Model::ConfigType::n_layers - 1
               and j == util::to_underlying( models::InferenceStage::Classification ) ) {
            hosting_table[i][j] = false;
          } else {
            hosting_table[i][j] = true;
          }
        }
      }
      return { model_path, hosting_table, batch_size, batch_size };
    }() )
    , vocabulary_( tokenizer_path )
    , breakdown_stages_( breakdown_stages )
  {
    create_initial_state();

    for ( size_t i = 0; i < batch_size_; ++i ) {
      contexts_.push_back( make_shared<typename Model::ContextType>( model_.settings() ) );
    }
  }

  void ramble()
  {
    while ( true ) {
      cout << endl << oof::fg_color( oof::color { 0, 128, 0 } ) << ">>> " << oof::reset_formatting();

      while ( true ) {
        state_.set_temperature( 0, current_temp() );

        if ( not breakdown_stages_ ) {
          GlobalScopeTimer<Timer::Category::TokenGeneration> _;
          model_.forward( state_, contexts_ );
        } else {
          for ( size_t i = 0; i < Model::ConfigType::n_layers; i++ ) {
            {
              GlobalScopeTimer<Timer::Category::ForwardPreAttention> _;
              model_.forward_pre_attention( state_ );
            }

            {
              GlobalScopeTimer<Timer::Category::ForwardAttention> _;
              model_.forward_attention( state_, contexts_ );
            }

            {
              GlobalScopeTimer<Timer::Category::ForwardPostAttention> _;
              model_.forward_post_attention( state_ );
            }

            if ( i == Model::ConfigType::n_layers - 1 ) {
              GlobalScopeTimer<Timer::Category::ForwardClassification> _;
              model_.forward_classify( state_ );
            }
          }
        }

        if ( state_.finished( 0 ) ) {
          break;
        }

        // check for completed dummy prompts, and restart them
        for ( size_t i = 1; i < batch_size_; ++i ) {
          if ( state_.finished( i ) ) {
            state_.set_prompt( i, {}, i, Model::ConfigType::token_bos, 0, 1.0, 1, 0, 0, 0 );
          }
        }

        cout << vocabulary_.get_word( state_.token( 0 ) ) << flush;
      }

      cout << endl;
      create_initial_state();
    }
  }
};

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0
       << " <model_dir> <model_name> (paged|static) <tokenizer_path> [<batch_size=1>] [<breakdown_stages=0>]" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc < 5 || argc > 7 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  signal( SIGINT, signal_handler );

  try {
    const filesystem::path model_dir_path { argv[1] };
    const string model_name { argv[2] };
    const string context_name { argv[3] };
    const filesystem::path tokenizer_path { argv[4] };
    const size_t batch_size = ( argc >= 6 ) ? stoul( argv[5] ) : 1u;
    const bool breakdown_stages = ( argc >= 7 ) ? static_cast<bool>( stoul( argv[6] ) ) : false;

#define CREATE_AND_RUN( MODEL_NAME, CLASS_NAME )                                                                       \
  if ( model_name == MODEL_NAME and context_name == "paged" ) {                                                        \
    using namespace llama2;                                                                                            \
                                                                                                                       \
    using DType = _GLINTHAWK_DTYPE_;                                                                                   \
    using ConfigType = configs::CLASS_NAME;                                                                            \
    using ContextType = _GLINTHAWK_ARCH_NS_::DynamicContext<ConfigType, DType>;                                        \
    using OperationsType = _GLINTHAWK_ARCH_NS_::LlamaOperations<ConfigType, DType, ContextType>;                       \
    using ModelType = Llama2<ConfigType, DType, OperationsType, ContextType>;                                          \
                                                                                                                       \
    Rambler<ModelType> rambler( model_dir_path, tokenizer_path, batch_size, breakdown_stages );                        \
    rambler.ramble();                                                                                                  \
  } else if ( model_name == MODEL_NAME and context_name == "static" ) {                                                \
    using namespace llama2;                                                                                            \
                                                                                                                       \
    using DType = _GLINTHAWK_DTYPE_;                                                                                   \
    using ConfigType = configs::CLASS_NAME;                                                                            \
    using ContextType = _GLINTHAWK_ARCH_NS_::Context<ConfigType, DType>;                                               \
    using OperationsType = _GLINTHAWK_ARCH_NS_::LlamaOperations<ConfigType, DType, ContextType>;                       \
    using ModelType = Llama2<ConfigType, DType, OperationsType, ContextType>;                                          \
                                                                                                                       \
    Rambler<ModelType> rambler( model_dir_path, tokenizer_path, batch_size, breakdown_stages );                        \
    rambler.ramble();                                                                                                  \
  }

    // XXX(sadjad): ugly af
    // clang-format off
    CREATE_AND_RUN( "stories-110m", Stories_110M )
    else CREATE_AND_RUN( "llama2-7b-chat", Llama2_7B_Chat )
    else CREATE_AND_RUN( "llama2-13b-chat", Llama2_13B_Chat )
    else CREATE_AND_RUN( "llama2-70b-chat", Llama2_70B_Chat )
    else CREATE_AND_RUN( "llama3-8b-instruct", Llama3_8B_Instruct )
    else LOG( FATAL ) << "Unknown model name: " << model_name << ", or context name: " << context_name;
    // clang-format on

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
