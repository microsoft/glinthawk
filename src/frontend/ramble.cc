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

  Model model_;
  llama2::Vocabulary vocabulary_;

  shared_ptr<typename Model::ContextType> context_;
  StateType state_ {};

  void create_initial_state()
  {
    state_ = { 1, DataType::_GLINTHAWK_DTYPE_NAME_, {}, {}, false, false, false };
    state_.set_next_layer( 0 );
    state_.set_next_stage( InferenceStage::PreAttention );
    state_.set_prompt( 0, {}, 1 /* BOS */, 0, 1.0, 1 );
  }

  float current_temp() const
  {
    constexpr float T0 = 0.1f;
    constexpr float T1 = 0.9f;
    constexpr size_t N = 5;
    // i.e., first N tokens will be sampled with temperature T0, then T1

    const auto n = state_.token_pos( 0 );

    return ( n <= N ) ? ( ( T1 - T0 ) / sqrt( N * 1.f ) * sqrt( n * 1.f ) + T0 ) : T1;
  }

public:
  Rambler( const filesystem::path& model_path, const filesystem::path& tokenizer_path )
    : model_( model_path, 0, std::numeric_limits<uint32_t>::max(), 1, 1 )
    , vocabulary_( tokenizer_path )
    , context_( make_shared<typename Model::ContextType>( model_.settings() ) )
  {
    create_initial_state();
  }

  void ramble()
  {
    while ( true ) {
      cout << endl << oof::fg_color( oof::color { 0, 128, 0 } ) << ">>> " << oof::reset_formatting();

      while ( true ) {
        state_.set_temperature( 0, current_temp() );

        {
          GlobalScopeTimer<Timer::Category::TokenGeneration> _;
          model_.forward( state_, vector<decltype( context_ )> { context_ } );
        }

        const auto token = state_.token( 0 );

        if ( state_.finished( 0 ) ) {
          break;
        }

        cout << vocabulary_.get_word( token ) << flush;
      }

      cout << endl;
      create_initial_state();
    }
  }
};

void usage( const char* argv0 ) { cerr << "Usage: " << argv0 << " <model_dir> <model_name> <tokenizer_path>" << endl; }

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 4 ) {
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
    const filesystem::path tokenizer_path { argv[3] };

#define CREATE_AND_RUN( MODEL_NAME, CLASS_NAME )                                                                       \
  if ( model_name == MODEL_NAME ) {                                                                                    \
    using ModelType = llama2::_GLINTHAWK_ARCH_NS_::CLASS_NAME<_GLINTHAWK_DTYPE_>;                                      \
    Rambler<ModelType> rambler( model_dir_path, tokenizer_path );                                                      \
    rambler.ramble();                                                                                                  \
  }

    // XXX(sadjad): ugly af
    // clang-format off
    CREATE_AND_RUN( "stories-110m", Stories_110M )
    else CREATE_AND_RUN( "llama2-7b-chat", Llama2_7B_Chat )
    else CREATE_AND_RUN( "llama2-13b-chat", Llama2_13B_Chat )
    else CREATE_AND_RUN( "llama2-70b-chat", Llama2_70B_Chat )
    else LOG( FATAL ) << "Unknown model name: " << model_name;
    // clang-format on

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
