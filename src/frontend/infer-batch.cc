#include <filesystem>
#include <iostream>
#include <vector>

#include <glog/logging.h>

#define OOF_IMPL
#include "oof/oof.hh"

#include "models/common/state.hh"
#include "models/llama2/model.hh"
#include "util/timer.hh"

#include "arch/platform_macros.hh"

using namespace std;
using namespace glinthawk;

template<class Model>
class BatchInference
{
private:
  using StateType = models::BatchedInferenceState<typename Model::ConfigType>;

  Model model_;
  models::llama2::Vocabulary vocabulary_;
  StateType state_;
  vector<typename Model::ContextPtr> contexts_ {};

  StateType ser_des( StateType&& state )
  {
    const std::string ser = state.serialize();
    state = {};

    return StateType { ser };
  }

public:
  BatchInference( const filesystem::path& model_path,
                  const filesystem::path& tokenizer_path,
                  const size_t batch_size,
                  const float temp )
    : model_( model_path, 0, std::numeric_limits<uint32_t>::max(), batch_size, batch_size )
    , vocabulary_( tokenizer_path )
    , state_( batch_size, DataType::_GLINTHAWK_DTYPE_NAME_, {}, {}, false, false, false )
  {
    state_.set_next_layer( 0 );
    state_.set_next_stage( decltype( state_ )::Stage::PreAttention );

    for ( size_t i = 0; i < batch_size; ++i ) {
      PromptID id;
      util::digest::sha256( to_string( i ), id );

      state_.set_prompt( i, id, 1 /* BOS */, 0, temp, 1 );
      contexts_.push_back( make_shared<typename Model::ContextType>( model_.settings() ) );
    }
  }

  void run()
  {
    for ( size_t i = 0; i < Model::ConfigType::seq_len; i++ ) {
      for ( size_t layer = 0; layer < Model::ConfigType::n_layers; layer++ ) {
        state_ = ser_des( move( state_ ) );
        state_ = model_.pre_attention_forward( move( state_ ), contexts_ );

        state_ = ser_des( move( state_ ) );
        state_ = model_.attention_forward( move( state_ ), contexts_ );

        state_ = ser_des( move( state_ ) );
        state_ = model_.post_attention_forward( move( state_ ) );

        if ( state_.next_stage() == decltype( state_ )::Stage::Classification ) {
          state_ = ser_des( move( state_ ) );
          state_ = model_.classify_forward( move( state_ ) );
        }
      }

      cout << vocabulary_.get_word( state_.token( 0 ) ) << flush;
    }
  }
};

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_dir> <model_name> <tokenizer_path> <batch_size> <temperature>" << endl;
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
    const filesystem::path model_dir_path { argv[1] };
    const string model_name { argv[2] };
    const filesystem::path tokenizer_path { argv[3] };
    const size_t batch_size = atoi( argv[4] );
    const float temp = atof( argv[5] );

    using ModelType = models::llama2::_GLINTHAWK_ARCH_NS_::Stories_110M<_GLINTHAWK_DTYPE_>;
    BatchInference<ModelType> inference { model_dir_path, tokenizer_path, batch_size, temp };
    inference.run();

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
