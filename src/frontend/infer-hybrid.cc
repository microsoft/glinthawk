#include <filesystem>
#include <iostream>
#include <vector>

#include <glog/logging.h>

#define OOF_IMPL
#include "oof/oof.hh"

#include "compute/kernel_hybrid.hh"
#include "models/common/state.hh"
#include "models/llama2/model.hh"
#include "util/timer.hh"

#include "arch/platform_macros.hh"

using namespace std;
using namespace glinthawk;
using namespace glinthawk::models;

template<typename ModelA, typename ModelB>
class BatchInference
{
private:
  using StateType = BatchedInferenceState<typename ModelA::ConfigType>;

  const uint32_t batch_size_;
  const float temp_;

  compute::HybridComputeKernel<ModelA, ModelB> kernel_;
  llama2::Vocabulary vocabulary_;
  StateType state_;

  std::vector<uint32_t> tokens_ { 1 /* BOS */ };

  size_t auto_id_ { 0 };

  PromptID next_prompt_id()
  {
    PromptID id;
    util::digest::sha256( to_string( auto_id_++ ), id );
    return id;
  }

  StateType ser_des( StateType&& state )
  {
    const std::string ser = state.serialize();
    state = {};

    return StateType { ser };
  }

  StateType make_state()
  {
    StateType st { batch_size_, DataType::_GLINTHAWK_DTYPE_NAME_, {}, {}, false, false, false };
    for ( size_t i = 0; i < batch_size_; ++i ) {
      st.set_prompt( i, next_prompt_id(), 1 /* BOS */, 0, temp_, 1 );
    }

    return st;
  }

public:
  BatchInference( const filesystem::path& model_path,
                  const filesystem::path& tokenizer_path,
                  const size_t batch_size,
                  const float temp )
    : batch_size_( batch_size )
    , temp_( temp )
    , kernel_( make_unique<ModelA>( model_path, 0, std::numeric_limits<uint32_t>::max(), batch_size, batch_size ),
               make_unique<ModelB>( model_path, 0, std::numeric_limits<uint32_t>::max(), batch_size, batch_size ),
               { batch_size, 0, batch_size, batch_size },
               { 0, batch_size, 0, 0 } )
    , vocabulary_( tokenizer_path )
    , state_( batch_size, DataType::_GLINTHAWK_DTYPE_NAME_, {}, {}, false, false, false )
  {
    state_.set_next_layer( 0 );
    state_.set_next_stage( InferenceStage::PreAttention );

    for ( size_t i = 0; i < batch_size_; ++i ) {
      state_.set_prompt( i, next_prompt_id(), 1 /* BOS */, 0, temp_, 1 );
    }
  }

  void run()
  {
    std::random_device rd;
    std::mt19937 gen { rd() };
    std::uniform_real_distribution<float> dis { 0.0, 1.0 };

    kernel_.event_fd().set_blocking( true );

    for ( size_t pos = 0; pos < ModelA::ConfigType::seq_len; pos++ ) {
      DLOG( INFO ) << "Processing state: " << state_.debug_string( false );

      kernel_.push( move( state_ ) );
      kernel_.event_fd().read_event();
      kernel_.pop( state_ );

      if ( not temp_ ) {
        // if temperature is 0, we expect all prompts in the batch to have the same output; the following checks this.
        for ( size_t i = 0; i < state_.batch_size(); ++i ) {
          if ( state_.token_pos( i ) < tokens_.size() ) {
            CHECK_EQ( state_.token( i ), tokens_[state_.token_pos( i )] );
          } else if ( state_.token_pos( i ) == tokens_.size() ) {
            tokens_.push_back( state_.token( i ) );
          } else {
            LOG( FATAL ) << "Unexpected token pos: " << state_.token_pos( i ) << " vs " << tokens_.size();
          }
        }

        // random chance to terminate a prompt early (otherwise they will all have the same length)
        for ( size_t i = 0; i < state_.batch_size(); ++i ) {
          if ( state_.token_pos( i ) >= 128 and dis( gen ) < 0.05 ) {
            state_.set_finished( i );
          }
        }

        bool any_finished = false;
        for ( size_t i = 0; i < state_.batch_size(); ++i ) {
          if ( state_.finished( i ) ) {
            state_.discard( i );
            any_finished = true;
          }
        }

        if ( any_finished ) {
          auto new_state = make_state();
          state_.replenish_from( new_state );
          CHECK_EQ( state_.free_slots(), 0 );
        }
      }

      cerr << vocabulary_.get_word( state_.token( 0 ) );
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

    using ModelTypeA = llama2::cuda::Stories_110M<glinthawk::float16_t>;
    using ModelTypeB = llama2::amd64::Stories_110M<glinthawk::float32_t>;

    BatchInference<ModelTypeA, ModelTypeB> inference { model_dir_path, tokenizer_path, batch_size, temp };
    inference.run();

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
