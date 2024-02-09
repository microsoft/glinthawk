#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>

#include <glog/logging.h>

#include "models/llama2/model.hh"
#include "platform_macros.hh"
#include "util/random.hh"

using namespace std;
using namespace std::chrono;
using namespace glinthawk;

#define STRING( s ) #s
#define TO_LITERAL( s ) STRING( s )

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0 << " <model_root> <stage=(pre|att|post)> <batch_size> <token_pos> <duration_s> <log_file>"
       << endl;
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
    // const string model_name { argv[2] }; // XXX let's fix the model to Llama2_70B_Chat for now
    const string stage { argv[2] };
    const uint64_t batch_size = atoi( argv[3] );
    const int64_t token_pos = atoi( argv[4] );
    const uint64_t duration_s = atoi( argv[5] );
    const string log_file { argv[6] };

    ofstream fout { log_file, ios::out | ios::trunc };

    const int start_layer = 1;
    const int end_layer = 1;

    using ModelType = models::llama2::_GLINTHAWK_ARCH_NS_::Llama2_70B_Chat<_GLINTHAWK_DTYPE_>;
    using ContextType = ModelType::ContextType;
    using ConfigType = ModelType::ConfigType;

    ModelType model { model_dir, start_layer, end_layer, batch_size, batch_size, true /* random params */ };

    constexpr size_t REPEATS = 1'000;

    vector<shared_ptr<ContextType>> contexts( batch_size );
    vector<vector<models::InferenceState>> all_states( REPEATS );

    for ( size_t i = 0; i < batch_size; i++ ) {
      contexts[i] = make_shared<ContextType>( model.settings() );
      model.ops().randomize_device_buffer( contexts[i]->layer( start_layer ).token( 0 ).key(),
                                           contexts[i]->max_size( model.settings().n_layers_loaded() )
                                             / sizeof( _GLINTHAWK_DTYPE_ ),
                                           -10.0 / sqrtf( ConfigType::dim ),
                                           10.0 / sqrtf( ConfigType::dim ) );
    }

    auto prepare_states = [&]( const bool first_time ) {
      for ( size_t r = 0; r < REPEATS; r++ ) {
        auto& states = all_states[r];
        states.resize( batch_size );

        for ( size_t i = 0; i < batch_size; i++ ) {
          DataBuffer state_buffer;

          state_buffer = DataBuffer { ( 2 * ConfigType::dim + 2 * ConfigType::kv_dim ) * sizeof( _GLINTHAWK_DTYPE_ ) };

          if ( first_time ) {
            util::randomize_buffer( reinterpret_cast<_GLINTHAWK_DTYPE_*>( state_buffer.data() ),
                                    state_buffer.len() / sizeof( _GLINTHAWK_DTYPE_ ),
                                    -10.0 / sqrtf( ConfigType::dim ),
                                    10.0 / sqrtf( ConfigType::dim ) );
          }

          states[i] = { DataType::_GLINTHAWK_DTYPE_NAME_ };
          states[i].set_token_pos( token_pos < 0 ? ConfigType::seq_len - 1 : token_pos );
          states[i].set_activations( move( state_buffer ) );

          states[i].set_next_layer( start_layer );
          states[i].set_next_stage( stage == "pre"   ? models::InferenceState::Stage::PreAttention
                                    : stage == "att" ? models::InferenceState::Stage::Attention
                                                     : models::InferenceState::Stage::PostAttention );
        }
      }
    };

    auto end_time = steady_clock::now() + seconds( duration_s );

    fout << "# " << TO_LITERAL( _GLINTHAWK_PLATFORM_NAME_ ) << "-" << TO_LITERAL( _GLINTHAWK_DTYPE_NAME_ )
         << " stage=" << stage << " " << " batch_size=" << batch_size << " token_pos=" << token_pos
         << " duration_s=" << duration_s << '\n';

    fout << "repeat,timestamp_ms,duration_us\n";

    for ( size_t r = 0;; r++ ) {
      if ( r % REPEATS == 0 ) {
        prepare_states( r == 0 );
      }

      if ( r == 0 ) {
        end_time = steady_clock::now() + seconds( duration_s );
      }

      auto& states = all_states[r % REPEATS];

      const auto now = system_clock::now();
      const auto start = steady_clock::now();

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

      const auto end = steady_clock::now();
      const auto duration = duration_cast<microseconds>( end - start ).count();
      fout << r << "," << duration_cast<milliseconds>( now.time_since_epoch() ).count() << "," << duration << '\n';

      if ( end >= end_time ) {
        break;
      }
    }
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
