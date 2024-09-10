#pragma once

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <typeinfo>
#include <vector>

#include "models/common/state.hh"
#include "util/util.hh"

namespace glinthawk {

template<typename Model>
class Profiler
{
public:
  using Stage = models::InferenceStage;
  using ContextType = typename Model::ContextType;
  using ConfigType = typename Model::ConfigType;
  using ModelDataType = typename Model::ModelDataType;
  using StateType = models::BatchedInferenceState<ConfigType>;

private:
  static constexpr size_t REPEATS = 1'000;

  Model model_; /* we just load one layer of the model */

  const Stage stage_;
  const Stage starting_stage_;
  const uint32_t batch_size_;
  const uint64_t token_pos_;
  const std::chrono::seconds duration_;
  std::vector<std::shared_ptr<ContextType>> contexts_ { batch_size_ };
  std::vector<StateType> all_states_ { REPEATS };

  std::thread thread_ {};

  const bool run_serialization_ { false };
  std::thread serialization_thread_ {};

  const std::filesystem::path log_path_;
  std::ofstream lout_ { log_path_, std::ios::out | std::ios::trunc };

public:
  Profiler( const std::filesystem::path& log_path,
            const std::filesystem::path& model_root,
            const Stage stage,
            const uint32_t batch_size,
            const uint64_t token_pos,
            const size_t duration_s,
            const bool run_serialization )
    : model_( [&]() -> Model {
      std::array<std::array<bool, util::to_underlying( Stage::__COUNT__ )>, ConfigType::n_layers> hosting_table;
      if ( stage == Stage::__ALL_NO_CLS__ ) {
        for ( size_t i = 0; i < util::to_underlying( Stage::Classification ); i++ ) {
          hosting_table[ConfigType::n_layers - 1][i] = true;
        }
      } else if ( stage == Stage::__ALL__ ) {
        for ( size_t i = 0; i < util::to_underlying( Stage::__COUNT__ ); i++ ) {
          hosting_table[ConfigType::n_layers - 1][i] = true;
        }
      } else {
        hosting_table[ConfigType::n_layers - 1][util::to_underlying( stage )] = true;
      }
      return { model_root, hosting_table, batch_size, batch_size, true };
    }() )
    , stage_( stage )
    , starting_stage_( stage_ == Stage::__ALL__ or stage_ == Stage::__ALL_NO_CLS__ ? Stage::PreAttention : stage_ )
    , batch_size_( batch_size )
    , token_pos_( token_pos )
    , duration_( duration_s )
    , run_serialization_( run_serialization )
    , log_path_( log_path )
  {
    for ( size_t i = 0; i < batch_size; i++ ) {
      contexts_[i] = std::make_shared<ContextType>( model_.settings() );
      if ( i == 0 ) {
        LOG( WARNING ) << "Randomizing context ("
                       << ( batch_size * contexts_[i]->max_size( model_.settings().num_attention_layers_hosted() )
                            >> 20 )
                       << " MiB)...";
      }

      model_.ops().randomize_device_buffer( contexts_[i]->layer( ConfigType::n_layers - 1 ).token( 0 ).key(),
                                            contexts_[i]->max_size( model_.settings().num_attention_layers_hosted() )
                                              / sizeof( ModelDataType ),
                                            -10.0 / sqrtf( ConfigType::dim ),
                                            10.0 / sqrtf( ConfigType::dim ) );
    }

    prepare_states( true );

    lout_ << "# " << "model='" << typeid( typename Model::ConfigType ).name() << "', stage=" << stage
          << ", batch_size=" << batch_size << ", token_pos=" << token_pos << ", duration_s=" << duration_s << '\n';

    lout_ << "repeat,timestamp_ms,duration_us\n";
  }

  void prepare_states( const bool first_time = false )
  {
    for ( size_t r = 0; r < REPEATS; r++ ) {
      auto& state = all_states_[r];
      state = StateType {
        batch_size_, ( sizeof( ModelDataType ) == 2 ? DataType::Float16 : DataType::Float32 ), RouteID {}, ModelID {}
      };

      state.set_next_layer( ConfigType::n_layers - 1 );
      state.set_next_stage( starting_stage_ );

      if ( first_time ) {
        util::randomize_buffer( reinterpret_cast<ModelDataType*>( state.activations().data() ),
                                state.activations().len() / sizeof( ModelDataType ),
                                -10.0 / sqrtf( ConfigType::dim ),
                                10.0 / sqrtf( ConfigType::dim ) );

        util::randomize_buffer( reinterpret_cast<ModelDataType*>( state.queries().data() ),
                                state.queries().len() / sizeof( ModelDataType ),
                                -10.0 / sqrtf( ConfigType::kv_dim ),
                                10.0 / sqrtf( ConfigType::kv_dim ) );

        util::randomize_buffer( reinterpret_cast<ModelDataType*>( state.kvs().data() ),
                                state.kvs().len() / sizeof( ModelDataType ),
                                -10.0 / sqrtf( ConfigType::kv_dim ),
                                10.0 / sqrtf( ConfigType::kv_dim ) );
      }

      for ( size_t i = 0; i < batch_size_; i++ ) {
        state.set_prompt( i, PromptID {}, ContextID {}, 1, token_pos_, 0.5, 1, 0, 0, 0 );
      }
    }
  }

  void run_serialization()
  {
    const auto end_time = std::chrono::steady_clock::now() + duration_;

    StateType state {
      batch_size_, ( sizeof( ModelDataType ) == 2 ? DataType::Float16 : DataType::Float32 ), RouteID {}, ModelID {}
    };

    for ( auto now = std::chrono::steady_clock::now(); now < end_time; now = std::chrono::steady_clock::now() ) {
      const auto sleep_util = now + std::chrono::microseconds( 1000 );
      const auto state_str = state.serialize();
      state = { state_str };
      std::this_thread::sleep_until( sleep_util );
    }
  }

  void run()
  {
    const auto end_time = std::chrono::steady_clock::now() + duration_;

    for ( size_t r = 0;; r++ ) {
      if ( r and r % REPEATS == 0 ) {
        prepare_states( false );
      }

      auto& states = all_states_[r % REPEATS];

      const auto now = std::chrono::system_clock::now();
      const auto start = std::chrono::steady_clock::now();

      if ( stage_ == Stage::PreAttention ) {
        model_.forward_pre_attention( states );
      } else if ( stage_ == Stage::Attention ) {
        model_.forward_attention( states, contexts_ );
      } else if ( stage_ == Stage::PostAttention ) {
        model_.forward_post_attention( states );
      } else if ( stage_ == Stage::Classification ) {
        model_.forward_classify( states );
      } else if ( stage_ == Stage::__ALL__ ) {
        model_.forward( states, contexts_ );
      } else if ( stage_ == Stage::__ALL_NO_CLS__ ) {
        model_.forward( states, contexts_ );
      } else {
        LOG( FATAL ) << "Unknown stage";
      }

      const auto end = std::chrono::steady_clock::now();
      const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
      lout_ << r << "," << std::chrono::duration_cast<std::chrono::milliseconds>( now.time_since_epoch() ).count()
            << "," << duration << '\n';

      if ( end >= end_time ) {
        break;
      }
    }
  }

  void run_in_thread()
  {
    if ( thread_.joinable() ) {
      LOG( FATAL ) << "Profiler thread is already running";
    }

    thread_ = std::thread( &Profiler::run, this );

    if ( run_serialization_ and not serialization_thread_.joinable() ) {
      serialization_thread_ = std::thread( &Profiler::run_serialization, this );
    }
  }

  void wait()
  {
    if ( thread_.joinable() ) {
      thread_.join();
    }

    if ( serialization_thread_.joinable() ) {
      serialization_thread_.join();
    }
  }
};

} // namespace glinthawk
