#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "kernel.hh"
#include "kernel_pipes.hh"
#include "models/common/model.hh"
#include "monitoring/measurement.hh"
#include "prompt/prompt.hh"
#include "util/eventfd.hh"

namespace glinthawk::compute {

template<typename Model_GPU, typename Model_CPU>
class ComputeKernelMerged
{
private:
  using ContextPtrGPU = std::shared_ptr<typename Model_GPU::ContextType>;
  using ContextPtrCPU = std::shared_ptr<typename Model_CPU::ContextType>;
  using StateContextPairGPU = std::pair<models::InferenceState, ContextPtrGPU>;
  using StateContextPairCPU = std::pair<models::InferenceState, ContextPtrCPU>;

  std::unique_ptr<Model_GPU> model_gpu_;
  std::unique_ptr<Model_CPU> model_cpu_;
  ContextManagerPreAllocated<Model_GPU> context_manager_gpu_;
  ContextManagerPreAllocated<Model_CPU> context_manager_cpu_;

  const uint64_t target_conc_pre_gpu_size_;
  const uint64_t target_conc_att_gpu_size_;
  const uint64_t target_conc_post_gpu_size_;
  const uint64_t target_conc_cls_gpu_size_;

  const uint64_t target_conc_att_cpu_size_;

  const uint64_t start_layer_gpu_;
  const uint64_t end_layer_gpu_;
  const uint64_t n_layers_gpu_;

  uint32_t mode = 2;

  std::vector<std::queue<StateContextPairGPU>> processing_pre_attention_gpu_;
  std::vector<std::queue<models::InferenceState>> processing_post_attention_gpu_;

  std::queue<StateContextPairGPU> processing_attention_gpu_ {};
  std::queue<models::InferenceState> processing_classification_gpu_ {};

  std::queue<StateContextPairCPU> processing_attention_cpu_ {};

  std::queue<models::InferenceState> incoming_ {}, waiting_attention_ {}, outgoing_ {};

  std::mutex ctx_mgr_gpu_mutex_ {}, ctx_mgr_cpu_mutex_ {}, outgoing_mutex_ {}, incoming_mutex_ {},
    waiting_attention_mutex_ {}, processing_gpu_mutex_ {}, processing_cpu_mutex_ {};

  std::condition_variable ctx_mgr_gpu_cv_ {}, ctx_mgr_cpu_cv_ {}, incoming_cv_ {}, processing_gpu_cv_ {},
    processing_cpu_cv_ {}, waiting_attention_cv_ {};

  EventFD event_fd_ {};

  std::atomic<bool> running_ { true };

  Measurement& __stats__ { global_measurement() };

  void execution_thread_gpu_func();
  void execution_thread_cpu_func();
  void bookkeeping_thread_func();
  void backlog_thread_func();
  void qmeasure_thread_func();

  std::thread execution_thread_gpu_;
  std::thread execution_thread_cpu_;
  std::thread bookkeeping_thread_;
  std::thread backlog_thread_;
  std::thread qmeasure_thread_;

public:
  ComputeKernelMerged( std::unique_ptr<Model_GPU>&& model_gpu,
                       const uint64_t target_conc_pre_gpu_size,
                       const uint64_t target_conc_att_gpu_size,
                       const uint64_t target_conc_post_gpu_size,
                       const uint64_t target_conc_cls_gpu_size,
                       const uint64_t start_layer_gpu,
                       const uint64_t end_layer_gpu,
                       std::unique_ptr<Model_CPU>&& model_cpu,
                       const uint64_t target_conc_att_cpu_size )
    : model_gpu_( std::move( model_gpu ) )
    , context_manager_gpu_( model_gpu_->settings() )
    , target_conc_pre_gpu_size_( target_conc_pre_gpu_size )
    , target_conc_att_gpu_size_( target_conc_att_gpu_size )
    , target_conc_post_gpu_size_( target_conc_post_gpu_size )
    , target_conc_cls_gpu_size_( target_conc_cls_gpu_size )
    , start_layer_gpu_( start_layer_gpu )
    , end_layer_gpu_( end_layer_gpu )
    , n_layers_gpu_( end_layer_gpu_ - start_layer_gpu_ + 1 )
    , model_cpu_( std::move( model_cpu ) )
    , context_manager_cpu_( model_cpu_->settings() )
    , target_conc_att_cpu_size_( target_conc_att_cpu_size )
    , processing_pre_attention_gpu_( n_layers_gpu_ )
    , processing_post_attention_gpu_( n_layers_gpu_ )
    , running_( true )
    , execution_thread_gpu_( &ComputeKernelMerged::execution_thread_gpu_func, this )
    , execution_thread_cpu_( &ComputeKernelMerged::execution_thread_cpu_func, this )
    , bookkeeping_thread_( &ComputeKernelMerged::bookkeeping_thread_func, this )
    , backlog_thread_( &ComputeKernelMerged::backlog_thread_func, this )
    , qmeasure_thread_( &ComputeKernelMerged::qmeasure_thread_func, this )
  {
  }

  void push( glinthawk::models::InferenceState&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      incoming_.push( std::move( state ) );
    }

    incoming_cv_.notify_one();
  }

  void push( std::vector<glinthawk::models::InferenceState>&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      for ( auto& s : state ) {
        incoming_.push( std::move( s ) );
      }
    }

    incoming_cv_.notify_one();
  }

  bool pop( glinthawk::models::InferenceState& state )
  {
    std::lock_guard lock( outgoing_mutex_ );
    if ( outgoing_.empty() )
      return false;
    state = std::move( outgoing_.front() );
    outgoing_.pop();
    return true;
  }

  void push_finished( glinthawk::models::InferenceState&& state )
  {
    // Release the context in GPU and CPU managers
    {
      std::lock_guard lock( ctx_mgr_gpu_mutex_ );
      if ( context_manager_gpu_.release( state.prompt_id() ) )
        ctx_mgr_gpu_cv_.notify_one();
    }
    {
      std::lock_guard lock( ctx_mgr_cpu_mutex_ );
      if ( context_manager_cpu_.release( state.prompt_id() ) )
        ctx_mgr_cpu_cv_.notify_one();
    }

    // do a "fake" forward: remove self from propagation list and set next worker
    for ( int i = 0; i < n_layers_gpu_; i++ ) {
      model_gpu_->dummy_forward( state );
      model_cpu_->dummy_forward( state );
      model_gpu_->dummy_forward( state );
    }

    if ( state.next_stage() == models::InferenceState::Stage::Classification ) {
      // drop release message as it has fully propagated
      DLOG( INFO ) << "Dropping empty (release) inference state: " << state.to_string();
    } else {
      // propagate the release message to the next worker
      {
        std::lock_guard lock( outgoing_mutex_ );
        outgoing_.emplace( std::move( state ) );
      }

      event_fd_.write_event();
    }
  }

  void check_finished( glinthawk::models::InferenceState& state )
  {
    if ( model_gpu_->is_finished( state ) ) {
      state.set_finished();
    }
  }

  EventFD& event_fd() { return event_fd_; }

  ~ComputeKernelMerged()
  {
    running_ = false;
    execution_thread_gpu_.join();
    execution_thread_cpu_.join();
    bookkeeping_thread_.join();
    backlog_thread_.join();
    qmeasure_thread_.join();
  }
};

template<typename Model_GPU, typename Model_CPU>
void ComputeKernelMerged<Model_GPU, Model_CPU>::execution_thread_gpu_func()
{
  LOG( INFO ) << "ComputeKernelMerged execution thread for GPU started.";

  while ( running_ ) {
    std::vector<models::InferenceState> input_states;
    std::vector<ContextPtrGPU> contexts;
    models::InferenceState::Stage next_stage;
    uint32_t next_layer_idx;
    {
      // hold lock until one queue has enough data for one batch
      std::unique_lock<std::mutex> lock( processing_gpu_mutex_ );
      // TODO: With splitting KV cache to in-context and out-context, will large batch sizes cause deadlocks?
      // TODO: Is there any reason we shouldn't always use the best batch size for any pipe?
      processing_gpu_cv_.wait( lock, [this, &next_stage, &next_layer_idx] {
        if ( processing_classification_gpu_.size() >= target_conc_cls_gpu_size_ and target_conc_cls_gpu_size_ > 0 ) {
          next_stage = models::InferenceState::Stage::Classification;
          next_layer_idx = static_cast<uint32_t>( -1 );
          return true;
        }

        //        if ( processing_attention_gpu_.size() >= target_conc_att_gpu_size_ and target_conc_att_gpu_size_ > 0 )
        //        {
        //          next_stage = models::InferenceState::Stage::Attention;
        //          next_layer_idx = static_cast<uint32_t>( -1 );
        //          return true;
        //        }

        for ( int layer_idx = static_cast<int>( n_layers_gpu_ - 1 ); layer_idx >= 0; layer_idx-- ) {
          if ( processing_post_attention_gpu_[layer_idx].size() >= target_conc_post_gpu_size_
               and target_conc_post_gpu_size_ > 0 and (mode == 0 or (layer_idx == 0 and start_layer_gpu_ == 0)) ) {
            next_stage = models::InferenceState::Stage::PostAttention;
            next_layer_idx = static_cast<uint32_t>( layer_idx );
            return true;
          }
          if ( processing_pre_attention_gpu_[layer_idx].size() >= target_conc_pre_gpu_size_
               and target_conc_pre_gpu_size_ > 0 and (mode > 0 or (layer_idx == 0 and start_layer_gpu_ == 0)) ) {
            next_stage = models::InferenceState::Stage::PreAttention;
            next_layer_idx = static_cast<uint32_t>( layer_idx );
            return true;
          }
        }
        return false;
      } );

      // find the queue and pop the data to input_states and possibly contexts
      switch ( next_stage ) {
        case models::InferenceState::Stage::PreAttention: {
          for ( size_t j = 0; j < target_conc_pre_gpu_size_; j++ ) {
            StateContextPairGPU action = std::move( processing_pre_attention_gpu_[next_layer_idx].front() );
            processing_pre_attention_gpu_[next_layer_idx].pop();
            input_states.push_back( std::move( action.first ) );
            contexts.push_back( action.second );
          }
          mode -= 1;
          break;
        }
        case models::InferenceState::Stage::Attention: {
          for ( size_t j = 0; j < target_conc_att_gpu_size_; j++ ) {
            StateContextPairGPU action = std::move( processing_attention_gpu_.front() );
            processing_attention_gpu_.pop();
            input_states.push_back( std::move( action.first ) );
            contexts.push_back( action.second );
          }
          break;
        }
        case models::InferenceState::Stage::PostAttention: {
          for ( size_t j = 0; j < target_conc_post_gpu_size_; j++ ) {
            models::InferenceState action = std::move( processing_post_attention_gpu_[next_layer_idx].front() );
            processing_post_attention_gpu_[next_layer_idx].pop();
            input_states.push_back( std::move( action ) );
          }
          mode = 1;
          break;
        }
        case models::InferenceState::Stage::Classification: {
          for ( size_t j = 0; j < target_conc_cls_gpu_size_; j++ ) {
            models::InferenceState action = std::move( processing_classification_gpu_.front() );
            processing_classification_gpu_.pop();
            input_states.push_back( std::move( action ) );
          }
          break;
        }
        default: LOG( FATAL ) << "Invalid stage";
      }
    }

    LOG(INFO) << "[COMPUTE]" << std::chrono::steady_clock::now().time_since_epoch().count() << "," << next_stage << "," << mode;
    const auto start = std::chrono::steady_clock::now();
    std::vector<models::InferenceState> results;
    switch ( next_stage ) {
      case models::InferenceState::Stage::PreAttention: {
        for ( auto& state : input_states ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cuda_batch";
          __stats__.add_point<IntDistributions::MergedPreKernelIncoming2BatchingTime>( start.time_since_epoch().count()
                                                                                       - state.timestamp() );
        }
        results = model_gpu_->pre_attention_forward( std::move( input_states ), contexts );
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        __stats__.add_point<IntDistributions::KernelPreAttentionForwardTime>( duration.count() );
        for ( auto& result : results ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << result.to_log() << ",proc_cuda_done";
          result.set_timestamp( end.time_since_epoch().count() );
          result.set_batch_timestamp( end.time_since_epoch().count() );
          result.set_batch_last( false );
        }
        results.back().set_batch_last( true );
      } break;
      case models::InferenceState::Stage::Attention: {
        for ( auto& state : input_states ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cuda_batch";
          __stats__.add_point<IntDistributions::MergedAttContext2BatchingTime>( start.time_since_epoch().count()
                                                                                - state.timestamp() );
          if ( state.batch_last() ) {
            __stats__.add_point<IntDistributions::MergedAttContext2BatchingTimeBatch>( start.time_since_epoch().count()
                                                                                       - state.batch_timestamp() );
          }
        }
        results = model_gpu_->attention_forward( std::move( input_states ), contexts );
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        __stats__.add_point<IntDistributions::CUDAKernelAttentionForwardTime>( duration.count() );
        for ( auto& result : results ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << result.to_log() << ",proc_cuda_done";
          result.set_timestamp( end.time_since_epoch().count() );
          result.set_batch_timestamp( end.time_since_epoch().count() );
          result.set_batch_last( false );
        }
        results.back().set_batch_last( true );
      } break;
      case models::InferenceState::Stage::PostAttention: {
        for ( auto& state : input_states ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cuda_batch";
          __stats__.add_point<IntDistributions::MergedAttInference2PostBatchingTime>( start.time_since_epoch().count()
                                                                                      - state.timestamp() );
          if ( state.batch_last() ) {
            __stats__.add_point<IntDistributions::MergedAttInference2PostBatchingTimeBatch>(
              start.time_since_epoch().count() - state.batch_timestamp() );
          }
        }
        results = model_gpu_->post_attention_forward( std::move( input_states ) );
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        __stats__.add_point<IntDistributions::KernelPostAttentionForwardTime>( duration.count() );
        for ( auto& result : results ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << result.to_log() << ",proc_cuda_done";
          result.set_timestamp( end.time_since_epoch().count() );
          result.set_batch_timestamp( end.time_since_epoch().count() );
          result.set_batch_last( false );
        }
        results.back().set_batch_last( true );
      } break;
      case models::InferenceState::Stage::Classification: {
        for ( auto& state : input_states ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cuda_batch";
          __stats__.add_point<IntDistributions::MergedClsKernelIncoming2BatchingTime>( start.time_since_epoch().count()
                                                                                       - state.timestamp() );
        }
        results = model_gpu_->classify_forward( std::move( input_states ) );
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        __stats__.add_point<IntDistributions::KernelClassificationForwardTime>( duration.count() );
        for ( auto& result : results ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << result.to_log() << ",proc_cuda_done";
          result.set_timestamp( end.time_since_epoch().count() );
          result.set_batch_timestamp( end.time_since_epoch().count() );
          result.set_batch_last( false );
        }
        results.back().set_batch_last( true );
      } break;
      default: LOG( FATAL ) << "Invalid stage";
    }

    std::vector<models::InferenceState> outgoing_states;
    std::vector<StateContextPairCPU> processing_states_cpu;
    std::vector<StateContextPairGPU> processing_states_gpu;
    std::vector<models::InferenceState> waiting_states;
    switch ( next_stage ) {
      case models::InferenceState::Stage::PreAttention: {
        // the next stage is attention, so if we hold the context and serve attention, add it directly to processing
        for ( size_t j = 0; j < target_conc_pre_gpu_size_; j++ ) {
          CHECK_GT( target_conc_att_cpu_size_, 0 );
          CHECK_EQ( contexts[j].get()->empty(), true );
          ContextPtrCPU context_cpu;
          {
            std::lock_guard lock( ctx_mgr_cpu_mutex_ );
            context_cpu = context_manager_cpu_.get_context( results[j].prompt_id(), false );
          }
          if ( context_cpu->empty() ) {
            waiting_states.emplace_back( std::move( results[j] ) );
          } else {
            const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
            __stats__.add_point<IntDistributions::MergedPreInference2AttContextTime>( current_time
                                                                                      - results[j].timestamp() );
            if ( results[j].batch_last() ) {
              __stats__.add_point<IntDistributions::MergedPreInference2AttContextTimeBatch>(
                current_time - results[j].batch_timestamp() );
            }
            results[j].set_timestamp( current_time );
            processing_states_cpu.emplace_back( std::move( results[j] ), context_cpu );
          }
        }
        break;
      }
      case models::InferenceState::Stage::Attention:
        // The next stage is post-attention, so if we serve that specific layer, add it directly to processing
        for ( size_t j = 0; j < target_conc_att_gpu_size_; j++ ) {
          if ( target_conc_post_gpu_size_ > 0 and results[j].next_layer() <= end_layer_gpu_
               and results[j].next_layer() >= start_layer_gpu_ ) {
            processing_states_gpu.emplace_back( std::move( results[j] ), nullptr );
          } else {
            outgoing_states.emplace_back( std::move( results[j] ) );
          }
        }
        break;
      case models::InferenceState::Stage::PostAttention:
        if ( results[0].next_stage() == models::InferenceState::Stage::PreAttention and target_conc_pre_gpu_size_ > 0
             and results[0].next_layer() <= end_layer_gpu_ and results[0].next_layer() >= start_layer_gpu_ ) {
          // If the next stage is pre-attention, and we serve that layer, get context and add it directly to processing
          std::lock_guard lock( ctx_mgr_gpu_mutex_ );
          for ( size_t j = 0; j < target_conc_post_gpu_size_; j++ ) {
            processing_states_gpu.emplace_back(
              std::move( results[j] ), context_manager_gpu_.get_context( results[j].prompt_id(), true, true ) );
          }
        } else if ( results[0].next_stage() == models::InferenceState::Stage::Classification
                    and target_conc_cls_gpu_size_ > 0 ) {
          // If next stage is classification, and we serve it, add it directly to processing
          for ( size_t j = 0; j < target_conc_post_gpu_size_; j++ ) {
            processing_states_gpu.emplace_back( std::move( results[j] ), nullptr );
          }
        } else {
          for ( size_t j = 0; j < target_conc_post_gpu_size_; j++ ) {
            outgoing_states.emplace_back( std::move( results[j] ) );
          }
        }

        break;
      case models::InferenceState::Stage::Classification:
        // the next stage is pre-attention in layer 0, so we have to return the state from the kernel path for logging
        for ( size_t j = 0; j < target_conc_cls_gpu_size_; j++ ) {
          check_finished( results[j] );
          CHECK_EQ( results[j].next_layer(), 0 );
          outgoing_states.emplace_back( std::move( results[j] ) );
        }
        break;
      default: LOG( FATAL ) << "Invalid stage";
    }

    {
      std::lock_guard lock( outgoing_mutex_ );
      for ( auto& state : outgoing_states ) {
        LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cuda_out";
        outgoing_.emplace( std::move( state ) );
      }
    }

    if ( outgoing_states.size() > 0 ) {
      event_fd_.write_event();
    }

    {
      std::lock_guard lock( processing_gpu_mutex_ );
      switch ( next_stage ) {
        case models::InferenceState::Stage::PreAttention:
          for ( auto& action : processing_states_gpu ) {
            LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << action.first.to_log() << ",proc_cuda_to_proc_cuda";
            processing_attention_gpu_.emplace( std::move( action.first ), action.second );
          }
          break;
        case models::InferenceState::Stage::Attention:
          for ( auto& action : processing_states_gpu ) {
            LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << action.first.to_log() << ",proc_cuda_to_proc_cuda";
            processing_post_attention_gpu_[action.first.next_layer() - start_layer_gpu_].emplace(
              std::move( action.first ) );
          }
          break;
        case models::InferenceState::Stage::PostAttention:
          if ( processing_states_gpu.size() > 0 ) {
            switch ( processing_states_gpu[0].first.next_stage() ) {
              case models::InferenceState::Stage::PreAttention:
                for ( auto& action : processing_states_gpu ) {
                  LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << action.first.to_log() << ",proc_cuda_to_proc_cuda";
                  processing_pre_attention_gpu_[action.first.next_layer() - start_layer_gpu_].emplace(
                    std::move( action.first ), action.second );
                }
                break;
              case models::InferenceState::Stage::Classification:
                for ( auto& action : processing_states_gpu ) {
                  LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << action.first.to_log() << ",proc_cuda_to_proc_cuda";
                  processing_classification_gpu_.emplace( std::move( action.first ) );
                }
                break;
              default: LOG( FATAL ) << "Invalid stage";
            }
          }
          break;
          // we should not fast-path states after classification, otherwise they won't be logged
        case models::InferenceState::Stage::Classification: break;
        default: LOG( FATAL ) << "Invalid stage";
      }
    }

    if ( processing_states_cpu.size() > 0 and next_stage == models::InferenceState::Stage::PreAttention ) {
      std::lock_guard lock( processing_cpu_mutex_ );
      for ( auto& action : processing_states_cpu ) {
        LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << action.first.to_log() << ",proc_cuda_to_proc_cpu";
        processing_attention_cpu_.emplace( std::move( action.first ), action.second );
      }
      processing_cpu_cv_.notify_one();
    }

    if ( waiting_states.size() > 0 and next_stage == models::InferenceState::Stage::PreAttention ) {
      std::lock_guard lock( waiting_attention_mutex_ );
      for ( auto& state : waiting_states ) {
        LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cuda_to_wait";
        waiting_attention_.emplace( std::move( state ) );
      }
      waiting_attention_cv_.notify_one();
    }
  }
}

template<typename Model_GPU, typename Model_CPU>
void ComputeKernelMerged<Model_GPU, Model_CPU>::execution_thread_cpu_func()
{
  LOG( INFO ) << "ComputeKernelMerged execution thread for CPU started.";

  while ( running_ ) {
    std::vector<models::InferenceState> input_states;
    std::vector<ContextPtrCPU> contexts;
    {
      // hold lock until one queue has enough data for one batch
      std::unique_lock<std::mutex> lock( processing_cpu_mutex_ );
      processing_cpu_cv_.wait( lock, [this] {
        return processing_attention_cpu_.size() >= target_conc_att_cpu_size_ and target_conc_att_cpu_size_ > 0;
      } );

      for ( size_t j = 0; j < target_conc_att_cpu_size_; j++ ) {
        StateContextPairCPU action = std::move( processing_attention_cpu_.front() );
        processing_attention_cpu_.pop();
        input_states.push_back( std::move( action.first ) );
        contexts.push_back( action.second );
      }
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<models::InferenceState> results;

    for ( auto& state : input_states ) {
      LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cpu_batch";
      __stats__.add_point<IntDistributions::MergedAttContext2BatchingTime>( start.time_since_epoch().count()
                                                                            - state.timestamp() );
      if ( state.batch_last() ) {
        __stats__.add_point<IntDistributions::MergedAttContext2BatchingTimeBatch>( start.time_since_epoch().count()
                                                                                   - state.batch_timestamp() );
      }
    }
    results = model_cpu_->attention_forward( std::move( input_states ), contexts );
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
    __stats__.add_point<IntDistributions::AMD64KernelAttentionForwardTime>( duration.count() );
    for ( auto& result : results ) {
      LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << result.to_log() << ",proc_cpu_done";
      result.set_timestamp( end.time_since_epoch().count() );
      result.set_batch_timestamp( end.time_since_epoch().count() );
      result.set_batch_last( false );
    }
    results.back().set_batch_last( true );

    std::vector<models::InferenceState> outgoing_states;
    std::vector<models::InferenceState> processing_states;
    // The next stage is post-attention, so if we serve that specific layer, add it directly to processing
    for ( size_t j = 0; j < target_conc_att_cpu_size_; j++ ) {
      if ( target_conc_post_gpu_size_ > 0 and results[j].next_layer() <= end_layer_gpu_
           and results[j].next_layer() >= start_layer_gpu_ ) {
        processing_states.emplace_back( std::move( results[j] ) );
      } else {
        outgoing_states.emplace_back( std::move( results[j] ) );
      }
    }

    if ( outgoing_states.size() > 0 ) {
      std::lock_guard lock( outgoing_mutex_ );
      for ( auto& state : outgoing_states ) {
        LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cpu_out";
        outgoing_.emplace( std::move( state ) );
      }
    }

    if ( outgoing_states.size() > 0 ) {
      event_fd_.write_event();
    }

    if ( processing_states.size() > 0 ) {
      std::lock_guard lock( processing_gpu_mutex_ );
      for ( auto& state : processing_states ) {
        LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << state.to_log() << ",proc_cpu_to_cuda";
        processing_post_attention_gpu_[state.next_layer() - start_layer_gpu_].emplace( std::move( state ) );
      }
      processing_gpu_cv_.notify_one();
    }
  }
}

template<typename Model_GPU, typename Model_CPU>
void ComputeKernelMerged<Model_GPU, Model_CPU>::bookkeeping_thread_func()
{
  LOG( INFO ) << "ComputeKernelMerged bookkeeping thread started.";

  while ( running_ ) {
    models::InferenceState action;
    // let's get an action from the incoming_
    {
      std::unique_lock<std::mutex> lock( incoming_mutex_ );
      incoming_cv_.wait( lock, [this] { return !incoming_.empty(); } );
      action = std::move( incoming_.front() );
      incoming_.pop();
    }

    // make sure this action is for our serving layers
    const uint32_t next_layer_index = action.next_layer() - start_layer_gpu_;
    CHECK_EQ( next_layer_index < n_layers_gpu_ or action.next_stage() == models::InferenceState::Stage::Attention,
              true )
      << "InferenceState can not be processed in this machine, original next layer/stage was: " << action.next_layer()
      << "/" << action.next_stage() << ", but we host " << start_layer_gpu_ << " to " << end_layer_gpu_;

    const auto log_t = action.to_log();
    LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << log_t << ",incoming" ;

    const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
    switch ( action.next_stage() ) {
      case models::InferenceState::Stage::PreAttention: {
        __stats__.add_point<IntDistributions::MergedPreWorker2KernelIncomingTime>( current_time - action.timestamp() );
        action.set_timestamp( current_time );

        // for action in pre-attention stage, get (try to create) context and just push to compute.
        CHECK_GT( target_conc_pre_gpu_size_, 0 ) << "This machine does not service the PreAttention pipeline";
        ContextPtrGPU context;
        {
          // let's get the context for this action, but it doesn't matter if it's empty
          std::lock_guard lock( ctx_mgr_gpu_mutex_ );
          //          if ( ctx_balance_ < 0 and context_manager_gpu_.free() > 0 ) {
          //            context = context_manager_gpu_.get_context( action.prompt_id(), true, false );
          //            ctx_balance += ratio_ctx_;
          //          } else {
          context = context_manager_gpu_.get_context( action.prompt_id(), true, true );
          //          }
        }
        {
          std::lock_guard lock( processing_gpu_mutex_ );
          processing_pre_attention_gpu_[next_layer_index].emplace( std::move( action ), context );
        }

        processing_gpu_cv_.notify_one();
        break;
      }

      case models::InferenceState::Stage::Attention: {
        // for action in attention stage, get non-empty context and just push to compute.
        // if the context doesn't exist, just wait
        CHECK_GT( target_conc_att_cpu_size_, 0 ) << "This machine does not service the Attention pipeline";
        CHECK_EQ( target_conc_att_gpu_size_, 0 ) << "GPU Attention is closed for now";

        bool scheduled = false;

        ContextPtrCPU context;
        {
          // let's get the context for this action
          std::lock_guard lock( ctx_mgr_cpu_mutex_ );
          context = context_manager_cpu_.get_context( action.prompt_id(), false );
        }
        if ( not context.get()->empty() ) {
          LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << log_t << ",context_alloc_inc";
          {
            std::lock_guard lock( processing_cpu_mutex_ );
            processing_attention_cpu_.emplace( std::move( action ), context );
          }

          processing_cpu_cv_.notify_one();
          scheduled = true;
        }

        if ( not scheduled ) {
          {
            std::lock_guard lock( waiting_attention_mutex_ );
            waiting_attention_.emplace( std::move( action ) );
          }

          waiting_attention_cv_.notify_one();
        }

        break;
      }

      case models::InferenceState::Stage::PostAttention: {
        // for action in post-attention stage, push to compute without context
        CHECK_GT( target_conc_post_gpu_size_, 0 ) << "This machine does not service the PostAttention pipeline";
        {
          std::lock_guard lock( processing_gpu_mutex_ );
          processing_post_attention_gpu_[next_layer_index].emplace( std::move( action ) );
        }

        processing_gpu_cv_.notify_one();
        break;
      }

      case models::InferenceState::Stage::Classification: {
        __stats__.add_point<IntDistributions::MergedClsWorker2KernelIncomingTime>( current_time - action.timestamp() );
        action.set_timestamp( current_time );
        // for action in classification stage, push to compute without context
        CHECK_GT( target_conc_cls_gpu_size_, 0 ) << "This machine does not service the Classification pipeline";
        {
          std::lock_guard lock( processing_gpu_mutex_ );
          processing_classification_gpu_.emplace( std::move( action ) );
        }

        processing_gpu_cv_.notify_one();
        break;
      }
      default: LOG( FATAL ) << "Invalid stage";
    }

    LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << log_t << ",incoming_routed";
  }
}

template<typename Model_GPU, typename Model_CPU>
void ComputeKernelMerged<Model_GPU, Model_CPU>::backlog_thread_func()
{
  LOG( INFO ) << "ComputeKernelMerged backlog thread started.";

  while ( running_ ) {
    models::InferenceState action;
    ContextPtrCPU context;

    // let's get an action from the waiting_attention_
    {
      std::unique_lock<std::mutex> lock( waiting_attention_mutex_ );
      waiting_attention_cv_.wait( lock, [this] { return !waiting_attention_.empty(); } );
      action = std::move( waiting_attention_.front() );
      waiting_attention_.pop();
    }

    // let's get a free context from context_manager_
    {
      std::unique_lock<std::mutex> lock( ctx_mgr_cpu_mutex_ );
      ctx_mgr_cpu_cv_.wait( lock, [this] { return context_manager_cpu_.free() > 0; } );
      context = context_manager_cpu_.get_context( action.prompt_id(), false );
    }

    CHECK_EQ( context.get()->empty(), false ) << "Context should not be empty";

    LOG(INFO) << "[EVENT]," << std::chrono::steady_clock::now().time_since_epoch().count() << "," << action.to_log() << ",context_alloc";

    const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
    __stats__.add_point<IntDistributions::MergedPreInference2AttContextTime>( current_time - action.timestamp() );
    if ( action.batch_last() ) {
      __stats__.add_point<IntDistributions::MergedPreInference2AttContextTimeBatch>( current_time
                                                                                     - action.batch_timestamp() );
    }
    action.set_timestamp( current_time );

    {
      std::lock_guard lock( processing_cpu_mutex_ );
      processing_attention_cpu_.emplace( std::move( action ), context );
    }

    processing_cpu_cv_.notify_one();
  }
}

template<typename Model_GPU, typename Model_CPU>
void ComputeKernelMerged<Model_GPU, Model_CPU>::qmeasure_thread_func()
{
  LOG( INFO ) << "ComputeKernelMerged queue measurement thread started.";

  while ( running_ ) {

    {
      std::lock_guard lock( processing_gpu_mutex_ );
      __stats__.add_point<IntDistributions::ProcessingClassificationQueue>( processing_classification_gpu_.size() );
      __stats__.add_point<IntDistributions::CUDAProcessingAttentionQueue>( processing_attention_gpu_.size() );
      for ( uint64_t layer_idx = 0; layer_idx < n_layers_gpu_; layer_idx++ ) {
        __stats__.add_point<IntDistributions::ProcessingPreAttentionQueue>(
          processing_pre_attention_gpu_[layer_idx].size() );
        __stats__.add_point<IntDistributions::ProcessingPostAttentionQueue>(
          processing_post_attention_gpu_[layer_idx].size() );
      }
    }

    {
      std::lock_guard lock( processing_cpu_mutex_ );
      __stats__.add_point<IntDistributions::AMD64ProcessingAttentionQueue>( processing_attention_cpu_.size() );
    }

    {
      std::lock_guard lock( waiting_attention_mutex_ );
      __stats__.add_point<IntDistributions::WaitingQueue>( waiting_attention_.size() );
    }

    {
      std::lock_guard lock( incoming_mutex_ );
      __stats__.add_point<IntDistributions::IncomingQueue>( incoming_.size() );
    }

    {
      std::lock_guard lock( outgoing_mutex_ );
      __stats__.add_point<IntDistributions::OutgoingQueue>( outgoing_.size() );
    }

    {
      std::lock_guard lock( ctx_mgr_gpu_mutex_ );
      __stats__.add_point<IntDistributions::CUDAAllocatedContexts>( context_manager_gpu_.allocated() );
      __stats__.add_point<IntDistributions::CUDAFreeContexts>( context_manager_gpu_.free() );
      __stats__.add_point<IntDistributions::CUDAEmptyContexts>( context_manager_gpu_.empty() );
    }

    {
      std::lock_guard lock( ctx_mgr_cpu_mutex_ );
      __stats__.add_point<IntDistributions::AMD64AllocatedContexts>( context_manager_cpu_.allocated() );
      __stats__.add_point<IntDistributions::AMD64FreeContexts>( context_manager_cpu_.free() );
      __stats__.add_point<IntDistributions::AMD64EmptyContexts>( context_manager_cpu_.empty() );
    }

    std::this_thread::sleep_for( std::chrono::seconds { 2 } );
  }
}

} // namespace glinthawk::compute
