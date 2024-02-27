#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "monitoring/measurement.hh"
#include "util/eventfd.hh"

namespace glinthawk::compute {

template<typename Model>
class PreallocatingContextManager
{
public:
  PreallocatingContextManager( const typename Model::SettingsType& settings );

  std::shared_ptr<typename Model::ContextType> get_context( const PromptID& prompt_id );
  bool release_context( const PromptID& prompt_id );

  size_t free() const;
  size_t allocated() const;
  size_t empty() const;
  size_t total() const;
};

template<typename ModelA, typename ModelB>
requires std::same_as<typename ModelB::ConfigType, typename ModelA::ConfigType>
class HybridComputeKernel
{
public:
  using ConfigType = typename ModelA::ConfigType;

  struct Concurrency
  {
    uint64_t pre {}, att {}, post {}, classify {};
  };

public:
  HybridComputeKernel( std::unique_ptr<ModelA>&& model_a,
                       std::unique_ptr<ModelB>&& model_b,
                       const Concurrency& concurrency_a,
                       const Concurrency& concurrency_b );

  ~HybridComputeKernel();

  void push( glinthawk::models::BatchedInferenceState<ConfigType>&& state );
  bool pop( glinthawk::models::BatchedInferenceState<ConfigType>& state );

  EventFD& event_fd();

private:
  using StateType = glinthawk::models::BatchedInferenceState<ConfigType>;

  template<typename M>
  requires std::same_as<M, ModelA> || std::same_as<M, ModelB>
  struct ModelData
  {
    struct ProcessingQueues
    {
      using ContextType = std::shared_ptr<typename M::ContextType>;
      using StateContextPair = std::pair<StateType, std::vector<ContextType>>;

      std::array<std::queue<StateContextPair>, ConfigType::n_layers> pre {};
      std::queue<StateContextPair> att {};
      std::array<std::queue<StateType>, ConfigType::n_layers> post {};
      std::queue<StateType> classify {};
    };

    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;
    const Concurrency concurrency;
    const typename M::SettingsType settings;

    std::mutex mutex {};
    std::condition_variable cv {};
    ProcessingQueues queues {};

    ModelData( std::unique_ptr<M>&& in_model, const Concurrency& in_concurrency );
  };

  // ... -> [pre(a|b) -> att(a|b) -> post(a|b)] * n_layers -> classify(a|b)
  ModelData<ModelA> a_;
  ModelData<ModelB> b_;

  EventFD event_fd_ {};
  std::atomic<bool> running_ { true };

  // <queues>
  std::queue<StateType> incoming_ {};
  std::queue<StateType> pre_to_att_ {};
  std::queue<StateType> att_to_post_ {};
  std::queue<StateType> outgoing_ {};
  // </queues>

  // <threads>
  template<typename M1, typename M2>
  void execution_thread_func( ModelData<M1>& model_data, ModelData<M2>& other_model_data );

  void bookkeeping_thread_func();
  void backlog_thread_func();

  std::vector<std::thread> threads_;
  // </threads>
};

template<typename ModelA, typename ModelB>
template<typename M>
HybridComputeKernel<ModelA, ModelB>::ModelData<M>::ModelData( std::unique_ptr<M>&& in_model,
                                                              const Concurrency& in_concurrency )
  : model( std::move( in_model ) )
  , context_manager( model->settings() )
  , concurrency( in_concurrency )
  , settings( model->settings() )
{
}

template<typename ModelA, typename ModelB>
HybridComputeKernel<ModelA, ModelB>::HybridComputeKernel( std::unique_ptr<ModelA>&& model_a,
                                                          std::unique_ptr<ModelB>&& model_b,
                                                          const Concurrency& concurrency_a,
                                                          const Concurrency& concurrency_b )
  : a_( std::move( model_a ), concurrency_a )
  , b_( std::move( model_b ), concurrency_b )
{
  // check the concurrency settings to be permissible
  CHECK_EQ( concurrency_a.pre + concurrency_b.pre, concurrency_a.att + concurrency_b.att );
  CHECK_EQ( concurrency_a.att + concurrency_b.att, concurrency_a.post + concurrency_b.post );
  CHECK_EQ( concurrency_a.post + concurrency_b.post, concurrency_a.classify + concurrency_b.classify );

  threads_.emplace_back( &HybridComputeKernel::backlog_thread_func, this );
  threads_.emplace_back( &HybridComputeKernel::bookkeeping_thread_func, this );
  threads_.emplace_back( &HybridComputeKernel::execution_thread_func<ModelA>, this, std::ref( a_ ) );
  threads_.emplace_back( &HybridComputeKernel::execution_thread_func<ModelB>, this, std::ref( b_ ) );
}

template<typename ModelA, typename ModelB>
HybridComputeKernel<ModelA, ModelB>::~HybridComputeKernel()
{
  running_ = false;
  for ( auto& t : threads_ ) {
    t.join();
  }
}

template<typename ModelA, typename ModelB>
template<typename M1, typename M2>
void HybridComputeKernel<ModelA, ModelB>::execution_thread_func( ModelData<M1>& model_data,
                                                                 ModelData<M2>& other_model_data )
{
  while ( running_ ) {
    // let's see what we have in the queues, with the following priority:
    // (1) classification
    // (2) attention
    // (3) postattention-from-(self+other) for the layers in reverse order
    // (4) preattention for the layers in reverse order
    uint32_t next_layer = 0;
    InferenceStage next_stage;

    std::unique_lock<std::mutex> lock { model_data.mutex };
    cv.wait( lock, [this, &next_layer, &next_stage] {
      // (1)
      if ( model_data.concurrency.classify && !model_data.queues.classify.empty() ) {
        next_stage = InferenceStage::Classification;
        next_layer = std::numeric_limits<uint32_t>::max();
        return true;
      }

      // (2)
      if ( model_data.concurrency.att && !model_data.queues.att.empty() ) {
        next_stage = InferenceStage::Attention;
        next_layer = std::numeric_limits<uint32_t>::max();
        return true;
      }

      // (3) || (4)
      for ( size_t layer_idx = model_data.settings().end_layer_num; layer_idx >= model_data.settings().start_layer_num;
            layer_idx-- ) {
        if ( model_data.concurrency.post && !model_data.queues.post_from_self[layer_idx].empty()
             && !model_data.queues.post_from_other[layer_idx].empty() ) {
          next_stage = InferenceStage::PostAttention;
          next_layer = layer_idx;
          return true;
        }

        if ( model_data.concurrency.pre && !model_data.queues.pre.empty() ) {
          next_stage = InferenceStage::PreAttention;
          next_layer = std::numeric_limits<uint32_t>::max();
          return true;
        }
      }
    } );

    StateType input_state;
    std::vector<typename M1::ContextType> contexts;

    switch ( next_stage ) {
      case InferenceStage::PreAttention:
      case InferenceStage::Attention:
      case InferenceStage::PostAttention:
      case InferenceStage::Classification:
    }
  }
}

} // namespace glinthawk::compute
