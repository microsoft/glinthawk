#pragma once

#include <atomic>
#include <concepts>
#include <memory>
#include <queue>
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
  using ContextA = std::shared_ptr<typename ModelA::ContextType>;
  using ContextB = std::shared_ptr<typename ModelB::ContextType>;

  using StateContextsPairA = std::pair<StateType, std::vector<ContextA>>;
  using StateContextsPairB = std::pair<StateType, std::vector<ContextB>>;

  template<typename M>
  struct ModelData
  {
    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;

    const Concurrency concurrency;
    const typename M::SettingsType settings;
  };

  // ... -> [pre(a|b) -> att(a|b) -> post(a|b)] -> ... -> classify
  ModelData<ModelA> a_;
  ModelData<ModelB> b_;

  EventFD event_fd_ {};
  std::atomic<bool> running_ { true };

  // <threads>
  void execution_thread_a_func();
  void execution_thread_b_func();
  void bookkeeping_thread_func();
  void backlog_thread_func();

  std::thread execution_thread_a_;
  std::thread_execution_thread_b_;
  std::thread bookkeeping_thread_;
  std::thread backlog_thread_;
  // </threads>
};

} // namespace glinthawk::compute
