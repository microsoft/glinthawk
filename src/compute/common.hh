#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace glinthawk::compute {

enum class KernelType
{
  Batched,     /* kernel.hh */
  Hybrid,      /* kernel_hybrid.hh */
  SimpleHybrid /* kernel_hybrid_simple.hh */
};

template<typename Kernel, typename StateType>
concept KernelConcept = requires( Kernel k, StateType s ) {
  std::is_same_v<typename std::decay<decltype( Kernel::Type )>::type, KernelType>;
  { k.push( std::move( s ) ) } -> std::same_as<void>;
  { k.pop( s ) } -> std::same_as<bool>;
};

class Concurrency
{
private:
  std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )> v_;

public:
  Concurrency( const size_t pre, const size_t att, const size_t post, const size_t classify )
    : v_ { pre, att, post, classify }
  {
    CHECK_GT( pre + att + post + classify, 0 ) << "At least one stage must be enabled";
  }

  void set( const models::InferenceStage stage, const size_t value ) { v_[util::to_underlying( stage )] = value; }
  size_t get( const models::InferenceStage stage ) const { return v_[util::to_underlying( stage )]; }
};

// std::priority_queue does not allow for moving elements, so we need to wrap the state in a struct
// to be able to move it around. The struct keeps the comparison key separate from the state itself, so the state
// can be moved out without affecting the queue's invariant.
struct StateQueueItem
{
  std::pair<size_t, size_t> _comp_key; /* (layer, stage) */
  mutable glinthawk::models::BatchedInferenceState<ConfigType> state;

  StateQueueItem( glinthawk::models::BatchedInferenceState<ConfigType>&& in_state )
    : state( std::move( in_state ) )
    , _comp_key( state.next_layer(), util::to_underlying( state.next_stage() ) )
  {
  }
};

struct StateCompOp
{
  bool operator()( const StateQueueItem& lhs, const StateQueueItem& rhs ) const
  {
    return lhs._comp_key > rhs._comp_key;
  }
};

struct GlobalQueue
{
  std::priority_queue<StateQueueItem, std::deque<StateQueueItem>, StateCompOp> queue;
  std::mutex mutex;
  std::condition_variable cv;
};

} // namespace glinthawk::compute
