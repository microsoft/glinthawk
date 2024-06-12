#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "util/eventfd.hh"
#include "util/util.hh"

#include "common.hh"
#include "contextman.hh"

namespace glinthawk::compute {

/// @brief
/// TierRouter is a middle-man between the worker and kernel. It's job is to break down states between machines in
/// tiers, while ensuring context does not run out and deadlocks don't happen.
/// TierRouter is only run on `rank-0' machines, i.e., it only runs on one node in each layer. The other nodes have a
/// dummy TierRouter that pass states through with no delay.
/// TierRouter distinguishes states as two types:
///     Parent BatchInferenceState: which is a full batch across all machines in both tiers.
///     Child BatchInferenceState: which is a specific part of a parent dedicated to a specific machine.
/// TierRouter receives parent/child states through push, and returns child states back. It sends to and receives local
/// child states from the compute kernel directly.
/// The worker reads an outgoing queue from TierRouter to send out outbound parent states.
/// TierRouter fully handles context management, and guarantees that if a child state arrives at the kernel, that kernel
/// has space for it.
template<typename Model>
class TierRouter
{
private:
  using ConfigType = typename Model::ConfigType;
  using StateType = typename glinthawk::models::BatchedInferenceState<ConfigType>;

  /// @brief
  /// For now we assume all tier 1 machines are alike and all tier 2 machines are alike.
  /// The TierRouter does not need to know if the kernel is hybrid or not. It treats hybrid kernels as a sum of two
  /// concurrencies.
  size_t n_tier_1;
  size_t kv_slots_tier_1;
  const Concurrency concurrency_tier_1;
  size_t n_tier_2;
  size_t kv_slots_tier_2;
  const Concurrency concurrency_tier_2;

  std::vector<VirtualPreallocatingContextManager<Model>> cms_tier_1;
  std::vector<VirtualPreallocatingContextManager<Model>> cms_tier_2;

  /// @brief
  /// assign_sub_groups assigns tier_routing_group indices to states that have not been assigned them before. The
  /// assignment is based on the index of each prompt in the state. Input states are either fully assigned (all slots
  /// in the batch inference state were assigned before) or none were assigned. This function is called a finite number
  /// of times before all context slots are taken, and never called again. Note that when worker replaces a discarded
  /// prompt, it keeps any assigned tier_routing_group indices. This is important to avoid situations where new prompts
  /// are assigned to tier sub groups that their neighbors do not belong in, which causes fragmentation.
  void assign_sub_groups( StateType& state );

  /// @brief
  /// process_parent_state breaks the `parent' state to `child' states by the tier_routing_group in each prompt. Each
  /// child is only sent out if the remote/local machine can allot context for it. Note that we do not care if some
  /// other machine has context for its own child state or not. If said context does not exist, TierRouter holds onto
  /// that child state until it does. The latter only occurs on releasing past context, which also happens in push. So,
  /// push may return older child states or return none at all.
  std::vector<StateType> process_parent_state( const StateType& state );

  /// @brief
  /// process_child_state manages an internal memory to merge states of various tiers together. If it receives a child
  /// state, it saves it. Upon merging, it may process it the same way it does the parent. If the parent does not need
  /// processing (is not local to that machine anymore, e.g., is for a different layer) it is sent out via an outgoing
  /// queue that the worker communicates with
  std::vector<StateType> process_child_state( const StateType& state );

public:
  /// @brief
  /// if the state was unbroken calls process_parent_state
  /// if the state was broken calls process_child_state
  /// Push is a synchronous function. This may/may not be good.
  /// Pros: Avoids another async queue
  /// Cons: Locks the worker thread, potentially delaying communications (?)
  std::vector<StateType> push_from_worker( StateType& state );

  /// @brief
  /// When the compute kernel finishes a child state, it sends it to TierRouter. If the TierRouter has enough
  /// children, it can merge them and move to the next step, possibly breaking them to children again.
  std::vector<StateType> push_from_kernel( StateType& state );

  /// @brief
  /// pop is called by the worker to receive states, that (1) are parent states that are no longer local to the node, or
  /// (2) are child states that resulted from the local node pushing the last
  bool pop_to_worker( StateType& state );
};

} // namespace glinthawk::compute
