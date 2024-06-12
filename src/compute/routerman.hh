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

template<typename Model>
class TierRouter
{
private:
  using ConfigType = typename Model::ConfigType;
  using StateType = typename glinthawk::models::BatchedInferenceState<ConfigType>;

  void assign_to_sub_groups( StateType& state );

  // TODO(pouya):
  //  1. Where does concurrency go?
  //  2. Where does virtual context manager go?
  //  3. Where does tier_router sit between kernel and worker?
  //  4. Does tier_router know about hybrid kernel vs. simple kernel?

public:
  std::pair<std::vector<StateType>, std::vector<StateType>> split_to_sub_groups( const StateType& state );
  void add_sub_group( const StateType& state )

};

} // namespace glinthawk::compute
