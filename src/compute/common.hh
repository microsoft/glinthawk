#pragma once

#include "util/eventfd.hh"
#include <concepts>
#include <type_traits>
#include <utility>

namespace glinthawk::compute {

enum class KernelType
{
  Batched,      /* kernel.hh */
  Hybrid,       /* kernel_hybrid.hh */
  SimpleHybrid, /* kernel_hybrid_simple.hh */
  SimplePiped,  /* kernel_piped.hh */
};

template<typename Kernel, typename StateType>
concept KernelConcept = requires( Kernel k, StateType s, EventFD e ) {
  std::is_same_v<typename std::decay<decltype( Kernel::Type )>::type, KernelType>;
  { k.push( std::move( s ) ) } -> std::same_as<void>;
  { k.pop( s ) } -> std::same_as<bool>;
  { k.set_event_fd( e ) };
};

class NodeConcurrency
{
private:
  std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )> v_;

public:
  NodeConcurrency( const size_t pre, const size_t att, const size_t post, const size_t classify )
    : v_ { pre, att, post, classify }
  {
    CHECK_GT( pre + att + post + classify, 0 ) << "At least one stage must be enabled";
  }

  void set( const models::InferenceStage stage, const size_t value ) { v_[util::to_underlying( stage )] = value; }
  size_t get( const models::InferenceStage stage ) const { return v_[util::to_underlying( stage )]; }
  size_t max() const { return std::max( v_ ); }
};

class SliceConcurrency
{
private:
  size_t monolith_batch_size_;
  std::vector<uint8_t> n_tier_s_;
  std::vector<std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )>> v_;
  std::array<std::vector<size_t>, util::to_underlying( models::InferenceStage::__COUNT__ )> shard_cut_cache_;
  std::array<std::vector<int8_t>, util::to_underlying( models::InferenceStage::__COUNT__ )> tier_index_cache_;
  std::array<std::vector<uint8_t>, util::to_underlying( models::InferenceStage::__COUNT__ )> rank_index_cache_;

public:
  SliceConcurrency( std::vector<uint8_t> n_tier_s,
                    std::vector<std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )>> v )
    : n_tier_s_( n_tier_s )
    , v_( v )
  {
    CHECK_GT( n_tier_s_.size(), 0 ) << "At least one tier";
    CHECK_LT( n_tier_s_.size(), 128 ) << "No more than 127 tiers";
    for ( auto n_tier : n_tier_s_ ) {
      CHECK_GT( n_tier, 0 ) << "No empty tiers";
      CHECK_LT( n_tier, 256 ) << "No more than 255 nodes in a tier";
    }
    for ( auto v_tier& : v_ ) {
      CHECK_GT( std::accumulate( v_tier.begin() + v_tier.end(), 0 ), 0 ) << "At least one stage must be enabled";
    }
    // Build the tier/rank lookup cache, and the sharding
    for ( int stage_i = 0; stage_i < util::to_underlying( models::InferenceStage::__COUNT__ ); stage_i++ ) {
      for ( int8_t tier_i = 0; tier_i < n_tier_s_.size(); tier_i++ ) {
        for ( uint8_t rank_i = 0; rank_i < n_tier_s_[tier_i]; rank_i++ ) {
          if ( v_[tier_i][stage_i] > 0 ) {
            shard_cut_cache_[stage_i].push_back( v_[tier_i][stage_i] );
          }
          for ( int batch_i = 0; batch_i < v_[tier_i][stage_i]; batch_i++ ) {
            tier_index_cache_[stage_i].push_back( tier_i );
            rank_index_cache_[stage_i].push_back( rank_i );
          }
        }
      }
    }
    monolith_batch_size_ = std::accumulate( shard_cut_cache_[0].begin(), shard_cut_cache_[0].end(), 0 );
    for ( int i = 1; i < util::to_underlying( models::InferenceStage::__COUNT__ ); i++ ) {
      CHECK_EQ( monolith_batch_size_, std::accumulate( shard_cut_cache_[i].begin(), shard_cut_cache_[i].end(), 0 ) );
    }
  }

  std::pair<int8_t, uint8_t> tier_rank( const models::InferenceStage stage, const size_t batch_index ) const
  {
    CHECK_GE( batch_index, 0 );
    CHECK_LT( batch_index, monolith_batch_size_ );
    return std::make_pair( tier_index_cache_[util::to_underlying( stage )][batch_index],
                           rank_index_cache_[util::to_underlying( stage )][batch_index] );
  }

  std::vector<size_t>& cutting_plan( const models::InferenceStage stage ) const
  {
    return shard_cut_cache_[util::to_underlying( stage )];
  }

  int8_t num_tiers() const { return n_tier_s_.size(); }

  uint8_t num_ranks( const int8_t tier_i ) const { return n_tier_s_[tier_i]; }

  size_t get( const int8_t tier_i, const models::InferenceStage stage ) const
  {
    return v_[tier_i][util::to_underlying( stage )];
  }

  size_t full_batch() const { return monolith_batch_size_; }

  NodeConcurrency& node_concurrency( const int8_t tier_i ) const
  {
    return { v_[tier_i][0], v_[tier_i][1], v_[tier_i][2], v_[tier_i][3] }
  }
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
