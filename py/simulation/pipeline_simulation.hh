//
// Created by Pouya Hamadanian on 9/16/24.
//

#pragma once

#include "algorithm"
#include "assert.h"
#include "iostream"
#include "numeric"
#include "vector"

using namespace std;

class PipelineElement
{
private:
  vector<double> delay_array_;
  double next_start_time_ {};
  const bool delay_only_;

  struct QueueItem
  {
    size_t step;
    double entry_time;
    size_t batch_id;
  };

  struct QueueItemCompOp
  {
    bool operator()( const QueueItem& lhs, const QueueItem& rhs ) const { return lhs.entry_time < rhs.entry_time; }
  };

  vector<QueueItem> queue_ {};

public:
  PipelineElement( vector<double>& delay_array, bool delay_only )
    : delay_array_( delay_array )
    , delay_only_( delay_only ) {};
  ~PipelineElement() = default;

  double min_event() const
  {
    if ( queue_.size() == 0 ) {
      return numeric_limits<double>::infinity();
    } else {
      if ( next_start_time_ < queue_[0].entry_time ) {
        return queue_[0].entry_time;
      } else {
        return next_start_time_;
      }
    }
  }

  QueueItem queue_pop( double current_time )
  {
    assert( queue_.size() > 0 );
    size_t best_i = 0;
    for ( size_t i = 0; i < queue_.size(); i++ ) {
      if ( queue_[i].entry_time > current_time ) {
        break;
      }
      if ( queue_[i].step > queue_[best_i].step ) {
        best_i = i;
      }
    }
    auto item = queue_[best_i];
    queue_.erase( queue_.begin() + best_i );
    return item;
  }

  void push( size_t step, double entry_time, size_t batch_id )
  {
    QueueItem new_item { step, entry_time, batch_id };
    queue_.insert( upper_bound( queue_.begin(), queue_.end(), new_item, QueueItemCompOp() ), new_item );
  }

  QueueItem pop( double current_time )
  {
    const auto next_batch = queue_pop( current_time );
    assert( current_time >= next_batch.entry_time );
    assert( current_time >= next_start_time_ );
    assert( next_batch.step < delay_array_.size() );
    if ( not delay_only_ ) {
      const double job_end_time = current_time + delay_array_[next_batch.step];
      next_start_time_ = job_end_time;
      return { next_batch.step + 1, job_end_time, next_batch.batch_id };
    } else {
      return { next_batch.step + 1, next_batch.entry_time + delay_array_[next_batch.step], next_batch.batch_id };
    }
  }
};

vector<double> run_simulation( size_t max_iters,
                               const double min_interval,
                               vector<double>& next_event_times,
                               vector<PipelineElement>& workers,
                               vector<size_t>& next_worker_map );

vector<double> single_tier_pipeline( vector<size_t> t1_layers,
                                     size_t in_flight,
                                     size_t max_iters,
                                     double mid_comp,
                                     double last_comp,
                                     double transit,
                                     double latency );

vector<double> two_tier_pipeline( vector<size_t> t1_layers,
                                  size_t in_flight,
                                  size_t max_iters,
                                  double mid_t1_comp,
                                  double last_t1_comp,
                                  double t2_comp,
                                  double t1_to_t2_transit,
                                  double t1_to_t2_latency,
                                  double t2_to_t1_transit,
                                  double t2_to_t1_latency );

vector<double> two_tier_pipeline( vector<size_t> t1_layers,
                                  size_t in_flight,
                                  size_t max_iters,
                                  double mid_t1_comp,
                                  double last_t1_comp,
                                  double t2_comp,
                                  double t1_to_t2_transit,
                                  double t1_to_t2_latency,
                                  double t2_to_t1_transit,
                                  double t2_to_t1_latency )
{
  vector<PipelineElement> workers {};
  vector<size_t> next_worker_map {};
  vector<double> delay_arr;

  const size_t n_layers = accumulate( t1_layers.begin(), t1_layers.end(), 0 );

  for ( size_t i = 0; i < n_layers; i++ ) {
    delay_arr.push_back( i == n_layers - 1 ? last_t1_comp : mid_t1_comp );
    delay_arr.push_back( t1_to_t2_transit );
    delay_arr.push_back( t1_to_t2_latency );
    delay_arr.push_back( t2_comp );
    delay_arr.push_back( t2_to_t1_transit );
    delay_arr.push_back( t2_to_t1_latency );
  }

  for ( size_t i = 0; i < t1_layers.size(); i++ ) {
    workers.push_back( PipelineElement( delay_arr, false ) );
    workers.push_back( PipelineElement( delay_arr, false ) );
    workers.push_back( PipelineElement( delay_arr, true ) );
    workers.push_back( PipelineElement( delay_arr, false ) );
    workers.push_back( PipelineElement( delay_arr, false ) );
    workers.push_back( PipelineElement( delay_arr, true ) );

    for ( size_t j = 0; j < t1_layers[i]; j++ ) {
      next_worker_map.push_back( i * 6 + 0 );
      next_worker_map.push_back( i * 6 + 1 );
      next_worker_map.push_back( i * 6 + 2 );
      next_worker_map.push_back( i * 6 + 3 );
      next_worker_map.push_back( i * 6 + 4 );
      next_worker_map.push_back( i * 6 + 5 );
    }
  }

  assert( next_worker_map.size() == delay_arr.size() );

  vector<double> next_event_times {};
  for ( size_t i = 0; i < workers.size(); i++ ) {
    next_event_times.push_back( workers[i].min_event() );
  }

  for ( size_t i = 0; i < in_flight; i++ ) {
    workers[0].push( 0, 0, i );
  }
  next_event_times[0] = workers[0].min_event();

  const double min_interval
    = ( mid_t1_comp + t2_comp + t1_to_t2_transit + t1_to_t2_latency + t2_to_t1_transit + t2_to_t1_latency ) * n_layers
      - mid_t1_comp + last_t1_comp;

  return run_simulation( max_iters, min_interval, next_event_times, workers, next_worker_map );
}

vector<double> single_tier_pipeline( vector<size_t> t1_layers,
                                     size_t in_flight,
                                     size_t max_iters,
                                     double mid_comp,
                                     double last_comp,
                                     double transit,
                                     double latency )
{
  vector<PipelineElement> workers {};
  vector<size_t> next_worker_map {};
  vector<double> delay_arr;

  const size_t n_layers = accumulate( t1_layers.begin(), t1_layers.end(), 0 );

  for ( size_t i = 0; i < t1_layers.size(); i++ ) {
    for ( size_t j = 0; j < t1_layers[i]; j++ ) {
      delay_arr.push_back( i == t1_layers.size() - 1 and j == t1_layers[i] - 1 ? last_comp : mid_comp );
    }
    delay_arr.push_back( i == t1_layers.size() - 1 ? 0 : transit );
    delay_arr.push_back( latency );
  }

  for ( size_t i = 0; i < t1_layers.size(); i++ ) {
    workers.push_back( PipelineElement( delay_arr, false ) );
    workers.push_back( PipelineElement( delay_arr, false ) );
    workers.push_back( PipelineElement( delay_arr, true ) );

    for ( size_t j = 0; j < t1_layers[i]; j++ ) {
      next_worker_map.push_back( i * 3 + 0 );
    }
    next_worker_map.push_back( i * 3 + 1 );
    next_worker_map.push_back( i * 3 + 2 );
  }

  vector<double> next_event_times {};
  for ( size_t i = 0; i < workers.size(); i++ ) {
    next_event_times.push_back( workers[i].min_event() );
  }

  for ( size_t i = 0; i < in_flight; i++ ) {
    workers[0].push( 0, 0, i );
  }
  next_event_times[0] = workers[0].min_event();

  const double min_interval = ( mid_comp + transit + latency ) * n_layers - mid_comp + last_comp;

  return run_simulation( max_iters, min_interval, next_event_times, workers, next_worker_map );
}

vector<double> run_simulation( size_t max_iters,
                               const double min_interval,
                               vector<double>& next_event_times,
                               vector<PipelineElement>& workers,
                               vector<size_t>& next_worker_map )
{
  double last_arrival = 0;

  vector<double> batch_0_complete {};
  while ( batch_0_complete.size() < max_iters ) {
    auto next_time = min_element( next_event_times.begin(), next_event_times.end() );
    auto wid = distance( next_event_times.begin(), next_time );
    auto next_item = workers[wid].pop( *next_time );
    next_event_times[wid] = workers[wid].min_event();

    if ( next_item.step == next_worker_map.size() ) {
      next_item.step = 0;
    }

    const auto next_wid = next_worker_map[next_item.step];
    workers[next_wid].push( next_item.step, next_item.entry_time, next_item.batch_id );
    next_event_times[next_wid] = workers[next_wid].min_event();

    if ( next_item.batch_id == 0 and next_item.step == 0 ) {
      batch_0_complete.push_back( next_item.entry_time );
      assert( next_item.entry_time * 1.001 > min_interval + last_arrival );
      last_arrival = next_item.entry_time;
    }
  }

  return batch_0_complete;
}
