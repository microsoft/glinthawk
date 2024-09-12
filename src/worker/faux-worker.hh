#pragma once

#include "monitoring/measurement.hh"
#include "worker.hh"

namespace glinthawk::core {

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
class FauxBatchedWorker : public BatchedWorker<ModelConfig, ComputeKernel>
{
protected:
  using _base_ = BatchedWorker<ModelConfig, ComputeKernel>;
  void handle_tier_router_event() override;
  Measurement& __faux_stats__ { global_measurement() };

public:
  /// \brief Construct a new Worker object
  ///
  /// \param worker_address The address of the worker
  /// \param coordinator_address The address of the coordinator
  /// \param model_root The root directory of the model
  FauxBatchedWorker( const net::Address& worker_address,
                     const net::Address& coordinator_address,
                     const std::filesystem::path& model_root )
    : _base_( worker_address, coordinator_address, model_root )
  {
  }

  ~FauxBatchedWorker() = default;
};

template<typename ModelConfig, typename ComputeKernel>
requires models::llama2::ModelConfig<ModelConfig>
         && compute::KernelConcept<ComputeKernel, models::BatchedInferenceState<ModelConfig>>
void FauxBatchedWorker<ModelConfig, ComputeKernel>::handle_tier_router_event()
{
  this->tier_router_->event_fd().read_event();

  models::BatchedInferenceState<ModelConfig> state;

  while ( this->tier_router_->pop( state ) ) {
    __faux_stats__.increment<Counters::StatesProcessed>( state.batch_size() );

    // If tier router emits a state not hosted in this slice, route it back to the first node in this slice
    if ( not _base_::slice_hosting_table_[state.next_layer()][util::to_underlying( state.next_stage() )] ) {
      const bool first_parent_derived_ = _base_::first_parent_;
      DCHECK( first_parent_derived_ ) << "A state not belonging to this slice should only surface in the first parent";

      // Starting place should always be tier 0, rank 0. Otherwise, we have to deal with shards
      state.set_next_tier( 0 );
      state.set_next_rank( 0 );
      state.set_next_stage( models::InferenceStage::PreAttention );
      state.set_next_layer( 0 );

      // Correct activation/queries/kv data based on stage and layer
      state.deallocate_activations();
      state.deallocate_queries();
      state.deallocate_kvs();

      // Move token position and check for finish
      for ( size_t i = 0; i < state.batch_size(); i++ ) {
        state.set_token_pos( i, state.token_pos( i ) + 1 );
        if ( state.check_finished( i ) ) {
          // Discarding the prompt entry is left to the caller, we just set the finished flag here
          state.set_finished( i );
        }
      }

      // Send it back to tier router
      _base_::handle_batch_inference_state( std::move( state ) );

    } else {

      const auto next_worker = _base_::find_next_worker( _base_::route_set_.at( state.route_id() ), state );
      auto peer_it = _base_::peers_.find( next_worker );

      // are we connected to this?
      if ( peer_it == _base_::peers_.end() ) {
        LOG( INFO ) << "Connecting to peer at " << next_worker.to_string();
        net::TCPSocket socket;
        socket.connect( next_worker );
        socket.set_blocking( false );

        std::tie( peer_it, std::ignore )
          = _base_::peers_.emplace( std::piecewise_construct,
                                    std::forward_as_tuple( next_worker ),
                                    std::forward_as_tuple( next_worker, std::move( socket ) ) );

        _base_::setup_peer( peer_it );
      }

      peer_it->second.outgoing_states.push_back( std::move( state ) );
    }
  }
}

} // namespace glinthawk::core
