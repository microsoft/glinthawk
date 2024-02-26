#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "models/types.hh"
#include "net/address.hh"
#include "util/digest.hh"

#include "state.hh"

namespace glinthawk::models {

class InferenceState
{
private:
  struct __attribute__( ( packed ) ) Data
  {
    PromptID prompt_id_ {};
    RouteID route_id_ {};
    ModelID model_id_ { 0 };

    uint32_t token_ { 1 };
    uint32_t token_pos_ { 0 };
    uint32_t next_layer_ { 0 };
    InferenceStage next_stage_ { InferenceStage::PreAttention };
    uint32_t prompt_length_ { 1 };
    float temperature_ { 0.0f };
    bool finished_ { false };
    bool last_on_cpu_ { false };

    // XXX temporary hack for getting some measurements
    uint64_t timestamp_ { 0 };
    uint64_t loop_start_timestamp_ { 0 };
    uint64_t time_in_node_ { 0 };
    uint64_t batch_timestamp_ { 0 };
    bool batch_last_ { false };

    DataType dtype_ { DataType::Float32 };

    Data() = default;
    Data( const char* buffer ) { *this = *reinterpret_cast<const Data*>( buffer ); }

    Data( const Data& ) = default;
    Data& operator=( const Data& ) = default;

  } data_ {};

  DataBuffer activations_ {};

  // mapping from layer to worker address for this inference state, only local and does not get passed along
  std::map<std::pair<uint32_t, InferenceStage>, glinthawk::net::Address> layer_workers_ {};

  size_t serialized_size() const;

public:
  InferenceState() = default;
  InferenceState( const DataType dtype ) { data_.dtype_ = dtype; }
  InferenceState( const std::string_view serialized_state );

  /* movable, not copyable */
  InferenceState( const InferenceState& other ) = delete;
  InferenceState& operator=( const InferenceState& other ) = delete;
  InferenceState( InferenceState&& other ) = default;
  InferenceState& operator=( InferenceState&& other ) = default;

  std::string serialize() const;
  std::string to_string() const;

  PromptID prompt_id() const { return data_.prompt_id_; }
  RouteID route_id() const { return data_.route_id_; }
  ModelID model_id() const { return data_.model_id_; }

  uint32_t token() const { return data_.token_; }
  uint32_t token_pos() const { return data_.token_pos_; }
  uint32_t next_layer() const { return data_.next_layer_; }
  InferenceStage next_stage() const { return data_.next_stage_; }
  uint32_t prompt_length() const { return data_.prompt_length_; }
  float temperature() const { return data_.temperature_; }
  bool finished() const { return data_.finished_; }
  bool last_on_cpu() const { return data_.last_on_cpu_; }
  const decltype( layer_workers_ )& layer_workers() const { return layer_workers_; }
  DataType dtype() const { return data_.dtype_; }
  uint64_t timestamp() const { return data_.timestamp_; }
  uint64_t loop_start_timestamp() const { return data_.loop_start_timestamp_; }
  uint64_t time_in_node() const { return data_.time_in_node_; }
  uint64_t batch_timestamp() const { return data_.batch_timestamp_; }
  bool batch_last() const { return data_.batch_last_; }

  DataBuffer& activations() { return activations_; }
  const DataBuffer& activations() const { return activations_; }

  void set_prompt_id( const PromptID prompt_id ) { data_.prompt_id_ = prompt_id; }
  void set_route_id( const RouteID route_id ) { data_.route_id_ = route_id; }
  void set_model_id( const ModelID model_id ) { data_.model_id_ = model_id; }
  void set_token( const uint32_t token ) { data_.token_ = token; }
  void set_token_pos( const uint32_t token_pos ) { data_.token_pos_ = token_pos; }
  void set_next_layer( const uint32_t next_layer ) { data_.next_layer_ = next_layer; }
  void set_next_stage( const InferenceStage next_stage ) { data_.next_stage_ = next_stage; }
  void set_prompt_length( const uint32_t prompt_length ) { data_.prompt_length_ = prompt_length; }
  void set_temperature( const float temperature ) { data_.temperature_ = temperature; }
  void set_activations( DataBuffer&& activations ) { activations_ = std::move( activations ); }
  void set_layer_workers( const decltype( layer_workers_ )& layer_workers ) { layer_workers_ = layer_workers; }
  void set_finished() { data_.finished_ = true; }
  void set_last_on_cpu( const bool last_on_cpu ) { data_.last_on_cpu_ = last_on_cpu; }
  void set_timestamp( const uint64_t timestamp ) { data_.timestamp_ = timestamp; }
  void set_loop_start_timestamp( const uint64_t t ) { data_.loop_start_timestamp_ = t; }
  void set_time_in_node( const uint64_t time_in_node ) { data_.time_in_node_ = time_in_node; }
  void set_batch_timestamp( const uint64_t batch_timestamp ) { data_.batch_timestamp_ = batch_timestamp; }
  void set_batch_last( const bool batch_last ) { data_.batch_last_ = batch_last; }

  glinthawk::net::Address next_worker() const;

  void loop_till_next_worker( const uint32_t n_layers );
  void erase_from_workers( const uint32_t next_layer, const InferenceStage next_stage );
};

} // namespace glinthawk::models

std::ostream& operator<<( std::ostream& os, const glinthawk::DataType& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::DataBuffer& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::models::InferenceStage& v );
std::ostream& operator<<( std::ostream& os, const glinthawk::models::InferenceState& v );
