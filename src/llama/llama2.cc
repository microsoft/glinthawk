#include "llama2.hh"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

Llama2::Llama2( const filesystem::path& weights_path ) { init_weights( weights_path ); }

template<class T, size_t alignment>
unique_ptr<T[]> allocate_memory_aligned( size_t size )
{
  GlobalScopeTimer<Timer::Category::MemoryAllocation> _;

  void* ptr = aligned_alloc( alignment, size * sizeof( T ) );

  if ( not ptr ) {
    throw runtime_error( "Failed to allocate memory." );
  }

  return unique_ptr<T[]>( static_cast<T*>( ptr ) );
}

void Llama2::init_weights( const filesystem::path& weights_path )
{
  ifstream weights_file { weights_path, ios::binary };

  if ( not weights_file.good() ) {
    throw runtime_error( "Could not open weights file: " + weights_path.string() );
  }

  // get weights_file size
  weights_file.seekg( 0, ios::end );
  const auto weights_file_size = weights_file.tellg();
  weights_file.seekg( 0, ios::beg );

  LOG( INFO ) << "Weights file size: " << weights_file_size << " bytes";

  // allocate the buffer
  weights_buffer_ = allocate_memory_aligned<float, 64>( weights_file_size / sizeof( float ) );

  // read the weights
  {
    GlobalScopeTimer<Timer::Category::DiskIO> _;

    if ( weights_file.read( reinterpret_cast<char*>( weights_buffer_.get() ), weights_file_size ) ) {
      LOG( INFO ) << "Read weights file successfully.";
    } else {
      throw runtime_error( "Failed to read the weights file." );
    }
  }

  // load the configuration
  memcpy( &config_, weights_buffer_.get(), sizeof( Config ) );

  // initialize TransformerWeights
  float* ptr = weights_buffer_.get();

  weights_.token_embedding_table = ( ptr += sizeof( Config ) / sizeof( float ) );
  weights_.rms_att_weight = ( ptr += config_.vocab_size * config_.dim );

  weights_.wq = ( ptr += config_.n_layers * config_.dim );
  weights_.wk = ( ptr += config_.n_layers * config_.dim * config_.dim );
  weights_.wv = ( ptr += config_.n_layers * config_.dim * config_.dim );
  weights_.wo = ( ptr += config_.n_layers * config_.dim * config_.dim );

  weights_.rms_ffn_weight = ( ptr += config_.n_layers * config_.dim * config_.dim );

  weights_.w1 = ( ptr += config_.n_layers * config_.dim );
  weights_.w2 = ( ptr += config_.n_layers * config_.dim * config_.hidden_dim );
  weights_.w3 = ( ptr += config_.n_layers * config_.hidden_dim * config_.dim );

  weights_.rms_final_weight = ( ptr += config_.n_layers * config_.dim * config_.hidden_dim );
  weights_.freq_cis_real = ( ptr += config_.dim );

  const int head_size = config_.dim / config_.n_heads;
  weights_.freq_cis_imag = ( ptr += config_.seq_len * head_size / 2 );
  weights_.wcls
    = ( config_.vocab_size > 0 ) ? weights_.token_embedding_table : ( ptr += config_.seq_len * head_size / 2 );
}

void Llama2::init_state()
{
  // allocate the state
  state_.x = allocate_memory_aligned<float, 64>( config_.dim );
  state_.xb = allocate_memory_aligned<float, 64>( config_.dim );
  state_.xb2 = allocate_memory_aligned<float, 64>( config_.dim );
  state_.hb = allocate_memory_aligned<float, 64>( config_.hidden_dim );
  state_.hb2 = allocate_memory_aligned<float, 64>( config_.hidden_dim );
  state_.q = allocate_memory_aligned<float, 64>( config_.dim );
  state_.k = allocate_memory_aligned<float, 64>( config_.dim );
  state_.v = allocate_memory_aligned<float, 64>( config_.dim );
  state_.att = allocate_memory_aligned<float, 64>( config_.n_heads * config_.seq_len );
  state_.logits = allocate_memory_aligned<float, 64>( config_.vocab_size );

  state_.key_cache = allocate_memory_aligned<float, 64>( config_.n_layers * config_.seq_len * config_.dim );
  state_.value_cache = allocate_memory_aligned<float, 64>( config_.n_layers * config_.seq_len * config_.dim );
}
