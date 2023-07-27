#include "llama2.hh"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

string Llama2::Config::to_string() const
{
  ostringstream oss;
  oss << "{\n";
  oss << "  dim: " << dim << ',' << endl;
  oss << "  hidden_dim: " << hidden_dim << ',' << endl;
  oss << "  n_layers: " << n_layers << ',' << endl;
  oss << "  n_heads: " << n_heads << ',' << endl;
  oss << "  n_kv_heads: " << n_kv_heads << ',' << endl;
  oss << "  vocab_size: " << vocab_size << ',' << endl;
  oss << "  seq_len: " << seq_len << endl;
  oss << "}";
  return oss.str();
}

Llama2::Llama2( const std::filesystem::path& tokenizer_path, const filesystem::path& weights_path )
{
  init_weights( weights_path ); // also initializes config_
  init_vocabulary( tokenizer_path );
  init_state();
}

template<class T, size_t alignment>
unique_ptr<T[]> make_unique_aligned( size_t size )
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
  weights_buffer_ = make_unique_aligned<float, 64>( weights_file_size / sizeof( float ) );

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
  LOG( INFO ) << "Configuration: \n" << config_.to_string();

  // XXX fucking ugly hack
  const bool shared_weights = config_.vocab_size > 0;
  config_.vocab_size = abs( config_.vocab_size );

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
  weights_.wcls = shared_weights ? weights_.token_embedding_table : ( ptr += config_.seq_len * head_size / 2 );
}

void Llama2::init_vocabulary( const std::filesystem::path& vocabulary_path )
{
  ifstream fin { vocabulary_path, ios::binary };
  CHECK_GT( config_.vocab_size, 0 ) << "Vocabulary size must be positive.";

  int len = 0;

  for ( int i = 0; i < config_.vocab_size; i++ ) {
    fin.read( reinterpret_cast<char*>( &len ), sizeof( int ) );
    CHECK_GT( len, 0 ) << "Vocabulary entry length must be positive.";
    string val;
    val.resize( len );
    fin.read( val.data(), val.length() );
    vocabulary_.push_back( move( val ) );
  }
}

void Llama2::init_state()
{
  // allocate the state
  state_.x = make_unique_aligned<float, 64>( config_.dim );
  state_.xb = make_unique_aligned<float, 64>( config_.dim );
  state_.xb2 = make_unique_aligned<float, 64>( config_.dim );
  state_.hb = make_unique_aligned<float, 64>( config_.hidden_dim );
  state_.hb2 = make_unique_aligned<float, 64>( config_.hidden_dim );
  state_.q = make_unique_aligned<float, 64>( config_.dim );
  state_.k = make_unique_aligned<float, 64>( config_.dim );
  state_.v = make_unique_aligned<float, 64>( config_.dim );
  state_.att = make_unique_aligned<float, 64>( config_.n_heads * config_.seq_len );
  state_.logits = make_unique_aligned<float, 64>( config_.vocab_size );

  state_.key_cache = make_unique_aligned<float, 64>( config_.n_layers * config_.seq_len * config_.dim );
  state_.value_cache = make_unique_aligned<float, 64>( config_.n_layers * config_.seq_len * config_.dim );
}
