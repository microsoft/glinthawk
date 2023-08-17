#include "base.hh"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda_fp16.h>

#include <glog/logging.h>

using namespace std;
using namespace glinthawk::models::llama2;

/* MODEL CONFIG */

namespace {

template<typename DType>
DType* _advance_pointer( DType*& ptr, const size_t size )
{
  auto old = ptr;
  ptr += size;
  return old;
}

}

Config::Config( const filesystem::path& config_file )
{
  ifstream fin { config_file, ios::binary };
  CHECK( fin ) << "Failed to open config file: " << config_file;

  fin.read( reinterpret_cast<char*>( this ), sizeof( *this ) );

  vocab_size = abs( vocab_size );

  CHECK_GT( dim, 0 ) << "Transformer dimension must be positive.";
  CHECK_GT( hidden_dim, 0 ) << "FFN hidden dimension must be positive.";
  CHECK_GT( n_layers, 0 ) << "Number of layers must be positive.";
  CHECK_GT( n_heads, 0 ) << "Number of query heads must be positive.";
  CHECK_GT( n_kv_heads, 0 ) << "Number of key/value heads must be positive.";
  CHECK_GT( vocab_size, 0 ) << "Vocabulary size must be positive.";
  CHECK_GT( seq_len, 0 ) << "Sequence length must be positive.";

  LOG( INFO ) << "Loaded config: " << to_string();
}

string Config::to_string() const
{
  ostringstream oss;
  oss << "{ ";
  oss << "dim: " << dim << ", ";
  oss << "hidden_dim: " << hidden_dim << ", ";
  oss << "n_layers: " << n_layers << ", ";
  oss << "n_heads: " << n_heads << ", ";
  oss << "n_kv_heads: " << n_kv_heads << ", ";
  oss << "vocab_size: " << vocab_size << ", ";
  oss << "seq_len: " << seq_len;
  oss << " }";
  return oss.str();
}

/* VOCABULARY */

Vocabulary::Vocabulary( const std::filesystem::path& vocabulary_path )
{
  ifstream fin { vocabulary_path, ios::binary };
  int len = 0;

  int i;
  for ( i = 0;; i++ ) {
    if ( not fin.read( reinterpret_cast<char*>( &len ), sizeof( int ) ) ) {
      break;
    }

    CHECK_GT( len, 0 ) << "Vocabulary entry length must be positive.";

    string val;
    val.resize( len );
    CHECK( fin.read( val.data(), val.length() ) ) << "Failed to read vocabulary entry.";

    token_to_word_.push_back( val );
    word_to_token_.emplace( val, i );
  }

  LOG( INFO ) << "Loaded vocabulary of size " << i << " from " << vocabulary_path;
}

string Vocabulary::get_word( int token ) const
{
  CHECK_GE( token, 0 ) << "Token index must be non-negative.";
  CHECK_LT( token, token_to_word_.size() ) << "Token index out of bounds.";
  return token_to_word_[token];
}

int Vocabulary::get_token( const string& word ) const
{
  auto it = word_to_token_.find( word );
  CHECK( it != word_to_token_.end() ) << "Unknown word: " << word;
  return it->second;
}

/* BASE WEIGHTS */

template<typename DType>
BaseWeights<DType>::BaseWeights( const Config& config, const DType* model )
{
  auto ptr = model;
  this->token_embedding_table = _advance_pointer( ptr, config.vocab_size * config.dim );

  const int head_size = config.dim / config.n_heads;

  this->rms_final_weight = _advance_pointer( ptr, config.dim );
  this->freq_cis_real = _advance_pointer( ptr, config.seq_len * head_size / 2 );
  this->freq_cis_imag = _advance_pointer( ptr, config.seq_len * head_size / 2 );

  // TODO shared_weights is assumed to be true, fix
  // wcls = true ? token_embedding_table : ( ptr += config.seq_len * head_size / 2 );
  this->wcls = token_embedding_table;
}

/* LAYER WEIGHTS */

template<typename DType>
LayerWeights<DType>::LayerWeights( const Config& config, const DType* model )
{
  auto ptr = model;

  // base pointers
  this->rms_att_weight = _advance_pointer( ptr, config.dim );
  this->rms_ffn_weight = _advance_pointer( ptr, config.dim );
  this->wq = _advance_pointer( ptr, config.dim * config.dim );
  this->wk = _advance_pointer( ptr, config.dim * config.dim );
  this->wv = _advance_pointer( ptr, config.dim * config.dim );
  this->wo = _advance_pointer( ptr, config.dim * config.dim );
  this->w1 = _advance_pointer( ptr, config.dim * config.hidden_dim );
  this->w2 = _advance_pointer( ptr, config.dim * config.hidden_dim );
  this->w3 = _advance_pointer( ptr, config.dim * config.hidden_dim );
}

template<typename DType>
size_t LayerWeights<DType>::layer_size( const Config& config )
{
  return sizeof( DType ) * ( 2 * config.dim + 4 * config.dim * config.dim + 3 * config.dim * config.hidden_dim );
}

/* RUN STATE */

template<typename DType>
RunState<DType>::RunState( const Config& config, DType* buffer )
  : buffer_( buffer )
  , x( buffer_ )
  , xb( buffer_ + config.dim )
  , xb2( xb + config.dim )
  , q( xb2 + config.dim )
  , k( q + config.dim )
  , v( k + config.dim )
  , hb( v + config.dim )
  , hb2( hb + config.hidden_dim )
  , att( hb2 + config.hidden_dim )
  , logits( att + config.n_heads * config.seq_len )
  , temp_softmax( logits + config.vocab_size )
{
}

template<typename DType>
size_t RunState<DType>::state_size( const Config& config )
{
  return sizeof( DType )
         * ( config.dim * 5 + config.hidden_dim * 2 + config.n_heads * config.seq_len + config.vocab_size
             + config.n_heads );
}

/* KV CACHE */

template<typename DType>
KVCache<DType>::KVCache( const Config& config, DType* buffer, const int32_t start_layer, const int32_t end_layer )
  : start_layer_( start_layer )
  , end_layer_( end_layer == -1 ? config.n_layers - 1 : end_layer )
  , buffer_( buffer )
  , seq_len_( config.seq_len )
  , dim_( config.dim )
  , n_layers_( end_layer_ - start_layer_ + 1 )
  , head_size_( config.dim / config.n_heads )
{
}

template<typename DType>
size_t KVCache<DType>::cache_size( const Config& config, const int32_t start_layer, const int32_t end_layer )
{
  return sizeof( DType ) * config.seq_len * config.dim * 2 * ( end_layer - start_layer + 1 );
}

template<typename DType>
DType* KVCache<DType>::key( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_ + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_;
}

template<typename DType>
DType* KVCache<DType>::value( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_ + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_ + dim_;
}

/* BaseLlama2 */

template<typename DType>
BaseLlama2<DType>::BaseLlama2( const Config& config,
                               unique_ptr<DType, void ( * )( DType* )>&& base_weights,
                               unique_ptr<DType, void ( * )( DType* )>&& layers_weights,
                               unique_ptr<DType, void ( * )( DType* )>&& run_state,
                               unique_ptr<DType, void ( * )( DType* )>&& kv_cache,
                               const int32_t start_layer,
                               const int32_t end_layer )
  : base_weights_buffer_( move( base_weights ) )
  , layers_buffer_( move( layers_weights ) )
  , run_state_buffer_( move( run_state ) )
  , kv_cache_buffer_( move( kv_cache ) )
  , config_( config )
  , start_layer_num_( start_layer )
  , end_layer_num_( end_layer == -1 ? config_.n_layers - 1 : end_layer )
  , state_( config_, run_state_buffer_.get() )
  , kv_cache_( config_, kv_cache_buffer_.get(), start_layer_num_, end_layer_num_ )
  , base_weights_( config_, base_weights_buffer_.get() )
  , layer_weights_( [&] {
    CHECK_GE( start_layer_num_, 0 ) << "Start layer must be non-negative.";
    CHECK_LT( end_layer_num_, config_.n_layers ) << "End layer must be less than the number of layers.";
    CHECK_LE( start_layer_num_, end_layer_num_ ) << "Start layer must be less than or equal to end layer.";

    std::vector<LayerWeights<DType>> layers {};
    layers.resize( config_.n_layers );

    const size_t layer_size = LayerWeights<DType>::layer_size( config_ );
    auto ptr = layers_buffer_.get();

    for ( int i = start_layer_num_; i <= end_layer_num_; i++ ) {
      layers[i] = LayerWeights { config_, ptr + ( i - start_layer_num_ ) * layer_size };
    }

    return layers;
  }() )
{
}

namespace glinthawk::models::llama2 {
template class RunState<float>;
template class BaseWeights<float>;
template class LayerWeights<float>;
template class KVCache<float>;
template class BaseLlama2<float>;

template class RunState<__half>;
template class BaseWeights<__half>;
template class LayerWeights<__half>;
template class KVCache<__half>;
template class BaseLlama2<__half>;
}
