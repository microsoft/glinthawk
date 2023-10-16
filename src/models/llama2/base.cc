#include "base.hh"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifdef GLINTHAWK_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

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

Config::Config( const filesystem::path& config_file,
                const int32_t start_layer,
                const int32_t end_layer,
                uint64_t concurrency_limit_ )
  : concurrency_limit( concurrency_limit_ )
{
  ifstream fin { config_file, ios::binary };
  CHECK( fin ) << "Failed to open config file: " << config_file;

  string raw_config;
  raw_config.resize( config_size() );

  fin.read( raw_config.data(), config_size() );
  if ( fin.gcount() != static_cast<streamsize>( config_size() ) ) {
    LOG( FATAL ) << "Failed to read config file: " << config_file;
  }

  auto ptr = raw_config.data();

  this->dim = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->hidden_dim = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->n_layers = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->n_heads = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->n_kv_heads = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );

  // if vocab size is negative, that means that wcls is present
  const auto original_vocab_size = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );

  this->vocab_size = abs( original_vocab_size );
  this->seq_len = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );

  if ( original_vocab_size < 0 ) {
    wcls_present = true;
  }

  this->gqa_size = this->n_heads / this->n_kv_heads;
  this->kv_dim = this->dim / this->gqa_size;

  CHECK_GT( dim, 0 ) << "Transformer dimension must be positive.";
  CHECK_GT( kv_dim, 0 ) << "key/value dimension must be positive.";
  CHECK_GT( hidden_dim, 0 ) << "FFN hidden dimension must be positive.";
  CHECK_GT( n_layers, 0 ) << "Number of layers must be positive.";
  CHECK_GT( n_heads, 0 ) << "Number of query heads must be positive.";
  CHECK_GT( n_kv_heads, 0 ) << "Number of key/value heads must be positive.";
  CHECK_GT( gqa_size, 0 ) << "GQA sharing rate must be positive.";
  CHECK_GT( vocab_size, 0 ) << "Vocabulary size must be positive.";
  CHECK_GT( seq_len, 0 ) << "Sequence length must be positive.";
  CHECK_GT( concurrency_limit, 0 ) << "Max concurrent inference size must be positive.";

  start_layer_num = start_layer;
  end_layer_num = ( end_layer == -1 ) ? ( n_layers - 1 ) : end_layer;

  CHECK_GE( start_layer_num, 0 ) << "Start layer must be non-negative.";
  CHECK_LT( end_layer_num, n_layers ) << "End layer must be less than the number of layers.";
  CHECK_LE( start_layer_num, end_layer_num ) << "Start layer must be less than or equal to end layer.";

  LOG( INFO ) << "Loaded config: " << to_string();
}

string Config::to_string() const
{
  ostringstream oss;
  oss << "{ ";
  oss << "dim: " << dim << ", ";
  oss << "kv_dim: " << kv_dim << ", ";
  oss << "hidden_dim: " << hidden_dim << ", ";
  oss << "n_layers: " << n_layers << ", ";
  oss << "n_heads: " << n_heads << ", ";
  oss << "n_kv_heads: " << n_kv_heads << ", ";
  oss << "gqa_size: " << gqa_size << ", ";
  oss << "vocab_size: " << vocab_size << ", ";
  oss << "seq_len: " << seq_len << ", ";
  oss << "concurrency_limit: " << concurrency_limit << ", ";
  oss << "wcls_present: " << wcls_present << ", ";
  oss << "start_layer_num: " << start_layer_num << ", ";
  oss << "end_layer_num: " << end_layer_num;
  oss << " }";
  return oss.str();
}

/* VOCABULARY */

Vocabulary::Vocabulary( const filesystem::path& vocabulary_path )
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
  const int head_size = config.dim / config.n_heads;

  auto ptr = model;
  token_embedding_table = _advance_pointer( ptr, config.vocab_size * config.dim );
  rms_final_weight = _advance_pointer( ptr, config.dim );
  freq_cis_real = _advance_pointer( ptr, config.seq_len * head_size / 2 );
  freq_cis_imag = _advance_pointer( ptr, config.seq_len * head_size / 2 );
  wcls = config.wcls_present ? ptr : token_embedding_table;
}

template<typename DType>
size_t BaseWeights<DType>::base_size( const Config& config )
{
  return sizeof( DType )
         * ( config.vocab_size * config.dim + config.dim + config.seq_len * config.dim / config.n_heads
             + ( config.wcls_present ? ( config.vocab_size * config.dim ) : 0 ) );
}

/* LAYER WEIGHTS */

template<typename DType>
LayerWeights<DType>::LayerWeights( const Config& config, const DType* model )
{
  auto ptr = model;

  // base pointers
  rms_att_weight = _advance_pointer( ptr, config.dim );
  this->wq = _advance_pointer( ptr, config.dim * config.dim );
  this->wk = _advance_pointer( ptr, config.dim * config.kv_dim );
  this->wv = _advance_pointer( ptr, config.dim * config.kv_dim );
  this->wo = _advance_pointer( ptr, config.dim * config.dim );
  this->rms_ffn_weight = _advance_pointer( ptr, config.dim );
  this->w1 = _advance_pointer( ptr, config.dim * config.hidden_dim );
  this->w2 = _advance_pointer( ptr, config.dim * config.hidden_dim );
  this->w3 = _advance_pointer( ptr, config.dim * config.hidden_dim );
}

template<typename DType>
size_t LayerWeights<DType>::layer_size( const Config& config )
{
  return sizeof( DType ) * ( 2 * config.dim + 2 * config.dim * config.dim + 2 * config.dim * config.kv_dim + 3 * config.dim * config.hidden_dim );
}

/* RUN STATE */

// TODO: optimize run state memory usage
template<typename DType>
RunState<DType>::RunState( const Config& config, DType* buffer )
  : buffer_( buffer )
  , x( buffer_ )
  , xb( buffer_ + config.dim * config.concurrency_limit )
  , xb2( xb + config.dim * config.concurrency_limit )
  , q( xb2 + config.dim * config.concurrency_limit )
  , k( q + config.dim * config.concurrency_limit )
  , v( k + config.kv_dim * config.concurrency_limit )
  , hb( v + config.kv_dim * config.concurrency_limit )
  , hb2( hb + config.hidden_dim * config.concurrency_limit )
  , att( hb2 + config.hidden_dim * config.concurrency_limit )
  , logits( att + config.n_heads * config.seq_len * config.concurrency_limit )
  , temp_softmax( logits + config.vocab_size * config.concurrency_limit )
  , rng_state( reinterpret_cast<curandState*>(temp_softmax + config.n_heads * config.concurrency_limit ) )
{
}

template<typename DType>
size_t RunState<DType>::state_size( const Config& config )
{
  return sizeof( DType ) * config.concurrency_limit
           * ( config.dim * 4 + config.kv_dim * 2 + config.hidden_dim * 2 + config.n_heads * config.seq_len + config.vocab_size
               + config.n_heads) + sizeof(curandState) * config.concurrency_limit * config.vocab_size;
}

/* InferenceContext */

template<typename DType>
size_t InferenceContext<DType>::context_size( const Config& config )
{
  return sizeof( DType ) * config.seq_len * config.kv_dim * 2 * config.n_layers_loaded();
}

template<typename DType>
DType* InferenceContext<DType>::key( const Config& config, int layer_num, const int token_num, const int head_num )
{
  return buffer_ + token_num * ( config.n_layers_loaded() * config.kv_dim * 2 )
         + ( layer_num - config.start_layer_num ) * ( config.kv_dim * 2 ) + head_num * ( config.dim / config.n_heads );
}

template<typename DType>
DType* InferenceContext<DType>::value( const Config& config, int layer_num, const int token_num, const int head_num )
{
  return key( config, layer_num, token_num, head_num ) + config.kv_dim;
}

/* BaseLlama2 */

template<typename DType, typename Context>
BaseLlama2<DType, Context>::BaseLlama2( const Config& config,
                                        unique_ptr<DType, void ( * )( DType* )>&& base_weights,
                                        unique_ptr<DType, void ( * )( DType* )>&& layers_weights,
                                        unique_ptr<DType, void ( * )( DType* )>&& run_state )
  : base_weights_buffer_( move( base_weights ) )
  , layers_buffer_( move( layers_weights ) )
  , run_state_buffer_( move( run_state ) )
  , config_( config )
  , state_( config_, run_state_buffer_.get() )
  , base_weights_( config_, base_weights_buffer_.get() )
  , layer_weights_( [&] {
    vector<LayerWeights<DType>> layers {};
    layers.resize( config_.n_layers );

    const size_t layer_size = LayerWeights<DType>::layer_size( config_ );
    auto ptr = layers_buffer_.get();

    for ( auto i = config_.start_layer_num; i <= config_.end_layer_num; i++ ) {
      layers[i] = LayerWeights { config_,
                                 reinterpret_cast<DType*>( reinterpret_cast<uint8_t*>( ptr )
                                                           + ( i - config_.start_layer_num ) * layer_size ) };
    }

    return layers;
  }() )
{
}

namespace glinthawk::models::llama2 {
template class RunState<float>;
template class BaseWeights<float>;
template class LayerWeights<float>;
template class InferenceContext<float>;

#ifdef GLINTHAWK_CUDA_ENABLED
template class RunState<__half>;
template class BaseWeights<__half>;
template class LayerWeights<__half>;
template class InferenceContext<__half>;

// forward declare Context
namespace cuda {
template<typename DType>
class Context;
}

template class BaseLlama2<__half, cuda::Context<__half>>;
#endif
}
