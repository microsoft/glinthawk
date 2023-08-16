#include "base.hh"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <glog/logging.h>

using namespace std;
using namespace glinthawk::models::llama2;

/* MODEL CONFIG */

Config::Config( const filesystem::path& weights_path )
{
  ifstream fin { weights_path, ios::binary };
  CHECK( fin ) << "Failed to open weights file: " << weights_path;

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

Vocabulary::Vocabulary( const Config& config, const std::filesystem::path& vocabulary_path )
{
  ifstream fin { vocabulary_path, ios::binary };
  int len = 0;

  for ( int i = 0; i < config.vocab_size; i++ ) {
    CHECK( fin.read( reinterpret_cast<char*>( &len ), sizeof( int ) ) ) << "Failed to read vocabulary entry length.";
    CHECK_GT( len, 0 ) << "Vocabulary entry length must be positive.";

    string val;
    val.resize( len );
    CHECK( fin.read( val.data(), val.length() ) ) << "Failed to read vocabulary entry.";

    token_to_word_.push_back( val );
    word_to_token_.emplace( val, i );
  }

  LOG( INFO ) << "Loaded vocabulary of size " << config.vocab_size << " from " << vocabulary_path;
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
  this->token_embedding_table = ptr;

  // skip over all the layer weights
  ptr += config.vocab_size * config.dim
         + config.n_layers * ( 2 * config.dim + 4 * config.dim * config.dim + 3 * config.dim * config.hidden_dim );

  const int head_size = config.dim / config.n_heads;

  this->rms_final_weight = ptr;
  this->freq_cis_real = ( ptr += config.dim );
  this->freq_cis_imag = ( ptr += config.seq_len * head_size / 2 );

  // TODO shared_weights is assumed to be true, fix
  // wcls = true ? token_embedding_table : ( ptr += config.seq_len * head_size / 2 );
  this->wcls = token_embedding_table;
}

/* LAYER WEIGHTS */

template<typename DType>
LayerWeights<DType>::LayerWeights( const Config& config, const DType* model, const int layer_num )
{
  auto ptr = model;

  // base pointers
  DType* const base_rms_att_weight = ( ptr += config.vocab_size * config.dim );
  DType* const base_wq = ( ptr += config.n_layers * config.dim );
  DType* const base_wk = ( ptr += config.n_layers * config.dim * config.dim );
  DType* const base_wv = ( ptr += config.n_layers * config.dim * config.dim );
  DType* const base_wo = ( ptr += config.n_layers * config.dim * config.dim );
  DType* const base_rms_ffn_weight = ( ptr += config.n_layers * config.dim * config.dim );
  DType* const base_w1 = ( ptr += config.n_layers * config.dim );
  DType* const base_w2 = ( ptr += config.n_layers * config.dim * config.hidden_dim );
  DType* const base_w3 = ( ptr += config.n_layers * config.hidden_dim * config.dim );

  this->rms_att_weight = base_rms_att_weight + layer_num * config.dim;
  this->rms_ffn_weight = base_rms_ffn_weight + layer_num * config.dim;
  this->wq = base_wq + layer_num * config.dim * config.dim;
  this->wk = base_wk + layer_num * config.dim * config.dim;
  this->wv = base_wv + layer_num * config.dim * config.dim;
  this->wo = base_wo + layer_num * config.dim * config.dim;
  this->w1 = base_w1 + layer_num * config.dim * config.hidden_dim;
  this->w2 = base_w2 + layer_num * config.hidden_dim * config.dim;
  this->w3 = base_w3 + layer_num * config.hidden_dim * config.dim;
}

/* RUN STATE */

template<typename DType>
RunState<DType>::RunState( const Config& config, DType* buffer, const int32_t start_layer, const int32_t end_layer )
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
  , kv_cache( config, start_layer, end_layer )
{
}

template<typename DType>
size_t RunState<DType>::state_size( const Config& config )
{
  return sizeof( DType )
         * ( config.dim * 5 + config.hidden_dim * 2 + config.n_heads * config.seq_len + config.vocab_size
             + config.n_heads );
}

template<typename DType>
RunState<DType>::KVCache::KVCache( const Config& config,
                                   DType* buffer,
                                   const int32_t start_layer,
                                   const int32_t end_layer )
  : start_layer_( start_layer )
  , end_layer_( end_layer )
  , buffer_( buffer )
  , seq_len_( config.seq_len )
  , dim_( config.dim )
  , n_layers_( end_layer_ - start_layer_ + 1 )
  , head_size_( config.dim / config.n_heads )
{
}

template<typename DType>
DType* RunState<DType>::KVCache::key( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_ + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_;
}

template<typename DType>
DType* RunState<DType>::KVCache::value( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_ + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_ + dim_;
}

template<typename DType>
void RunState<DType>::KVCache::pop()
{
  throw runtime_error( "KVCache::pop() not implemented" );
}
