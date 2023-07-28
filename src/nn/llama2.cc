#include "llama2.hh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "nn/ops.hh"
#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

Llama2::Config::Config( const filesystem::path& weights_path )
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

string Llama2::Config::to_string() const
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

Llama2::Vocabulary::Vocabulary( const Config& config, const std::filesystem::path& vocabulary_path )
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

string Llama2::Vocabulary::get_word( int token ) const
{
  CHECK_GE( token, 0 ) << "Token index must be non-negative.";
  CHECK_LT( token, token_to_word_.size() ) << "Token index out of bounds.";
  return token_to_word_[token];
}

int Llama2::Vocabulary::get_token( const string& word ) const
{
  auto it = word_to_token_.find( word );
  CHECK( it != word_to_token_.end() ) << "Unknown word: " << word;
  return it->second;
}

Llama2::Llama2( const std::filesystem::path& tokenizer_path, const filesystem::path& weights_path )
  : config_( weights_path )
  , weights_( config_, weights_path )
  , vocabulary_( config_, tokenizer_path )
  , state_( config_ )
{
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

Llama2::TransformerWeights::TransformerWeights( const Config& config, const filesystem::path& weights_path )
{
  ifstream weights_file { weights_path, ios::binary };

  CHECK( weights_file ) << "Failed to open weights file: " << weights_path;

  // get weights_file size
  weights_file.seekg( 0, ios::end );
  auto weights_file_size = weights_file.tellg();
  weights_file.seekg( 0, ios::beg );

  // TODO not great, but want to keep compatibility with the .bin files; should be removed in the future
  CHECK_GT( weights_file_size, sizeof( Config ) ) << "Weights file is empty.";
  weights_file.seekg( sizeof( Config ), ios::beg );
  weights_file_size -= sizeof( Config );

  LOG( INFO ) << "Weights file size: " << weights_file_size << " bytes";

  // allocate the buffer
  buffer_ = make_unique_aligned<float, 64>( weights_file_size / sizeof( float ) );

  // read the weights
  {
    GlobalScopeTimer<Timer::Category::DiskIO> _;
    CHECK( weights_file.read( reinterpret_cast<char*>( buffer_.get() ), weights_file_size ) )
      << "Failed to read weights file.";
  }

  // initialize TransformerWeights
  float* ptr = buffer_.get();

  layers = make_unique<TransformerWeights::LayerWeights[]>( config.n_layers );
  token_embedding_table = ptr;

  auto rms_att_weight = ( ptr += config.vocab_size * config.dim );
  auto wq = ( ptr += config.n_layers * config.dim );
  auto wk = ( ptr += config.n_layers * config.dim * config.dim );
  auto wv = ( ptr += config.n_layers * config.dim * config.dim );
  auto wo = ( ptr += config.n_layers * config.dim * config.dim );
  auto rms_ffn_weight = ( ptr += config.n_layers * config.dim * config.dim );
  auto w1 = ( ptr += config.n_layers * config.dim );
  auto w2 = ( ptr += config.n_layers * config.dim * config.hidden_dim );
  auto w3 = ( ptr += config.n_layers * config.hidden_dim * config.dim );

  for ( int i = 0; i < config.n_layers; i++ ) {
    layers[i].rms_att_weight = rms_att_weight + i * config.dim;
    layers[i].wq = wq + i * config.dim * config.dim;
    layers[i].wk = wk + i * config.dim * config.dim;
    layers[i].wv = wv + i * config.dim * config.dim;
    layers[i].wo = wo + i * config.dim * config.dim;
    layers[i].rms_ffn_weight = rms_ffn_weight + i * config.dim;
    layers[i].w1 = w1 + i * config.dim * config.hidden_dim;
    layers[i].w2 = w2 + i * config.hidden_dim * config.dim;
    layers[i].w3 = w3 + i * config.hidden_dim * config.dim;
  }

  rms_final_weight = ( ptr += config.n_layers * config.dim * config.hidden_dim );
  freq_cis_real = ( ptr += config.dim );

  const int head_size = config.dim / config.n_heads;
  freq_cis_imag = ( ptr += config.seq_len * head_size / 2 );

  // TODO shared_weights is assumed to be true, fix
  // wcls = true ? token_embedding_table : ( ptr += config.seq_len * head_size / 2 );
  wcls = token_embedding_table;
}

Llama2::RunState::RunState( const Config& config )
  : buffer_( make_unique_aligned<float, 64>( sizeof( float ) * config.dim * 6          /* x, xb, xb2, q, k, v */
                                             + sizeof( float ) * config.hidden_dim * 2 /* hb, hb2 */
                                             + sizeof( float ) * config.n_heads * config.seq_len /* att */
                                             + sizeof( float ) * config.vocab_size               /* logits */
                                             ) )
  , kv_cache( config )
{
  auto ptr = buffer_.get();

  x = ptr;
  xb = ( ptr += config.dim );
  xb2 = ( ptr += config.dim );
  q = ( ptr += config.dim );
  k = ( ptr += config.dim );
  v = ( ptr += config.dim );
  hb = ( ptr += config.dim );
  hb2 = ( ptr += config.hidden_dim );
  att = ( ptr += config.hidden_dim );
  logits = ( ptr += config.n_heads * config.seq_len );
}

Llama2::RunState::KVCache::KVCache( const Config& config )
  : buffer_( make_unique<float[]>( sizeof( float ) * config.seq_len * config.n_layers * config.dim * 2 ) )
  , seq_len_( config.seq_len )
  , dim_( config.dim )
  , n_layers_( config.n_layers )
  , head_size_( config.dim / config.n_heads )
{
}

float* Llama2::RunState::KVCache::key( int layer, const int step, const int head )
{
  return buffer_.get() + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_;
}

float* Llama2::RunState::KVCache::value( int layer, const int step, const int head )
{
  return buffer_.get() + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + dim_ + head * head_size_;
}

void Llama2::RunState::KVCache::pop() { throw runtime_error( "KVCache::pop() not implemented" ); }

void Llama2::transformer( const int token )
{
  // a few convenience variables
  float* x = state_.x;
  const int dim = config_.dim;
  const int hidden_dim = config_.hidden_dim;
  const int head_size = dim / config_.n_heads;

  // copy the token embedding into x
  const float* content_row = weights_.token_embedding_table + token * dim;
  memcpy( x, content_row, dim * sizeof( *x ) );

  // pluck out the "pos" row of freq_cis_real and freq_cis_imag
  float* freq_cis_real_row = weights_.freq_cis_real + current_pos_ * head_size / 2;
  float* freq_cis_imag_row = weights_.freq_cis_imag + current_pos_ * head_size / 2;

  for ( int layer_num = 0; layer_num < config_.n_layers; layer_num++ ) {
    const auto& layer_weights = weights_.layers[layer_num];

    // attention rmsnorm
    ops::rmsnorm( state_.xb, x, layer_weights.rms_att_weight, dim );

    // qkv matmuls for this position
    ops::matmul( state_.q, state_.xb, layer_weights.wq, dim, dim );
    ops::matmul( state_.k, state_.xb, layer_weights.wk, dim, dim );
    ops::matmul( state_.v, state_.xb, layer_weights.wv, dim, dim );

    // apply RoPE rotation to the q and k vectors for each head
    for ( int head_num = 0; head_num < config_.n_heads; head_num++ ) {
      // get the q and k vectors for this head
      float* q = state_.q + head_num * head_size;
      float* k = state_.k + head_num * head_size;

      // rotate q and k by the freq_cis_real and freq_cis_imag
      for ( int i = 0; i < head_size; i += 2 ) {
        const float q0 = q[i];
        const float q1 = q[i + 1];
        const float k0 = k[i];
        const float k1 = k[i + 1];
        const float fcr = freq_cis_real_row[i / 2];
        const float fci = freq_cis_imag_row[i / 2];
        q[i] = q0 * fcr - q1 * fci;
        q[i + 1] = q0 * fci + q1 * fcr;
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
      }
    }

    // save key,value at this time step (pos) to our kv cache
    memcpy( state_.kv_cache.key( layer_num, current_pos_ ), state_.k, dim * sizeof( float ) );
    memcpy( state_.kv_cache.value( layer_num, current_pos_ ), state_.v, dim * sizeof( float ) );

    // multihead attention. iterate over all heads
    int head_num;
#pragma omp parallel for private( head_num )
    for ( head_num = 0; head_num < config_.n_heads; head_num++ ) {
      // get the query vector for this head
      const float* q = state_.q + head_num * head_size;

      // attention scores for this head
      float* att = state_.att + head_num * config_.seq_len;

      // iterate over all timesteps, including the current one
      for ( int t = 0; t <= current_pos_; t++ ) {
        // get the key vector for this head and at this timestep
        const float* k = state_.kv_cache.key( layer_num, t, head_num );

        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for ( int i = 0; i < head_size; i++ ) {
          score += q[i] * k[i];
        }
        score /= sqrtf( head_size );

        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      ops::softmax( att, current_pos_ + 1 );

      // weighted sum of the values, store back into xb
      float* xb = state_.xb + head_num * head_size;
      memset( xb, 0, head_size * sizeof( float ) );

      for ( int t = 0; t <= current_pos_; t++ ) {
        // get the value vector for this head and at this timestep
        const float* v = state_.kv_cache.value( layer_num, t, head_num );

        // get the attention weight for this timestep
        const float a = att[t];

        // accumulate the weighted value into xb
        for ( int i = 0; i < head_size; i++ ) {
          xb[i] += a * v[i];
        }
      }
    }
    // end of multihead attention

    // final matmul to get the output of the attention
    ops::matmul( state_.xb2, state_.xb, layer_weights.wo, dim, dim );

    // residual connection back into x
    ops::accum( x, state_.xb2, dim );

    // ffn rmsnorm
    ops::rmsnorm( state_.xb, x, layer_weights.rms_ffn_weight, dim );

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    ops::matmul( state_.hb, state_.xb, layer_weights.w1, dim, hidden_dim );
    ops::matmul( state_.hb2, state_.xb, layer_weights.w3, dim, hidden_dim );

    // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for ( int i = 0; i < hidden_dim; i++ ) {
      state_.hb[i] = state_.hb[i] * ( 1.0f / ( 1.0f + expf( -state_.hb[i] ) ) );
    }

    // elementwise multiply with w3(x)
    for ( int i = 0; i < hidden_dim; i++ ) {
      state_.hb[i] = state_.hb[i] * state_.hb2[i];
    }

    // final matmul to get the output of the ffn
    ops::matmul( state_.xb, state_.hb, layer_weights.w2, hidden_dim, dim );

    // residual connection
    ops::accum( x, state_.xb, dim );
  }

  // final rmsnorm
  ops::rmsnorm( x, x, weights_.rms_final_weight, dim );

  // classifier into logits
  ops::matmul( state_.logits, x, weights_.wcls, config_.dim, config_.vocab_size );
}

string Llama2::next_token()
{
  if ( current_pos_ >= config_.seq_len ) {
    return string {};
  }

  int next_token;

  // forward the transformer to get logits for the next token
  transformer( current_token_ );

  // sample the next token
  if ( temperature_ == 0.0f ) {
    // greedy argmax sampling
    next_token = ops::argmax( state_.logits, config_.vocab_size );
  } else {
    // apply the temperature to the logits
    for ( int q = 0; q < config_.vocab_size; q++ ) {
      state_.logits[q] /= temperature_;
    }

    // apply softmax to the logits to get the probabilities for next token
    ops::softmax( state_.logits, config_.vocab_size );

    // we now want to sample from this distribution to get the next token
    next_token = ops::sample( state_.logits, config_.vocab_size );
  }

  // advance forward
  current_token_ = next_token;
  current_pos_++;

  return vocabulary_.get_word( current_token_ );
}
