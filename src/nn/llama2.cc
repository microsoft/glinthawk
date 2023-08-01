#include "llama2.hh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include "nn/ops.hh"
#include "util/exception.hh"
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

Llama2::BaseWeights::BaseWeights( const Config& config, const float* model )
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

Llama2::LayerWeights::LayerWeights( const Config& config, const float* model, const int layer_num )
{
  auto ptr = model;

  // base pointers
  auto base_rms_att_weight = ( ptr += config.vocab_size * config.dim );
  auto base_wq = ( ptr += config.n_layers * config.dim );
  auto base_wk = ( ptr += config.n_layers * config.dim * config.dim );
  auto base_wv = ( ptr += config.n_layers * config.dim * config.dim );
  auto base_wo = ( ptr += config.n_layers * config.dim * config.dim );
  auto base_rms_ffn_weight = ( ptr += config.n_layers * config.dim * config.dim );
  auto base_w1 = ( ptr += config.n_layers * config.dim );
  auto base_w2 = ( ptr += config.n_layers * config.dim * config.hidden_dim );
  auto base_w3 = ( ptr += config.n_layers * config.hidden_dim * config.dim );

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

Llama2::Llama2( const std::filesystem::path& tokenizer_path,
                const filesystem::path& model_path,
                const int32_t start_layer,
                const int32_t end_layer )
  : model_mmap_region_( [&] {
    const auto model_fs = filesystem::file_size( model_path );
    FileDescriptor model_fd { CHECK_SYSCALL( "open", open( model_path.c_str(), O_RDONLY ) ) };
    return MMap_Region { nullptr, model_fs, PROT_READ, MAP_PRIVATE, model_fd.fd_num(), 0 };
  }() )
  , model_ptr_( reinterpret_cast<const float*>( model_mmap_region_.addr() ) + sizeof( Config ) / sizeof( float ) )
  , config_( model_path )
  , start_layer_num_( start_layer )
  , end_layer_num_( end_layer == -1 ? config_.n_layers - 1 : end_layer )
  , base_weights_( config_, model_ptr_ )
  , layer_weights_( [&] {
    CHECK_GE( start_layer_num_, 0 ) << "Start layer must be non-negative.";
    CHECK_LT( end_layer_num_, config_.n_layers ) << "End layer must be less than the number of layers.";

    std::vector<LayerWeights> layers {};
    layers.resize( config_.n_layers );

    for ( int i = start_layer_num_; i <= end_layer_num_; i++ ) {
      layers[i] = LayerWeights { config_, model_ptr_, i };
    }

    return layers;
  }() )
  , vocabulary_( config_, tokenizer_path )
  , state_( config_, start_layer_num_, end_layer_num_ )
{
}

template<class T, size_t alignment>
unique_ptr<T[]> make_unique_aligned( size_t size )
{
  GlobalScopeTimer<Timer::Category::MemoryAllocation> _;
  void* ptr = aligned_alloc( alignment, size * sizeof( T ) );
  CHECK( ptr ) << "Failed to allocate memory.";
  return unique_ptr<T[]>( static_cast<T*>( ptr ) );
}

Llama2::RunState::RunState( const Config& config, const int32_t start_layer, const int32_t end_layer )
  : buffer_( make_unique_aligned<float, 64>( sizeof( float ) * config.dim * 6          /* x, xb, xb2, q, k, v */
                                             + sizeof( float ) * config.hidden_dim * 2 /* hb, hb2 */
                                             + sizeof( float ) * config.n_heads * config.seq_len /* att */
                                             + sizeof( float ) * config.vocab_size               /* logits */
                                             ) )
  , kv_cache( config, start_layer, end_layer )
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

Llama2::RunState::KVCache::KVCache( const Config& config, const int32_t start_layer, const int32_t end_layer )
  : start_layer_( start_layer )
  , end_layer_( end_layer )
  , buffer_(
      make_unique<float[]>( sizeof( float ) * config.seq_len * ( end_layer - start_layer + 1 ) * config.dim * 2 ) )
  , seq_len_( config.seq_len )
  , dim_( config.dim )
  , n_layers_( end_layer_ - start_layer_ + 1 )
  , head_size_( config.dim / config.n_heads )
{
}

float* Llama2::RunState::KVCache::key( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_.get() + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_;
}

float* Llama2::RunState::KVCache::value( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_.get() + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + dim_ + head * head_size_;
}

void Llama2::RunState::KVCache::pop() { throw runtime_error( "KVCache::pop() not implemented" ); }

void Llama2::pass_begin( const int token )
{
  // copy the token embedding into the state
  const float* content_row = base_weights_.token_embedding_table + token * config_.dim;
  memcpy( state_.x, content_row, config_.dim * sizeof( *state_.x ) );
}

void Llama2::transformer_layer( const int32_t layer_num, const int token_pos )
{
  float* x = state_.x;
  const int dim = config_.dim;
  const int hidden_dim = config_.hidden_dim;
  const int head_size = dim / config_.n_heads;

  // pluck out the "pos" row of freq_cis_real and freq_cis_imag
  const float* freq_cis_real_row = base_weights_.freq_cis_real + token_pos * head_size / 2;
  const float* freq_cis_imag_row = base_weights_.freq_cis_imag + token_pos * head_size / 2;

  const auto& layer_weights = layer_weights_[layer_num];

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
  memcpy( state_.kv_cache.key( layer_num, token_pos ), state_.k, dim * sizeof( float ) );
  memcpy( state_.kv_cache.value( layer_num, token_pos ), state_.v, dim * sizeof( float ) );

  // multihead attention. iterate over all heads
  int head_num;
#pragma omp parallel for private( head_num )
  for ( head_num = 0; head_num < config_.n_heads; head_num++ ) {
    // get the query vector for this head
    const float* q = state_.q + head_num * head_size;

    // attention scores for this head
    float* att = state_.att + head_num * config_.seq_len;

    // iterate over all timesteps, including the current one
    for ( int t = 0; t <= token_pos; t++ ) {
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
    ops::softmax( att, token_pos + 1 );

    // weighted sum of the values, store back into xb
    float* xb = state_.xb + head_num * head_size;
    memset( xb, 0, head_size * sizeof( float ) );

    for ( int t = 0; t <= token_pos; t++ ) {
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

void Llama2::pass_end()
{
  // final rmsnorm
  ops::rmsnorm( state_.x, state_.x, base_weights_.rms_final_weight, config_.dim );

  // classifier into logits
  ops::matmul( state_.logits, state_.x, base_weights_.wcls, config_.dim, config_.vocab_size );
}

InferenceResult Llama2::forward( const InferenceState& inference_state )
{
  CHECK( start_layer_num_ == inference_state.next_layer ) << "Cannot apply to this instance.";

  InferenceResult result;

  if ( inference_state.next_layer == 0 ) {
    pass_begin( inference_state.token );
  } else {
    CHECK_EQ( inference_state.activations.len, config_.dim ) << "Invalid activations.";
    memcpy( state_.x, inference_state.activations.ptr, config_.dim * sizeof( float ) );
  }

  for ( int i = start_layer_num_; i <= end_layer_num_; i++ ) {
    transformer_layer( i, inference_state.token_pos );
  }

  if ( end_layer_num_ == config_.n_layers - 1 ) {
    pass_end();

    // token extraction
    auto [next_token, next_word] = extract_output( inference_state );

    result.inference_state.token = next_token;
    result.inference_state.token_pos = inference_state.token_pos + 1;
    result.inference_state.next_layer = 0;
    result.inference_state.activations = { state_.x, config_.dim };
    result.word.emplace( move( next_word ) );
  } else {
    result.inference_state.token = inference_state.token;
    result.inference_state.token_pos = inference_state.token_pos;
    result.inference_state.next_layer = end_layer_num_ + 1;
    result.inference_state.activations = { state_.x, config_.dim };
    result.word.reset();
  }

  return result;
}

std::pair<int, std::string> Llama2::extract_output( const InferenceState& inference_state )
{
  if ( inference_state.token_pos >= config_.seq_len ) {
    return { -1, string {} };
  }

  int next_token;

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

  return { next_token, vocabulary_.get_word( next_token ) };
}
