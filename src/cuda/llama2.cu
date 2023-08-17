#include "llama2.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <source_location>

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

using namespace std;
using namespace glinthawk::gpu;

void CHECK_CUDA( const cudaError_t err, const source_location location = source_location::current() )
{
  if ( err != cudaSuccess ) {
    throw runtime_error( "CUDA error: " + string( cudaGetErrorString( err ) ) + " (" + location.file_name() + ":"
                         + std::to_string( location.line() ) + ")" );
  }
}

namespace {

static cublasHandle_t handle;

}

namespace ops {

__global__ void normalize_and_scale( float* output,
                                     const float* x,
                                     const float* weight,
                                     const int size,
                                     const float ss )
{
  const int i = threadIdx.x;
  output[i] = weight[i] * ss * x[i];
}

void rmsnorm( float* output, const float* x, const float* weight, const int size )
{
  // calculate sum of squares
  float ss = 0.0f;

  cublasSdot( handle, size, x, 1, x, 1, &ss );
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale<<<1, size>>>( output, x, weight, size, ss );
}

void softmax( float* _x, const int size )
{
  thrust::device_ptr<float> x { _x };

  const float max_val = *thrust::max_element( x, x + size );
  const float sum = thrust::transform_reduce(
    x, x + size, [max_val] __device__( const float x ) { return expf( x - max_val ); }, 0.0f, thrust::plus<float>() );
  thrust::transform( x, x + size, x, [sum] __device__( const float x ) { return x / sum; } );
}

void sample( const float* probabilities, const int n, int* output ) { throw runtime_error( "not implemented" ); }

void argmax( const float* _v, const int n, int* _output )
{
  thrust::device_ptr<const float> v { _v };
  thrust::device_ptr<int> output { _output };

  const auto it = thrust::max_element( v, v + n );
  *output = thrust::distance( v, it );
}

void accum( float* a, const float* b, const int size )
{
  float alpha = 1.0f;
  cublasSaxpy( handle, size, &alpha, b, 1, a, 1 );
}

// void rmsnorm( float* o, const float* x, const float* weight, const int size );
// void softmax( float* x, const int size );

void matmul( float* xout, const float* x, const float* W, const int n, const int d )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(d,n) @ x(n,) -> xout(d,)
  cublasSgemv( handle, CUBLAS_OP_T, n, d, &alpha, W, n, x, 1, &beta, xout, 1 );
}

void silu( float* _hb, float* _hb2, const int hidden_dim )
{
  thrust::device_ptr<float> hb { _hb };
  thrust::device_ptr<float> hb2 { _hb2 };

  thrust::transform(
    hb, hb + hidden_dim, hb, [] __device__( float x ) { return x * ( 1.0f / ( 1.0f + expf( -x ) ) ); } );
  thrust::transform( hb, hb + hidden_dim, hb2, hb, thrust::multiplies<float>() );
}

// int sample( const float* probabilities, const int n );
// int argmax( const float* v, const int n );

}

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
  : model_ptr_( [&] {
    const auto model_size = filesystem::file_size( model_path );
    FileDescriptor model_fd { CHECK_SYSCALL( "open", open( model_path.c_str(), O_RDONLY ) ) };
    MMap_Region model_mmap { nullptr, model_size, PROT_READ, MAP_PRIVATE, model_fd.fd_num(), 0 };
    void* ptr;

    CHECK_CUDA( cudaMalloc( &ptr, model_size ) );
    CHECK_CUDA( cudaMemcpy( ptr, model_mmap.addr(), model_size, cudaMemcpyHostToDevice ) );

    return reinterpret_cast<const float*>( ptr ) + sizeof( Config ) / sizeof( float );
  }() )
  , config_( model_path )
  , start_layer_num_( start_layer )
  , end_layer_num_( end_layer == -1 ? config_.n_layers - 1 : end_layer )
  , base_weights_( config_, model_ptr_ )
  , layer_weights_( [&] {
    CHECK_GE( start_layer_num_, 0 ) << "Start layer must be non-negative.";
    CHECK_LT( end_layer_num_, config_.n_layers ) << "End layer must be less than the number of layers.";

    vector<LayerWeights> layers( config_.n_layers );
    for ( int i = start_layer_num_; i <= end_layer_num_; i++ ) {
      new ( &layers[i] ) LayerWeights { config_, model_ptr_, i };
    }

    return layers;
  }() )
  , vocabulary_( config_, tokenizer_path )
  , state_( config_, start_layer_num_, end_layer_num_ )
{
  cublasCreate( &handle );
}

Llama2::RunState::RunState( const Config& config, const int32_t start_layer, const int32_t end_layer )
  : buffer_( [&] {
    void* ptr;
    const auto size = sizeof( float )
                      * ( config.dim * 5 + config.hidden_dim * 2 + config.n_heads * config.seq_len + config.vocab_size
                          + config.n_heads );

    CHECK_CUDA( cudaMalloc( &ptr, size ) );
    return reinterpret_cast<float*>( ptr );
  }() )
  , x( [&] {
    void* ptr;
    const auto size = sizeof( float ) * config.dim;
    CHECK_CUDA( cudaMalloc( &ptr, size ) );
    return reinterpret_cast<float*>( ptr );
  }() )
  , xb( buffer_ )
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

Llama2::RunState::KVCache::KVCache( const Config& config, const int32_t start_layer, const int32_t end_layer )
  : start_layer_( start_layer )
  , end_layer_( end_layer )
  , buffer_( [&] {
    void* ptr;
    const auto size = sizeof( float ) * config.seq_len * ( end_layer - start_layer + 1 ) * config.dim * 2;
    CHECK_CUDA( cudaMalloc( &ptr, size ) );
    return reinterpret_cast<float*>( ptr );
  }() )
  , seq_len_( config.seq_len )
  , dim_( config.dim )
  , n_layers_( end_layer_ - start_layer_ + 1 )
  , head_size_( config.dim / config.n_heads )
{
}

float* Llama2::RunState::KVCache::key( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_ + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_;
}

float* Llama2::RunState::KVCache::value( int layer, const int step, const int head )
{
  layer -= start_layer_;
  return buffer_ + step * ( n_layers_ * dim_ * 2 ) + layer * ( dim_ * 2 ) + head * head_size_ + dim_;
}

void Llama2::RunState::KVCache::pop() { throw runtime_error( "KVCache::pop() not implemented" ); }

void Llama2::pass_begin( const int token )
{
  // copy the token embedding into the state
  const float* content_row = base_weights_.token_embedding_table + token * config_.dim;
  CHECK_CUDA( cudaMemcpy( state_.x, content_row, config_.dim * sizeof( float ), cudaMemcpyDeviceToDevice ) );
}

__global__ void do_rope( const int head_size,
                         const int n_heads,
                         const float* freq_cis_real_row,
                         const float* freq_cis_imag_row,
                         float* state_q,
                         float* state_k )
{
  const int head_num = blockIdx.x;
  const int elem_idx = 2 * threadIdx.x;

  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  float* q = state_q + head_num * head_size;
  float* k = state_k + head_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const float q0 = q[elem_idx];
  const float q1 = q[elem_idx + 1];
  const float k0 = k[elem_idx];
  const float k1 = k[elem_idx + 1];
  const float fcr = freq_cis_real_row[elem_idx / 2];
  const float fci = freq_cis_imag_row[elem_idx / 2];
  q[elem_idx] = q0 * fcr - q1 * fci;
  q[elem_idx + 1] = q0 * fci + q1 * fcr;
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;
}

__global__ void attention_0( const float* all_q,
                             const float* kv_cache,
                             float* att,
                             const int layer_num,
                             const int n_layers,
                             const int seq_len,
                             const int head_size,
                             const int dim )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  const float* q = all_q + head_num * head_size;
  const float* k = kv_cache + token_pos * ( n_layers * dim * 2 ) + layer_num * ( dim * 2 ) + head_num * head_size;

  float score = 0.0f;
  for ( int i = 0; i < head_size; i++ ) {
    score += q[i] * k[i];
  }
  score /= sqrtf( head_size );

  // save the score to the attention buffer
  att[token_pos] = score;
}

__global__ void find_max_for_rows( const float* att,
                                   float* output,
                                   const int token_pos,
                                   const int n_heads,
                                   const int seq_len )
{
  const int head_num = threadIdx.x;
  att += head_num * seq_len;

  float max_value = att[0];
  for ( int i = 1; i <= token_pos; i++ ) {
    max_value = max( max_value, att[i] );
  }

  output[head_num] = max_value;
}

__global__ void subtract_and_expf( const float* values, float* att, const int n_heads, const int seq_len )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] = expf( att[token_pos] - values[head_num] );
}

__global__ void sum_rows( float* att, float* output, const int token_pos, const int n_heads, const int seq_len )
{
  const int head_num = threadIdx.x;
  att += head_num * seq_len;

  float sum = 0.0;
  for ( int i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

__global__ void normalize_by_sum( float* att, const float* sums, const int n_heads, const int seq_len )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

void attention_softmax( float* att, const int token_pos, const int seq_len, const int n_heads, float* temp_buffer )
{
  float* head_values = temp_buffer;

  // (1) find the max value for each head (each row)
  find_max_for_rows<<<1, n_heads>>>( att, head_values, token_pos, n_heads, seq_len );

  // (2) exp(att - max)
  subtract_and_expf<<<token_pos + 1, n_heads>>>( head_values, att, n_heads, seq_len );

  // (3) sum each row
  sum_rows<<<1, n_heads>>>( att, head_values, token_pos, n_heads, seq_len );

  // (4) normalize each row by its sum
  normalize_by_sum<<<token_pos + 1, n_heads>>>( att, head_values, n_heads, seq_len );
}

__global__ void attention_2( float* att,
                             const float* kv_cache,
                             float* xb,
                             const int layer_num,
                             const int n_layers,
                             const int seq_len,
                             const int head_size,
                             const int dim )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  xb += head_num * head_size;

  const float a = att[token_pos];
  const float* v = kv_cache + token_pos * ( n_layers * dim * 2 ) + layer_num * ( dim * 2 ) + head_num * head_size + dim;

  for ( int i = 0; i < head_size; i++ ) {
    atomicAdd( &xb[i], a * v[i] );
  }
}

void Llama2::transformer_layer( const int32_t layer_num, const int token_pos )
{
  float* const x = state_.x;
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

  do_rope<<<config_.n_heads, head_size / 2>>>(
    head_size, config_.n_heads, freq_cis_real_row, freq_cis_imag_row, state_.q, state_.k );

  float* k_cache_pos = state_.kv_cache.key( layer_num, token_pos );
  float* v_cache_pos = state_.kv_cache.value( layer_num, token_pos );

  // save key,value at this time step (pos) to our kv cache
  CHECK_CUDA( cudaMemcpy( k_cache_pos, state_.k, dim * sizeof( float ), cudaMemcpyDeviceToDevice ) );
  CHECK_CUDA( cudaMemcpy( v_cache_pos, state_.v, dim * sizeof( float ), cudaMemcpyDeviceToDevice ) );

  // multihead attention. for each head and for each token up to and including the current one
  attention_0<<<token_pos + 1, config_.n_heads>>>(
    state_.q, state_.kv_cache.buffer_, state_.att, layer_num, config_.n_layers, config_.seq_len, head_size, dim );

  // softmax
  attention_softmax( state_.att, token_pos, config_.seq_len, config_.n_heads, state_.temp_softmax );

  CHECK_CUDA( cudaMemset( state_.xb, 0, dim * sizeof( float ) ) );

  attention_2<<<token_pos + 1, config_.n_heads>>>(
    state_.att, state_.kv_cache.buffer_, state_.xb, layer_num, config_.n_layers, config_.seq_len, head_size, dim );
  // end of multihead attention

  // final matmul to get the output of the attention
  ops::matmul( state_.xb2, state_.xb, layer_weights.wo, dim, dim );

  // residual connection back into x
  ops::accum( x, state_.xb2, dim );

  // ffn rmsnorm
  ops::rmsnorm( state_.xb, x, layer_weights.rms_ffn_weight, dim );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops::matmul( state_.hb, state_.xb, layer_weights.w1, dim, hidden_dim );
  ops::matmul( state_.hb2, state_.xb, layer_weights.w3, dim, hidden_dim );

  ops::silu( state_.hb, state_.hb2, hidden_dim );

  // final matmul to get the output of the ffn
  ops::matmul( state_.xb, state_.hb, layer_weights.w2, hidden_dim, dim );

  // residual connection
  ops::accum( x, state_.xb, dim );
}

void Llama2::pass_end()
{
  float* x = state_.x;

  // final rmsnorm
  ops::rmsnorm( x, x, base_weights_.rms_final_weight, config_.dim );

  // classifier into logits
  ops::matmul( state_.logits, x, base_weights_.wcls, config_.dim, config_.vocab_size );
}

pair<int, string> Llama2::forward( const int token )
{
  if ( token_pos >= config_.seq_len ) {
    return { 2, {} };
  }

  pass_begin( token );

  for ( int layer_num = start_layer_num_; layer_num <= end_layer_num_; layer_num++ ) {
    transformer_layer( layer_num, token_pos );
  }

  pass_end();

  int next_token;
  int* next_token_device;
  CHECK_CUDA( cudaMalloc( &next_token_device, sizeof( int ) ) );

  if ( temperature_ == 0.0f ) {
    // greedy argmax sampling
    ops::argmax( state_.logits, config_.vocab_size, next_token_device );
  } else {
    // apply the temperature to the logits
    for ( int q = 0; q < config_.vocab_size; q++ ) {
      state_.logits[q] /= temperature_;
    }

    // apply softmax to the logits to get the probabilities for next token
    ops::softmax( state_.logits, config_.vocab_size );

    // we now want to sample from this distribution to get the next token
    ops::sample( state_.logits, config_.vocab_size, next_token_device );
  }

  token_pos++;
  CHECK_CUDA( cudaMemcpy( &next_token, next_token_device, sizeof( int ), cudaMemcpyDeviceToHost ) );
  return { next_token, vocabulary_.get_word( next_token ) };
}
