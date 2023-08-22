#include "model.cuh"

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <source_location>

#include <cuda_fp16.h>

#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

#include "models/common/cuda/ops.cuh"
#include "models/llama2/base.hh"

using namespace std;
using namespace glinthawk::models;
using namespace glinthawk::models::common::cuda;

namespace glinthawk::models::llama2::cuda {

void CHECK_CUDA( const cudaError_t err, const source_location location = source_location::current() )
{
  if ( err != cudaSuccess ) {
    throw runtime_error( "CUDA error " + string( cudaGetErrorName( err ) ) + ": " + string( cudaGetErrorString( err ) )
                         + " (" + location.file_name() + ":" + to_string( location.line() ) + ")" );
  }
}

template<typename DType>
void cuda_deleter( DType* ptr )
{
  cudaFree( ptr );
}

template<typename DType>
std::string dtype_str()
{
  if constexpr ( is_same_v<DType, float> ) {
    return "FP32";
  } else if constexpr ( is_same_v<DType, __half> ) {
    return "FP16";
  } else {
    return "unknown";
  }
}

template<typename DType>
Llama2<DType>::~Llama2()
{
  ops::destroy();
}

template<typename DType>
Llama2<DType> Llama2<DType>::load( const filesystem::path& model_path,
                                   const int32_t start_layer,
                                   const int32_t end_layer_raw,
                                   const uint64_t batch_size )
{
  ops::init();

  const string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Config config { config_path, batch_size };

  const int32_t end_layer = ( end_layer_raw == -1 ) ? ( config.n_layers - 1 ) : end_layer_raw;
  const int32_t layer_count = end_layer - start_layer + 1;

  CHECK_GE( start_layer, 0 ) << "Start layer must be non-negative.";
  CHECK_LE( start_layer, end_layer ) << "Start layer must be less than or equal to end layer.";
  CHECK_LT( end_layer, config.n_layers ) << "End layer must be less than the number of layers.";

  const auto run_state_size = RunState<DType>::state_size( config );
  const auto base_size = BaseWeights<DType>::base_size( config );
  const auto layer_size = LayerWeights<DType>::layer_size( config );
  const auto kv_cache_size = KVCache<DType>::cache_size( config, start_layer, end_layer );

  DType* base_raw_ptr;
  DType* layers_raw_ptr;
  DType* run_state_raw_ptr;
  DType* kv_cache_raw_ptr;

  // Allocate memory for the base weights
  CHECK_CUDA( cudaMalloc( &base_raw_ptr, base_size ) );
  unique_ptr<DType, void ( * )( DType* )> base { base_raw_ptr, cuda_deleter };

  // Allocate memory for the layers
  CHECK_CUDA( cudaMalloc( &layers_raw_ptr, layer_size * layer_count ) );
  unique_ptr<DType, void ( * )( DType* )> layers { layers_raw_ptr, cuda_deleter };

  // Allocate memory for the run state
  CHECK_CUDA( cudaMalloc( &run_state_raw_ptr, run_state_size ) );
  unique_ptr<DType, void ( * )( DType* )> run_state { run_state_raw_ptr, cuda_deleter };

  // Allocate memory for the kv cache
  CHECK_CUDA( cudaMalloc( &kv_cache_raw_ptr, kv_cache_size ) );
  unique_ptr<DType, void ( * )( DType* )> kv_cache { kv_cache_raw_ptr, cuda_deleter };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    CHECK_CUDA( cudaMemcpy( base.get(), base_mmap.addr(), base_size, cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  }

  // (2) load the layers
  for ( auto i = start_layer; i <= end_layer; i++ ) {
    const auto layer_path = model_path / ( "LAYER" + to_string( i ) + filename_suffix );

    CHECK_EQ( filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

    FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
    MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

    CHECK_CUDA( cudaMemcpy( reinterpret_cast<uint8_t*>( layers.get() ) + ( i - start_layer ) * layer_size,
                            layer_mmap.addr(),
                            layer_size,
                            cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
  }

  return { config, move( base ), move( layers ), move( run_state ), move( kv_cache ), start_layer, end_layer };
}

template<typename DType>
__global__ void do_rope( const uint64_t head_size,
                         const DType* freq_cis_real_row,
                         const DType* freq_cis_imag_row,
                         DType* state_q,
                         DType* state_k )
{
  const uint64_t head_num = blockIdx.x;
  const uint64_t elem_idx = 2 * threadIdx.x;

  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  DType* q = state_q + head_num * head_size;
  DType* k = state_k + head_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const DType q0 = q[elem_idx];
  const DType q1 = q[elem_idx + 1];
  const DType k0 = k[elem_idx];
  const DType k1 = k[elem_idx + 1];
  const DType fcr = freq_cis_real_row[elem_idx / 2];
  const DType fci = freq_cis_imag_row[elem_idx / 2];
  q[elem_idx] = q0 * fcr - q1 * fci;
  q[elem_idx + 1] = q0 * fci + q1 * fcr;
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;
}

template<typename DType>
__global__ void find_max_for_rows( const DType* att,
                                   DType* output,
                                   const uint64_t token_pos,
                                   const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType max_value = att[0];
  for ( uint64_t i = 1; i <= token_pos; i++ ) {
    if constexpr ( is_same_v<DType, __half> ) {
      max_value = __hmax( max_value, att[i] );
    } else {
      max_value = max( max_value, att[i] );
    }
  }

  output[head_num] = max_value;
}

template<typename DType>
__global__ void subtract_and_expf( const DType* values, DType* att, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] = expf( att[token_pos] - values[head_num] );
}

template<typename DType>
__global__ void sum_rows( DType* att,
                          DType* output,
                          const uint64_t token_pos,
                          const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType sum = 0.0;
  for ( uint64_t i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

template<typename DType>
__global__ void normalize_by_sum( DType* att, const DType* sums, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

template<typename DType>
void attention_softmax( DType* att,
                        const uint64_t token_pos,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size )
{
  // TODO: fix thread and block assignments, threads might overflow
  // (1) find the max value for each head (each row)
  find_max_for_rows<<<1, n_heads * batch_size>>>( att, temp_buffer, token_pos, seq_len );

  // (2) exp(att - max)
  subtract_and_expf<<<token_pos + 1, n_heads * batch_size>>>( temp_buffer, att, seq_len );

  // (3) sum each row
  sum_rows<<<1, n_heads * batch_size>>>( att, temp_buffer, token_pos, seq_len );

  // (4) normalize each row by its sum
  normalize_by_sum<<<token_pos + 1, n_heads * batch_size>>>( att, temp_buffer, seq_len );
}

template<typename DType>
void Llama2<DType>::pass_begin( const std::vector<uint32_t>& token )
{
  // copy the token embedding into the state
  for (size_t i = 0; i < token.size(); i++){
    const DType* content_row = this->base_weights_.token_embedding_table + token[i] * this->config_.dim;
    CHECK_CUDA(
      cudaMemcpy( this->state_.x + i * this->config_.dim, content_row, this->config_.dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  }
}

template<typename DType>
__global__ void check_eq_cuda_arrs( const DType* x1,
                          const DType* x2,
                          const int size )
{
  DType eps_ = DType(1e-5);
  bool differ = false;
  for ( uint64_t i = 0; i < size; i++ ) {
    if ( x1[i] < x2[i] && x2[i]-x1[i] > eps_){
      differ = true;
      printf("%f, %f, %" PRIu64 "\n", __half2float(x1[i]), __half2float(x2[i]), i);
    }
    if ( x1[i] > x2[i] && x1[i]-x2[i] > eps_){
      differ = true;
      printf("%f, %f, %" PRIu64 "\n", __half2float(x1[i]), __half2float(x2[i]), i);
    }
  }
  if (differ)
    printf("The two arrays differ\n");
}

template<typename DType>
void Llama2<DType>::transformer_layer( const int32_t layer_num, const uint64_t token_pos )
{
  DType* const x = this->state_.x;
  const uint64_t dim = this->config_.dim;
  const uint64_t hidden_dim = this->config_.hidden_dim;
  const uint64_t head_size = dim / this->config_.n_heads;

//  printf("Input\n");
//  check_eq_cuda_arrs<<<1, 1>>>(x, x + dim, dim);
//  cudaDeviceSynchronize();

  // pluck out the "pos" row of freq_cis_real and freq_cis_imag
  const DType* freq_cis_real_row = this->base_weights_.freq_cis_real + token_pos * head_size / 2;
  const DType* freq_cis_imag_row = this->base_weights_.freq_cis_imag + token_pos * head_size / 2;

  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops::rmsnorm( this->state_.xb, x, layer_weights.rms_att_weight, dim, this->curr_batch_size );

  // qkv matmuls for this position
  ops::matmul( this->state_.q, this->state_.xb, layer_weights.wq, this->curr_batch_size, dim, dim );
  ops::matmul( this->state_.k, this->state_.xb, layer_weights.wk, this->curr_batch_size, dim, dim );
  ops::matmul( this->state_.v, this->state_.xb, layer_weights.wv, this->curr_batch_size, dim, dim );

  do_rope<<<this->config_.n_heads * this->curr_batch_size, head_size / 2>>>(
    head_size, freq_cis_real_row, freq_cis_imag_row, this->state_.q, this->state_.k );

  DType* k_cache_pos = this->kv_cache_.key( layer_num, token_pos );
  DType* v_cache_pos = this->kv_cache_.value( layer_num, token_pos );

  // save key,value at this time step (pos) to our kv cache
  CHECK_CUDA( cudaMemcpy( k_cache_pos, this->state_.k, this->curr_batch_size * dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  CHECK_CUDA( cudaMemcpy( v_cache_pos, this->state_.v, this->curr_batch_size * dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );

  // multihead attention. for each head and for each token up to and including the current one
  ops::attention_0_gemm( this->state_.q,
                         this->kv_cache_.buffer_ + layer_num * this->config_.batch_size * ( dim * 2 ),
                         this->state_.att,
                         this->config_.n_layers,
                         this->config_.seq_len,
                         head_size,
                         this->config_.n_heads,
                         token_pos + 1,
                         this->curr_batch_size,
                         this->config_.batch_size);
  
  // softmax
  attention_softmax(
    this->state_.att, token_pos, this->config_.seq_len, this->config_.n_heads, this->state_.temp_softmax, this->curr_batch_size );

  ops::attention_2_gemm( this->state_.att,
                         this->kv_cache_.buffer_ + layer_num * this->config_.batch_size * ( dim * 2 ) + this->config_.batch_size * dim,
                         this->state_.xb,
                         this->config_.n_layers,
                         this->config_.seq_len,
                         head_size,
                         this->config_.n_heads,
                         token_pos + 1,
                         this->curr_batch_size,
                         this->config_.batch_size );
  // end of multihead attention

  // final matmul to get the output of the attention
  ops::matmul( this->state_.xb2, this->state_.xb, layer_weights.wo, this->curr_batch_size, dim, dim );

  // residual connection back into x
  ops::accum( x, this->state_.xb2, dim, this->curr_batch_size );

  // ffn rmsnorm
  ops::rmsnorm( this->state_.xb, x, layer_weights.rms_ffn_weight, dim, this->curr_batch_size );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops::matmul( this->state_.hb, this->state_.xb, layer_weights.w1, this->curr_batch_size, dim, hidden_dim );
  ops::matmul( this->state_.hb2, this->state_.xb, layer_weights.w3, this->curr_batch_size, dim, hidden_dim );

  ops::silu( this->state_.hb, this->state_.hb2, hidden_dim, this->curr_batch_size );

  // final matmul to get the output of the ffn
  ops::matmul( this->state_.xb, this->state_.hb, layer_weights.w2, this->curr_batch_size, hidden_dim, dim );

  // residual connection
  ops::accum( x, this->state_.xb, dim, this->curr_batch_size );
}

template<typename DType>
void Llama2<DType>::pass_end()
{
  // final rmsnorm
  ops::rmsnorm( this->state_.x, this->state_.x, this->base_weights_.rms_final_weight, this->config_.dim, this->curr_batch_size );

  // classifier into logits
  ops::matmul(
    this->state_.logits, this->state_.x, this->base_weights_.wcls, this->curr_batch_size, this->config_.dim, this->config_.vocab_size );
}

template<typename DType>
uint32_t extract_token( const RunState<DType>& state, const Config& config, const float temp, const uint64_t batch_index = 0 )
{
  uint32_t next_token;

  if ( temp == 0.0f ) {
    // greedy argmax sampling
    next_token = ops::argmax( state.logits + batch_index * config.vocab_size, config.vocab_size);
  } else {
    // apply the temperature to the logits
    for ( auto q = batch_index * config.vocab_size; q < ( batch_index + 1 ) * config.vocab_size; q++ ) {
      state.logits[q] /= temp;
    }

    // apply softmax to the logits to get the probabilities for next token
    ops::softmax( state.logits + batch_index * config.vocab_size, config.vocab_size );

    // we now want to sample from this distribution to get the next token
    next_token = ops::sample( state.logits + batch_index * config.vocab_size, config.vocab_size );
  }

  return next_token;
}

template<typename DType>
std::vector<uint32_t> extract_batch_token( const RunState<DType>& state, const Config& config, const std::vector<float>& temp )
{
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  std::vector<uint32_t> next_tokens;
  for (size_t i = 0; i < temp.size(); i++)
    next_tokens.push_back(extract_token(state, config, temp[i], i));
  return next_tokens;
}

template<typename DType>
std::vector<InferenceState<DType>> Llama2<DType>::forward( const std::vector<std::reference_wrapper<const InferenceState<DType>>>& inference_state_s )
{
  CHECK_GT( inference_state_s.size(), 0 ) << "batch size must be at least 1";
  token_pos_ = inference_state_s[0].get().token_pos();
  for (auto & item : inference_state_s) {
    CHECK_EQ( item.get().token_pos(), token_pos_ ) << "current implementation expects all inference states to be at the same token position";
    CHECK_EQ( item.get().next_layer(), this->start_layer_num_ ) << "next_layer must be the start layer";
  }

  this->curr_batch_size = inference_state_s.size();

  if ( inference_state_s[0].get().next_layer() == 0 ) {
    std::vector<uint32_t> token_vector;
    for (size_t i = 0; i < inference_state_s.size(); i++)
      token_vector.push_back(inference_state_s[i].get().token());
    pass_begin( token_vector );
  } else {
    for (size_t i = 0; i < inference_state_s.size(); i++)
      // load the activations
      CHECK_CUDA( cudaMemcpy( this->state_.x + i * this->config_.dim * sizeof( DType ),
                              inference_state_s[i].get().activations().ptr.get(),
                              this->config_.dim * sizeof( DType ),
                              cudaMemcpyHostToDevice ) );
  }

  for ( int layer_num = this->start_layer_num_; layer_num <= this->end_layer_num_; layer_num++ ) {
    transformer_layer( layer_num, token_pos_ );
  }

  std::vector<InferenceState<DType>> token_vector;

  if ( this->end_layer_num_ == this->config_.n_layers - 1 ) {
    pass_end();

    std::vector<float> batch_temps;
    for (size_t i = 0; i < inference_state_s.size(); i++)
      batch_temps.push_back(inference_state_s[i].get().temperature());
    std::vector<uint32_t> next_tokens = extract_batch_token(this->state_, this->config_, batch_temps);
    for (size_t i = 0; i < inference_state_s.size(); i++)
      token_vector.emplace_back(
        next_tokens[i],                                                                 // token
        token_pos_ + 1,                                                                 // token_pos
        0,                                                                              // next_layer
        inference_state_s[i].get().temperature(),                                       // temperature
        DataBuffer<DType> {}                                                            // activations
      );

    return token_vector;
  }

  for (size_t i = 0; i < inference_state_s.size(); i++){
    DataBuffer<DType> activations { make_unique<DType[]>( this->config_.dim ), this->config_.dim };

    CHECK_CUDA(
      cudaMemcpy( activations.ptr.get(), this->state_.x + i * this->config_.dim * sizeof( DType ), this->config_.dim * sizeof( DType ), cudaMemcpyDeviceToHost ) );

    token_vector.emplace_back(
        inference_state_s[i].get().token(),                                             // token
        token_pos_,                                                                     // token_pos
        static_cast<uint32_t>( this->end_layer_num_ ) + 1,                              // next_layer
        inference_state_s[i].get().temperature(),                                       // temperature
        move( activations )                                                             // activations
      );
  }
  

  return token_vector;
}

template<typename DType>
std::vector<uint32_t> Llama2<DType>::forward( const std::vector<uint32_t>& token_s )
{
  CHECK_GT( token_s.size(), 0 ) << "batch size must be at least 1";
  if ( token_pos_ >= this->config_.seq_len ) {
    return std::vector<uint32_t>(token_s.size(), 2); /* EOS */
  }

  this->curr_batch_size = token_s.size();

  pass_begin( token_s );

   for ( int layer_num = this->start_layer_num_; layer_num <= this->end_layer_num_; layer_num++ ) {
     transformer_layer( layer_num, token_pos_ );
   }

   /////////////////////////////////////////////////////////// profile batching //////////////////////////////////////////////////////////////
//  for ( int layer_num = 0; layer_num <= 31; layer_num++ ) {
//    transformer_layer( 0, token_pos_ );
//  }
   /////////////////////////////////////////////////////////// profile batching //////////////////////////////////////////////////////////////


  pass_end();

  token_pos_++;
  return extract_batch_token( this->state_, this->config_, std::vector<float>(token_s.size(), temperature_) );
}

template<typename DType>
std::vector<InferenceState<DType>> Llama2<DType>::forward( const std::vector<InferenceState<DType>>& inference_state_s ) {
  std::vector<std::reference_wrapper<const InferenceState<DType>>> res;
  for (auto & state : inference_state_s)
    res.push_back(std::ref(state));
  return forward(res);
}

template<typename DType>
InferenceState<DType> Llama2<DType>::forward( const InferenceState<DType>& inference_state )
{
  std::vector<std::reference_wrapper<const InferenceState<DType>>> token_vector;
  token_vector.push_back( std::ref( inference_state ) );
  return move( forward(token_vector)[0] );
}

template<typename DType>
uint32_t Llama2<DType>::forward( const uint32_t& token )
{
  std::vector<uint32_t> token_vector = {token};
  return forward(token_vector)[0];
}

template class Llama2<float>;
template class Llama2<__half>;

} // namespace glinthawk::models::llama2::cuda
