#include "model.cuh"

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <set>
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
unique_ptr<Llama2<DType>> Llama2<DType>::load( const filesystem::path& model_path,
                                               const int32_t start_layer,
                                               const int32_t end_layer_raw,
                                               const uint64_t kv_prompt_limit,
                                               const uint64_t concurrency_limit )
{
  ops::init( concurrency_limit );

  const string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Config config { config_path, kv_prompt_limit, concurrency_limit };

  CHECK_GT( 1025, config.n_heads ) << "Attention softmax has n_heads threads, and this cannot surpass 1024.";
  CHECK_GT( 1 << 16, config.n_heads ) << "RoPE has n_heads blocks, and this cannot surpass 2^16.";
  CHECK_GT( 1025, config.dim / config.n_heads / 2 ) << "RoPE has head_size / 2 threads, and this cannot surpass 1024.";
  CHECK_GT( 1 << 16, config.seq_len ) << "Attention softmax has seq_len blocks, and this cannot surpass 2^16.";
  const size_t tpb = ops::TPB;
  CHECK_GT( 1025, tpb ) << "Threads per block cannot surpass 1024.";
  CHECK_GT( 1 << 16, ( config.dim + tpb - 1 ) / tpb ) << "RMS Norm blocks cannot surpass 2^16.";
  CHECK_GT( 1 << 16, ( config.dim * config.concurrency_limit + tpb - 1 ) / tpb ) << "Accum blocks cannot surpass 2^16.";
  CHECK_GT( 1 << 16, ( config.hidden_dim * config.concurrency_limit + tpb - 1 ) / tpb )
    << "Silu blocks cannot surpass 2^16.";

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
  ops::CHECK_CUDA( cudaMalloc( &base_raw_ptr, base_size ) );
  unique_ptr<DType, void ( * )( DType* )> base { base_raw_ptr, cuda_deleter };

  // Allocate memory for the layers
  ops::CHECK_CUDA( cudaMalloc( &layers_raw_ptr, layer_size * layer_count ) );
  unique_ptr<DType, void ( * )( DType* )> layers { layers_raw_ptr, cuda_deleter };

  // Allocate memory for the run state
  ops::CHECK_CUDA( cudaMalloc( &run_state_raw_ptr, run_state_size ) );
  unique_ptr<DType, void ( * )( DType* )> run_state { run_state_raw_ptr, cuda_deleter };

  // Allocate memory for the kv cache
  ops::CHECK_CUDA( cudaMalloc( &kv_cache_raw_ptr, kv_cache_size ) );
  unique_ptr<DType, void ( * )( DType* )> kv_cache { kv_cache_raw_ptr, cuda_deleter };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    ops::CHECK_CUDA( cudaMemcpy( base.get(), base_mmap.addr(), base_size, cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  }

  // (2) load the layers
  for ( auto i = start_layer; i <= end_layer; i++ ) {
    const auto layer_path = model_path / ( "LAYER" + to_string( i ) + filename_suffix );

    CHECK_EQ( filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

    FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
    MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

    ops::CHECK_CUDA( cudaMemcpy( reinterpret_cast<uint8_t*>( layers.get() ) + ( i - start_layer ) * layer_size,
                                 layer_mmap.addr(),
                                 layer_size,
                                 cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
  }

  auto model = unique_ptr<Llama2<DType>>( new Llama2<DType> {
    config, move( base ), move( layers ), move( run_state ), move( kv_cache ), start_layer, end_layer } );

  return model;
}

template<typename DType>
__global__ void print_arr_point( const DType* const* x_mega, const int i, const int size, const int stride )
{
  const DType* x1 = x_mega[i];
  for ( uint64_t i = 0; i < size; i++ ) {
    if constexpr ( is_same_v<DType, __half> ) {
      printf( "%.9f, ", __half2float( x1[i * stride] ) );
    } else {
      printf( "%.9f, ", x1[i * stride] );
    }
  }
  printf( "\n" );
}

template<typename DType>
__global__ void print_arr( const DType* x1, const int size, const int stride )
{
  for ( uint64_t i = 0; i < size; i++ ) {
    if constexpr ( is_same_v<DType, __half> ) {
      printf( "%.9f, ", __half2float( x1[i * stride] ) );
    } else {
      printf( "%.9f, ", x1[i * stride] );
    }
  }
  printf( "\n" );
}

template<typename DType>
void Llama2<DType>::pass_begin( const std::vector<uint32_t>& token )
{
  // copy the token embedding into the state
  for ( size_t i = 0; i < token.size(); i++ ) {
    CHECK_LT( token[i], this->config_.vocab_size ) << "token index must not surpass vocab size";
    const DType* content_row = this->base_weights_.token_embedding_table + token[i] * this->config_.dim;
    ops::CHECK_CUDA( cudaMemcpyAsync( this->state_.x + i * this->config_.dim,
                                      content_row,
                                      this->config_.dim * sizeof( DType ),
                                      cudaMemcpyDeviceToDevice ) );
  }
}

template<typename DType>
void Llama2<DType>::transformer_layer( const int32_t layer_num )
{
  DType* const x = this->state_.x;
  const uint64_t dim = this->config_.dim;
  const uint64_t hidden_dim = this->config_.hidden_dim;
  const uint64_t head_size = dim / this->config_.n_heads;

  const auto& layer_weights = this->layer_weights_[layer_num];
//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
//      std::cout << "\nLayer " << layer_num << "\n" << std::endl;
//    }
//  }

  //  std::cout << "Pre attention" << std::endl;
  //  print_arr<<<1, 1>>>(x, 768, 1);
  //  cudaDeviceSynchronize();

  // attention rmsnorm
  ops::rmsnorm( this->state_.xb, x, layer_weights.rms_att_weight, dim, this->curr_concurrency_size );
//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
//      std::cout << "Post First RMS Norm, token " << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->state_.xb+i*dim, 12, 1 );
//      cudaDeviceSynchronize();
//    }
//  }

  // qkv matmuls for this position
  ops::matmul( this->state_.q, this->state_.xb, layer_weights.wq, this->curr_concurrency_size, dim, dim );
  ops::matmul( this->state_.k, this->state_.xb, layer_weights.wk, this->curr_concurrency_size, dim, dim );
  ops::matmul( this->state_.v, this->state_.xb, layer_weights.wv, this->curr_concurrency_size, dim, dim );

  //  std::cout << "Queries" << std::endl;
  //  print_arr<<<1, 1>>>(this->state_.q+64+62, 1, 1);
  //  cudaDeviceSynchronize();
  //
  //  std::cout << "Keys" << std::endl;
  //  print_arr<<<1, 1>>>(this->state_.k, 8, 1);
  //  cudaDeviceSynchronize();
  //
  //  std::cout << "Values" << std::endl;
  //  print_arr<<<1, 1>>>(this->state_.v, 8, 1);
  //  cudaDeviceSynchronize();

//  if (layer_num == 0)
//  for (size_t i = 0; i < this->curr_concurrency_size; i++){
//    if (this->token_pos_[i] == 9){
//      std::cout << "Post att0, batch " << i << ", token " << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->state_.k + i * dim, 11, 1 );
//      cudaDeviceSynchronize();
//    }
//  }

  ops::apply_rope( head_size,
                   this->config_.n_heads,
                   this->curr_concurrency_size,
                   this->token_pos_,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->state_.q,
                   this->state_.k );

  // save key,value at this time step (pos) to our kv cache
  ops::copy_kv_cache( this->state_.k,
                      this->state_.v,
                      this->kv_cache_.key( layer_num, 0, 0 ),
                      this->kv_cache_.value( layer_num, 0, 0 ),
                      dim,
                      this->kv_cache_.n_layers_,
                      this->curr_concurrency_size,
                      this->config_.kv_prompt_limit,
                      this->id_allocation_,
                      this->token_pos_ );

//  cudaDeviceSynchronize();
//  if (layer_num == 0)
//  for (size_t i = 0; i < this->curr_concurrency_size; i++){
//    if (this->token_pos_[i] == 9){
//            std::cout << "Post att0, batch " << i << ", token " << this->token_pos_[i] << std::endl << std::flush;
//      //      print_arr<<<1, 1>>>( this->state_.q+i*dim, head_size, 1 );
//      //      cudaDeviceSynchronize();
////      std::cout << "" << this->token_pos_[i] << std::endl << std::flush;
//            print_arr<<<1, 1>>>( this->state_.k + i * dim, 11, 1 );
//      print_arr<<<1, 1>>>( this->kv_cache_.key( layer_num, 9, this->id_allocation_[i], 0 ), 11, 1 );
//      cudaDeviceSynchronize();
//      //      std::cout << "" << this->token_pos_[i] << std::endl << std::flush;
//      //      print_arr<<<1, 1>>>( this->state_.att+i*this->config_.n_heads*this->config_.seq_len, 10, 1 );
//      //      cudaDeviceSynchronize();
//    }
//  }

//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
//            std::cout << "Post att0, batch " << i << ", token " << this->token_pos_[i] << std::endl << std::flush;
//      //      print_arr<<<1, 1>>>( this->state_.q+i*dim, head_size, 1 );
//      //      cudaDeviceSynchronize();
////      std::cout << "" << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->kv_cache_.key( layer_num, 9, this->id_allocation_[i], 0 ), 11, 1 );
//      cudaDeviceSynchronize();
//      //      std::cout << "" << this->token_pos_[i] << std::endl << std::flush;
//      //      print_arr<<<1, 1>>>( this->state_.att+i*this->config_.n_heads*this->config_.seq_len, 10, 1 );
//      //      cudaDeviceSynchronize();
//    }
//  }

  // multihead attention. for each head and for each token up to and including the current one
  ops::attention_0_gemm( this->state_.q,
                         this->kv_cache_.key( layer_num, 0, 0, 0 ),
                         this->state_.att,
                         this->kv_cache_.n_layers_,
                         this->config_.seq_len,
                         head_size,
                         this->config_.n_heads,
                         this->curr_concurrency_size,
                         this->config_.kv_prompt_limit,
                         this->id_allocation_,
                         this->token_pos_ );

  ops::matmul( this->state_.xb2, this->state_.xb, layer_weights.wo, this->curr_concurrency_size, dim, dim );

//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
////      std::cout << "Post att0, token " << this->token_pos_[i] << std::endl << std::flush;
////      print_arr<<<1, 1>>>( this->state_.q+i*dim, head_size, 1 );
////      cudaDeviceSynchronize();
//      std::cout << "" << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->kv_cache_.key( layer_num, 9, i, 0 ), head_size, 1 );
//      cudaDeviceSynchronize();
////      std::cout << "" << this->token_pos_[i] << std::endl << std::flush;
////      print_arr<<<1, 1>>>( this->state_.att+i*this->config_.n_heads*this->config_.seq_len, 10, 1 );
////      cudaDeviceSynchronize();
//    }
//  }

  //  std::cout << "Q in attention head 2" << std::endl;
  //  print_arr_point<<<1, 1>>>(this->state_.q_p, 1, 64, 1);
  //  cudaDeviceSynchronize();
  //
  //  std::cout << "K in attention head 2" << std::endl;
  //  print_arr_point<<<1, 1>>>(this->state_.k_p, 1, 64, 1);
  //  cudaDeviceSynchronize();

  //  std::cout << "Attention" << std::endl;
  //  print_arr<<<1, 1>>>(this->state_.att, 8, this->config_.seq_len);
  //  cudaDeviceSynchronize();

  // softmax
  ops::attention_softmax( this->state_.att,
                          this->token_pos_,
                          this->config_.seq_len,
                          this->config_.n_heads,
                          this->state_.temp_softmax,
                          this->curr_concurrency_size );

  //  std::cout << "Softmax Attention" << std::endl;
  //  print_arr<<<1, 1>>>(this->state_.att, 8, this->config_.batch_size * this->config_.seq_len);
  //  cudaDeviceSynchronize();


//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
//      std::cout << "Post att_soft, token " << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->state_.att+i*this->config_.n_heads*this->config_.seq_len, 12, this->config_.seq_len );
//      cudaDeviceSynchronize();
//    }
//  }

  ops::attention_2_gemm( this->state_.att,
                         this->kv_cache_.value( layer_num, 0, 0, 0 ),
                         this->state_.xb,
                         this->kv_cache_.n_layers_,
                         this->config_.seq_len,
                         head_size,
                         this->config_.n_heads,
                         this->curr_concurrency_size,
                         this->config_.kv_prompt_limit,
                         this->id_allocation_,
                         this->token_pos_ );
  // end of multihead attention

  //  std::cout << "Post MHA" << std::endl;
  //  print_arr<<<1, 1>>>(this->state_.xb, 8, 1);
  //  cudaDeviceSynchronize();
//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
//      std::cout << "Post MHA, token " << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->state_.xb+i*dim, 12, 1 );
//      cudaDeviceSynchronize();
//    }
//  }

  // final matmul to get the output of the attention
  ops::matmul( this->state_.xb2, this->state_.xb, layer_weights.wo, this->curr_concurrency_size, dim, dim );
//  if (layer_num == 0)
//  for (size_t i = 0; i < 2; i++){
//    if (this->token_pos_[i] == 772){
//      std::cout << "Post Wo, token " << this->token_pos_[i] << std::endl << std::flush;
//      print_arr<<<1, 1>>>( this->state_.xb2+i*dim, 10, 1 );
//      cudaDeviceSynchronize();
//    }
//  }

  // residual connection back into x
  ops::accum( x, this->state_.xb2, dim, this->curr_concurrency_size );

  // ffn rmsnorm
  ops::rmsnorm( this->state_.xb, x, layer_weights.rms_ffn_weight, dim, this->curr_concurrency_size );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops::matmul( this->state_.hb, this->state_.xb, layer_weights.w1, this->curr_concurrency_size, dim, hidden_dim );
  ops::matmul( this->state_.hb2, this->state_.xb, layer_weights.w3, this->curr_concurrency_size, dim, hidden_dim );

  ops::silu( this->state_.hb, this->state_.hb2, hidden_dim, this->curr_concurrency_size );

  // final matmul to get the output of the ffn
  ops::matmul( this->state_.xb, this->state_.hb, layer_weights.w2, this->curr_concurrency_size, hidden_dim, dim );

  // residual connection
  ops::accum( x, this->state_.xb, dim, this->curr_concurrency_size );

//  if (layer_num == 0)
//    for (size_t i = 0; i < 2; i++){
//      if (this->token_pos_[i] == 772){
//        std::cout << "Post Layer, token " << this->token_pos_[i] << std::endl << std::flush;
//        print_arr<<<1, 1>>>( x+i*dim, 12, 1 );
//        cudaDeviceSynchronize();
//      }
//    }
}

template<typename DType>
void Llama2<DType>::pass_end()
{
  // final rmsnorm
  ops::rmsnorm( this->state_.x,
                this->state_.x,
                this->base_weights_.rms_final_weight,
                this->config_.dim,
                this->curr_concurrency_size );

  // classifier into logits
  ops::matmul( this->state_.logits,
               this->state_.x,
               this->base_weights_.wcls,
               this->curr_concurrency_size,
               this->config_.dim,
               this->config_.vocab_size );
}

template<typename DType>
uint32_t extract_token( const RunState<DType>& state,
                        const Config& config,
                        const float temp,
                        const uint64_t batch_index = 0 )
{
  uint32_t next_token;

  if ( temp == 0.0f ) {
    // greedy argmax sampling
    next_token = ops::argmax( state.logits + batch_index * config.vocab_size, config.vocab_size );
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
std::vector<uint32_t> extract_batch_token( const RunState<DType>& state,
                                           const Config& config,
                                           const std::vector<float>& temp )
{
  std::vector<uint32_t> next_tokens;
  for ( size_t i = 0; i < temp.size(); i++ )
    next_tokens.push_back( extract_token( state, config, temp[i], i ) );
  return next_tokens;
}

template<typename DType>
std::vector<InferenceState<DType>> Llama2<DType>::forward(
  const std::vector<std::reference_wrapper<const InferenceState<DType>>>& inference_state_s,
  const std::vector<uint32_t>& prompt_id_s )
{
  CHECK_GT( inference_state_s.size(), 0 ) << "batch size must be at least 1";
  CHECK_EQ( inference_state_s.size(), prompt_id_s.size() ) << "token size must be the same as prompt_id size";
  CHECK_LE( inference_state_s.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";
  for ( auto& item : inference_state_s ) {
    CHECK_EQ( item.get().next_layer(), this->start_layer_num_ ) << "next_layer must be the start layer";
  }

  for ( size_t i = 0; i < prompt_id_s.size(); i++ ) {
    this->id_allocation_[i] = prompt_id_s[i];
    this->token_pos_[i] = inference_state_s[i].get().token_pos();
    CHECK_LT( this->token_pos_[i], this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  this->curr_concurrency_size = inference_state_s.size();

  if ( inference_state_s[0].get().next_layer() == 0 ) {
    std::vector<uint32_t> token_vector;
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      token_vector.push_back( inference_state_s[i].get().token() );
    pass_begin( token_vector );
  } else {
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      // load the activations
      ops::CHECK_CUDA( cudaMemcpyAsync( this->state_.x + i * this->config_.dim * sizeof( DType ),
                                        inference_state_s[i].get().activations().ptr.get(),
                                        this->config_.dim * sizeof( DType ),
                                        cudaMemcpyHostToDevice ) );
  }

  for ( int layer_num = this->start_layer_num_; layer_num <= this->end_layer_num_; layer_num++ ) {
    transformer_layer( layer_num );
  }

  std::vector<InferenceState<DType>> token_vector;

  if ( this->end_layer_num_ == this->config_.n_layers - 1 ) {
    pass_end();

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      batch_temps.push_back( inference_state_s[i].get().temperature() );
    std::vector<uint32_t> next_tokens = extract_batch_token( this->state_, this->config_, batch_temps );
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      token_vector.emplace_back( next_tokens[i],                             // token
                                 inference_state_s[i].get().token_pos() + 1, // token_pos
                                 0,                                          // next_layer
                                 inference_state_s[i].get().temperature(),   // temperature
                                 DataBuffer<DType> {}                        // activations
      );

    return token_vector;
  }

  for ( size_t i = 0; i < inference_state_s.size(); i++ ) {
    DataBuffer<DType> activations { make_unique<DType[]>( this->config_.dim ), this->config_.dim };

    ops::CHECK_CUDA( cudaMemcpy( activations.ptr.get(),
                                 this->state_.x + i * this->config_.dim * sizeof( DType ),
                                 this->config_.dim * sizeof( DType ),
                                 cudaMemcpyDeviceToHost ) );

    token_vector.emplace_back( inference_state_s[i].get().token(),                // token
                               inference_state_s[i].get().token_pos(),            // token_pos
                               static_cast<uint32_t>( this->end_layer_num_ ) + 1, // next_layer
                               inference_state_s[i].get().temperature(),          // temperature
                               move( activations )                                // activations
    );
  }

  return token_vector;
}

template<typename DType>
std::vector<uint32_t> Llama2<DType>::forward( const std::vector<uint32_t>& token_s,
                                              const std::vector<uint32_t>& prompt_id_s,
                                              const std::vector<uint32_t>& token_pos_s )
{
  CHECK_GT( token_s.size(), 0 ) << "batch size must be at least 1";
  CHECK_EQ( token_s.size(), prompt_id_s.size() ) << "token size must be the same as prompt_id size";
  CHECK_EQ( token_s.size(), token_pos_s.size() ) << "token size must be the same as token_pos size";
  CHECK_LE( token_s.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  for ( size_t i = 0; i < prompt_id_s.size(); i++ ) {
    this->id_allocation_[i] = prompt_id_s[i];
    this->token_pos_[i] = token_pos_s[i];
    CHECK_LT( token_pos_s[i], this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  this->curr_concurrency_size = token_s.size();

  pass_begin( token_s );

  for ( int layer_num = this->start_layer_num_; layer_num <= this->end_layer_num_; layer_num++ ) {
    transformer_layer( layer_num );
  }

  pass_end();

  return extract_batch_token( this->state_, this->config_, std::vector<float>( token_s.size(), temperature_ ) );
}

template<typename DType>
std::vector<InferenceState<DType>> Llama2<DType>::forward( const std::vector<InferenceState<DType>>& inference_state_s,
                                                           const std::vector<uint32_t>& prompt_id_s )
{
  std::vector<std::reference_wrapper<const InferenceState<DType>>> res;
  for ( auto& state : inference_state_s )
    res.push_back( std::ref( state ) );
  return forward( res, prompt_id_s );
}

template<typename DType>
InferenceState<DType> Llama2<DType>::forward( const InferenceState<DType>& inference_state, const uint32_t& prompt_id )
{
  std::vector<std::reference_wrapper<const InferenceState<DType>>> token_vector;
  token_vector.push_back( std::ref( inference_state ) );
  std::vector<uint32_t> prompt_id_vector = { prompt_id };
  return move( forward( token_vector, prompt_id_vector )[0] );
}

template<typename DType>
uint32_t Llama2<DType>::forward( const uint32_t& token, const uint32_t& prompt_id, const uint32_t& token_pos )
{
  std::vector<uint32_t> token_vector = { token };
  std::vector<uint32_t> prompt_id_vector = { prompt_id };
  std::vector<uint32_t> token_pos_vector = { token_pos };
  return forward( token_vector, prompt_id_vector, token_pos_vector )[0];
}

template class Llama2<float>;
template class Llama2<__half>;

} // namespace glinthawk::models::llama2::cuda
