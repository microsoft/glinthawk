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

  max_steps_ = config_.seq_len;

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

void Llama2::transformer( const int token, const int pos )
{
  // a few convenience variables
  float* x = state_.x.get();
  const int dim = config_.dim;
  const int hidden_dim = config_.hidden_dim;
  const int head_size = dim / config_.n_heads;

  // copy the token embedding into x
  const float* content_row = &( weights_.token_embedding_table[token * dim] );
  memcpy( x, content_row, dim * sizeof( *x ) );

  // Question(sadjad): wtf is this?
  // pluck out the "pos" row of freq_cis_real and freq_cis_imag
  float* freq_cis_real_row = weights_.freq_cis_real + pos * head_size / 2;
  float* freq_cis_imag_row = weights_.freq_cis_imag + pos * head_size / 2;

  for ( int layer_num = 0; layer_num < config_.n_layers; layer_num++ ) {
    // attention rmsnorm
    ops::rmsnorm( state_.xb.get(), x, weights_.rms_att_weight + layer_num * dim, dim );

    const off64_t weights_offset = layer_num * dim * dim;

    // qkv matmuls for this position
    ops::matmul( state_.q.get(), state_.xb.get(), weights_.wq + weights_offset, dim, dim );
    ops::matmul( state_.k.get(), state_.xb.get(), weights_.wk + weights_offset, dim, dim );
    ops::matmul( state_.v.get(), state_.xb.get(), weights_.wv + weights_offset, dim, dim );

    // apply RoPE rotation to the q and k vectors for each head
    for ( int head_num = 0; head_num < config_.n_heads; head_num++ ) {
      // get the q and k vectors for this head
      float* q = state_.q.get() + head_num * head_size;
      float* k = state_.k.get() + head_num * head_size;

      // rotate q and k by the freq_cis_real and freq_cis_imag
      for ( int i = 0; i < head_size; i += 2 ) {
        float q0 = q[i];
        float q1 = q[i + 1];
        float k0 = k[i];
        float k1 = k[i + 1];
        float fcr = freq_cis_real_row[i / 2];
        float fci = freq_cis_imag_row[i / 2];
        q[i] = q0 * fcr - q1 * fci;
        q[i + 1] = q0 * fci + q1 * fcr;
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
      }
    }

    // save key,value at this time step (pos) to our kv cache
    const off64_t loff = layer_num * config_.seq_len * dim; // kv cache layer offset for convenience
    float* key_cache_row = state_.key_cache.get() + loff + pos * dim;
    float* value_cache_row = state_.value_cache.get() + loff + pos * dim;
    memcpy( key_cache_row, state_.k.get(), dim * sizeof( *key_cache_row ) );
    memcpy( value_cache_row, state_.v.get(), dim * sizeof( *value_cache_row ) );

    // multihead attention. iterate over all heads
    int head_num;
#pragma omp parallel for private( head_num )
    for ( head_num = 0; head_num < config_.n_heads; head_num++ ) {
      // get the query vector for this head
      const float* q = state_.q.get() + head_num * head_size;

      // attention scores for this head
      float* att = state_.att.get() + head_num * config_.seq_len;

      // iterate over all timesteps, including the current one
      for ( int t = 0; t <= pos; t++ ) {
        // get the key vector for this head and at this timestep
        const float* k = state_.key_cache.get() + loff + t * dim + head_num * head_size;

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
      ops::softmax( att, pos + 1 );

      // weighted sum of the values, store back into xb
      float* xb = state_.xb.get() + head_num * head_size;
      memset( xb, 0, head_size * sizeof( float ) );

      for ( int t = 0; t <= pos; t++ ) {
        // get the value vector for this head and at this timestep
        const float* v = state_.value_cache.get() + loff + t * dim + head_num * head_size;

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
    ops::matmul( state_.xb2.get(), state_.xb.get(), weights_.wo + layer_num * dim * dim, dim, dim );

    // residual connection back into x
    ops::accum( x, state_.xb2.get(), dim );

    // ffn rmsnorm
    ops::rmsnorm( state_.xb.get(), x, weights_.rms_ffn_weight + layer_num * dim, dim );

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    ops::matmul( state_.hb.get(), state_.xb.get(), weights_.w1 + layer_num * dim * hidden_dim, dim, hidden_dim );
    ops::matmul( state_.hb2.get(), state_.xb.get(), weights_.w3 + layer_num * dim * hidden_dim, dim, hidden_dim );

    // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for ( int i = 0; i < hidden_dim; i++ ) {
      state_.hb[i] = state_.hb[i] * ( 1.0f / ( 1.0f + expf( -state_.hb[i] ) ) );
    }

    // elementwise multiply with w3(x)
    for ( int i = 0; i < hidden_dim; i++ ) {
      state_.hb[i] = state_.hb[i] * state_.hb2[i];
    }

    // final matmul to get the output of the ffn
    ops::matmul( state_.xb.get(), state_.hb.get(), weights_.w2 + layer_num * dim * hidden_dim, hidden_dim, dim );

    // residual connection
    ops::accum( x, state_.xb.get(), dim );
  }

  // final rmsnorm
  ops::rmsnorm( x, x, weights_.rms_final_weight, dim );

  // classifier into logits
  ops::matmul( state_.logits.get(), x, weights_.wcls, config_.dim, config_.vocab_size );
}

string Llama2::next_token()
{
  if ( current_pos_ >= max_steps_ ) {
    return string {};
  }

  int next_token;

  // forward the transformer to get logits for the next token
  transformer( current_token_, current_pos_ );

  // sample the next token
  if ( temperature_ == 0.0f ) {
    // greedy argmax sampling
    next_token = ops::argmax( state_.logits.get(), config_.vocab_size );
  } else {
    // apply the temperature to the logits
    for ( int q = 0; q < config_.vocab_size; q++ ) {
      state_.logits[q] /= temperature_;
    }

    // apply softmax to the logits to get the probabilities for next token
    ops::softmax( state_.logits.get(), config_.vocab_size );

    // we now want to sample from this distribution to get the next token
    next_token = ops::sample( state_.logits.get(), config_.vocab_size );
  }

  // advance forward
  current_token_ = next_token;
  current_pos_++;

  return vocabulary_[current_token_];
}
