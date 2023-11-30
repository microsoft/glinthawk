#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace glinthawk::models::llama2 {

template<class T>
concept ModelConfig = requires( T t ) {
  T::dim;
  requires std::is_unsigned_v<decltype( T::dim )>;

  T::kv_dim;
  requires std::is_unsigned_v<decltype( T::kv_dim )>;

  T::hidden_dim;
  requires std::is_unsigned_v<decltype( T::hidden_dim )>;

  T::n_layers;
  requires std::is_unsigned_v<decltype( T::n_layers )>;

  T::head_size;
  requires std::is_unsigned_v<decltype( T::head_size )>;

  T::n_heads;
  requires std::is_unsigned_v<decltype( T::n_heads )>;

  T::n_kv_heads;
  requires std::is_unsigned_v<decltype( T::n_kv_heads )>;

  T::gqa_size;
  requires std::is_unsigned_v<decltype( T::gqa_size )>;

  T::vocab_size;
  requires std::is_unsigned_v<decltype( T::vocab_size )>;

  T::seq_len;
  requires std::is_unsigned_v<decltype( T::seq_len )>;

  T::wcls_present;
  requires std::is_convertible_v<decltype( T::wcls_present ), bool>;
};

namespace configs {

struct Llama2_70B_Chat
{
  constexpr static uint64_t dim = 8192;
  constexpr static uint64_t kv_dim = 1024;
  constexpr static uint64_t hidden_dim = 28672;
  constexpr static uint64_t n_layers = 80;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 64;
  constexpr static uint64_t n_kv_heads = 8;
  constexpr static uint64_t gqa_size = 8;
  constexpr static uint64_t vocab_size = 32000;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
};

struct Llama2_13B_Chat
{
  constexpr static uint64_t dim = 5120;
  constexpr static uint64_t kv_dim = 5120;
  constexpr static uint64_t hidden_dim = 13824;
  constexpr static uint64_t n_layers = 40;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 40;
  constexpr static uint64_t n_kv_heads = 40;
  constexpr static uint64_t gqa_size = 1;
  constexpr static uint64_t vocab_size = 32000;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
};

struct Llama2_7B_Chat
{
  constexpr static uint64_t dim = 4096;
  constexpr static uint64_t kv_dim = 4096;
  constexpr static uint64_t hidden_dim = 11008;
  constexpr static uint64_t n_layers = 32;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 32;
  constexpr static uint64_t n_kv_heads = 32;
  constexpr static uint64_t gqa_size = 1;
  constexpr static uint64_t vocab_size = 32000;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
};

struct Stories_110M
{
  constexpr static uint64_t dim = 768;
  constexpr static uint64_t kv_dim = 768;
  constexpr static uint64_t hidden_dim = 2048;
  constexpr static uint64_t n_layers = 12;
  constexpr static uint64_t head_size = 64;
  constexpr static uint64_t n_heads = 12;
  constexpr static uint64_t n_kv_heads = 12;
  constexpr static uint64_t gqa_size = 1;
  constexpr static uint64_t vocab_size = 32000;
  constexpr static uint64_t seq_len = 1024;
  constexpr static bool wcls_present = false;
};

static_assert( ModelConfig<Llama2_70B_Chat> );
static_assert( ModelConfig<Llama2_13B_Chat> );
static_assert( ModelConfig<Llama2_7B_Chat> );
static_assert( ModelConfig<Stories_110M> );

} // namespace configs

} // namespace glinthawk::models::llama2
