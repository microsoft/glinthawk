#include <concepts>
#include <cstdint>

namespace glinthawk::models {

template<class T>
concept Model = requires( T t ) {
  {
    T::dim
  } -> std::unsigned_integral;
  {
    T::kv_dim
  } -> std::unsigned_integral;
  {
    T::hidden_dim
  } -> std::unsigned_integral;
  {
    T::n_layers
  } -> std::unsigned_integral;
  {
    T::head_size
  } -> std::unsigned_integral;
  {
    T::n_heads
  } -> std::unsigned_integral;
  {
    T::n_kv_heads
  } -> std::unsigned_integral;
  {
    T::gqa_size
  } -> std::unsigned_integral;
  {
    T::vocab_size
  } -> std::unsigned_integral;
  {
    T::seq_len
  } -> std::unsigned_integral;
};

template<typename T>
  requires Model<T>
struct Llama2_70B
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
};

struct Llama2_13B
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
};

struct Llama2_7B
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
};

}