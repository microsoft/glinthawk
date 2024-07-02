#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "arch/float.hh"
#include "variants.hh"

namespace glinthawk::models::llama2 {

/// @brief "Context" for Llama2 model is the KV-cache. InferenceState should be accompanied by its Context when passing
/// to the model, i.e., Llama2::forward(inference_state, context). Context is mutated after the forward pass and needs
/// to be kept for the next forward pass.

template<typename T, typename DType>
concept TokenContextConcept = requires( T t ) {
  { T() };
  { t.key() } -> std::same_as<DType*>;
  { t.value() } -> std::same_as<DType*>;
  { t.key_head( 0ull ) } -> std::same_as<DType*>;
  { t.value_head( 0ull ) } -> std::same_as<DType*>;
  { t.empty() } -> std::same_as<bool>;
  { T::size() } -> std::same_as<size_t>;
};

template<typename T, typename DType>
concept LayerContextConcept = TokenContextConcept<typename T::TokenContextType, DType> && requires( T t ) {
  { T() };
  { t.token( 0ull ) } -> std::same_as<typename T::TokenContextType>;
  { T::max_size() } -> std::same_as<size_t>;
};

template<typename T, typename DType>
concept ContextConcept = LayerContextConcept<typename T::LayerContextType, DType>
                         && TokenContextConcept<typename T::TokenContextType, DType> && requires( T t ) {
                              { T() };
                              { t.layer( 0ull ) } -> std::same_as<typename T::LayerContextType>;
                              { t.empty() } -> std::same_as<bool>;
                              { T::max_size( 0ull ) } -> std::same_as<size_t>;
                            };

// <forward declarations>
template<typename Config>
requires ModelConfig<Config>
class ConfigRuntime;

template<typename Config, typename DType>
requires ModelConfig<Config>
class LayerContext;

template<typename Config, typename DType>
requires ModelConfig<Config>
class Context;
// </forward declarations>

template<typename Config, typename DType>
requires ModelConfig<Config>
class TokenContext
{
  friend class LayerContext<Config, DType>;

protected:
  DType* buffer_ { nullptr };

  // Only LayerContext can construct non-empty TokenContext
  TokenContext( DType* buffer )
    : buffer_( buffer )
  {
  }

public:
  TokenContext() {}

  DType* key() { return buffer_; }
  const DType* key() const { return buffer_; }

  DType* value() { return buffer_ + Config::kv_dim; }
  const DType* value() const { return buffer_ + Config::kv_dim; }

  DType* key_head( const size_t h_idx ) { return key() + h_idx * Config::head_size; }
  const DType* key_head( const size_t h_idx ) const { return key() + h_idx * Config::head_size; }

  DType* value_head( const size_t h_idx ) { return value() + h_idx * Config::head_size; }
  const DType* value_head( const size_t h_idx ) const { return value() + h_idx * Config::head_size; }

  constexpr static size_t size() { return Config::kv_dim * 2 * sizeof( DType ); }
  bool empty() const { return buffer_ == nullptr; }
};

template<typename Config, typename DType>
requires ModelConfig<Config>
class LayerContext
{
  friend class Context<Config, DType>;

protected:
  DType* buffer_ { nullptr };

  // Only Context can construct non-empty LayerContext
  LayerContext( DType* buffer )
    : buffer_( buffer )
  {
  }

public:
  using TokenContextType = TokenContext<Config, DType>;

  LayerContext() {}

  TokenContextType token( const int token_num )
  {
    if ( buffer_ == nullptr )
      return { nullptr };
    return { buffer_ + token_num * Config::kv_dim * 2 };
  }

  const TokenContextType token( const int token_num ) const
  {
    if ( buffer_ == nullptr )
      return { nullptr };
    return { buffer_ + token_num * Config::kv_dim * 2 };
  }

  constexpr static size_t max_size() { return Config::seq_len * TokenContext<Config, DType>::size(); }
  constexpr static bool is_contiguous() { return true; } // i.e., all kv-pairs for a layer are contiguous in memory
};

template<typename Config, typename DType>
requires ModelConfig<Config>
class Context
{
protected:
  size_t nlayers_loaded_;
  size_t start_layer_num_;
  DType* buffer_;
  // TODO: Is there a benefit to maintaining context state (such as token count)?
  // TODO: The CPU attention machines do not need to know about layers for reserving context. Should we redesign them
  //  that way?

public:
  using LayerContextType = LayerContext<Config, DType>;
  using TokenContextType = TokenContext<Config, DType>;

  Context( const ConfigRuntime<Config>& settings, DType* buffer )
    : nlayers_loaded_( settings.n_layers_loaded() )
    , start_layer_num_( settings.start_layer_num )
    , buffer_( buffer )
  {
  }

  Context( const ConfigRuntime<Config>& settings )
    : nlayers_loaded_( settings.n_layers_loaded() )
    , start_layer_num_( settings.start_layer_num )
    , buffer_( nullptr )
  {
  }

  Context()
    : nlayers_loaded_( 0 )
    , start_layer_num_( 0 )
    , buffer_( nullptr )
  {
  }

  // This function is always called before processing a state, with the current layer number
  // and token position. For dynamic contexts that allocate memory differently, this function
  // should be overridden. Returns true on success, false otherwise.
  bool prepare( [[maybe_unused]] const size_t layer_num, [[maybe_unused]] const size_t token_pos ) { return true; }

  LayerContextType layer( const int layer_num ) const
  {
    if ( buffer_ == nullptr )
      return { nullptr };
    return { buffer_ + ( layer_num - start_layer_num_ ) * LayerContextType::max_size() / sizeof( DType ) };
  }

  static size_t max_size( const size_t n_layers ) { return n_layers * LayerContextType::max_size(); }
  bool empty() const { return buffer_ == nullptr; }
};

static_assert( ContextConcept<Context<configs::Stories_110M, glinthawk::float32_t>, glinthawk::float32_t> );
static_assert( LayerContextConcept<LayerContext<configs::Stories_110M, glinthawk::float32_t>, glinthawk::float32_t> );
static_assert( TokenContextConcept<TokenContext<configs::Stories_110M, glinthawk::float32_t>, glinthawk::float32_t> );

} // namespace glinthawk::models::llama2
