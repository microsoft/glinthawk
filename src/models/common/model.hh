#pragma once

namespace glinthawk::models {

template<typename DType>
struct DataBuffer
{
  std::unique_ptr<DType[]> ptr { nullptr };
  uint64_t len { 0 };

  DataBuffer() = default;
  DataBuffer( std::unique_ptr<DType[]>&& other_ptr, const uint64_t other_len )
    : ptr( std::move( other_ptr ) )
    , len( other_len )
  {
  }
};

template<typename DType>
class InferenceState
{
private:
  uint32_t token_ { 1 };
  uint32_t token_pos_ { 0 };
  uint32_t next_layer_ { 0 };
  float temperature_ { 0.0f };
  DataBuffer<DType> activations_ {};

public:
  InferenceState( const uint32_t token,
                  const uint32_t token_pos,
                  const uint32_t next_layer,
                  const float temperature,
                  DataBuffer<DType>&& activations )
    : token_( token )
    , token_pos_( token_pos )
    , next_layer_( next_layer )
    , temperature_( temperature )
    , activations_( std::move( activations ) )
  {
  }

  InferenceState( const std::string_view serialized );
  std::string serialize();

  uint32_t token() const { return token_; }
  uint32_t token_pos() const { return token_pos_; }
  uint32_t next_layer() const { return next_layer_; }
  float temperature() const { return temperature_; }
  const DataBuffer<DType>& activations() const { return activations_; }
};

template<typename DType>
class Model
{
public:
  virtual ~Model() = default;
  virtual InferenceState<DType> forward( const InferenceState<DType>& inference_state ) = 0;
  virtual std::vector<InferenceState<DType>> forward( const std::vector<InferenceState<DType>>& inference_state_s ) = 0;
};

} // namespace glinthawk::models
