#pragma once

namespace glinthawk::models {

template<typename DType>
class InferenceState
{};

template<typename DType>
class Model
{
public:
  InferenceState<DType> forward( const InferenceState<DType>& inference_state );
};

} // namespace glinthawk::models
