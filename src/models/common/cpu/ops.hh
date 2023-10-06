#pragma once

#include <cstdint>

namespace glinthawk::models::common::cpu {

void accum( float* a, const float* b, const uint64_t size );
void rmsnorm( float* o, const float* x, const float* weight, const uint64_t size );
void softmax( float* x, const uint64_t size );
void matmul( float* xout, const float* x, const float* w, const uint64_t n, const uint64_t d );

uint32_t sample( const float* probabilities, const uint64_t n );
uint32_t argmax( const float* v, const uint64_t n );

}
