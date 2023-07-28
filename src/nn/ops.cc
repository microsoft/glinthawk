#include "ops.hh"

#include <cmath>
#include <pmmintrin.h>

namespace glinthawk {

namespace ops {

void accum( float* a, const float* b, const int size )
{
#ifdef __SSE3__
  for ( int i = 0; i < size; i += 4 ) {
    __m128 avec = _mm_load_ps( &a[i] );
    __m128 bvec = _mm_load_ps( &b[i] );
    avec = _mm_add_ps( avec, bvec );
    _mm_store_ps( &a[i], avec );
  }
#else
  for ( int i = 0; i < size; i++ ) {
    a[i] += b[i];
  }
#endif
}

void rmsnorm( float* output, const float* x, const float* weight, const int size )
{
#ifdef __SSE3__
  __m128 ss = _mm_setzero_ps();
  const __m128 epsilon = _mm_set1_ps( 1e-5f );
  for ( int j = 0; j < size; j += 4 ) {
    __m128 xvec = _mm_load_ps( &x[j] );
    ss = _mm_add_ps( ss, _mm_mul_ps( xvec, xvec ) );
  }
  ss = _mm_hadd_ps( ss, ss );
  ss = _mm_hadd_ps( ss, ss );

  ss = _mm_div_ps( ss, _mm_set1_ps( static_cast<float>( size ) ) );
  ss = _mm_add_ps( ss, epsilon );
  ss = _mm_rsqrt_ps( ss ); // reciprocal of square root

  for ( int j = 0; j < size; j += 4 ) {
    __m128 xvec = _mm_load_ps( &x[j] );
    __m128 wvec = _mm_load_ps( &weight[j] );
    __m128 res = _mm_mul_ps( wvec, _mm_mul_ps( ss, xvec ) );
    _mm_store_ps( &output[j], res );
  }
#else
  // calculate sum of squares
  float ss = 0.0f;
  for ( int j = 0; j < size; j++ ) {
    ss += x[j] * x[j];
  }

  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  // normalize and scale
  for ( int j = 0; j < size; j++ ) {
    output[j] = weight[j] * ( ss * x[j] );
  }
#endif
}

void softmax( float* x, const int size )
{
  // find max value (for numerical stability)
  float max_val = x[0];
  for ( int i = 1; i < size; i++ ) {
    if ( x[i] > max_val ) {
      max_val = x[i];
    }
  }

  // exp and sum
  float sum = 0.0f;
  for ( int i = 0; i < size; i++ ) {
    x[i] = expf( x[i] - max_val );
    sum += x[i];
  }

  // normalize
  for ( int i = 0; i < size; i++ ) {
    x[i] /= sum;
  }
}

// W(d,n) @ x(n,) -> xout(d,)
void matmul( float* xout, const float* x, const float* w, const int n, const int d )
{
  int i;
#pragma omp parallel for private( i )
#ifdef __SSE3__
  for ( i = 0; i < d; i++ ) {
    __m128 val = _mm_setzero_ps();
    for ( int j = 0; j < n; j += 4 ) {
      __m128 xVec = _mm_load_ps( &x[j] );
      __m128 wVec = _mm_load_ps( &w[i * n + j] );
      val = _mm_add_ps( val, _mm_mul_ps( xVec, wVec ) );
    }
    val = _mm_hadd_ps( val, val );
    val = _mm_hadd_ps( val, val );
    _mm_store_ss( &xout[i], val );
  }
#else
  for ( i = 0; i < d; i++ ) {
    float val = 0.0f;
    for ( int j = 0; j < n; j++ ) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
#endif
}

int sample( const float* probabilities, const int n )
{
  // sample index from probabilities, they must sum to 1
  float r = static_cast<float>( rand() ) / RAND_MAX;
  float cdf = 0.0f;

  for ( int i = 0; i < n; i++ ) {
    cdf += probabilities[i];
    if ( r < cdf ) {
      return i;
    }
  }

  return n - 1; // in case of rounding errors
}

int argmax( const float* v, const int n )
{
  // return argmax of v in elements 0..n
  int max_i = 0;
  float max_p = v[0];

  for ( int i = 1; i < n; i++ ) {
    if ( v[i] > max_p ) {
      max_i = i;
      max_p = v[i];
    }
  }

  return max_i;
}

}

}
