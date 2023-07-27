#include "ops.hh"

#include <cmath>

namespace glinthawk {

namespace ops {

void accum( float* a, const float* b, const int size )
{
  for ( int i = 0; i < size; i++ ) {
    a[i] += b[i];
  }
}

void rmsnorm( float* output, const float* x, const float* weight, const int size )
{
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

  for ( i = 0; i < d; i++ ) {
    float val = 0.0f;
    for ( int j = 0; j < n; j++ ) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

int sample( const float* probabilities, const int n )
{
  // sample index from probabilities, they must sum to 1
  float r = (float)rand() / (float)RAND_MAX;
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
