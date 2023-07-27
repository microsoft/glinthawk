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

}

}
