#pragma once

namespace glinthawk {

namespace ops {

void accum( float* a, const float* b, const int size );
void rmsnorm( float* o, const float* x, const float* weight, const int size );
void softmax( float* x, const int size );
void matmul( float* xout, const float* x, const float* w, const int n, const int d );

}

}
