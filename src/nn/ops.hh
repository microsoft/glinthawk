#pragma once

namespace glinthawk {

namespace ops {

void accum( float* a, float* b, int size );
void rmsnorm( float* o, float* x, float* weight, int size );
void softmax( float* x, int size );
void matmul( float* xout, float* x, float* w, int n, int d );

}

}
