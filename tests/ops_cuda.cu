#include <gtest/gtest.h>

#include <models/common/cuda/ops.cuh>

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <vector>

using namespace std;
using namespace glinthawk::models::common::cuda;

using DTypes = ::testing::Types<float, __half>;

template<class>
struct OperationsCUDA : public ::testing::Test
{
protected:
  void SetUp() override { ops::init( 8 ); }
  void TearDown() override { ops::destroy(); }
};

TYPED_TEST_SUITE( OperationsCUDA, DTypes );

TYPED_TEST( OperationsCUDA, AccumBasic )
{
  const uint64_t size = 16;
  const uint64_t batch_size = 4;

  thrust::device_vector<TypeParam> a( size * batch_size, static_cast<TypeParam>( 0.0f ) );
  thrust::device_vector<TypeParam> b( size * batch_size, static_cast<TypeParam>( 1.0f ) );

  ops::accum( a.data().get(), b.data().get(), size, batch_size );

  thrust::host_vector<TypeParam> result { a };

  for ( uint64_t i = 0; i < size * batch_size; ++i ) {
    ASSERT_EQ( result[i], static_cast<TypeParam>( 1.0f ) );
  }
}

TYPED_TEST( OperationsCUDA, MatMulBasic )
{
  const uint64_t a = 4;
  const uint64_t b = 5;
  const uint64_t c = 3;

  thrust::device_vector<TypeParam> A { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
  thrust::device_vector<TypeParam> B { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  thrust::device_vector<TypeParam> C( a * c, static_cast<TypeParam>( 0.0f ) );

  ops::matmul( C.data().get(), A.data().get(), B.data().get(), a, b, c );

  thrust::host_vector<TypeParam> result { C };
  thrust::host_vector<TypeParam> expected { 55, 130, 205, 130, 330, 530, 205, 530, 855, 280, 730, 1180 };

  EXPECT_TRUE( thrust::equal( result.begin(), result.end(), expected.begin() ) );
}

int main( int argc, char* argv[] )
{
  testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
