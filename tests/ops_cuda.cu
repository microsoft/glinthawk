#include <gtest/gtest.h>

#include <models/common/cuda/ops.cuh>

#include <thrust/device_vector.h>
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

int main( int argc, char* argv[] )
{
  testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
