#include <gtest/gtest.h>

#include <models/common/cpu/ops.hh>

#include <algorithm>
#include <vector>

using namespace std;
using namespace glinthawk::models::common::cpu;

using DTypes = ::testing::Types<float, _Float16>;

template<class>
struct OperationsCPU : public ::testing::Test
{};
TYPED_TEST_SUITE( OperationsCPU, DTypes );

TYPED_TEST( OperationsCPU, AccumBasic )
{
  const uint64_t size = 16;
  const uint64_t batch_size = 4;

  vector<TypeParam> a( size * batch_size, static_cast<TypeParam>( 0.0f ) );
  vector<TypeParam> b( size * batch_size, static_cast<TypeParam>( 1.0f ) );

  ops::accum( a.data(), b.data(), size, batch_size );

  for ( uint64_t i = 0; i < size * batch_size; ++i ) {
    EXPECT_EQ( a[i], 1.0f );
  }
}

TYPED_TEST( OperationsCPU, MatMulBasic )
{
  const uint64_t a = 4;
  const uint64_t b = 5;
  const uint64_t c = 3;

  vector<TypeParam> A { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
  vector<TypeParam> B { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  vector<TypeParam> C( a * c, static_cast<TypeParam>( 0.0f ) );

  ops::matmul( C.data(), A.data(), B.data(), a, b, c );

  vector<TypeParam> expected { 55, 130, 205, 130, 330, 530, 205, 530, 855, 280, 730, 1180 };

  EXPECT_TRUE( equal( C.begin(), C.end(), expected.begin() ) );
}

int main( int argc, char* argv[] )
{
  testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
