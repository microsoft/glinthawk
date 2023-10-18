#include <gtest/gtest.h>

#include <models/common/cpu/ops.hh>

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

int main( int argc, char* argv[] )
{
  testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
