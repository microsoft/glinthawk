#pragma once

#include <chrono>
#include <iostream>

#include "measurement.hh"

namespace glinthawk {

template<IntDistributions Category, typename F>
void timeit( Measurement& stats, F&& f )
{
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  stats.add_point<Category>( std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() );
}

} // namespace glinthawk
