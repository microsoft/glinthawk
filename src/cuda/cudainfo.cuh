#pragma once

namespace glinthawk::gpu {

class CUDAInfo
{
private:
  size_t max_threads_per_block_ { 0 };

public:
  CUDAInfo();

  size_t max_threads_per_block() const { return max_threads_per_block_; }
};

};
