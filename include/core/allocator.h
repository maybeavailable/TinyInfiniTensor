#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>
namespace infini {
  class Allocator
  {
  private:
    Runtime runtime;

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // =================================== 作业 ===================================
    // 空闲块表：key 为 block 的起始地址，value 为 block 的大小。
    // 利用 map 的有序性，可以快速找到相邻的空闲块用于合并
    // =================================== 作业 ===================================
    std::map<size_t, size_t> free_blocks;

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
  };
}
