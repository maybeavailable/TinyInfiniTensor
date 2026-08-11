#include "core/allocator.h"
#include <algorithm>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // 策略：first-fit，从头到尾找到第一个大小足够的空闲块并复用
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it)
        {
            if (it->second >= size)
            {
                size_t offset = it->first;
                // 若空闲块比所需更大，把剩余部分作为一个新的空闲块放回表里
                if (it->second > size)
                    free_blocks.emplace(offset + size, it->second - size);
                free_blocks.erase(it);
                return offset;
            }
        }

        // 没有可复用的空闲块，就在内存尾部（bump pointer）扩展分配
        size_t offset = used;
        used += size;
        peak = std::max(peak, used);
        return offset;
        // =================================== 作业 ===================================
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // 先与右侧相邻的空闲块合并：addr+size 正好是某个空闲块的起点
        auto rightIt = free_blocks.find(addr + size);
        if (rightIt != free_blocks.end())
        {
            size += rightIt->second;
            free_blocks.erase(rightIt);
        }

        // 再与左侧相邻的空闲块合并：找起点小于 addr 的最大空闲块，
        // 若它的终点正好是 addr，则两者相邻
        auto it = free_blocks.lower_bound(addr);
        if (it != free_blocks.begin())
        {
            --it;
            if (it->first + it->second == addr)
            {
                addr = it->first;
                size += it->second;
                free_blocks.erase(it);
            }
        }

        // 若合并后的空闲块正好位于内存尾部，则可以直接回退 bump pointer，
        // 下次 alloc 从 used 处扩展时自然能复用这段空间，无需记入空闲表
        if (addr + size == used)
            used = addr;
        else
            free_blocks.emplace(addr, size);
        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
