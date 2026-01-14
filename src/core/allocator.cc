#include "core/allocator.h"
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
        // 采用 free-list + 合并 的方式进行模拟分配：
        // 1) 优先从空闲块中找可用块（best-fit，减少碎片）
        // 2) 若没有合适空闲块，则从末尾 bump 分配
        // 返回分配块的起始 offset
        // =================================== 作业 ===================================
        size_t bestStart = 0;
        size_t bestSize = 0;
        bool found = false;
        for (const auto &kv : freeBlocks)
        {
            const size_t start = kv.first;
            const size_t blkSize = kv.second;
            if (blkSize < size)
                continue;
            if (!found || blkSize < bestSize)
            {
                bestStart = start;
                bestSize = blkSize;
                found = true;
                if (blkSize == size)
                    break;
            }
        }

        if (found)
        {
            auto it = freeBlocks.find(bestStart);
            IT_ASSERT(it != freeBlocks.end());
            if (bestSize == size)
            {
                freeBlocks.erase(it);
            }
            else
            {
                freeBlocks.erase(it);
                freeBlocks.emplace(bestStart + size, bestSize - size);
            }
            return bestStart;
        }

        const size_t offset = this->used;
        this->used += size;
        if (this->used > this->peak)
            this->peak = this->used;
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // 回收逻辑：
        // 1) 若释放的是末尾块，直接回退 used，并持续吞并末尾相邻的空闲块
        // 2) 否则插入 freeBlocks，并与前后相邻空闲块合并
        // =================================== 作业 ===================================
        IT_ASSERT(size > 0);
        IT_ASSERT(addr + size <= this->used);

        // Case 1: free at the end -> shrink
        if (addr + size == this->used)
        {
            this->used = addr;
            // continue shrinking if there are free blocks at the new end
            while (true)
            {
                if (freeBlocks.empty())
                    break;
                auto it = freeBlocks.upper_bound(this->used);
                if (it == freeBlocks.begin())
                    break;
                --it;
                const size_t start = it->first;
                const size_t blkSize = it->second;
                if (start + blkSize != this->used)
                    break;
                this->used = start;
                freeBlocks.erase(it);
            }
            return;
        }

        // Case 2: insert + coalesce
        size_t newStart = addr;
        size_t newSize = size;

        auto next = freeBlocks.lower_bound(newStart);
        if (next != freeBlocks.begin())
        {
            auto prev = std::prev(next);
            if (prev->first + prev->second == newStart)
            {
                newStart = prev->first;
                newSize += prev->second;
                freeBlocks.erase(prev);
            }
        }

        next = freeBlocks.lower_bound(newStart);
        if (next != freeBlocks.end() && newStart + newSize == next->first)
        {
            newSize += next->second;
            freeBlocks.erase(next);
        }

        freeBlocks.emplace(newStart, newSize);
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
