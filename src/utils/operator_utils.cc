#include "utils/operator_utils.h"
#include "core/runtime.h"
#include <algorithm>

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // ONNX 双向广播（numpy 规则）：
    //   - 从右往左逐维对齐
    //   - 每维要么相等，要么有一个是 1（1 会广播成另一边）
    //   - 输出 shape = 逐维取 max
    // 例如 {1,3,2,4} 与 {4} 广播 → {1,3,2,4}
    // =================================== 作业 ===================================
    int rank = std::max(A.size(), B.size());
    Shape result(rank, 1);
    for (int i = 1; i <= rank; ++i)
    {
        int a = i <= (int)A.size() ? A[A.size() - i] : 1;
        int b = i <= (int)B.size() ? B[B.size() - i] : 1;
        IT_ASSERT(a == b || a == 1 || b == 1,
                  "Incompatible shapes for broadcast");
        result[rank - i] = std::max(a, b);
    }
    return result;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
