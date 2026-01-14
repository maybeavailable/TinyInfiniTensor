#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================

        IT_ASSERT(inputs.size() == 2);
        const auto A = inputs[0];
        const auto B = inputs[1];

        const Shape dimsA = A->getDims();
        const Shape dimsB = B->getDims();
        const int rankA = static_cast<int>(dimsA.size());
        const int rankB = static_cast<int>(dimsB.size());
        IT_ASSERT(rankA >= 2 && rankB >= 2);

        const int a0 = dimsA[rankA - 2];
        const int a1 = dimsA[rankA - 1];
        const int b0 = dimsB[rankB - 2];
        const int b1 = dimsB[rankB - 1];

        const int m_ = transA ? a1 : a0;
        const int kA = transA ? a0 : a1;
        const int kB = transB ? b1 : b0;
        const int n_ = transB ? b0 : b1;

        IT_ASSERT(kA == kB);

        m = m_;
        n = n_;
        k = kA;

        Shape batchA, batchB;
        if (rankA > 2)
            batchA = Shape(dimsA.begin(), dimsA.end() - 2);
        if (rankB > 2)
            batchB = Shape(dimsB.begin(), dimsB.end() - 2);

        Shape out = infer_broadcast(batchA, batchB);
        out.push_back(m);
        out.push_back(n);
        return {{out}};
    }

} // namespace infini