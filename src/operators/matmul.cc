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
        // matmul：最后两维是矩阵，前面的维度是 batch，batch 之间做广播。
        // transA/transB 只交换矩阵维，不改变 batch 维。
        // 输出 = broadcast(batchA, batchB) + {m, n}
        // =================================== 作业 ===================================
        const auto A = inputs[0], B = inputs[1];
        auto dimsA = A->getDims(), dimsB = B->getDims();
        auto rankA = A->getRank(), rankB = B->getRank();

        // 有效矩阵维度
        m = transA ? dimsA[rankA - 1] : dimsA[rankA - 2];
        int kA = transA ? dimsA[rankA - 2] : dimsA[rankA - 1];
        int kB = transB ? dimsB[rankB - 1] : dimsB[rankB - 2];
        n = transB ? dimsB[rankB - 2] : dimsB[rankB - 1];
        IT_ASSERT(kA == kB, "Matmul: inner dims mismatch");
        k = kA;

        // batch 维度广播
        Shape batchA(dimsA.begin(), dimsA.end() - 2);
        Shape batchB(dimsB.begin(), dimsB.end() - 2);
        auto batch = infer_broadcast(batchA, batchB);

        batch.emplace_back(m);
        batch.emplace_back(n);
        return {{batch}};
    }

} // namespace infini