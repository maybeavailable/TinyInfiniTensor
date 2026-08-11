#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // 图优化规则：
        //   1. 删除相邻且互逆的 transpose 算子（两个 transpose 复合为单位置换时互相抵消）
        //   2. 把 matmul 输入前的、仅交换最后两维的 transpose 融合进 transA / transB
        // =================================== 作业 ===================================
        // perm 是否恰好是"交换最后两维"的置换
        auto isSwapLastTwo = [](const std::vector<int> &perm, int rank) -> bool {
            if ((int)perm.size() != rank)
                return false;
            for (int d = 0; d < rank; ++d)
            {
                int expected =
                    (d == rank - 2) ? rank - 1 : (d == rank - 1) ? rank - 2 : d;
                if (perm[d] != expected)
                    return false;
            }
            return true;
        };
        // 两个置换 P、Q 复合后是否为单位置换：transpose(Q, transpose(P, x)) == x
        auto composeIsIdentity = [](const std::vector<int> &P,
                                    const std::vector<int> &Q) -> bool {
            if (P.size() != Q.size())
                return false;
            for (size_t j = 0; j < Q.size(); ++j)
                if (P[Q[j]] != (int)j)
                    return false;
            return true;
        };
        // 删除 op 时同步清理所有反向连接，避免残留悬空的 weak_ptr
        auto removeOp = [this](const Operator &op) {
            for (auto &in : op->getInputs())
                if (in)
                    in->removeTarget(op);
            for (auto &out : op->getOutputs())
                if (out)
                    out->setSource(nullptr);
            for (auto &pred : op->getPredecessors())
                if (pred)
                    pred->removeSuccessors(op);
            for (auto &succ : op->getSuccessors())
                if (succ)
                    succ->removePredecessors(op);
            removeOperator(op);
        };

        std::unordered_set<OperatorObj *> deadOps;
        std::unordered_set<UidBaseType> deadTensors;

        // ---- 规则 2：把 matmul 输入前的 transpose 融合进 transA / transB ----
        for (auto &op : ops)
        {
            if (op->getOpType() != OpType::MatMul)
                continue;
            auto matmul = as<MatmulObj>(op);

            // 沿输入 0（-> transA）或输入 1（-> transB）一路跳过可融合的 transpose
            auto fold = [&](int inputIdx, bool &flipped) {
                auto in = matmul->getInputs(inputIdx);
                while (in && in->getSource() &&
                       in->getSource()->getOpType() == OpType::Transpose)
                {
                    auto trans = as<TransposeObj>(in->getSource());
                    if (!isSwapLastTwo(trans->getPermute(), in->getRank()))
                        break;

                    flipped = !flipped;
                    auto next = trans->getInputs(0);
                    matmul->replaceInput(in, next); // 输入改为跳过这个 transpose
                    next->addTarget(matmul);        // matmul 现在消费 next
                    in->removeTarget(matmul);       // matmul 不再消费 in
                    deadOps.insert(trans.get());
                    deadTensors.insert(in->getFuid());
                    in = next;
                }
            };
            bool flipA = false, flipB = false;
            fold(0, flipA);
            fold(1, flipB);
            if (flipA)
                matmul->setTransA(!matmul->getTransA());
            if (flipB)
                matmul->setTransB(!matmul->getTransB());
        }

        // ---- 规则 1：删除相邻且互逆的 transpose 对 ----
        // 用 while 循环，因为消掉一对后可能暴露出新的可消除对
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (auto &op : ops)
            {
                if (deadOps.count(op.get()) || op->getOpType() != OpType::Transpose)
                    continue;
                auto out = op->getOutput();
                auto in = op->getInputs(0);
                if (!in || !in->getSource() || deadOps.count(in->getSource().get()))
                    continue;
                auto pred = in->getSource();
                if (pred->getOpType() != OpType::Transpose)
                    continue;

                auto opTrans = as<TransposeObj>(op);
                auto predTrans = as<TransposeObj>(pred);
                if (!composeIsIdentity(predTrans->getPermute(), opTrans->getPermute()))
                    continue;

                // pred: pIn -> in ；op: in -> out ，两者复合为单位置换，
                // 所以所有消费 out 的算子可以直接改用 pIn
                auto pIn = predTrans->getInputs(0);
                for (auto &consumer : out->getTargets())
                {
                    consumer->replaceInput(out, pIn);
                    out->removeTarget(consumer);
                    pIn->addTarget(consumer);
                }
                deadOps.insert(op.get());
                deadOps.insert(pred.get());
                deadTensors.insert(in->getFuid());
                deadTensors.insert(out->getFuid());
                changed = true;
            }
        }

        // ---- 清理：删除死 op 与死 tensor ----
        std::vector<Operator> removeOps;
        for (auto &op : ops)
            if (deadOps.count(op.get()))
                removeOps.emplace_back(op);
        for (auto &op : removeOps)
            removeOp(op);

        std::vector<Tensor> removeTensors;
        for (auto &t : tensors)
            if (deadTensors.count(t->getFuid()))
                removeTensors.emplace_back(t);
        for (auto &t : removeTensors)
            removeTensor(t);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // 记录每个 tensor 剩余的消费者个数：减到 0 说明之后不会再被使用，内存可回收
        std::unordered_map<UidBaseType, int> refCount;
        for (auto &t : tensors)
            refCount[t->getFuid()] = t->getTargets().size();

        // 每个 tensor 在 allocator 中分到的偏移
        std::unordered_map<UidBaseType, size_t> offsets;

        // ① 图的输入 tensor（没有 source）全程存活，最先分配、永不回收
        for (auto &input : getInputs())
            offsets[input->getFuid()] = allocator.alloc(input->getBytes());

        // ② 按拓扑序遍历 op：先回收本 op 消费完的输入，再给输出分配
        for (auto &op : ops)
        {
            // 本 op 消费一次输入，引用计数减到 0 后内存可交给 allocator 回收
            for (auto &input : op->getInputs())
            {
                if (input && input->getSource()) // 图输入无 source，不回收
                {
                    auto fuid = input->getFuid();
                    if (--refCount[fuid] == 0)
                        allocator.free(offsets[fuid], input->getBytes());
                }
            }

            // 给输出 tensor 分配内存
            for (auto &output : op->getOutputs())
            {
                if (output)
                    offsets[output->getFuid()] = allocator.alloc(output->getBytes());
            }
        }

        // ③ 记账已完成，peak 已定；真正 malloc 一次（getPtr），再给所有 tensor 绑定
        void *base = allocator.getPtr();
        for (auto &t : tensors)
        {
            if (auto it = offsets.find(t->getFuid()); it != offsets.end())
                t->setDataBlob(make_ref<BlobObj>(runtime, (char *)base + it->second));
        }
        // =================================== 作业 ===================================

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini