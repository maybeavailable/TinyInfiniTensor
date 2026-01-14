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
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        IT_ASSERT(topo_sort() == true);

        auto is_swap_last_two = [](const vector<int> &perm, int rank) -> bool
        {
            if (rank < 2)
                return false;
            if ((int)perm.size() != rank)
                return false;
            for (int i = 0; i < rank - 2; ++i)
                if (perm[i] != i)
                    return false;
            return perm[rank - 2] == rank - 1 && perm[rank - 1] == rank - 2;
        };

        auto is_inverse_perm = [](const vector<int> &p, const vector<int> &q,
                                  int rank) -> bool
        {
            if ((int)p.size() != rank || (int)q.size() != rank)
                return false;
            vector<int> inv(rank, -1);
            for (int i = 0; i < rank; ++i)
            {
                IT_ASSERT(p[i] >= 0 && p[i] < rank);
                inv[p[i]] = i;
            }
            for (int i = 0; i < rank; ++i)
            {
                if (inv[i] != q[i])
                    return false;
            }
            return true;
        };

        auto cleanup_dangling_tensors = [&]()
        {
            for (auto it = tensors.begin(); it != tensors.end();)
            {
                auto &t = *it;
                if (!t)
                {
                    it = tensors.erase(it);
                    continue;
                }
                if (t->targets.empty() && t->source.expired())
                    it = tensors.erase(it);
                else
                    ++it;
            }
        };

        auto detach_op = [&](const Operator &op)
        {
            // Disconnect from predecessor/successor bookkeeping and tensor edges.
            for (const auto &in : op->inputs)
            {
                if (!in)
                    continue;
                in->removeTarget(op);
            }
            for (const auto &out : op->outputs)
            {
                if (!out)
                    continue;
                if (out->source.lock() == op)
                    out->source.reset();
            }

            for (const auto &pred : op->getPredecessors())
                pred->removeSuccessors(op);
            for (const auto &succ : op->getSuccessors())
                succ->removePredecessors(op);

            op->predecessors.clear();
            op->successors.clear();
        };

        bool changed = false;
        do
        {
            changed = false;

            std::unordered_set<OperatorObj *> eraseOps;

            // Rule 2: fuse Transpose(swapping last two dims) into Matmul's transA/transB.
            for (size_t opIdx = 0; opIdx < ops.size(); ++opIdx)
            {
                auto op = ops[opIdx];
                if (!op || eraseOps.count(op.get()))
                    continue;
                if (op->getOpType() != OpType::MatMul)
                    continue;
                auto mm = std::dynamic_pointer_cast<MatmulObj>(op);
                if (!mm)
                    continue;

                for (int inputIdx = 0; inputIdx < 2; ++inputIdx)
                {
                    auto in = op->inputs[inputIdx];
                    if (!in)
                        continue;
                    auto pred = in->getSource();
                    if (!pred || eraseOps.count(pred.get()) ||
                        pred->getOpType() != OpType::Transpose)
                        continue;

                    // Only safe to fuse if transpose output is used only by this matmul.
                    if (in->getTargets().size() != 1)
                        continue;

                    auto tp = std::dynamic_pointer_cast<TransposeObj>(pred);
                    if (!tp)
                        continue;
                    const auto perm = tp->getPermute();
                    const int rank = static_cast<int>(in->getRank());
                    if (!is_swap_last_two(perm, rank))
                        continue;

                    auto orig = pred->inputs[0];
                    if (!orig)
                        continue;

                    // Rewire matmul to consume transpose input directly.
                    op->replaceInput(in, orig);
                    in->removeTarget(op);
                    orig->addTarget(op);

                    // Update predecessor/successor relation.
                    op->removePredecessors(pred);
                    pred->removeSuccessors(op);
                    if (auto origPred = orig->getSource())
                    {
                        origPred->addSuccessors(op);
                        op->addPredecessors(origPred);
                    }

                    // Toggle trans flag.
                    if (inputIdx == 0)
                        mm->setTransA(!mm->getTransA());
                    else
                        mm->setTransB(!mm->getTransB());

                    // If transpose becomes unused, remove it.
                    if (in->getTargets().empty())
                    {
                        detach_op(pred);
                        eraseOps.insert(pred.get());
                        changed = true;
                    }
                }
            }

            if (!eraseOps.empty())
            {
                ops.erase(std::remove_if(ops.begin(), ops.end(),
                                         [&](const Operator &op)
                                         {
                                             return !op || eraseOps.count(op.get());
                                         }),
                          ops.end());
            }

            // Rule 1: remove adjacent inverse Transpose pairs.
            for (auto it = ops.begin(); it != ops.end();)
            {
                auto &op1 = *it;
                if (op1->getOpType() != OpType::Transpose)
                {
                    ++it;
                    continue;
                }
                auto t1 = std::dynamic_pointer_cast<TransposeObj>(op1);
                if (!t1)
                {
                    ++it;
                    continue;
                }
                auto y = op1->outputs[0];
                if (!y || y->getTargets().size() != 1)
                {
                    ++it;
                    continue;
                }
                auto op2 = y->getTargets()[0];
                if (!op2 || op2->getOpType() != OpType::Transpose)
                {
                    ++it;
                    continue;
                }
                auto t2 = std::dynamic_pointer_cast<TransposeObj>(op2);
                if (!t2)
                {
                    ++it;
                    continue;
                }
                auto z = op2->outputs[0];
                if (!z)
                {
                    ++it;
                    continue;
                }
                // Skip if z is a graph output (no targets) since we cannot safely
                // replace external tensor references.
                if (z->getTargets().empty())
                {
                    ++it;
                    continue;
                }

                auto x = op1->inputs[0];
                if (!x)
                {
                    ++it;
                    continue;
                }

                const auto p1 = t1->getPermute();
                const auto p2 = t2->getPermute();
                const int rank = static_cast<int>(y->getRank());
                if (!is_inverse_perm(p1, p2, rank))
                {
                    ++it;
                    continue;
                }

                // Rewire: replace uses of z with x.
                auto succs = z->getTargets();
                for (auto &succ : succs)
                {
                    succ->replaceInput(z, x);
                    z->removeTarget(succ);
                    x->addTarget(succ);

                    succ->removePredecessors(op2);
                    op2->removeSuccessors(succ);
                    if (auto xp = x->getSource())
                    {
                        xp->addSuccessors(succ);
                        succ->addPredecessors(xp);
                    }
                }

                // Remove the two transpose ops and their dangling tensors.
                detach_op(op1);
                detach_op(op2);
                ops.erase(std::remove(ops.begin(), ops.end(), op1), ops.end());
                ops.erase(std::remove(ops.begin(), ops.end(), op2), ops.end());
                cleanup_dangling_tensors();

                changed = true;
                // Restart since iterators invalidated.
                it = ops.begin();
            }

            if (changed)
            {
                sorted = false;
                IT_ASSERT(topo_sort() == true);
                cleanup_dangling_tensors();
            }
        } while (changed);
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
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        // Pass 1: simulate allocation to compute offsets and peak memory.
        std::unordered_map<TensorObj *, size_t> offsetMap;
        std::unordered_map<TensorObj *, size_t> remainingUses;
        std::unordered_set<TensorObj *> pinned;

        pinned.reserve(tensors.size());
        remainingUses.reserve(tensors.size());
        offsetMap.reserve(tensors.size());

        // Pin graph inputs/outputs: keep their storage alive.
        for (const auto &t : tensors)
        {
            if (!t)
                continue;
            if (!t->getSource() || t->getTargets().empty())
                pinned.insert(t.get());
            remainingUses.emplace(t.get(), t->getTargets().size());
        }

        // Allocate graph inputs first (they have no source op).
        for (const auto &t : tensors)
        {
            if (!t)
                continue;
            if (!t->getSource())
            {
                auto off = allocator.alloc(t->getBytes());
                offsetMap.emplace(t.get(), off);
            }
        }

        // Allocate outputs when produced; free intermediates after last use.
        for (const auto &op : ops)
        {
            // Allocate op outputs
            for (const auto &out : op->getOutputs())
            {
                if (!out)
                    continue;
                if (offsetMap.find(out.get()) == offsetMap.end())
                {
                    auto off = allocator.alloc(out->getBytes());
                    offsetMap.emplace(out.get(), off);
                }
            }

            // Consume op inputs; free when no longer needed.
            for (const auto &in : op->getInputs())
            {
                if (!in)
                    continue;
                auto *tp = in.get();
                if (pinned.find(tp) != pinned.end())
                    continue;
                auto it = remainingUses.find(tp);
                IT_ASSERT(it != remainingUses.end());
                IT_ASSERT(it->second > 0);
                it->second--;
                if (it->second == 0)
                {
                    auto offIt = offsetMap.find(tp);
                    IT_ASSERT(offIt != offsetMap.end());
                    allocator.free(offIt->second, in->getBytes());
                }
            }
        }

        // Pass 2: allocate the real arena once, then bind each tensor's blob.
        void *base = allocator.getPtr();
        for (const auto &t : tensors)
        {
            if (!t)
                continue;
            auto it = offsetMap.find(t.get());
            if (it == offsetMap.end())
                continue;
            auto ptr = static_cast<void *>(static_cast<char *>(base) + it->second);
            t->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
        return tensors.back();
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