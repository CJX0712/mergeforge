# MergeForge

> 世界级模型合并 / 任务算术（Model Merging & Task Arithmetic）基准系统。
> 纯 NumPy、零深度学习框架依赖、CPU 秒级复现、逐位确定性。
> 作者：**晨星**（CJX0712）

[![CI](https://github.com/CJX0712/mergeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/mergeforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Deterministic](https://img.shields.io/badge/reproducibility-bitwise%20deterministic-brightgreen)](#可复现性)

---

## 这是什么

MergeForge 把模型合并（Model Merging）文献中的五大方法——**Model Soups**、
**Task Arithmetic**、**TIES-Merging**、**DARE**、**Greedy Soup**——与旗舰
**MergeFuse**（TIES 符号选举 + AdaMerging 式逐张量 λ 校准 + Model Soups 非劣守护）
放进一个**可预注册门禁、可逐位复现**的合成基准里，回答一个问题：

> 什么样的任务结构下，参数合并能以 1× 算力恢复 K 专家集成的性能？

**核心结论（A11 规范设定，3 seeds）**：当 base 扮演「预训练路由器」、专家在其上
做「私有块分类」微调时，合并单模型恢复 **99%** 的 oracle 集成差距、**93%** 的
joint 联合训练差距——且全面碾压 uniform soup（+0.17）与不合并 base（+0.61）。

| 方法 | mean test acc (3 seeds) | 说明 |
|---|---|---|
| base（不合并） | 0.245 ± 0.008 | pretext 块路由器，块内随机 |
| uniform soup | 0.685 ± 0.019 | 平均稀释 1/K，丢失判别幅值 |
| **MergeFuse（旗舰）** | **0.857 ± 0.013** | TIES ρ=1 + 逐张量 λ + 守护 |
| TIES / TA / DARE | 0.858 ± 0.012 | 不相交 delta 下殊途同归 |
| joint 联合训练（2× 算力参照） | 0.901 ± 0.005 | 上界参照 |
| oracle 集成（K 模型 + 任务 ID） | 0.864 ± 0.015 | 恢复率分母 |
| 朴素概率集成 | 0.711 ± 0.028 | 私有标签下失效的对照 |

预注册门禁 **G1–G6 全部通过**（详见 `benchmark.json` 与
[docs/model_card.md](docs/model_card.md)）。

## 快速开始

```bash
pip install -e ".[dev]"

# 完整 3-seed 基准 + 预注册门禁（~50s，CPU）
python cli.py benchmark

# 附带同 seed 二次运行逐位确定性校验
python cli.py benchmark --determinism

# 最小可运行示例（~17s，含门禁判定）
python examples/run_demo.py
```

无需 GPU、无需 PyTorch，唯一依赖是 NumPy。

## 系统结构

```
cli.py / examples/run_demo.py        # 入口
pipeline/merge_pipeline.py           # 单 seed 实验 + 多 seed 基准 + 预注册门禁
finetune/trainer.py                  # pretext base + frozen-trunk block-loss 专家
data/tasks.py                        # 多任务 DGP（私有类别子空间 + 可复现）
merging/methods.py                   # Soup / TA / TIES / DARE / Greedy（与模型结构解耦）
merging/fuse.py                      # 旗舰 MergeFuse
models/mlp.py                        # 纯 NumPy MLP（解析梯度 + 数值自检）
eval/metrics.py                      # 逐任务准确率 / oracle 集成 / 插值障碍 / 恢复率
core/                                # 配置（ENV 覆盖 + schema 校验）/ 错误注册表 / 种子
```

调用方向单向无环：`cli → pipeline → {data, finetune, merging, eval} → core`。

## 科学设计（为什么这套基准成立）

任务 k 的样本来自高斯簇 `x = μ_k + Mz`，标签 `y = k·C + argmax(z·V_k)`：

- **任务身份**编码在 x 的正交补子空间（噪声less）→ 路由可学；
- **类内判别**只依赖潜因子 z → 是任务私有的头部知识；
- base 以块路由 pretext 训练（会路由、不会分类）→ base 弱在**可分解**的部分；
- 专家冻结 trunk、只训自有类别块 → delta 不相交、可无损组合；
- uniform soup 因 1/K 稀释丢失判别幅值 → 旗舰逐张量 λ 有明确超车机制。

设计迭代 A1→A11 的完整实证记录（含 4 次结构性失败与 2 个致命 bug 的解剖）
见 [docs/model_card.md](docs/model_card.md)——负结果与正结果同样完整。

## 预注册门禁

| 门禁 | 判据 | 结果 |
|---|---|---|
| G1 fuse vs soup | 配对差 ≥ 0.01，½σ 显著，多数 seed 胜出 | ✅ +0.173，3/3 胜 |
| G2 fuse vs base | ≥ +0.02 | ✅ +0.612 |
| G3 fuse vs specialist 均值 | ≥ 0 | ✅ +0.513 |
| G4 fuse vs greedy soup | ≥ 0 | ✅ +0.173 |
| G5 joint 参照 | joint > base（否则 recovery 无定义） | ✅ recovery 0.93 |
| G6 oracle 集成参照 | 恢复率 = (fuse−base)/(oracle−base) | ✅ 0.99 |

## 可复现性

- 全部随机性来自种子（init / shuffle / pretext 重标注无隐式随机）；
- BLAS 线程钉 1，legacy `RandomState`，float64 全程；
- 同 seed 两次运行**逐位一致**（`benchmark --determinism` 内置校验，38 项指标 max|Δ|=0）。

## 测试与质量

```bash
pytest              # 109 项：梯度校验 / 合并代数恒等式 / 端到端确定性 / CLI 冒烟
ruff check .        # 零告警
ruff format --check .
```

合并代数恒等式测试（黄金性质）：soup 置换不变、TIES K=1 ≡ TA、TIES 全体一致 ≡ 算术平均、
DARE δ=0 ≡ TA、DARE 期望保持、块损失梯度支撑严格限于自有块、class_block 路径
数值梯度校验 ≤1e-4。

## 文档

- [docs/architecture.md](docs/architecture.md) —— 模块设计、扁平参数表示一致性、扩展指南
- [docs/model_card.md](docs/model_card.md) —— A1→A11 实验迭代全记录、负结果、门禁与失败案例

## 引用

本项目实现与对标的方法学：

```bibtex
@article{wortsman2022modelsoups,  title={Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time}, author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Ya and et al.}, year={2022}}
@article{ilharco2023taskarithmetic, title={Editing models with task arithmetic}, author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and et al.}, year={2023}}
@article{yadav2023ties, title={TIES-Merging: Resolving Interference When Merging Models}, author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and et al.}, year={2023}}
@article{yu2024dare, title={Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch}, author={Yu, Le and Yu, Bowen and Yu, Haiyang and et al.}, year={2024}}
```

## License

MIT © 2026 晨星（CJX0712）
