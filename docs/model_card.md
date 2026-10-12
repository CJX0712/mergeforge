# MergeForge Model Card —— 设计迭代全记录（A1 → A11）

> 本文档是 MergeForge 科学设计的完整实证日志：每一次 regime 迭代、每一次失败、
> 每一个致命 bug 及其解剖。负结果与正结果同样完整——它们共同界定
> 「参数合并何时成立」的边界。

作者：晨星（CJX0712） · 版本 0.1.0 · 2026-10

---

## 1. 系统目标与评测哲学

**问题**：K 个任务专家模型，能否把参数合并成**一个**模型，以接近零额外推理成本
恢复 K 模型集成的性能？何时能、何时不能、为什么？

**评测哲学**（借鉴 MergeForge 系列前作 MatrixForge / FlowForge 的踩坑库）：

1. **预注册门禁**：G1–G6 在实验前定死判据，禁止根据结果迁就阈值；
2. **配对判据**：方法间比较用同 seed 配对差（消除 DGP 级方差），
   跨 seed 非配对 std 含 DGP 方差，不可作显著性判据；
3. **参照上界**：恢复率分母用 oracle 路由集成（任务身份下的最强 K 模型组合）；
4. **逐位确定性**：同 seed 两次运行全部指标逐位一致，否则视为实现泄漏；
5. **失败案例归因**：每个基准必须报告至少 3 条最差任务的归因链。

## 2. 合成 DGP（data/tasks.py）

任务 k：潜变量 `z ~ N(0, I_r)`，观测 `x = μ_k + Mz`（M ∈ R^{d×r} 共享），
标签 `y = k·C + argmax(z·V_k)` + 块内翻转噪声。

关键几何性质（A11 设计的根基）：

- `Mz` 只在 M 列空间（r=4 维）有方差 → **任务身份完整保存在 12 维正交补中
  （逐样本噪声less、任务内恒定）**；
- 类内判别信息只存在于 z（列空间）；
- 标签全局空间为 K×C，任务 k 独占类别块 [kC, (k+1)C)。

因此「路由」（补空间 → 块）是跨任务**共享函数**，「块内判别」（列空间 → 类）
是**任务私有**的头部知识——两者可分解，是合并增益的来源。

## 3. 迭代日志

### A1 —— 从联合训练 base 微调专家（已作废：代码 bug）

设定：任务私有独立 head + mean_shift ρ=1.0 + 从 base 微调。
**全部数字作废**：本轮实验跑在 flat 参数顺序 bug 的损坏代码上（见 §4.1）。

### A2 —— 共享 readout：失败（base 太快）

设定：`V_k = V0 + s·P_k` 共享结构。
结果：base 几个 epoch 即学会共享结构，soup 天花板 ≈ base，无增益空间。
**教训**：合并需要 base 弱，但 base 弱不能靠「任务结构太共享」。

### A3 —— 不相交类别子空间：机制成立但无 headroom

设定：全局 K×C 标签、任务独占类别块（即现行 DGP）。
结果：成分组合机制成立（合并 ≥ base），但 base 训练 100 epoch 后贴天花板。
**教训**：路由 + 块内判别 base 都学得会 → 需要人为切分「base 学什么」。

### A4 —— 从共享初始化训练专家：失败（特征旋转歧义）

设定：specialist 从 init 训练（hidden 64 ≫ r=4）。
结果：各专家学到潜特征的不同旋转，合并全面崩溃（soup 0.15-0.17）。
**教训**：无共享锚点时，神经网络的特征基不可对齐——这是文献统一采用
「同一预训练模型微调」设定的根本原因。

### A5 —— ρ=0.2 + freeze_trunk + block_loss：假性成功（两个 bug 叠加）

设定：ρ=0.2 弱化路由 + 专家冻结 trunk 只训输出层 + 私有块 softmax。
结果：fuse=soup=base=0.48，oracle 0.89（差距巨大但合并未解锁）。
事后解剖（见 §4）发现**两个 bug**：class_block 梯度漏项 + pipeline 锚点错位。

### A6 —— 24-way 专家（block_loss=False）：抑制 delta 跨任务泄漏

设定：同 A5 但专家用 24-way softmax。
测量：Δ 的 trunk 分量精确为 0（冻结正确）；head 内 off-diag（对他块抑制）
范数是 diag（本块正向）的 **2.2 倍**；TA λ 曲线单调下降（λ=0 最优）；
TIES ρ≥0.5 符号选举把正向票杀光（5 负 vs 1 正）。
**机制**：ρ=0.2 任务集群重叠 → 各专家的抑制方向在他人任务输入上同样激活
→ 求和后每块净抑制 ≈ −3.3 → 全模型被压制（TA λ=1 = 0.34 < base）。
**教训**：**base 缺失的能力是「路由」= 跨任务共享函数，per-task delta
相加无法构成共享路由器**。

### A7-A8 —— DGP 翻转扫描：prototype_scale 不进 Bayes 误差

扫描 (ρ ∈ [0.25,3.0], prototype_scale ∈ [1,3], base_epochs ∈ [40,100])：
- ρ≥0.5：base 100 epoch 已贴 Bayes（0.88+）——prototype_scale 对 Bayes
  准确率零影响（类内由 z 决定，给定任务即可完美恢复），只影响学习速度；
- ρ≤0.3：路由未学会（base 0.48-0.63），但专家检测器在重叠集群上不精确；
- 所有配置下 **TA λ=1 = soup = base 逐位相同**（合并无增益）。
**教训**：路由学习速度是 ρ 的阶跃函数；类内难度不影响 Bayes 上界。

### A9 —— 弱 base + unfrozen 专家：崩溃

设定：be ∈ {40,60,100} × unfrozen 24-way。
测量：专家 trunk 漂移**近正交**（平均成对余弦 ≈ −0.02），
soup 全崩（0.17-0.39 vs base 0.18-0.48）。
**教训**：100 epoch full-batch Adam 的微调不是「温和适配」而是「重建」——
base 欠训练时专家离开共享盆，任务向量失去可加性。

### A10 —— 中等 base + unfrozen：同一结论

扫描 ρ ∈ [0.4,0.5] × be ∈ [60,100]：trunk_cos ≈ 0，soup/TA/TIES 全部 < base。
**结论固化**：full-ft Adam 在本规模下永远产出正交任务向量。

### A11 —— pretext base（现行规范设定）：完全成功

**思想转向**：文献中合并成立的真正前提是「base = 预训练模型（见过域、
不会任务）、专家 = 任务微调」。对应本 DGP：

- base 以**块路由 pretext** 训练（标签 = k·C + rand(C)）：学会补空间路由，
  块内四行近似等幅（不含判别信息）→ base ≈ 0.25；
- 专家冻结 trunk、只训自有块（修复后的 block loss）→ own-task 0.86；
- 合并 = base 路由 + Σ 专家块判别 → **TA λ=1-1.5 = 0.85-0.87 ≈ oracle 0.858**；
- uniform soup 稀释 1/K 丢失判别幅值 → soup = 0.55-0.68，旗舰有明确超车机制。

3-seed 基准：**G1–G6 全过**，recovery vs oracle = 0.99，vs joint = 0.93。

## 4. 致命 bug 解剖（两处静默错误，测试套件立功）

### 4.1 flat 参数顺序不一致（A1 时代，已修复）

`get_flat` 用块序 [W1,W2,W3,b1,b2,b3]，`param_spec.split` 用交错序
[W1,b1,W2,b3,...] → `set_flat` 每步静默重组参数：训练循环每步损坏、
全部合并评测失真、梯度校验失败（1.84），且**不抛任何异常**。
修复：get_flat/grad_flat 统一逐层交错；新增 roundtrip 逐位一致测试。
**此 bug 导致整个 A1 实验战役作废重跑。**

### 4.2 class_block 梯度漏项（A5-A10 时代，已修复）

`loss_and_grad(class_block=...)` 的解析梯度只写了真类项
`delta[y] = p_y − 1`，漏掉块内竞争类的 `p_c` 项（正确实现：
`delta[:, lo:hi] = probs; delta[y] -= 1`）。
后果：block-loss 专家训练信号残缺——在全局标签 base 上勉强可用
（A5 专家 own-task 仍达 0.90，掩盖了问题），在 pretext base 上完全失效
（专家 0.25 = 随机，A11 首测直接暴露）。
根因：`numerical_grad` 从不传 `class_block`，数值校验从未覆盖该分支。
修复：梯度改为正确形式 + `numerical_grad`/`gradient_check` 增加
class_block 参数 + 新增块支撑不变量测试（块外列梯度严格为 0）。

### 4.3 pipeline 合并锚点错位（A5 时代，已修复）

`anchor_flat = init.get_flat()` 无视 `specialist_from`——当专家从 base
微调时，任务向量按 `specialist − init` 计算，混入 base 自身的训练位移，
污染全部 TA/TIES/DARE/fuse 的 λ 网格校准。
修复：`anchor_flat = anchor_model.get_flat()`。

**方法论教训**：手写数值系统的每条计算路径（尤其条件分支）都必须有
独立的数值金标准校验；「主干路径正确」不构成对分支路径的任何证据。

## 5. 规范配置（canonical benchmark）

| 参数 | 值 | 说明 |
|---|---|---|
| d / r | 16 / 4 | 输入维度 / 共享潜因子秩 |
| K / C | 6 / 4 | 任务数 / 每任务类别（全局 24 类） |
| n_train/val/test | 900/300/600 | train/val/test 互不相交 |
| label_noise | 0.05 | DGP 贝叶斯误差 |
| mean_shift_rho | 0.3 | 任务均值平移半径 |
| hidden_dims | (64, 64) | trunk |
| base_pretext | block | base = 块路由预训练 |
| freeze_trunk | True | 专家只训输出层 |
| specialist_block_loss | True | 私有块 softmax |
| base/ft/joint epochs | 100/100/200 | 算力口径：base 与专家总量对齐，joint 2× |
| lr / schedule / batch | 0.005 / cosine / 8192 | full-batch |
| seeds | (7, 11, 23) | 3-seed 报 mean±std |

## 6. 门禁结果（3 seeds，逐位确定性）

| 方法 | mean ± std | per-seed |
|---|---|---|
| base | 0.245 ± 0.008 | 0.240 / 0.242 / 0.254 |
| uniform_soup | 0.685 ± 0.019 | 0.669 / 0.706 / 0.678 |
| task_arithmetic | 0.858 ± 0.012 | 0.852 / 0.872 / 0.851 |
| ties | 0.858 ± 0.012 | 0.852 / 0.872 / 0.851 |
| dare | 0.858 ± 0.012 | 0.852 / 0.872 / 0.851 |
| **mergefuse** | **0.857 ± 0.013** | 0.849 / 0.873 / 0.850 |
| greedy_soup | 0.685 ± 0.019 | 0.669 / 0.706 / 0.678 |
| ensemble（朴素） | 0.711 ± 0.028 | 0.701 / 0.743 / 0.691 |
| oracle_ensemble | 0.864 ± 0.015 | 0.857 / 0.881 / 0.853 |
| joint（2× 算力参照） | 0.901 ± 0.005 | 0.907 / 0.899 / 0.898 |

门禁：G1 ✅（配对差 +0.173，½σ 显著，3/3 胜）G2 ✅（+0.612）G3 ✅ G4 ✅
G5 ✅（recovery vs joint 0.93）G6 ✅（recovery vs oracle 0.99）。

## 7. 失败案例（预注册归因链）

合并相对 specialist own-task 的残余差距（forgetting ≈ 0.003-0.013）归因：
「合并机制共性遗忘（旗舰优于 soup，仍不及 specialist）」——各专家块判别
在合并模型中需与 23 类竞争（专家训练时只与块内 3 类竞争），幅值竞争
造成 1-2% 的残余损失。这是私有块设定的固有代价，非缺陷。

## 8. 已知边界与适用性

- 本基准在合成 DGP 上界定机制边界，不直接外推到 LLM 规模；
- 「任务身份可从输入识别」是路由型合并的前提（与文献一致）；
- block_loss 要求任务标签空间已知可分块——对应多任务头部分类场景，
  不覆盖语言建模式共享词表设定（那正是 TIES/DARE 原论文的场景）；
- 旗舰相对 TA/TIES 的增益（fuse − ties ≈ 0）在不相交 delta 极限下趋近于零——
  逐张量 λ 校准的价值在 delta 部分重叠的中间地带，见 model_card §A6 的
  反例记录。
