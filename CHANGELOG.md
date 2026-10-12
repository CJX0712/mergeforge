# Changelog

All notable changes to MergeForge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-13

首个公开版本。作者：晨星（CJX0712）。

### Added

- 多任务合成 DGP：私有类别子空间（任务 k 独占类别块 [kC,(k+1)C)）、
  任务身份噪声less 编码于输入正交补、逐位可复现、标签噪声块内翻转。
- 纯 NumPy MLP：解析梯度 + 中心差分数值自检（含 class_block 分支）、
  扁平参数表示、梯度掩码训练。
- 训练流水线：pretext 块路由 base（预训练模型设定）+ 冻结 trunk +
  私有块 softmax 专家 + joint 全局标签参照。
- 合并方法库：uniform/greedy soup、Task Arithmetic、TIES-Merging、DARE
  （全部只依赖扁平参数 + 形状清单，与模型结构解耦）。
- 旗舰 MergeFuse：TIES 符号选举 + AdaMerging 式逐张量 λ 坐标下降 +
  Model Soups 非劣守护。
- 评测：逐任务准确率、朴素/oracle 路由集成、线性插值障碍、恢复率、
  预注册门禁 G1–G6、失败案例归因链。
- ENV_MERGEFORGE_* 配置覆盖 + schema 校验 + 错误目录（E100–E500）。
- 测试 109 项：梯度校验、合并代数恒等式、端到端确定性、CLI 冒烟。
- CI：Python 3.12/3.13 矩阵（ruff + pytest + demo 冒烟 + 密钥扫描）。

### Fixed

- class_block 解析梯度漏写块内竞争类 p_c 项（A11 首测暴露，
  numerical 校验此前未覆盖该分支）。
- flat 参数顺序与 param_spec 错位导致的静默参数重组（历史战役作废重跑）。
- pipeline 合并锚点无视 specialist_from 的锚点错位。

### Verified

- 3-seed 基准 G1–G6 全过：mergefuse 0.857±0.013，
  recovery vs oracle 0.99 / vs joint 0.93。
- 同 seed 二次运行 38 项指标逐位一致。
