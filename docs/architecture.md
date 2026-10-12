# MergeForge 架构

## 1. 设计原则

1. **扁平参数表示是合并的唯一契约**：一切合并方法只操作
   `(base_flat, model_flats, param_spec)`，与网络结构解耦——任何满足
   `TrainableModel` 协议（core/interfaces.py）的同构模型集合都可被合并。
2. **调用方向单向无环**：`cli → pipeline → {data, finetune, merging, eval} → core`。
   底层模块绝不反向 import。
3. **一切随机性显式种子化**：init（`seed+1`）、base shuffle（`seed+2`）、
   specialist（`seed+100+k`）、DARE 掩码（`seed*100_007+4242`）、
   pretext 重标注（`seed+900`）。无任何隐式全局随机。
4. **float64 全程 + BLAS 线程钉 1**：合并是逐元素运算，精度即正确性；
   线程数不确定会破坏浮点求和顺序 → 确定性不可复现。
5. **错误即数据**：错误目录（E100-E500）注册表在导入期检测重码/重名，
   配置校验、数据校验、模型校验抛带 code 的类型化异常。

## 2. 模块图

```
                    ┌──────────────────────────────┐
                    │   cli.py / examples/run_demo.py   │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   pipeline/merge_pipeline.py  │
                    │  run_seed / benchmark / 门禁   │
                    └──┬───────┬───────┬───────┬───┘
                       ▼       ▼       ▼       ▼
              data/tasks.py  finetune/  merging/  eval/metrics.py
              多任务 DGP     trainer.py  methods.py  指标/集成/障碍
                             (pretext     fuse.py
                              base +      (旗舰)
                              专家)
                       └───────┴───────┴───────┘
                                   ▼
                    core/  config · types · errors · seed · thread_pin · interfaces
```

## 3. 关键数据流（run_seed）

```
seed
 ├─ make_multitask_dataset(seed)         # train/val/test 互不相交
 ├─ make_init_model(seed+1)
 ├─ train_base(pretext=block, seed)      # 块路由预训练（base_pretext="block"）
 ├─ train_base(joint, pretext=False)     # 全局标签 joint 参照（2× 预算）
 ├─ train_specialist(base_k, seed+100+k) # 冻结 trunk + 私有块 loss × K
 ├─ anchor_flat = anchor_model.get_flat()# 合并锚点（与 specialist_from 一致！）
 ├─ deltas = task_vectors(anchor, flats) # (K, P)
 ├─ 合并方法 × N（val 网格校准，test 只在最终评测）
 ├─ oracle/朴素集成参照
 ├─ 线性插值障碍诊断
 └─ SeedResult
```

## 4. 扁平参数表示一致性（生命线）

`MLP.get_flat` / `loss_and_grad` 的梯度拼接 / `param_spec.split` 三者的
参数顺序必须**逐位一致**（逐层 W、b 交错）。历史上块序 vs 交错序的错位
让 `set_flat` 每步静默重组参数（训练每步损坏 + 全部评测失真 + 零异常），
整个实验战役作废（见 model_card §4.1）。

防护措施：
- `test_flat_roundtrip`：set(get(x)) == x 逐位；
- `test_param_spec_split_flatten_roundtrip`：split/flatten 互逆；
- `test_gradient_check_*`：解析梯度 vs 中心差分（含 class_block 分支）。

**扩展新架构时必须同时满足这三个测试。**

## 5. 合并方法契约

```python
def method(base_flat: np.ndarray,
           model_flats: list[np.ndarray],
           spec: ParamSpec,
           eval_fn: Callable[[np.ndarray], float] | None = None,
           cfg: MergeConfig | None = None) -> tuple[np.ndarray, dict]:
```

- 无验证需求的方法（soup/TA/TIES/DARE）不用 eval_fn；
- 校准型方法（greedy_soup、mergefuse）通过 eval_fn 在 **val 折**上校准，
  绝不接触 test；
- 返回 (合并参数, 诊断 dict)——诊断进入报告的 extra 字段。

黄金代数恒等式（tests/test_merging.py）：
- uniform soup 对输入顺序置换不变；
- TIES(ρ, K=1) ≡ TA(λ=1)；
- TIES 全体符号一致 ≡ 算术平均；
- DARE(δ=0) ≡ TA；DARE 的期望保持性；
- greedy soup 的贪心选择语义。

## 6. 门禁与报告

benchmark() 产出单一 JSON（benchmark.json）：config 全量、逐方法
mean±std+per-seed、G1–G6 门禁明细、消融（guard 触发数、fuse−ties、
fuse−soup）、失败案例归因链、插值障碍。`gates_pass` 是唯一布尔结论。

## 7. 扩展指南

- **换模型**：实现 `TrainableModel` 协议（forward/loss_and_grad/get_flat/
  set_flat/param_spec），保证 flat 序与 spec 一致，通过 §4 三个测试即可
  接入全部合并方法。
- **加合并方法**：实现 §5 契约，在 pipeline 挂接 + 网格校准 + 报告行；
  如需进核心指标，更新 CORE_METRIC_METHODS 与对应测试。
- **换 DGP**：data/tasks.py 的 `_sample_split` 保持「同序 rng 重放可复现」
  契约；标签空间变化需同步 model_output_dim 与校验。
