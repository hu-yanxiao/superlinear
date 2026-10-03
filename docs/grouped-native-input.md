# C++ 原生训练：分组物理输入

维护分支 `codex/radial-coefficient-gate` 的开发版 `mlp-sus2` 现在支持
分组 YAML 和物理语义 CLI。它是现有 `init-sh` / `train` 的输入适配层，
没有更换模型方程、参数布局、初始化算法、损失函数或优化器。
历史命令仍可直接使用；其他模型架构继续使用历史 CLI。

```bash
mlp-sus2 train run.yaml --check-config
mpirun -n 40 mlp-sus2 train run.yaml
mlp-sus2 init-sh model.yaml
mlp-sus2 train --config continue.yaml
```

`--check-config` 只检查输入、数值范围和参数翻译，不读取训练数据或模型，
不校验文件内容，不创建目录，也不开始训练。实际运行仍会检查模型、数据和输出。
日志会列出翻译后的原生选项；主 MPI rank 输出一次。

## 从物理过程组织参数

`coordination` 表示原子周围的局域配位环境描述，不是一个整数配位数。
它可以包含不同角动量和多体相关信息。避免让客户把 `two-layer-gate`
理解为通用神经网络的“两层门控”：

```text
邻居距离/角度 → coordination 配位环境 → response 径向系数响应
                                      → interaction 多体相互作用 → readout 原子能
```

三个通道数分别是 `K=radial_channels`（原始径向通道）、
`G=coordination.channels`（配位投影通道）和
`O=interaction.channels`（相互作用通道）。当前原生实现要求 `1 <= G,O <= K`。

| 分组字段 | 原生选项 | 含义 / 允许值 / 默认值 |
|---|---|---|
| `model.kind` | `superlinear-mode` | `coordination-adaptive`；分组新模型默认采用此架构 |
| `model.name` | `potential-name` | 模型名称；原生默认名称 |
| `model.elements` 或顶层 `elements` | `species-count` | 按 CFG type 顺序的元素列表；推导元素数，不启用 ZBL、不改 CFG 类型 |
| `model.species_count` | `species-count` | 不提供元素表时可单独指定；默认 2；和元素表冲突时报错 |
| `model.l_max` | `l-max` | 最大角动量 L；C++ 支持 0..6；默认 3 |
| `model.radial_channels` | `k-max` | K；默认 3 |
| `model.neighbor_cutoff` | `cutoff` | 邻域截断，单位 Å；默认 7.5 |
| `model.radial.basis` | `radial-basis-type` | `chebyshev-sss`（默认）、`laguerre-log1p` |
| `model.radial.basis_size` | `radial-basis-size` | 径向基函数数 N；默认 10 |
| `model.radial.basis_scale` | `scaling` | 全局径向基缩放；默认 0.01 |
| `model.radial.reference_min_distance` | `min-dist` | 径向基参考下界，单位 Å；默认 1.5；不是强制排斥或数据过滤阈值 |
| `model.radial.warp_sharing` | `scaling-map` | `channel`→K、`angular`→L、`channel-angular`→LK（默认）；只控制距离坐标变换的共享方式 |
| `model.coordination.channels` | `two-layer-gate-k-rank` | G；默认 K |
| `model.coordination.body_order` | `two-layer-gate-body-order` | 配位环境多体阶数，2..6；默认相互作用多体阶数 |
| `model.coordination.channel_coupling` | `two-layer-gate-factor-pruning` | `cross`→q-total（默认），`diagonal`→same-k |
| `model.coordination.distance_warp` | `two-layer-gate-decoupled-scaling` | `shared`（默认）或 `independent`；shared 时不传这个存在即启用的旧开关 |
| `model.coordination.tensor_l_max` | `tensor-message-lmax` | 0（默认）、1、2，且不超过 L |
| `model.interaction.channels` | `outer-k-rank` | O；必须显式提供，避免误用旧架构的默认 0 |
| `model.interaction.body_order` | `body-order` | 相互作用多体阶数，2..6；默认 5 |
| `model.interaction.channel_coupling` | `sh-factor-pruning` | `diagonal`→same-k（默认），`cross`→q-total |
| `model.angular_limit_by_body_order` | `body-l-max` | 如 `{2: 3, 3: 3, 4: 2, 5: 1}`；2..5 必须齐全，六体模型还需 6；也作用于配位图，保持原生语义 |
| `model.response.layout` | `adaptive-radial-projection` | `per-basis`→full（默认）、`shared-latent`→shared-n |
| `model.response.latent_channels` | `adaptive-radial-shared-n-rank` | 潜在通道 d；默认 1；per-basis 只能 1，shared-latent 要求 1..N |
| `model.response.activation` | `adaptive-radial-activation` | `linear`→none（默认）、`tanh` |
| `model.response.element_mode` | `adaptive-radial-amplitude-mode` | `shared`（默认）、`source-element`→species；响应幅度按来源邻居元素区分 |
| `model.readout.element_mode` | `readout-mode` | `shared`（默认）、`per-element`→species；能量读出按中心原子元素区分 |
| `model.readout.joint` | `joint-readout` | false（默认）或 true；联合读出遵守现有 L/张量约束 |

## 训练、损失和输出

参考 [完整示例](../examples/grouped-native-train.yaml)。示例的 20/20/5
是显式选择，不是被本次修改引入的新默认值。

| 字段 | 原生选项 / 语义 |
|---|---|
| `initialization.seed` | `init-random-seed`；uint64；平衡初始化默认 seed=1 |
| `initialization.parameters` | 分组训练固定 `same`；新模型由现有平衡初始化生成，续训保持已有物理系数 |
| `initialization.warp_slope_endpoints` | `scal-range`；两个端点列表；保持原生端点方向语义，不自动排序 |
| `initialization.warp_shift_endpoints` | `s-range`；两个端点列表 |
| `training.engine/device/dtype` | 仅 `native/cpu/float64`；可省略 |
| `training.profile` | 仅 `native-adaptive-v1`；可省略 |
| `training.optimizer` | 分组入口仅 `bfgs`；其他优化器可用性以历史 CLI 为准 |
| `training.iterations` | `max-iter`；BFGS 迭代上限，不是 Adam epoch；默认 1000 |
| `training.convergence_tolerance` | `optimizer-conv-tol`；非负收敛阈值，保持原生默认 |
| `training.trainable_groups` | 仅完整 checkpoint 续训；`shift/readout/radial/adaptive_amplitude/adaptive_weight` |
| `loss.profile` | 仅 `sus2-legacy`；不改变损失平均方式 |
| `loss.energy_weight/force_weight/virial_weight` | `energy-weight/force-weight/stress-weight`；默认 1/0.01/0.001；不转换 virial/stress 单位或数据 |
| `data.train/valid` | 训练 CFG / 可选验证 CFG；不读取或自动添加 test |
| `data.model` | 完整自适应模型 checkpoint，用于续训 |
| `output.directory` | 为五项输出提供默认路径：`initial.mtp/current.mtp/best.mtp/final.mtp/optimizer.csv`；续训不生成 initial |
| `output.initial_model/current/best/final/optimizer_trace` | 可分别指定路径；不使用 directory 时必须全部显式指定（续训不需 initial） |
| `output.model` | 仅 init-sh 的拓扑输出路径；也可从 directory 默认生成 initial.mtp |

归一化及正则化沿用现有原生自适应训练协议，不因采用 YAML 而变动。
以下四项可以省略，也可以显式写出固定值用于说明；不能修改其数值：

```yaml
regularization:
  radial_smoothness: 1e-6
  radial_smoothness_grid: 128
  energy_readout_l2: 1e-5
  coordination_response_l2: 1e-4
```

原生 CPU 会用全部训练配置做固定 RMS 标定；这不保证和 PT 有上限的标定子集
逐项数值相同。见 [现有训练协议](adaptive-pytorch-training-profile.md)。
分组输入不会增加 GPU/Adam 支持；`lr/batch/epochs` 等 PT 参数会直接报错。
`--std-weight`、`--stdd-weight` 仍可追加为非负原生正则参数。

## CLI 短名称和严格检查

所有分组标量也可以用 `--group.key=value` 或 `--group.key value`。
新增常用短名如下，不删除旧参数名：

```bash
mlp-sus2 train train.cfg \
  --radial-channels 8 --coord-channels 4 --interaction-channels 4 \
  --neighbor-cutoff 6.5 --coord-body-order 4 --interaction-body-order 4 \
  --coord-coupling cross --interaction-coupling diagonal \
  --distance-warp-sharing channel-angular --response-layout per-basis \
  --response-activation linear --iterations 2000 --virial-weight 5 \
  --output.directory runs/new-run --check-config
```

若同时给出 YAML、新 CLI 和允许的旧 CLI，对同一设置要求一致，不会隐式覆盖。
分组模式拒绝未支持的旧选项；复杂旧训练选项请单独使用历史 CLI。
不混用 YAML 的 data 路径和位置参数，允许位置数据路径搭配 CLI output.directory。

YAML 路径相对 YAML 所在目录解析；仅通过 CLI 新添的路径相对启动目录。
续训配置中不再写 `model.*` / `elements`，直接继承 checkpoint 架构。
实际执行前拒绝复用或覆盖已存在的输出文件，拒绝输出互相重名或等于输入。
该检查用于避免意外覆盖，不是并发作业锁；两个作业应使用不同输出目录。
输入数据和 checkpoint 永远不因配置翻译而改写。

本入口支持明确、无需外部依赖的 YAML 子集：缩进映射、标量、单/双引号、
注释、上述列表的行内或块列表，以及多体角动量的行内或块映射。
不支持 anchors、tags、tabs、多行标量、文档分隔符、任意嵌套 flow maps。
未知字段、重复字段、空组、非有限数字和非法通道数都报错。

## 回归检查

```bash
bash dev_test/grouped_input_check.sh
python dev_test/grouped_native_cli_check.py /absolute/path/to/mlp-sus2
bash dev_test/sh_train_cli_compatibility_check.sh /absolute/path/to/mlp-sus2
```

第一个检查不依赖 MKL/MPI。第二个对真实可执行程序检查 YAML/旧 CLI
拓扑一致、单步 BFGS 最终模型一致、续训和拒绝覆盖；需在计算资源上执行。

2026-10-03 qualification（CPU 队列 33；没有 GPU 或生产训练）：

- Clang 和 GCC C++11 严格编译的输入测试：36 项通过。
- 真实 MPI/MKL 可执行程序：21 项通过，包括 2-rank 输入检查。
- YAML 与旧 CLI 的拓扑、初始化模型、单步 BFGS 最终模型、优化器 CSV、
  预测能量/力/应力输出均逐字节相同。
- 历史 `sh_train_cli_compatibility_check.sh` 与
  `adaptive_selective_train_check.py --mpi-command 'mpiexec.hydra -n 2'` 通过。
- 服务端资格记录：
  `/work/phy-weigw/20260321_Test/superlinear-grouped-cli-20261003/logs/`。
