# C++ 原生训练：分组物理输入

维护分支 `codex/radial-coefficient-gate` 的开发版 `mlp-sus2` 现在支持
分组 YAML 和物理语义 CLI。它是现有 `init-sh` / `train` 的输入适配层，
复用现有模型方程、参数布局和训练实现，不新增求值算法。
支持配位自适应和无配位修正的静态径向模型；历史 CLI 的行为不变。

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
`O=interaction.channels`（相互作用通道）。启用配位层时要求 `1 <= G,O <= K`；
关闭后不再有 G，仍要求 `1 <= O <= K`。

| 分组字段 | 原生选项 | 含义 / 允许值 / 默认值 |
|---|---|---|
| `model.kind` | `superlinear-mode` | 默认 `coordination-adaptive`；`static-radial` 默认关闭配位层；显式 `coordination.enabled: false` 也可从默认架构选择静态模型 |
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
| `model.radial.warp_sharing` | `scaling-map` | `L`、`K`、`LK`（默认）；兼容 `angular`→L、`channel`→K、`channel-angular`→LK；只控制距离坐标变换的共享方式 |
| `model.coordination.enabled` | 输入适配层开关 | 默认 true；false 选择无 gate 的静态径向模型，不是冻结已有 U |
| `model.coordination.channels` | `two-layer-gate-k-rank` | G；默认 K |
| `model.coordination.body_order` | `two-layer-gate-body-order` | 配位环境多体阶数，2..6；默认相互作用多体阶数 |
| `model.coordination.channel_coupling` | `two-layer-gate-factor-pruning` | `cross`→q-total（默认），`diagonal`→same-k |
| `model.coordination.distance_warp` | `two-layer-gate-decoupled-scaling` | `shared`（默认）或 `independent`；shared 时不传这个存在即启用的旧开关 |
| `model.coordination.tensor_l_max` | `tensor-message-lmax` | 0（默认）、1、2，且不超过 L |
| `model.interaction.channels` | `outer-k-rank` | O；必须显式提供，避免误用旧架构的默认 0 |
| `model.interaction.body_order` | `body-order` | 相互作用多体阶数，2..6；默认 5 |
| `model.interaction.channel_coupling` | `sh-factor-pruning` | 自适应模式默认 `diagonal`→same-k，可选 `cross`→q-total；静态 ranked 模式仅 `cross`，也是该模式的默认值 |
| `model.angular_limit_by_body_order` | `body-l-max` | 如 `{2: 3, 3: 3, 4: 2, 5: 1}`；2..5 必须齐全，六体模型还需 6；也作用于配位图，保持原生语义 |
| `model.response.layout` | `adaptive-radial-projection` | `full`（默认）、`shared-n`；兼容 `per-basis`→full、`shared-latent`→shared-n |
| `model.response.latent_channels` | `adaptive-radial-shared-n-rank` | 共享投影数 d；默认 1；full 只能 1，shared-n 要求 1..N |
| `model.response.activation` | `adaptive-radial-activation` | `linear`→none（默认）、`tanh` |
| `model.response.element_mode` | `adaptive-radial-amplitude-mode` | `shared`（默认）、`source-element`→species；响应幅度按来源邻居元素区分 |
| `model.readout.element_mode` | `readout-mode` | `shared`（默认）、`per-element`→species；能量读出按中心原子元素区分 |
| `model.readout.joint` | `joint-readout` | false（默认）或 true；联合读出遵守现有 L/张量约束 |

距离 scaling 直接写 `L` / `K` / `LK`（区分大小写）：

```yaml
model:
  radial:
    warp_sharing: LK
```

每个元素对中，`L` 按角动量 l 区分、跨 k 共享；`K` 按径向通道 k 区分、
跨 l 共享；`LK` 对每个 (l,k) 单独设置。它不改变径向基数 N，也不控制
U 的 n 维度；U 仍由独立的 `model.response.layout` 设置。

U 的现有实现直接用以下三种配置（不是新增模型算法）：

```yaml
# 完整 U：U[l,o,n,q]，每个径向基 n 独立投影（默认）
model:
  response:
    layout: full
```

```yaml
# 单组跨 n 共享 U：U[l,o,q]，幅度 a[l,o,n] 仍依赖 n
model:
  response:
    layout: shared-n
    latent_channels: 1
```

```yaml
# 多组跨 n 共享 U：U[l,o,t,q]，幅度 a[l,o,n,t] 仍依赖 n
# 此处 d=2；允许 1 <= d <= radial.basis_size
model:
  response:
    layout: shared-n
    latent_channels: 2
```

跨 n 共享不删除径向基 n。`latent_channels` 增大也不自动变成 `full`，
两者仍是不同参数布局。以上结构在新模型初始化时确定；续训继承 checkpoint。

## 不使用配位效应层

新训练只需设置开关，并删除配位层和 response 的其他字段：

```yaml
model:
  l_max: 3
  radial_channels: 8
  neighbor_cutoff: 6.5
  radial:
    basis_size: 10
    warp_sharing: LK
  coordination:
    enabled: false
  interaction:
    channels: 4
    body_order: 4
    channel_coupling: cross
```

也可以写 `model.kind: static-radial`，省略 enabled；不能同时写 enabled: true。
显式 enabled: false 优先于 `kind: coordination-adaptive` 的默认配位行为。
完整可运行输入见 [静态模式示例](../examples/grouped-native-static-train.yaml)。

该模式的径向系数仍然可训练，但不再依赖原子的局域配位环境。它复用已有的
gate-free SH/ranked outer 路径：静态径向系数加环境无关的 K→O 投影 W，
没有配位图、U 或自适应响应幅度。O=1 时加载器会把 W 吸收到径向系数中，
O>1 时保留独立 W；不要把静态 W 误认成配位响应 U。

| 参数 | 关闭配位层后 |
|---|---|
| 元素、L、K、cutoff、径向基种类/N/scale/参考下界 | 可用，含义不变 |
| `radial.warp_sharing: L/K/LK` | 可用，仍可训练；不是环境修正开关 |
| `interaction.channels` O、多体阶数、各阶角动量上限 | 可用；1≤O≤K，多体阶数 2..6 |
| `interaction.channel_coupling` | 仅 `cross`；显式 `diagonal` 会报错，不会被自动替换 |
| `coordination.*` 除 enabled | 不可用，必须省略；G、配位多体阶数、独立配位 scaling、张量信息均不存在 |
| `response.*` | 不可用，必须省略；full/shared-n、latent_channels、activation、element_mode 都不再有作用 |
| `readout.element_mode/joint` | 仅 shared / false；per-element 和 joint:true 不支持 |
| `initialization.parameters` | 新模型可用 same（默认）或 random；续训仅 same |
| `initialization.seed` | 仅新模型显式 parameters:random 时可用，避免静默接受无效 seed |
| slope/shift 初始化端点 | 可用，沿用原生距离 scaling 初始化 |
| BFGS iterations/收敛阈值、E/F/virial 权重、train/valid、全部输出路径 | 可用；CPU/FP64；不添加 test |
| 径向平滑系数/网格、energy_readout_l2 | 可调；默认 1e-6 / 128 / 0，均为非负（网格为正整数） |
| `coordination_response_l2` | 不可用，必须省略，包括显式写 0 |
| `training.trainable_groups` | 不可用；当前分组选择性训练仅支持自适应 checkpoint |
| Adam 的 lr/batch/epochs | 不可用；分组 C++ 入口仍是 BFGS |

静态模式自动选择 `native-static-v1`：物理径向参数化、原生 vibrations 加权；
新模型保留原生 rescale + 线性读出 warmup，不使用自适应 fixed-RMS 标定。
为使 iterations 与主 BFGS 步数对应并隔离输出目录，这个新分组 profile
显式跳过历史额外 75 步 pretraining（不生成工作目录的 ini.mtp）。对应旧 CLI
加 `--skip-preinit` 即可得到相同路径；历史 CLI 本身的默认行为未修改。

静态模型续训必须显式声明 `training.profile: native-static-v1`，并省略所有
`model.*`。见 [静态续训示例](../examples/grouped-native-static-continue.yaml)。
省略 profile 的续训仍默认 native-adaptive-v1，实际加载后检查与 checkpoint
匹配。不允许以这个开关改变已有 checkpoint 的结构；新建模型才可选开关。

## 训练、损失和输出

参考 [完整示例](../examples/grouped-native-train.yaml)。示例的 20/20/5
是显式选择，不是被本次修改引入的新默认值。

| 字段 | 原生选项 / 语义 |
|---|---|
| `initialization.seed` | `init-random-seed`；uint64；平衡初始化默认 seed=1 |
| `initialization.parameters` | 自适应训练固定 same；静态新模型支持 same/random；续训保持已有物理系数 |
| `initialization.warp_slope_endpoints` | `scal-range`；两个端点列表；保持原生端点方向语义，不自动排序 |
| `initialization.warp_shift_endpoints` | `s-range`；两个端点列表 |
| `training.engine/device/dtype` | 仅 `native/cpu/float64`；可省略 |
| `training.profile` | `native-adaptive-v1` 或 `native-static-v1`；新训练按配位开关默认选择；静态续训必须显式声明 |
| `training.optimizer` | 分组入口仅 `bfgs`；其他优化器可用性以历史 CLI 为准 |
| `training.iterations` | `max-iter`；BFGS 迭代上限，不是 Adam epoch；默认 1000 |
| `training.convergence_tolerance` | `optimizer-conv-tol`；非负收敛阈值，保持原生默认 |
| `training.trainable_groups` | 仅完整 checkpoint 续训；`shift/readout/radial/adaptive_amplitude/adaptive_weight` |
| `loss.profile` | 仅 `sus2-legacy`；不改变损失平均方式 |
| `loss.energy_weight/force_weight/virial_weight` | `energy-weight/force-weight/stress-weight`；默认 1/0.01/0.001；不转换 virial/stress 单位或数据 |
| `data.train/valid` | 训练 CFG / 可选验证 CFG；不读取或自动添加 test |
| `data.model` | 完整自适应或无 gate ranked SH checkpoint，用于续训；profile 必须匹配 |
| `output.directory` | 为五项输出提供默认路径：`initial.mtp/current.mtp/best.mtp/final.mtp/optimizer.csv`；续训不生成 initial |
| `output.initial_model/current/best/final/optimizer_trace` | 可分别指定路径；不使用 directory 时必须全部显式指定（续训不需 initial） |
| `output.model` | 仅 init-sh 的拓扑输出路径；也可从 directory 默认生成 initial.mtp |

自适应模式的归一化及正则化沿用现有原生自适应训练协议，不因采用 YAML 而变动。
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
  --distance-warp-sharing LK --response-layout full \
  --response-activation linear --iterations 2000 --virial-weight 5 \
  --output.directory runs/new-run --check-config
```

静态新模型 CLI 示例（不附带任何 coord-channels/response 设置）：

```bash
mlp-sus2 train train.cfg --coord-enabled false \
  --radial-channels 8 --interaction-channels 4 --neighbor-cutoff 6.5 \
  --interaction-body-order 4 --distance-warp-sharing LK \
  --iterations 2000 --output.directory runs/new-static --check-config
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

同日直接选项名称扩展的资格检查：

- 输入检查 59 项、真实 CLI 检查 45 项（含 2-rank 检查）通过。
- `L/K/LK`、`full`、单组与多组 `shared-n` 的 YAML、兼容别名和旧 CLI
  生成的拓扑逐字节相同；单步 BFGS 和预测输出的等价检查继续通过。
- 历史 CLI 与选择性自适应训练 MPI 回归通过；模型求值算法未修改。
- 服务端资格记录：
  `/work/phy-weigw/20260321_Test/superlinear-grouped-options-20261003/logs/`。

同日无配位层模式的资格检查（CPU 队列 33；资格作业 4309286）：

- 输入检查 90 项、真实 CLI 检查 71 项通过，含静态双进程单步 BFGS。
- 静态 L/K/LK 拓扑、初始化/最终模型、优化器 CSV 和 E/F/stress 预测
  与对应旧 CLI（含 skip-preinit）逐字节一致。
- 静态默认初始化、续训、profile/无效字段拒绝检查通过。
- `sh_outer_k_rank_no_gate_check.sh` 的 rank 1/2/3、恒等映射和 E/F/stress
  方向导数检查通过；历史 CLI、自适应选择性训练 MPI 回归继续通过。
- 服务端资格记录：
  `/work/phy-weigw/20260321_Test/superlinear-grouped-static-qualified-20261003/logs/`。
