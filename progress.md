# 实验进度

## 当前阶段

- 阶段：6 / 6，文档与复核
- 状态：已完成；核心实验通过全部验收标准
- WIP：无；后续工作仅为可选扩展，不属于核心实验交付

## 2026-10-30 最小闭环启动修正

### 发现的问题

- 第一次正式启动在环境快照校验处停止，尚未进入正确性检查或性能计时。
- 运行时 `LD_LIBRARY_PATH` 为
  `/home/sunchi/miniconda3/envs/ai-infra-compile/lib:/opt/ros/noetic/lib`，
  与快照记录的单一 Conda 库目录不一致。
- 原因是旧版 `scripts/run_in_env.sh` 会把交互终端继承的 ROS 库目录
  拼接进实验子进程。

### 修正与验证

- `scripts/run_in_env.sh` 现在仅向子进程传入
  `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`，不再继承无关库目录。
- 已验证调用者终端的 `/opt/ros/noetic/lib` 在命令执行前后保持不变；
  只有实验子进程使用 Conda 库目录。
- 本次失败没有生成最小闭环 CSV，不属于实验测量数据。

## 2026-10-30 最小闭环验收

### 已完成内容

- 在 Ubuntu VM 的独立 `ai-infra-compile` 环境运行 CPU、batch size 1、
  4 线程的 Eager/compiled 最小闭环。
- 正确性检查在独立进程中通过：`allclose=true`，最大绝对误差为
  `2.2649765014648438e-06`，使用 `rtol=1e-4`、`atol=1e-5`。
- Eager 与 compiled 分别在新的工作进程和临时 Inductor 缓存中运行；
  首次调用与预热后稳态测量分开保存。
- 原始 CSV 的四个组合均存在：两种执行模式 × `first_call`/`steady`，
  延迟与吞吐量均为正值；元数据配置与 CSV 一致。
- 结果文件已从 VM 复制回本地项目的 `results/` 目录。

### 验证命令

```bash
scripts/run_in_env.sh python benchmark.py
scripts/run_in_env.sh python -m unittest -v tests.test_benchmark
```

### 结果文件

- `results/minimal_raw_results.csv`
- `results/minimal_raw_results.metadata.json`

### 暂定观察（非正式结论）

- Eager 首次调用：`32.230913 ms`；compiled 首次调用：`9967.199815 ms`。
- Eager 稳态中位延迟：`25.221801 ms`；compiled：`33.001973 ms`。
- 单次最小闭环的稳态加速比为 `0.7643x`；compiled 约慢 `30.85%`。
- 因 compiled 稳态没有比 Eager 快，该配置的回本次数应为 `null`。
- 当前只有一次独立重复，不足以形成性能结论；该结果只用于验收测量链路。

### 下一步

进入阶段 3“完整矩阵”：将 batch size 扩展为 `{1, 8, 32}`，线程数扩展为
去重后的 `{1, 4}`，每个配置至少进行 5 次独立测量，并继续保留新进程、
独立缓存、正确性检查和原始数据。

## 2026-10-30 完整矩阵脚本准备

### 已实现内容

- `benchmark.py --matrix` 自动使用 batch size `{1, 8, 32}` 与线程数
  `{1, 4, 物理核心数}`，重复项去重；当前 VM 因此使用 `{1, 4}`。
- 完整矩阵默认进行 5 次独立重复，少于 5 次会被参数校验拒绝。
- 每个 batch size/线程数组合在性能计时前单独运行正确性检查。
- 奇数次重复按 Eager→compiled、偶数次按 compiled→Eager 执行，降低固定
  执行顺序与时间漂移完全重合的风险。
- 每完成一种执行模式的一次独立测量，就原子更新 partial CSV 与状态文件；
  `--resume` 会校验环境快照、脚本版本和参数后跳过已完成的测量。
- 成功结束时预计写入 120 行原始数据；checkpoint 只在两个最终文件都成功
  写入后删除。

### 当前验证

- Python 语法检查通过。
- 本地与 Ubuntu VM 的 13 个单元测试通过，包括 120 行矩阵完整性、从
  checkpoint 续跑、VM 核心数变化拒绝、run ID 不匹配拒绝，以及续跑时
  无关最终文件拒绝测试。
- Ubuntu 上环境快照与运行时的物理/逻辑核心数均为 `4/4`，完整环境校验通过。
- 独立代码审查提出的 3 个 Important 问题均已修复并复核关闭；复核未发现
  新的 Critical 或 Important 问题。
- 正式输出、元数据和两个 checkpoint 路径均确认不存在，可以开始正式运行。
- 尚未在 Ubuntu VM 执行正式矩阵，阶段 3 仍处于进行中。

### 下一步

1. 用户在空闲 VM 中执行
   `scripts/run_in_env.sh python benchmark.py --matrix`。
2. 若运行中断，保留 checkpoint 并使用同一命令加 `--resume` 继续。
3. 成功后验收 120 行数据、6 个正确性结果和完整元数据，再结束阶段 3。

## 2026-10-30 完整矩阵验收

### 已完成内容

- Ubuntu VM 正式运行 `scripts/run_in_env.sh python benchmark.py --matrix`，
  从 2026-10-30 14:25 至 14:39（Asia/Shanghai）完成，无续跑。
- 6 个 batch size/线程数组合的正确性检查全部通过；最大绝对误差范围为
  `2.2649765e-06` 至 `2.9802322e-06`。
- `results/raw_results.csv` 共 120 行，覆盖 3 个 batch size × 2 个线程数 ×
  5 次独立重复 × 2 种执行模式 × 2 个阶段；无缺失或重复。
- 所有延迟、吞吐量和误差字段均为有限有效值，吞吐量可由 batch size 与延迟
  重新计算得到；稳态 Timer 计数与 IQR 字段完整。
- CSV 与元数据只有一个相同的 `run_id`；元数据记录 60 个已完成测量、6 个
  正确性结果和 0 次续跑。
- 本地与 VM 的两个结果文件 SHA-256 一致，成功运行后的 checkpoint 已清理。

### 结果文件

- `results/raw_results.csv`
- `results/raw_results.metadata.json`

### 初步测量观察（Profiler 前，不作因果解释）

| batch | 线程 | Eager 稳态中位数 (ms) | compiled 稳态中位数 (ms) | 加速比 | 回本次数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 46.622 | 56.051 | 0.832x | 不可回本 |
| 1 | 4 | 20.254 | 26.352 | 0.769x | 不可回本 |
| 8 | 1 | 347.652 | 278.104 | 1.250x | 约 137 |
| 8 | 4 | 124.851 | 106.664 | 1.171x | 约 534 |
| 32 | 1 | 1496.273 | 1092.562 | 1.370x | 约 23 |
| 32 | 4 | 560.265 | 409.589 | 1.368x | 约 64 |

- 上表使用每组 5 次独立重复的中位数；回本次数为当前校验阶段的复算值，
  后续仍需由 `analyze.py` 统一生成并由单元测试覆盖。
- 目前只能陈述 batch size 与观测加速模式相关，不能在 Profiler 前声称具体
  算子融合或调度机制导致了差异。

### 下一步

进入阶段 4“性能分析”：选择一个代表性配置运行 Eager/compiled Profiler，
保存算子表或 trace，并比较主要算子、自身 CPU 时间和调用结构。Profiler
数据只用于解释瓶颈，不作为正式延迟数据。

## 2026-10-30 Profiler 脚本准备

### 已实现内容

- 新增 `profile_inference.py`，避免使用会遮蔽 Python 标准库模块的
  `profile.py` 文件名。
- 默认代表性配置为 CPU、batch size 32、4 线程；该配置在正式矩阵中观测到
  `1.368x` 稳态加速。
- Eager 与 compiled 分别在新进程和新 Inductor 缓存中运行；一次 priming 与
  5 次预热全部在 Profiler 启动前完成，因此不采集编译过程。
- 使用 `torch.profiler` 采集 3 次推理，按输入 shape 聚合并保存全部算子，
  输出自身 CPU 时间、总 CPU 时间、调用次数、平均时间和自身 CPU 占比。
- Profiler 启动前再次运行独立正确性检查；结果表与元数据拒绝覆盖已有文件。
- 元数据明确标注 Profiler 墙钟时间不能作为正式基准延迟。

### 当前验证

- Python 语法检查通过。
- 本地完整测试集 18 个测试通过，其中 5 个覆盖 Profiler 排序、完整输出、
  参数校验和覆盖保护。
- Ubuntu VM 完整测试集 18 个测试通过。
- Ubuntu 真实 Profiler 冒烟测试通过：batch size 1、单线程、一次采样捕获
  Eager 112 组和 compiled 48 组聚合算子事件；测试使用临时缓存且未写正式结果。
- 独立代码审查未发现 Critical 或 Important 问题；确认编译/预热位于采样区间
  之外、模式间进程与缓存隔离、全部聚合事件均保留，且自身 CPU 占比计算
  不使用会重复计数的 inclusive 时间。
- 正式 Profiler 的 CSV 与元数据路径确认不存在，可以开始运行。

### 下一步

1. 用户运行 `scripts/run_in_env.sh python profile_inference.py`。
2. 验收 `results/profiler/operator_summary.csv` 与 `metadata.json`。
3. 比较 Eager/compiled 主要自身 CPU 算子、调用结构和占比后结束阶段 4。

## 2026-10-30 Profiler 验收与解释

### 已完成内容

- 正式运行 CPU、batch size 32、4 线程的 Eager/compiled Profiler；独立
  正确性检查通过，最大绝对误差为 `2.9802322e-06`。
- 保存 `results/profiler/operator_summary.csv` 和
  `results/profiler/metadata.json`，共 158 个按输入 shape 聚合的算子条目：
  Eager 112 个、compiled 46 个。
- CSV 的 rank 连续且无重复分组；两个模式的自身 CPU 占比各自精确加总为
  100%，所有时间与调用次数字段有效；CSV 与元数据 `run_id` 一致。
- 本地与 VM 的两个 Profiler 文件 SHA-256 完全一致。
- 元数据确认一次 priming 与 5 次预热发生在采样前，正式采样为 3 次推理，
  `compilation_profiled=false`。

### 主要观察

- Eager 的自身 CPU 时间主要分布为：`aten::mkldnn_convolution` 81.44%、
  `aten::max_pool2d_with_indices` 9.62%、`aten::native_batch_norm` 5.10%、
  `aten::clamp_min_` 2.06%、`aten::add_` 0.98%。
- compiled 的自身 CPU 时间主要分布为：`aten::mkldnn_convolution` 91.54%、
  生成的 `CompiledFxGraph` 调用 7.83%，其他单项均低于 0.2%。
- 两种模式都记录 60 次 `mkldnn_convolution` 调用，即 3 次推理中每次仍有
  20 次卷积；没有证据表明编译减少了卷积层调用数。
- Eager 中单独出现的 BatchNorm、ReLU、残差加法等主要条目在 compiled
  表中不再作为主要独立算子出现，且聚合事件组从 112 降到 46。这与
  `torch.compile` 将非卷积工作合并或下沉到生成代码、减少调度碎片的解释
  一致，但单凭聚合表不能证明每一个具体融合规则。
- compiled 后卷积仍占绝对多数，因此代表性配置的主要计算瓶颈仍是卷积；
  编译收益更可能来自卷积外工作与框架/算子调度开销的减少。
- Profiler 墙钟时间只用于交叉观察，不代替无 Profiler 的正式基准延迟。

### 下一步

进入阶段 5“结果分析”：实现 `analyze.py` 和回本公式单元测试，从原始 CSV
自动生成汇总表、延迟/吞吐量/回本图，并确保不可回本配置输出 `null`。

## 2026-10-30 结果分析验收

### 已完成内容

- 新增 `analyze.py`，从 `results/raw_results.csv` 自动生成 6 行配置级汇总，
  不修改原始 CSV；派生文件默认拒绝覆盖，只有显式 `--overwrite` 才会重建。
- 分析前强制校验 ResNet18、CPU、随机种子、容差、软件与硬件身份，以及
  batch size `{1, 8, 32}`、线程数 `{1, 4, 物理核心数}`、5 次独立重复、
  两种执行模式和两个测量阶段的完整笛卡尔积。
- 每个配置报告首次调用与稳态延迟的中位数和 min–max 范围、稳态吞吐量
  中位数和 min–max 范围、稳态加速比、额外编译开销、延迟节省和回本次数。
- compiled 稳态不快于 Eager 的两个 batch size 1 配置，其回本次数均写为
  `null`，没有输出负数或把负结果解释成加速。
- 生成 `figures/latency.png`、`figures/throughput.png` 和
  `figures/break_even.png`；三张图均由汇总脚本从原始数据生成并通过目视检查。
- `results/analysis_metadata.json` 记录原始 CSV、基准元数据、分析脚本和派生
  产物 SHA-256，以及 Linux、Python 3.12.16、Matplotlib 3.10.9/Agg 和 Git
  状态；明确 PNG 字节可能因渲染栈不同而变化。
- 在专用 `ai-infra-compile` 环境增加 `matplotlib-base`，并更新
  `environment.yml`、Conda 精确锁文件和 pip 版本清单；未修改其他环境。

### 验证命令

```bash
scripts/run_in_env.sh python -m unittest discover -s tests -p "test_*.py" -v
scripts/run_in_env.sh python analyze.py --overwrite
```

### 验证结果

- Ubuntu VM 完整测试集 27 个测试通过；分析测试覆盖回本公式、不可回本、
  中位数与范围、缺行/重行、正确性失败、吞吐量不一致、身份/矩阵错误、
  输入输出路径保护，以及真实 CLI 端到端生成与输入文件哈希不变。
- 正式分析验证 120 行原始数据和 6 个完整配置；汇总值由独立 PowerShell
  复算一致，所有已记录输入/脚本/输出哈希均与当前文件匹配。
- 独立代码审查第一轮提出的实验身份、必做矩阵、provenance、测试覆盖和
  进度记录问题均已修复；最终复核未发现 Critical 或 Important 问题。

### 正式结果（5 次独立重复的中位数）

| batch | 线程 | Eager 稳态 (ms) | compiled 稳态 (ms) | 加速比 | 回本次数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 46.622 | 56.051 | 0.832x | 不可回本 |
| 1 | 4 | 20.254 | 26.352 | 0.769x | 不可回本 |
| 8 | 1 | 347.652 | 278.104 | 1.250x | 136.7 |
| 8 | 4 | 124.851 | 106.664 | 1.171x | 534.0 |
| 32 | 1 | 1496.273 | 1092.562 | 1.370x | 22.7 |
| 32 | 4 | 560.265 | 409.589 | 1.368x | 63.9 |

### 结果文件

- `results/summary.csv`
- `results/analysis_metadata.json`
- `figures/latency.png`
- `figures/throughput.png`
- `figures/break_even.png`

### 下一步

进入阶段 6“文档与复核”：完成 README 的运行方法、实验设计、结果、
Profiler 解释、限制和失败结果；随后从干净环境重新运行关键命令。核心实验
尚不能在 README 与干净环境复核完成前宣称全部完成。

## 2026-10-29 环境探测

### 已完成内容

- 完整阅读 `AGENTS.md`，确认核心实验范围与验收标准。
- 只读检查项目目录：当前仅有 `AGENTS.md`，目录尚未初始化为 Git 仓库。
- 探测 Windows 宿主：Windows 11，Intel Core Ultra 5 125H，14 个物理核心、18 个逻辑核心；未检测到 NVIDIA GPU 或 CUDA。
- 检测到当前 PATH 中的 Python 是独立安装的 Python 3.13.12，不是 Conda 环境。
- 验证 PyTorch 2.12.0+cpu 与 torchvision 0.27.0+cpu 可导入。
- 运行极小 `torch.compile` CPU 冒烟测试；失败原因是 Inductor 找不到 MSVC `cl.exe`。
- 确认用户另有可通过 SSH 访问的 Ubuntu 虚拟机和 HPC，并偏好使用独立 Conda 环境。
- 新增跨平台环境采集脚本 `collect_environment.py`。
- 选定 VMware Ubuntu 虚拟机作为核心 CPU 实验主环境：Ubuntu 20.04.6、4 个 vCPU、约 7.7 GiB 内存，无 CUDA 设备。
- 在 `/home/sunchi/miniconda3` 安装并校验 Miniconda 26.9.4；未运行 `conda init`，未修改 `.bashrc`。
- 创建独立环境 `ai-infra-compile`：Python 3.12.16、PyTorch 2.12.1+cpu、torchvision 0.27.1+cpu。
- 在该环境内安装 GCC/G++ 13.4.0；系统 Python 3.8.10 与系统 GCC/G++ 9.4.0 保持不变。
- 新增 `scripts/env.sh`，仅在显式 source 后为本实验配置 `CC`、`CXX` 与 `LD_LIBRARY_PATH`。
- Ubuntu 上的极小 `torch.compile` CPU 冒烟测试通过：`allclose=true`，最大绝对误差为 `0.0`。
- 生成并复制正式环境文件及两份依赖锁定文件到本地 `results/`。

### 验证命令

```powershell
python --version
python -m pip show torch torchvision
python -m torch.utils.collect_env
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import torch; f=lambda x: torch.sin(x)+1; print(torch.compile(f)(torch.randn(8)))"
```

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-infra-compile
source scripts/env.sh
python collect_environment.py --output results/environment.json
python scripts/smoke_compile.py
```

### 证据与结果

- `python -m torch.utils.collect_env` 在中文 Windows OEM 解码处抛出 `UnicodeDecodeError`；这是诊断工具失败，不是 PyTorch 导入失败。
- `torch.compile` 冒烟测试抛出：`RuntimeError: Compiler: cl is not found.`
- Windows 环境不作为正式实验环境。
- Ubuntu 系统 G++ 9.4 首次冒烟测试失败：不支持 PyTorch 2.12.1 需要的 `-std=c++20`。
- 改用环境级 G++ 13.4 后首次加载失败：系统 `libstdc++.so.6` 缺少 `GLIBCXX_3.4.32`；通过仅对实验命令设置 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 解决。
- Conda 默认 channels 要求非交互接受服务条款；未代替用户接受，环境创建时显式使用 `conda-forge`。用户随后明确表示接受默认源条款，但当前环境保持既有来源不变。
- 正式环境记录：`results/environment.json`。
- 精确 Conda 锁定：`results/conda-linux-64.lock`。
- 完整 pip 版本：`results/pip-freeze.txt`。

### 当前问题

- VM 只分配了 4 个 vCPU，因此核心 CPU 线程矩阵去重后为 `{1, 4}`。
- VM 共享宿主 CPU，正式测量前必须记录空闲负载并尽量关闭其他高负载程序；结果只能解释为该 VM 配置下的性能。
- HPC 保留为核心实验通过后的复核或可选 CUDA 扩展，不与当前主环境混测。

### 下一步

进入阶段 2“最小闭环”，但一次只做 CPU、batch size 1：

1. 编写最小版 `benchmark.py`，固定模型参数、输入和随机种子。
2. 分别记录 Eager 与 compiled 首次调用，compiled 首次调用必须使用新进程。
3. 预热后测量稳态延迟，并检查 `torch.allclose` 与最大绝对误差。
4. 只生成 batch size 1 的原始 CSV，不提前扩展完整矩阵。
5. 验证结果文件和计时逻辑后再结束阶段 2。

## 2026-10-30 文档与干净环境复核

### 已完成内容

- 新增 `README.md`，记录项目目标、实验设计、正式结果、Profiler 观察、
  完整复现命令、失败记录和适用限制，并引用 PyTorch 官方资料。
- 新增 `.gitignore`，只忽略 Python 缓存、测试缓存、虚拟环境和完整矩阵的
  checkpoint 临时文件；正式原始数据、Profiler 结果和图表不被忽略。
- 收紧 `environment.yml`：固定 Python 3.12.16、pip 26.2.2、G++ 13.4.0
  和 Matplotlib 3.10.9，使用 `conda-forge` 与 `nodefaults`，其余 Python
  依赖继续由 `requirements.txt` 固定。
- 按用户此前明确同意的条款记录 Conda defaults 两个频道的 ToS 接受状态；
  该操作没有从 defaults 安装实验依赖，干净环境求解结果仍全部来自
  `conda-forge`。

### 干净环境复核

- 从 `environment.yml` 新建临时环境 `ai-infra-clean-verify`，没有改动正式
  `ai-infra-compile` 环境或系统 Python。
- 干净环境版本核对：PyTorch 2.12.1+cpu、torchvision 0.27.1+cpu、
  Matplotlib 3.10.9；编译冒烟测试 `allclose=true`、最大绝对误差为 0。
- 完整测试集 27 个测试全部通过。
- 在临时环境重新运行环境采集、最小闭环、正式原始数据分析和代表性
  Profiler。最小闭环正确性通过，生成 4 行结果；Profiler 正确性通过，
  生成 158 行且同时含 Eager/compiled 两种模式。
- 重新分析正式 120 行原始数据后，6 行 `summary.csv` 和三张 PNG 与正式
  产物的 SHA-256 均逐字节一致。
- 为了在不改动脚本正式环境名保护的前提下复核临时环境，只在复核命令
  子进程中把 `CONDA_DEFAULT_ENV` 映射为 `ai-infra-compile`；实际
  `CONDA_PREFIX` 始终指向临时环境。

### 验证命令

```bash
conda env create -n ai-infra-clean-verify -f environment.yml
conda activate ai-infra-clean-verify
CONDA_DEFAULT_ENV=ai-infra-compile scripts/run_in_env.sh python scripts/smoke_compile.py
CONDA_DEFAULT_ENV=ai-infra-compile scripts/run_in_env.sh python -m unittest discover -s tests -p "test_*.py" -v
```

临时复核输出写入 `/tmp/ai-infra-clean-verify.y2FYmW`，没有覆盖正式结果。
复核后已校验精确绝对路径并删除该目录及临时 Conda 环境；环境列表恢复为
`base` 和正式的 `ai-infra-compile`。正式环境随后再次通过编译冒烟测试。

### 当前问题与下一步

- 当前没有已知的核心实验阻塞项。
- 最终独立审查结论为 `Ready`，未发现 Critical 或 Important 问题。审查者
  独立核对了 120 行完整矩阵、6 个正确性配置、README 数字、Profiler
  统计、全部已记录 SHA-256 和 27 个测试。
- 审查者提出的非阻塞建议包括：可另存结构化干净环境复核清单、可在能恢复
  时补存 Windows 失败的完整 traceback，以及说明 Conda 锁文件代表最终
  可运行环境而非基准开始瞬间。最后一项已在 README 中补充；前两项不影响
  当前核心实验证据链。
- 核心实验至此完成。CUDA、HPC、其他模型或精度属于可选扩展，开始前应
  单独定义实验问题，且不得覆盖当前 CPU 正式结果。
