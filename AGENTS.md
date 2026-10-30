# PyTorch 推理编译性能实验协作说明

## 项目目标

本项目复现并扩展一个短周期 AI Infra 实验：比较 PyTorch Eager 与 `torch.compile` 在 ResNet18 推理中的性能，回答以下问题：

1. 不同 batch size、CPU 线程数和设备下，`torch.compile` 是否带来稳定的稳态收益？
2. 首次编译开销需要多少次推理才能收回？
3. 编译前后的主要算子瓶颈发生了什么变化？

最终产物应是一个可重复运行的 GitHub 项目，而不是一次性的 Notebook 或截图。实验允许得到负结果；不得为了“证明加速”筛选数据。

## 用户背景

- 用户：孙驰，哈尔滨工业大学（深圳）计算机科学与技术专业本科生。
- 已有能力：Python、C、C++、Java；机器学习 95，深度学习 97；正在学习操作系统与计算机网络。
- 相关经历：完成 MIT xv6 lock lab 的 buffer cache 并发与缓存策略优化，具备基线设计、重复测量和性能指标解释经验。
- 兴趣方向：AI Infra、操作系统、Agent 工程、机器人。
- 本实验用途：判断是否喜欢 AI Infra 研究，并作为联系本科毕设导师时的补充项目。

## 研究范围

### 必做范围

- 模型：`torchvision.models.resnet18(weights=None).eval()`。
- 输入：固定随机种子生成的 `[B, 3, 224, 224]` 张量，不下载数据集。
- 执行模式：PyTorch Eager、默认配置的 `torch.compile`。
- batch size：`1`、`8`、`32`。
- CPU 线程数：`1`、`4`、机器物理核心数；若物理核心数与前两项重复则去重。
- 每个配置至少进行 5 次独立测量。
- 必须测量首次调用和预热后的稳态性能。
- 必须检查 Eager 与 compiled 输出的数值一致性。
- 使用 `torch.profiler` 分析至少一个代表性配置。

### 可选扩展

只有必做范围通过验收后，才可以增加：

- CUDA 设备；
- MobileNetV3 与 ResNet18 的对比；
- `torch.compile` 的其他 mode；
- FP16/BF16 或量化实验。

可选扩展不得阻塞核心实验交付。

### 明确不做

- 不训练模型，不追求分类精度。
- 不下载 ImageNet 等大型数据集。
- 不构建网页或前后端界面。
- 不把随机输入下的推理性能描述为真实业务端到端性能。
- 不在核心实验完成前引入 ONNX Runtime、TensorRT 等额外后端。

## 实验环境

先检测环境，再决定使用 CPU 或增加 CUDA。不得假设机器一定有可用 GPU。

必须记录：

- 操作系统与版本；
- CPU 型号、物理核心数和逻辑核心数；
- GPU 型号与显存（若有）；
- Python、PyTorch、torchvision 版本；
- CUDA 与驱动版本（若有）；
- 实验所用命令和关键环境变量。

优先在 Linux 或 WSL2 中运行。若原生 Windows 下 `torch.compile` 因编译器或后端不兼容失败，应保留完整错误和环境信息，再评估是否切换到 Linux/WSL2；不得悄悄改成另一个实验并宣称完成。

## 测量规范

### 正确性

- 固定随机种子。
- 使用同一组模型参数和输入比较两种执行模式。
- 使用 `torch.inference_mode()`。
- 记录 `torch.allclose` 结果、最大绝对误差和所用 `rtol/atol`。
- 任一配置不满足数值一致性时，不得纳入性能结论，先定位原因。

### 首次调用

- Eager 首次调用和 compiled 首次调用分别记录。
- compiled 的首次调用包含图捕获与编译开销，必须与稳态时间分开。
- 冷启动的独立重复应使用新进程，避免内存缓存或已编译图污染后续测量。

### 稳态性能

- 正式计时前进行预热。
- 优先使用 `torch.utils.benchmark.Timer` 或等价的可靠计时方法。
- 若使用 CUDA，计时必须正确同步设备；不能直接用未经同步的 `time.time()`。
- 不只报告最快值。至少报告中位数和波动范围，原始重复结果必须保留。
- Profiler 用于定位瓶颈，不使用开启 Profiler 时的延迟作为正式基准数据。

### 指标

至少计算：

- 首次调用延迟，单位 ms；
- 稳态单批次延迟，单位 ms；
- 吞吐量，`samples / second`；
- 稳态加速比：`eager_latency / compiled_latency`；
- 额外编译开销；
- 回本次数：

```text
break_even_requests = compile_overhead_ms /
                      (eager_steady_ms - compiled_steady_ms)
```

如果 compiled 稳态下不比 Eager 快，回本次数记为不可回本（例如 `null`），不能写成负数或强行解释为加速。

## 原始数据格式

结果写入 CSV 或等价的结构化文件。每行至少包含：

```text
timestamp
os
cpu_model
physical_cores
gpu_model
python_version
torch_version
torchvision_version
seed
model
device
execution_mode
batch_size
num_threads
repetition
phase
latency_ms
throughput_samples_s
allclose
max_abs_error
```

图表和结论必须由原始数据生成，不能手工填写结果。

## 建议项目结构

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── benchmark.py
├── profile.py
├── analyze.py
├── tests/
│   └── test_analysis.py
├── results/
│   ├── environment.json
│   └── raw_results.csv
├── figures/
│   ├── latency.png
│   ├── throughput.png
│   └── break_even.png
└── progress.md
```

`results/` 中保留原始实验数据。若文件较大，再决定是否加入版本控制；不要在未征得用户同意时删除数据。

## 工作顺序

一次只推进一个阶段，保持 WIP=1：

1. **环境探测**：确认 Python、PyTorch、编译器和设备，生成 `environment.json`。
2. **最小闭环**：仅跑 CPU、batch size 1，完成 Eager/compiled 正确性与计时。
3. **完整矩阵**：加入全部 batch size、CPU 线程数和独立重复。
4. **性能分析**：运行 Profiler，保存表格或 trace，解释主要瓶颈。
5. **结果分析**：生成汇总表、图表和回本点。
6. **文档与复核**：完成 README，从干净环境重新运行关键命令。

每阶段完成后更新 `progress.md`，写明：已完成内容、验证命令、结果文件、当前问题和下一步。

## 验收标准

只有以下条件全部满足，才能称核心实验完成：

- 环境信息已结构化保存；
- Eager 与 compiled 输出一致性检查通过；
- CPU 核心实验矩阵无缺失，且每组至少 5 次独立测量；
- 冷启动与稳态性能分开记录；
- 原始数据可追溯，汇总和图表可由脚本重新生成；
- 至少一个配置完成 Profiler 分析；
- 回本点公式有单元测试，且正确处理“不可回本”；
- README 写明运行方法、实验设计、主要结果、限制和失败结果；
- 从项目根目录执行核心命令能够成功，不依赖未记录的手工步骤。

未经以上证据，不得使用“完成”“复现成功”或“获得 X 倍加速”等结论。

## 写作与诚信约束

- 区分测量结果与解释，不把相关性写成因果关系。
- 所有加速数字注明基线、硬件、batch size、线程数和统计口径。
- 若只在随机输入上测量，明确说明不代表真实业务工作负载。
- 负结果、异常波动和失败配置都要保留并解释。
- 不选择性删除不利数据；异常值只有在给出可复核理由时才能排除。
- README 中引用 PyTorch 官方文档，避免复制大段教程文字。

## 官方参考资料

- `torch.compile` 端到端教程：<https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html>
- PyTorch Benchmark：<https://docs.pytorch.org/tutorials/recipes/recipes/benchmark>
- PyTorch Profiler：<https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- `torch.utils.benchmark` API：<https://docs.pytorch.org/docs/main/utils/benchmark.html>

## 新对话启动指令

新对话开始后：

1. 完整阅读本文件；
2. 检查当前目录已有文件，保留用户的现有修改；
3. 先做只读环境探测；
4. 给出简短实施计划；
5. 从“环境探测 → 最小闭环”开始执行；
6. 遇到配置问题时先记录证据并定位，不跳过正确性和计时规范。

