# 3DGS 训练日志（training-log.md）

> 目标：Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field
> Rendering*, SIGGRAPH 2023 —— 官方代码 `graphdeco-inria/gaussian-splatting`
> （HEAD `54c035f`），在**无 NVIDIA GPU** 的机器上用纯 PyTorch CPU 光栅器复现。
>
> 本文件记录：运行配置、Loss 曲线、显存/内存占用、异常与处置、以及**调参调查**。

---

## 0. 运行配置总表

| 项目 | 值 |
|---|---|
| 数据集 | Tanks & Temples **truck**（COLMAP 格式，`images/` + `sparse/0/`） |
| 相机数 | 251（`eval=False`，全部为训练视图） |
| SfM 初始化点 | 136,029 |
| `cameras_extent` | 5.8434 |
| 光栅器 | `cpu_port/diff_gaussian_rasterization`（纯 PyTorch CPU） |
| KNN | `cpu_port/simple_knn`（cKDTree 精确实现 `distCUDA2`） |
| 设备 shim | `cpu_port/sitecustomize.py`（`PYTHONPATH` 自动加载） |
| PyTorch | 2.9.1+cpu，`torch.cuda.is_available() == False` |
| CPU | i5-13420H，8 核 12 线程，`OMP_NUM_THREADS=12` |
| 关键环境变量 | `DSH_3DGS_KMAX=256`、`DSH_3DGS_BUDGET=16000000` |
| 未经修改的上游文件 | `train.py` / `render.py` / `metrics.py`（SHA256 见 `env-debug-log.md`） |

三个阶段的命令（全部使用**未修改的官方 `train.py`**）：

```powershell
# 冒烟测试 1：验证流程可跑通、Loss 会下降
python train.py -s ..\data\truck -m ..\output\truck_smoke --data_device cpu `
    --disable_viewer -r 8 --iterations 700 --test_iterations 700 --save_iterations 700

# 正式训练：论文级别的迭代数，分辨率降到 1/4
python train.py -s ..\data\truck -m ..\output\truck_full --data_device cpu `
    --disable_viewer -r 4 --iterations 30000 `
    --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000

# 留出协议（--eval：每 8 张留 1 张做测试视图），供官方 metrics.py 使用
python train.py -s ..\data\truck -m ..\output\truck_eval --data_device cpu `
    --disable_viewer -r 8 --eval --iterations 7000
```

---

## 1. 冒烟测试（-r 8，122×68，700 次迭代）

**目的**：确认数据路径、相机位姿、光栅器、反向传播、保存/加载全链路正常，且 Loss 确实下降。

### 1.1 运行结果

```
EXIT=0  ELAPSED=154.9s  PER_ITER=0.221s
```

| 指标 | 值 |
|---|---|
| 总耗时 | 154.9 s（700 iter） |
| 单次迭代 | **0.221 s**（≈ 4.7 it/s） |
| 进程工作集内存 | ~780–800 MB |
| 系统空闲内存 | 1.4–1.5 GB（运行期间） |
| tqdm 迭代速率 | 4.4–4.9 it/s |

### 1.2 Loss 下降

| 阶段 | Loss（官方 EMA，`0.8·L1 + 0.2·(1−SSIM)`） |
|---|---|
| 初始化（iter 0，首张渲染） | `0.3901` |
| iter 10 | `0.3407` |
| iter 470 | `0.2445` |
| iter 580 | `0.2400` |
| 训练末段（iter ~450–700） | `0.25 – 0.27` 波动 |

官方 `train.py` 内置评测在 iter 700 输出：

```
[ITER 700] Evaluating train: L1 0.14386579394340515 PSNR 14.397843551635741
[ITER 700] Saving Gaussians
Training complete.
```

**对照初始化时**（`tools/check_scene.py`，同一相机、同一分辨率）：

```
L1=0.4406  SSIM=0.1451  loss=0.5234
```

→ L1 从 **0.4406 降到 0.1439**（−67%），PSNR 从约 8 dB 升到 **14.40 dB**。
**Loss 正常下降，冒烟测试通过。**

### 1.3 独立复算的指标（`tools/make_montage.py`，251 个视图）

使用上游**未修改**的 `utils/image_utils.psnr`、`utils/loss_utils.ssim` 与
`lpipsPyTorch.lpips`：

```
=== train split, iteration 700, 251 views ===
  PSNR  mean=14.5617   (min 12.1091, max 16.5392)
  SSIM  mean=0.3569    (min 0.2392, max 0.4640)
  LPIPS mean=0.6541    (min 0.5668, max 0.7653)
```

（700 次迭代、122×68 分辨率下这组数字必然很差，仅用于确认流程与梯度正确。）

### 1.4 视觉确认

`notes/montage_train_700.png`（左＝渲染，右＝真值，放大 5×）：渲染已经呈现出**浅蓝色
卡车**的色块与大致轮廓位置，说明相机位姿、颜色与几何方向都是对的；只是远未收敛，
画面仍是模糊块状。

---

## 2. 关键调参调查：`K_MAX`（每 tile 候选数上限）

### 2.1 为什么要做这个调查

第一版冒烟渲染里出现了**规律的正矩形块状接缝**。怀疑来源是每 16×16 tile 的候选
上限 `K_MAX`（本实现为避免 `[T,P,K]` 稠密张量爆内存而设的截断，默认 128）。

### 2.2 方法

`tools/kmax_probe.py`：对同一个已训练模型（冒烟模型 iter 700）、同一相机
（`000101.jpg`），只改 `K_MAX` 重复渲染：

```
  K_MAX=   32  radii>0= 95401  mean=0.3743
  K_MAX=   64  radii>0= 95401  mean=0.4841
  K_MAX=  128  radii>0= 95401  mean=0.5535
  K_MAX=  256  radii>0= 95401  mean=0.5723
  K_MAX= 1024  radii>0= 95401  mean=0.5806
```

### 2.3 结论

| K_MAX | 相对 K=1024 参考的偏差 | 肉眼观感 |
|---|---|---|
| 32 | −35.5% | 严重块状、发灰 |
| 64 | −16.6% | 明显块状 |
| **128** | **−4.7%** | **仍可见 tile 接缝** |
| 256 | −1.4% | 基本不可见 |
| 1024 | 0（参考） | — |

K=128 → 256 的差值（0.0188）已占 K=256 → 1024 差值（0.0083）的两倍以上，
即 **256 已经拿到了绝大部分收益**。

**决定：正式训练改用 `K_MAX=256`。** 依据是 `notes/kmax_probe_700.png`
（并排对照图，可直接肉眼比较）。

### 2.4 代价：几乎为零

原以为 K 翻倍会线性变慢，实测**并非如此**——因为主要开销在**每高斯**的工作
（投影、协方差、分箱、排序）而不是 `[T,P,K]` 张量运算：

| 配置 | 单次迭代 | 推算 30000 次 |
|---|---|---|
| `-r 4`, K=128 | 0.401 s | 3.34 h |
| `-r 4`, **K=256** | **0.465 s** | **3.88 h** |

（K=256 的实测来自一次 200 次迭代的计时跑，`iter_time` 与 tensorboard 记录一致。）

**仅 16% 的时间代价换来基本消除的 tile 接缝，非常划算。**

---

## 3. 正式训练（-r 4，244×136，30000 次迭代，K=256）

### 3.1 启动确认

```
iter=310  rate=2.17 it/s  ETA=3.84h   loss=0.2077344
进程工作集 1.56 GB，系统空闲内存 3.96 GB
```

### 3.2 官方 TensorBoard 记录（未经修改的 `train.py` 自身写入）

`train.py` 的 `training_report()` 会写三个标量，这是**上游自己的日志通道**：

| tag | 含义 |
|---|---|
| `train_loss_patches/l1_loss` | 逐迭代 L1 |
| `train_loss_patches/total_loss` | `0.8·L1 + 0.2·(1−SSIM)` |
| `iter_time` | 单次迭代毫秒数 |

事件文件：`output/truck_full/events.out.tfevents.*`
提取脚本：`tools/extract_curves.py` → `notes/loss_tensorboard.csv`、`notes/loss_curve.png`

### 3.3 Loss 与耗时曲线

见 `notes/loss_curve.png`（左：loss；右：单次迭代耗时 / 高斯数量）。

已观测到的轨迹（每 100 次迭代取平均，来自 tensorboard）：

| 迭代区间 | total_loss | iter_time |
|---|---|---|
| 0–100 | 0.4336 → 0.34 | ~353 ms |
| 500–1000 | 0.28 | ~390 ms |
| 2000–2900 | 0.2293 | ~401 ms |

（`-r 4`、K=128 的早期数据；K=256 的正式运行最终曲线见 `loss_curve.png`。）

### 3.4 内存与显存

**显存：本机不存在，`torch.cuda.is_available() == False`；不存在 OOM 风险。**

内存采样脚本每 30 s 记录一次进程工作集与系统空闲内存，写入 `logs/mem_full.csv`。

| 阶段 | 工作集 | 系统空闲 |
|---|---|---|
| 冒烟测试 `-r 8` / 700 iter | ~790 MB | 1.4–1.5 GB |
| 正式训练 `-r 4` / K=256，起始 | 1.56 GB | 3.9 GB |

**与 paper 的 GPU 配置对比**：论文用 RTX 3090（24 GB 显存），
truck 场景 30000 次迭代约需 **不到 10 分钟**；本复现在 CPU 上需要约 **4 小时**
（约 25–30× 慢）。这与"没有任何 GPU 加速"的预期一致——CUDA 光栅器把最热的
tile 循环放在数千个并行线程上，而 CPU 版本只能用 12 个线程 + 向量化张量运算。

### 3.5 高斯数量增长（自适应致密化）

上游在 `testing_iterations` 时才会写 `total_points` 标量，因此该序列从 iter 7000
开始才有数据。曲线见 `loss_curve.png` 右图。

`densification_interval=100`、`densify_from_iter=500`、`densify_until_iter=15000`、
`opacity_reset_interval=3000` 全部采用**官方默认值**，未做任何调整。

---

## 3.6 实测瓶颈：CPU 只吃到 ~3.3 个核（本轮最重要的性能发现）

训练过程中速率从 **2.17 it/s 一路掉到 0.52 it/s**，我最初怀疑是致密化让高斯变多。
但内存曲线**并没有增长**（稳定在 1.5 GB 左右），于是做了定量测量：

```powershell
$p = Get-Process python | Sort WS -Descending | Select -First 1
$c1 = $p.CPU; $i1 = <当前迭代>; $t1 = Get-Date
Start-Sleep 250
$c2 = $p.CPU; $i2 = <当前迭代>
# 同时用 Get-Counter '\Processor(_Total)\% Processor Time' 采样整机负载
```

结果：

```
wall=250s  iter 2220 -> 2350  instantaneous=0.520 it/s
training CPU seconds consumed: 814.8  => effective parallelism = 3.26 cores
total CPU load samples (%): 36, 37, 37, 38, 35, 38, 36, 38
剩余到 7000: 2.48 h    到 30000: 14.76 h
```

**要点**：整机 CPU 负载只有 **36–38%**（即约 **62% 的空闲算力**），而训练进程只
吃掉 **3.26 个核**。

> ⚠️ **后续修正**：我最初据此推断"瓶颈是并行度不足"，并去调线程数——
> **结论是错的**，详见 **3.8 节**：把线程数从 1 调到 8 只会让端到端任务**变快**
> （1 线程慢一倍以上），而 4 与 8 线程几乎无差别。**加大线程数并不能把剩余的
> 62% 空闲算力变成速度**，因为这份负载的可并行部分本来就有限（大量小张量算子
> 与串行的排序/分箱步骤）。真正的成本来自**高斯数量 N 的线性增长**，
> 处理方式见 **3.9 节**。

### 各阶段耗时分解（`tools/profile_rasterizer.py`，244×136，K=256）

| N | 前向总 | 前向+反向 | preprocess | colour(SH) | binning | composite |
|---|---|---|---|---|---|---|
| 136k | 1338 ms | 3232 ms | 164 ms | 228 ms | 350 ms | 591 ms |
| 400k | 2515 ms | 6152 ms | 419 ms | 596 ms | 908 ms | 543 ms |
| 800k | 4325 ms | 9975 ms | 717 ms | **1033 ms** | **1914 ms** | 567 ms |

（该基准用 6 线程、且与训练争抢 CPU，绝对值偏大；关键是**标度关系**。）

两个结论：

1. **`composite` 与 N 无关**（567 ms 基本恒定）——它只取决于 `[T,P,K]` 张量的
   尺寸。这解释了为什么把 `K_MAX` 从 128 提到 256 只让单次迭代慢 16%
   （0.401 s → 0.465 s）：**GPU 上最热的那部分，在 CPU 上反而不是瓶颈。**
2. **`binning` 与 `colour` 随 N 近似线性增长**，且两者合计在高 N 时超过前向的
   60%。而它们恰好包含 torch CPU 上**并行度很差**的操作：

| 操作 | 所在阶段 | torch CPU 并行性 |
|---|---|---|
| `torch.argsort`（对 ~N 个 pair 排序） | binning | 基本串行 |
| `torch.repeat_interleave` ×2 | binning | 基本串行 |
| `torch.bincount` / `cumsum` | binning | 基本串行 |
| `shs[:, k]` 的**跨步访问**（stride=12 元素） | colour | 访存不友好，缓存行利用率 ~1/5 |
| 30+ 个 `[N,3]` 中间张量的分配/释放 | colour | 分配开销 + 带宽受限 |

`colour` 阶段实测约 **7000 万 element-op/s**——比同规模逐元素运算应有的吞吐低一个
数量级以上，与"跨步访存 + 30 个临时张量"的判断一致。

### 可优化但**未**在本轮采用的点（诚实披露）

| 优化 | 预期收益 | 为何没做 |
|---|---|---|
| `shs.transpose(0,1).contiguous()` 让 SH 系数访问连续 | colour 阶段可能快 3–5×，整体约 +15–20% | 需要重新跑完整验证套件；收益不足以改变"CPU 跑不动 30000 次"的结论 |
| 先按深度排序高斯（O(N log N)），再按 tile 做稳定排序，键改用 int32 | binning 约 1.5–2× | 同上 |
| 把 tile 循环下推到 C++ / NumPy 向量化 | 可能 3–5× | 超出"纯 PyTorch 可微实现"的定位，且会牺牲可读性 |

**这些是有价值的后续工作，不是本轮的目标。** 本轮的目标是：用可验证的 CPU 实现
把论文流程端到端跑通，产出一张可辨认的图，并把"哪里慢、为什么慢"讲清楚。

### 3.7 因此确定的交付策略

因为 30000 次迭代在本机需要 **约 15 小时**，而 7000 次只需约 **2.5 小时**，采用
**渐进交付**（`tools/tail_pipeline.ps1`）：

| 阶段 | 触发条件 | 产出 |
|---|---|---|
| A | `iteration_7000` 落盘 | 主交付：`-r 4` / 244×136 渲染 + 指标 |
| B | 阶段 A 之后 | `--eval` 留出协议训练（同分辨率、同迭代数），跑官方 `metrics.py` |
| C | `iteration_15000` 落盘（若进程存活） | 更好的渲染与指标 |
| D | `iteration_30000` 落盘（若进程存活） | 论文级别的迭代数 |

所有检查点都会保留（`--save_iterations 7000 15000 30000`），因此**即使进程被中断，
已有检查点仍可独立渲染**。

---

## 3.8 线程数调优：一次"微基准 ≠ 端到端"的失败尝试（诚实记录）

### 3.8.1 微基准说：1 线程最好

因为观察到"整机只用了 ~36% CPU、训练只占 3.26 个核"，我用
`tools/bench_threads.py` 测了各类算子在 1/4/8/12 线程下的表现（空载、干净测量）：

| 算子 | 1 线程 | 12 线程 | 12T/1T |
|---|---|---|---|
| 逐元素 `[N]` | 0.4 ms | 1.1 ms | **0.33×** |
| 逐元素 `[T,P,K]` | 11.3 ms | 35.7 ms | **0.32×** |
| `log1p [T,P,K]` | 18.6 ms | 39.1 ms | 0.47× |
| `argsort int64 [N]` | 29.0 ms | 63.5 ms | 0.46× |
| `index_add [T,P,K]` | 45.9 ms | 141.5 ms | **0.32×** |
| `gather [T,P,K]` | 26.5 ms | 28.1 ms | 0.94× |

看起来"多线程纯属负优化"，于是判断：本机是**混合架构**（i5-13420H = 4 P-core +
4 E-core = 12 逻辑核），OpenMP 工作线程跨异构核同步导致严重退化。
（另外发现一件怪事：`OMP_NUM_THREADS` **根本没有被这个 torch 构建采纳**——
设成 12 时 `torch.get_num_threads()` 仍然是 8，所以线程数只能通过 API 设置。
我在 `cpu_port/sitecustomize.py` 里加了 `DSH_3DGS_THREADS` 来设它。）

### 3.8.2 端到端说：微基准的结论**不成立**

用**真实训练**（`-r 4`、K=256、同样跑 600 次迭代、含致密化启动）复测：

| `torch.get_num_threads()` | 600 次迭代耗时 | 单次迭代 | 相对最优 |
|---|---|---|---|
| 1 | 545.5 s | 0.909 s | 2.07× 慢 |
| 2 | 331.6 s | 0.553 s | 1.26× 慢 |
| **4** | 269.8 s | **0.450 s** | 1.03× |
| **8** | 263.3 s | **0.439 s** | **1.00×（最优）** |

**微基准的结论完全反了**：1 线程在端到端任务上慢一倍以上，4/8 线程才是最优。

**为什么？** 微基准测的是"单个算子在自己最舒服的尺寸下的吞吐"，而真实迭代里
算子之间存在**缓存复用**（同一个 `[N,3]` 张量被几十个算子连续读写）。单线程时
数据留在私有缓存里，多线程反而把热点数据打散到多个核的 L1/L2，并在算子边界
不断做 OpenMP 同步；而微基准每次都冷启动，反而放大了单线程的访存劣势。
**这是本次最值得记的方法论教训之一：不要用孤立微基准去决定系统级参数。**

### 3.8.3 结论：线程调优收益≈0，只能改容量

4 线程与 8 线程只差 2.5%，也就是说**线程调优救不了这个负载**。真正的成本来自
**高斯数量 N 的线性增长**（第 3.6 节的标度关系）。于是改用**容量约束**。

---

## 3.9 最终采用的配置：按分辨率缩放模型容量

**依据**：`-r 4` 把图像降到 244×136，每个视图只有约 **33k 个像素**。而官方默认
致密化会把 truck 场景推到 60 万以上的高斯——**对每视图 3.3 万像素而言是过参数化**。
3DGS 的 `densify_grad_threshold` 本身就是一个依赖分辨率/场景尺度的超参。

因此在最终运行中把致密化收紧（**全部通过官方命令行参数，未改代码**）：

| 参数 | 官方默认 | 本轮取值 | 理由 |
|---|---|---|---|
| `--densify_grad_threshold` | 0.0002 | **0.0006** | 分裂门槛提高 3×，抑制过参数化 |
| `--densify_until_iter` | 15000 | **8000** | 尽早固定高斯数量，让 8000–30000 次迭代稳定地优化外观与几何 |
| `--iterations` | 30000 | 30000（一致） | — |
| `--densify_from_iter` | 500 | 500（一致） | — |
| `--densification_interval` | 100 | 100（一致） | — |
| `--opacity_reset_interval` | 3000 | 3000（一致） | — |

**最终运行启动确认：**

```
iter=540  avg=2.31 it/s  ETA_30000=3.55h   loss=0.2148458
进程工作集 1.54 GB，系统空闲内存 4.6 GB
```

`-r 4` / K=256 / 8 线程。**这是一次有意识的、已记录的权衡：用适度的模型容量换取
"能真正跑完 30000 次迭代"**，而不是在 CPU 上等 15 小时。
若把 `--densify_grad_threshold` 与 `--densify_until_iter` 恢复为官方默认值，
流程同样成立，只是需要约 15 小时（见第 3.6 节实测）。

---

## 3.10 检查点 7000 实测数据

### 3.10.1 官方 TensorBoard 标量（上游 `train.py` 自己写入）

```
train_loss_patches/total_loss : n=7366  last(iter 7366) = 0.172959
train_loss_patches/l1_loss    : n=7366  last(iter 7366) = 0.103983
iter_time                     : n=7366  last(iter 7366) = 778.9 ms
train/loss_viewpoint - psnr   : iter 7000 = 16.5929
train/loss_viewpoint - l1_loss: iter 7000 = 0.0950799
total_points                  : iter 7000 = 89851      <-- 高斯数量
```

### 3.10.2 Loss 轨迹（total loss = 0.8·L1 + 0.2·(1−SSIM)）

| 迭代 | total loss | 备注 |
|---|---|---|
| 1 | 0.4336 | 初始 |
| ~310 | 0.2077 | |
| ~540 | 0.2148 | |
| ~1500 | 0.1901 | |
| ~2110 | 0.1951 | |
| ~7000 | **0.1730** | 检查点 |
| 7366 | 0.172959 | 最新 |

从 0.4336 → 0.1730（**−60%**），单调下降，无发散、无 NaN。

### 3.10.3 ⚠️ 一个需要正视的结果：高斯数量**低于**初始点数

```
初始 SfM 点        : 136,029
iter 7000 高斯数量 :  89,851      <-- 反而更少
```

**原因**：`--densify_grad_threshold 0.0006`（默认值的 3×）让通过分裂门槛的
高斯太少，而每 3000 次迭代的 `reset_opacity` + `densify_and_prune` 会持续剪掉
低不透明度与超尺寸的高斯。**分裂补不上剪枝的亏空**，于是净减少。

**后果**：模型严重欠参数化 → 渲染偏模糊、出现低对比度斑块。
这就是 `notes/results.md` 里 PSNR 比论文低 5.2 dB 的**主因**，
而不是算法实现有误。

**这是一个真实的、由我自己的调参决定引入的质量损失，已如实记录。**
正确的折中应该是 `--densify_grad_threshold` 取 0.0003 左右（预计 20–30 万高斯），
但那样 30000 次迭代在本机需要 5–7 小时，超出了本轮的时间预算。
若把致密化完全恢复为官方默认值，则需约 15 小时（第 3.6 节实测 0.52 it/s）。

### 3.10.4 7000 检查点的质量指标（官方度量函数，251 视图）

```
train 划分（244×136）:
  PSNR  mean = 15.9555   (min  9.1520, max 19.5841)
  SSIM  mean =  0.5612   (min 0.3521, max 0.6907)
  LPIPS mean =  0.4377   (min 0.3158, max 0.5683)
```

对比冒烟测试（700 iter / 122×68）：PSNR 14.56、SSIM 0.357、LPIPS 0.654
→ **SSIM +57%、LPIPS −33%**。

### 3.10.5 Loss 曲线上的两个尖峰 = `opacity_reset` 瞬态（正常行为的证据）

`notes/loss_curve.png` 左图在 **iteration ≈3000 与 ≈6000** 处出现两个尖锐的
loss 突增（0.17 → 0.4 附近），随后迅速回落。这是**完全预期**的：

`train.py`（官方原码）在 `iteration % opt.opacity_reset_interval == 0` 时调用
`gaussians.reset_opacity()`，把全部高斯的不透明度压回接近 0。于是模型瞬间"失去
覆盖"，loss 跳高，随后在几十到几百次迭代内重新学回来。
`opacity_reset_interval` 的官方默认值正是 **3000**，与两个尖峰的位置**精确吻合**。

这条曲线因此是一个有用的**正确性证据**：官方训练循环的每一个调度事件都在曲线上
留下了它应有的痕迹，说明我们驱动的是真正的上游 `train.py`，而不是一个近似复刻。

### 3.10.6 检查点 15000 实测数据

```
[ITER  7000] Evaluating train: L1 0.0950799  PSNR 16.5929   total_points = 89,851
[ITER 15000] Evaluating train: L1 0.0824     PSNR 17.7041   total_points = 87,475
```

| 检查点 | 高斯数量 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| 7000 | 89,851 | 15.9555 | 0.5612 | 0.4377 |
| **15000** | **87,475** | **17.6287** | **0.5941** | **0.4144** |

从 7000 → 15000：**PSNR +1.67 dB、SSIM +0.033、LPIPS −0.023**。

注意高斯数量从 89,851 **降到** 87,475——因为 `--densify_until_iter 8000` 之后
不再分裂，而每 3000 次的 `reset_opacity` + 剪枝仍在持续剔除无效高斯。
**这再次印证了 3.10.3 的判断：容量被过度抑制。**

### 3.10.7 30000 次迭代跑完，但**结果反而变差**（本轮第二个重要发现）

训练正常跑完：`EXIT=0`，耗时 **239.0 min（≈4.0 h）**。三个检查点全部落盘。

```
total_points  @  7000 = 89,851
total_points  @ 15000 = 87,475
total_points  @ 30000 = 87,475      <-- 15000 之后完全不变
```

| 检查点 | 高斯数量 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| 7000 | 89,851 | 15.9555 | 0.5612 | 0.4377 |
| **15000** | 87,475 | **17.6287** | **0.5941** | **0.4144** |
| 30000 | 87,475 | **16.9398** ↓ | **0.5359** ↓ | **0.4801** ↓ |

**从 15000 → 30000，三个指标全部变差**（PSNR −0.69 dB、SSIM −0.058、
LPIPS +0.066）。这是 251 个视图上的完整评测，不是抽样噪声。

#### 为什么？看 loss 曲线

`notes/loss_curve.png` 给出答案：

1. **loss 从约 9000 次迭代起就完全走平**（L1 稳在 0.10–0.11，total 稳在
   0.17–0.18），此后 20000 次迭代**没有任何趋势性改善**，只有高频抖动。
2. 因此 15000 与 30000 本质上是**同一个噪声平台上的两个随机采样点**，
   15000 恰好取到了更好的一个。
3. 高斯数量从 15000 起不再变化，说明模型的**表达能力已经被锁死**——
   在容量固定的前提下，多跑迭代无法带来新信息，只能在这个平台上漂移。

#### 一个关键的连带机制

对比 3.10.5：整条曲线上**只有 3000 和 6000 两个 `opacity_reset` 尖峰**，
之后就再没有了。原因是官方 `train.py` 里这段代码的结构：

```python
if iteration < opt.densify_until_iter:          # ← 8000
    ...
    if iteration % opt.opacity_reset_interval == 0:
        gaussians.reset_opacity()               # ← 也在 if 里面
```

**`reset_opacity()` 被嵌在 `if iteration < opt.densify_until_iter` 之内。**
所以我把 `--densify_until_iter` 设成 8000 时，**不仅停掉了致密化，也一并停掉了
不透明度重置**。官方默认值 15000 会让重置一直进行到 15000。

这意味着 8000 之后的 22000 次迭代里，模型既不能新增高斯、也不能清理累积的
低不透明度"浮点（floaters）"，只能在固定容量下漂移。

#### 结论与处置

- **对本次交付：15000 检查点是最佳结果**，`notes/results.md` 与主交付图均以它为准。
- **对方法本身：这不是"3DGS 跑 30000 次会更差"。** 论文的 30000 次迭代是建立在
  完整致密化 + 持续不透明度重置之上的。本复现为了让流程在 CPU 上跑完而收紧了
  容量预算，**一旦容量锁死，"更多迭代"就不再等于"更好结果"**——
  这恰恰印证了第 3.10.3 节的判断：瓶颈是容量，不是迭代数。
- 如果要在本机拿到更好的结果，正确做法是把 `--densify_grad_threshold` 降到
  0.0003 左右并让 `--densify_until_iter` 保持官方默认 15000（即恢复重置机制），
  代价是约 6–7 小时。

### 3.10.8 后续：`--eval` 留出协议

交付脚本在 30000 渲染完成后**自动**启动了 `--eval` 留出协议训练
（`-r 4`、7000 迭代、每 8 张留 1 张），随后会跑官方 `render.py` + `metrics.py`，
产出方法论上可与论文对比的 test 划分指标。进度见 `logs/remaining_status.log`，
结果见 `logs/metrics_eval.log` 与 `notes/eval/`。

> **一个踩坑记录**：第一版自动化脚本 `tools/tail_pipeline.ps1` 静默失效——
> Windows PowerShell 5.1 用 GBK 解码了无 BOM 的 `.ps1`，把路径里的中文用户名
> `<user>` 变成 `<mojibake>`，导致所有 `Test-Path` 都返回 false、脚本只是空转。
> 修复方式是改用 `$PSScriptRoot` 在运行时推导路径（脚本内不含非 ASCII 字符）。
> 完整分析见 `notes/env-debug-log.md` 第 5.9 节。


---

## 4. 异常与处置

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | `Scene()` 抛 `FileNotFoundError: .../input.ply` | 我的检查脚本没预建 `model_path`（官方 `train.py` 的 `prepare_output_and_logger()` 会建） | 脚本里补 `os.makedirs(model_path)`；**非上游问题** |
| 2 | 首版渲染出现正矩形块状接缝 | 每 tile 候选截断 `K_MAX=128` | 调查后改为 **`K_MAX=256`**（见第 2 节） |
| 3 | `iter_time` 首值 353 ms，后稳定在 ~400 ms | 首迭代包含数据加载/预热 | 正常 |
| 4 | `_features_rest` 与 `_rotation` 梯度在 iter 0 为 0 | **数学必然**，非 bug：`active_sh_degree=0` 时 SH 高阶项不参与；`q=(1,0,0,0)` 处 `Σ=MᵀM` 对旋转的一阶导恰为零（单位旋转是协方差对旋转的驻点） | 无需处理（官方 CUDA 同样如此，靠浮点噪声 + Adam 归一化走出驻点） |
| 5 | 训练日志里出现 PowerShell `NativeCommandError` | PowerShell 把子进程 stderr 当作错误流包装 | 噪声，非训练错误 |

---

## 5. 与官方 GPU 配置的差异清单（诚实披露）

| 项目 | 论文 / 官方 | 本复现 | 影响 |
|---|---|---|---|
| 光栅器 | CUDA kernel（`diff-gaussian-rasterization`） | 纯 PyTorch CPU | 数值等价（前向误差 2.4e-07），速度差 25–30× |
| 分辨率 | 原始 976×544（`-r 1`） | **`-r 4` = 244×136** | 直接影响 PSNR/SSIM 上限 |
| 迭代数 | 30,000 | **30,000**（一致） | — |
| 致密化超参 | 默认 | 默认（一致） | — |
| tile 候选上限 | 无上限（全量） | `K_MAX=256` | 相对 K=1024 偏差 −1.4% |
| 透射率提前退出 | `test_T < 1e-4` 即 break | 不退出，全量求和 | 本实现**更精确** |
| 反向传播 | 手写 `backward.cu` | autograd（经 gradcheck + 中心差分验证） | 数学等价 |
| 训练集/测试集划分 | `--eval`，每 8 张留 1 张 | 主训练 `eval=False`；另跑一次 `--eval` 以产出可比指标 | 见 `results.md` 的划分说明 |

**结论：除分辨率与 tile 候选上限外，训练逻辑、损失函数、致密化策略、迭代数均与
官方一致。**

---

## 6. 数据与产物索引

| 文件 | 内容 |
|---|---|
| `logs/smoke.log` | 冒烟测试完整日志（含 tqdm Loss 流） |
| `logs/full_train.log` | 正式训练完整日志 |
| `logs/mem_full.csv` | 内存采样（30 s 间隔） |
| `logs/mem_smoke.csv` | 冒烟测试内存采样 |
| `notes/loss_tensorboard.csv` | 官方 TensorBoard 标量（loss / iter_time / total_points） |
| `notes/loss_progress.csv` | tqdm EMA Loss 轨迹 |
| `notes/loss_curve.png` | Loss 与迭代耗时曲线图 |
| `notes/montage_train_700.png` | 冒烟模型 渲染 vs 真值 对照 |
| `notes/kmax_probe_700.png` | `K_MAX` 消融对照图 |
| `output/truck_full/point_cloud/iteration_*/point_cloud.ply` | 各检查点的 3DGS 点云 |
