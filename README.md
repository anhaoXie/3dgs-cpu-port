# 3D Gaussian Splatting —— 无 GPU 环境的 CPU 复现

复现 Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
(SIGGRAPH 2023)，官方代码 [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting)。

**本机没有 NVIDIA GPU**（只有 Intel UHD 核显，无 nvcc，`torch.cuda.is_available() == False`），
因此官方 CUDA 光栅器**物理上无法编译也无法运行**。本复现的做法是：

> **官方 `train.py` / `render.py` / `metrics.py` 一个字节都不改**，
> 只把两个算法性的 CUDA 扩展替换为纯 PyTorch CPU 实现，
> 并用一个启动期 shim 把与算法无关的 `device="cuda"` 调用重定向到 CPU。

验证：`git -C gaussian-splatting status --porcelain` 输出为空。

---

## 结果

**渲染图（左：本复现，右：真值）—— 浅蓝色木质平板卡车清晰可辨：**

![渲染 vs 真值](notes/montage_train_15000.png)

主模型（**train 划分**，iteration 15000，245×136）：

| 指标（245×136，iteration 15000，train 划分 251 视图） | K=256（交付口径） | K=1024（更忠实口径） |
|---|---|---|
| PSNR | **17.63** | **17.85** |
| SSIM | **0.594** | **0.637** |
| LPIPS | **0.414** | **0.351** |
| 高斯数量 | 87,475 | 同一检查点 |
| 训练耗时 | ≈3.4 h（CPU，8 线程） | 仅重渲染：2.1 min |

> 两列之差的原因见下面的《一处更正》。

**但 train 划分是乐观偏置的上界，不能拿去和论文比。** 与论文同口径的是独立留出
协议（`--eval`，每 8 张留 1 张；**官方默认致密化参数**；7000 次迭代；245×136）。
下表同时给出两个渲染口径——`K_MAX=1024` 是消除 tile 截断后的更忠实口径
（原因见下面的"一处更正"）：

| 划分 | 视图 | K_MAX | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| train | 219 | 256 | 16.0572 | 0.5645 | 0.4368 |
| **test（留出）** | **32** | 256 | **15.2789** | **0.5235** | **0.4639** |
| *train − test 间隙* | | *256* | *−0.778 dB* | *−0.041* | *+0.027* |
| train | 219 | 1024 | 16.0359 | 0.5951 | 0.3807 |
| **test（留出）** | **32** | **1024** | **15.3970** | **0.5674** | **0.3991** |
| *train − test 间隙* | | *1024* | *−0.639 dB* | *−0.028* | *+0.018* |

论文 Table 1 的 Tanks&Temples 平均值为 PSNR 23.14 / SSIM 0.841 / LPIPS 0.183
（30K 迭代，RTX 3090，约 27 分钟）。与留出集相差约 **7.7 dB**。

**这 7.7 dB 归给容量，不归给泛化。** 同一次运行内 train 与 test 只差
0.64–0.78 dB：模型在**没见过的**相机上和见过的相机上几乎一样吃力——这是
**欠拟合**的特征，而不是过拟合。根因是"无 GPU"导致的模型容量与迭代预算压缩，
而非算法简化——完整分析见 [`notes/results.md`](notes/results.md) 第 2.2 节。

### 一处更正：16×16 矩形接缝是 `K_MAX=256` 造成的

交付渲染曾出现正矩形块状接缝，项目文档一度记为"把 `K_MAX` 从 128 提到 **256
就已消除**"。**这个记录是错的。** 一次专门的证伪实验
（[`notes/kmax-ablation.md`](notes/kmax-ablation.md)）在交付检查点上实测：

| 同一检查点、同一划分（train，251 视图） | 接缝比率（边界梯度 / 内部梯度，横向） | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| `K_MAX=256`（交付口径） | **3.10** | 17.6287 | 0.5941 | 0.4144 |
| `K_MAX=1024` | 1.19 | **17.8468** | **0.6366** | **0.3505** |
| `K_MAX=2048`（收敛） | **1.04** | 17.7620 | 0.6351 | 0.3512 |

逐像素比对显示 **251/251 张渲染全部改变**（平均每像素差 4.2% 满量程）。
证据图：[`notes/probe/seam_compare_00051.png`](notes/probe/seam_compare_00051.png)
——左 K=256 / 右 K=1024，红线是 16px tile 网格；左图的块状跳变全部落在网格线上。

**为什么丢这么多**：直接实测了光栅器看到的候选数（`tools/raster_cap_probe.py`；
这个量**不依赖** `K_MAX`，所以是原始事实）——最忙的 16×16 tile 有
**约 11,000 个候选**，平均每 tile 也有 **1,478** 个。因此 `K_MAX=256`
只保留了最忙 tile 的 **2.3%**，并且**大多数 tile 都在截断**。
`K_MAX` 不是一个小旋钮，而是 CPU 近似 CUDA 时的**内存换质量折衷**：
完全不截断需要约 10–15 GB 内存，本机放不下。

旧结论之所以错，是因为它来自 **iteration 700** 的探针，而**那个探针测的是全图均值**。
事后直接实测发现，iteration 700 时这个上限其实**咬得更狠**
（`counts_max` 高达 20,618–38,481），接缝也更重（比率 **5.01**）。
全图均值对 tile 接缝这种**空间上高度集中**的伪影不敏感，
于是"均值只差 1.4%"被误读成了"接缝已消除"。

教训：**被测量必须直接对应目标现象，并且必须在最终检查点上复验。**

---

## 目录

```
3dgs/
├── notes/                    ← 交付文档
│   ├── env-debug-log.md        编译失败记录与修复方案（含网络排障全过程）
│   ├── training-log.md         Loss 曲线、内存占用、异常、三项调参调查
│   ├── results.md              渲染图、指标、与论文对比
│   ├── kmax-ablation.md        K_MAX 证伪实验：tile 接缝的真正成因与结论更正
│   ├── lessons-learned.md      最难的一步是什么？为什么？
│   ├── eval/                   各口径配对指标原始 JSON（train/test × K=256/1024）
│   ├── probe/                  接缝证据图与消融对照图
│   └── loss_curve.png / montage_*.png / kmax_probe_700.png
│
├── cpu_port/                 ← 本复现的核心：CPU 等价实现
│   ├── README.md               设计说明、与 CUDA 的逐项对照、已验证的偏差
│   ├── sitecustomize.py        CUDA→CPU 设备 shim（PYTHONPATH 自动加载）
│   ├── diff_gaussian_rasterization/   纯 PyTorch 可微光栅器
│   ├── simple_knn/_C.py        distCUDA2 的 cKDTree 精确实现
│   └── tests/                  前向/反向/梯度验证套件
│
├── tools/                    ← 可复用的工具脚本
│   ├── fetch_truck.py          从 hf-mirror 拉取 truck 数据集（带字节校验）
│   ├── check_scene.py          用官方 scene 加载器做端到端集成体检
│   ├── check_split.py          用官方读取器校验 train/test 划分并自对账
│   ├── eval_split_metrics.py   分划分配对指标（不写图，故不覆盖拼图）
│   ├── seam_metric.py          接缝定量：tile 边界梯度 / 内部梯度
│   ├── raster_cap_probe.py     实测 counts_max（上限实际丢了多少候选）
│   ├── seam_figure.py          接缝证据图（自动定位 + 16px 网格叠加）
│   ├── diff_renders.py         两目录渲染的逐像素差异统计
│   ├── test_cpu_rasterizer.py 验证套件（见 cpu_port/tests/）
│   ├── profile_rasterizer.py   分阶段性能剖析
│   ├── bench_threads.py        线程扩展性微基准
│   ├── kmax_probe.py           K_MAX 消融对照（iteration 700）
│   ├── kmax_ablation.ps1        K_MAX 消融驱动（非破坏性，输出到兄弟目录）
│   ├── make_montage.py         渲染/真值拼图 + 官方指标
│   ├── extract_curves.py       TensorBoard 曲线提取与绘图
│   ├── setup_lpips_weights.py  LPIPS 权重准备（绕过被墙的下载源）
│   ├── run_eval_holdout.ps1    --eval 留出集训练 + 渲染 + 官方指标
│   └── deliver_remaining.ps1   30000 检查点自动渲染 + --eval 留出指标
│
├── gaussian-splatting/       ← 官方仓库（未修改）
├── data/truck/               ← Tanks & Temples truck，COLMAP 格式（187 MB）
├── output/truck_full/        ← 训练产物（检查点 7000 / 15000 / 30000）
├── output/truck_eval/        ← --eval 留出集训练产物（检查点 7000）
├── output/*_k1024/           ← K_MAX=1024 重渲染（消融，非破坏性）
├── logs/                     ← 完整运行日志与内存采样
├── venv/                     ← Python 3.11 + torch 2.9.1+cpu
└── wheels/                   ← 本地 torch wheel 缓存
```

---

## 复现步骤

```powershell
$w = "<work>"

# 0) 一次性环境准备（详见 notes/env-debug-log.md）
#    - SJTU 镜像下载 torch 2.9.1+cpu / torchvision 0.24.1+cpu
#    - 清华镜像装 numpy/scipy/opencv/plyfile/tqdm/matplotlib/tensorboard
& $w\venv\Scripts\python.exe $w\tools\fetch_truck.py --dest $w\data\truck

# 1) 验证 CPU 光栅器（应输出 ALL CHECKS PASSED）
$env:PYTHONPATH = "$w\cpu_port"
cd $w\gaussian-splatting
& $w\venv\Scripts\python.exe $w\cpu_port\tests\test_cpu_rasterizer.py

# 2) 用官方 train.py 训练（三个文件均未修改）
$env:TORCH_HOME     = "$w\torch_home"
$env:DSH_3DGS_THREADS = "8"
$env:DSH_3DGS_KMAX    = "256"
$env:DSH_3DGS_BUDGET  = "16000000"
& $w\venv\Scripts\python.exe train.py -s $w\data\truck -m $w\output\truck_full `
    --data_device cpu --disable_viewer -r 4 --iterations 30000 `
    --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 `
    --densify_grad_threshold 0.0006 --densify_until_iter 8000

# 3) 官方 render.py 渲染
& $w\venv\Scripts\python.exe render.py -m $w\output\truck_full --iteration 15000 --skip_test

# 4) 官方 metrics.py 评测（需要 --eval 划分，见 notes/results.md 第 2.2 节）
& $w\venv\Scripts\python.exe metrics.py -m $w\output\truck_eval
```

**三个环境变量的作用**（都由 `cpu_port/sitecustomize.py` 读取，不改官方代码）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `DSH_3DGS_THREADS` | torch 默认(8) | 设置 torch 线程数（本机 4/8 最优；`OMP_NUM_THREADS` 对该 torch 构建无效） |
| `DSH_3DGS_KMAX` | 128 | 每 16×16 tile 保留的最近高斯数上限。**有效值是它与 `counts_max`、`BUDGET` 三者取小**。交付脚本用 256，但实测 256 **并不能**消除接缝，只有 1024 才可以——见《一处更正》 |
| `DSH_3DGS_BUDGET` | 6e6 | `[T,256,K]` 稠密张量的元素总量上限，用于约束内存。**注意它会反过来钳住 K**：245×136 有 144 个 tile，默认 6e6 只允许 `BUDGET//(144×256) = 162`；要真正用上 K=1024，必须**同时**把 BUDGET 抬到 ≥37748736 |

---

## 许可与上游代码

本仓库**不包含**上游 `gaussian-splatting` 的源码——它由 INRIA 以
**非商业研究用途**许可发布，因此更合适的做法是引用而非再分发。
请用 `tools/setup_upstream.ps1` 克隆并固定到本次验证的 commit：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_upstream.ps1
```

该脚本除了 clone + checkout `54c035f`，还会**校验**三个关键文件的 SHA-256 前缀
（`train.py` `078CB36BDE69090F`、`render.py` `E756466AB0A19AFC`、
`metrics.py` `A1334B4F421DC8E5`）——所以"官方代码未被修改"是**被验证的事实**，
而不是一句声明。

`data/`、`output/`、`venv/`、`wheels/`、`torch_home/` 同样不入库（合计约 1.9 GB），
原因见 `.gitignore` 中的注释；它们都可以由 `tools/fetch_truck.py` 与上面的训练命令
重新生成。
