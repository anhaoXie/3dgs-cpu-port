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

| 指标（244×136，iteration 15000，train 划分 251 视图） | 值 |
|---|---|
| PSNR | **17.63** |
| SSIM | **0.594** |
| LPIPS | **0.414** |
| 高斯数量 | 87,475 |
| 训练耗时 | ≈3.4 h（CPU，8 线程） |

论文 Table 1 的 Tanks&Temples 平均值为 PSNR 23.14 / SSIM 0.841 / LPIPS 0.183
（30K 迭代，RTX 3090，约 27 分钟）。差距约 5.5 dB，**根因是"无 GPU"导致的
模型容量与迭代预算压缩，而非算法简化**——完整分析见 `notes/results.md`。

---

## 目录

```
3dgs/
├── notes/                    ← 交付文档
│   ├── env-debug-log.md        编译失败记录与修复方案（含网络排障全过程）
│   ├── training-log.md         Loss 曲线、内存占用、异常、三项调参调查
│   ├── results.md              渲染图、指标、与论文对比
│   └── lessons-learned.md      最难的一步是什么？为什么？
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
│   ├── test_cpu_rasterizer.py 验证套件（见 cpu_port/tests/）
│   ├── profile_rasterizer.py   分阶段性能剖析
│   ├── bench_threads.py        线程扩展性微基准
│   ├── kmax_probe.py           K_MAX 消融对照
│   ├── make_montage.py         渲染/真值拼图 + 官方指标
│   ├── extract_curves.py       TensorBoard 曲线提取与绘图
│   ├── setup_lpips_weights.py  LPIPS 权重准备（绕过被墙的下载源）
│   └── deliver_remaining.ps1   30000 检查点自动渲染 + --eval 留出指标
│
├── gaussian-splatting/       ← 官方仓库（未修改）
├── data/truck/               ← Tanks & Temples truck，COLMAP 格式（187 MB）
├── output/truck_full/        ← 训练产物（检查点 7000 / 15000 / 30000）
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
| `DSH_3DGS_KMAX` | 128 | 每 16×16 tile 保留的最近高斯数上限；`kmax_probe` 实测 256 基本消除接缝 |
| `DSH_3DGS_BUDGET` | 6e6 | `[T,256,K]` 稠密张量的元素总量上限，用于约束内存 |

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
