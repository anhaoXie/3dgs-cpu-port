# cpu_port — 纯 PyTorch CPU 的 3D Gaussian Splatting 可微光栅器

上游 `graphdeco-inria/gaussian-splatting` 的核心是一个**纯 CUDA** 的可微光栅器。
本目录提供它的 CPU 等价实现，使得在**没有 NVIDIA GPU** 的机器上也能复现论文流程。

> **本目录之外的上游仓库一个字节都没有改。**
> `git -C gaussian-splatting status --porcelain` 输出为空即可验证
> （同时 `train.py` / `render.py` / `metrics.py` 的 SHA256 与 HEAD 一致）。

---

## 目录结构

```
cpu_port/
├── sitecustomize.py                      # CUDA→CPU 设备重映射 shim
├── diff_gaussian_rasterization/
│   └── __init__.py                       # 光栅器（前向 + autograd 反向）
├── simple_knn/
│   ├── __init__.py
│   └── _C.py                             # distCUDA2 的 CPU 实现
└── tests/
    ├── test_cpu_rasterizer.py            # 前向/反向/梯度验证套件
    └── diag_gradcheck.py                 # 梯度问题诊断工具
```

## 为什么需要 `sitecustomize.py`

上游代码里有若干**与算法无关**的硬编码 CUDA 调用：`device="cuda"`、
`.cuda()`、`.to("cuda")`、`torch.cuda.Event`、`torch.cuda.empty_cache()`。
它们只是"张量放哪"的问题，与 3DGS 的数学毫无关系。

把 `cpu_port` 放进 `PYTHONPATH` 后，CPython 启动时会自动导入 `sitecustomize`，
在 `torch` 之上把这些调用透明地重定向到 CPU；于是官方脚本可以**逐字节不改**地运行。

shim 覆盖的内容：

| 类别 | 处理 |
|---|---|
| `torch.zeros/ones/rand/tensor/... (device="cuda")` | 工厂函数包装，参数改写为 `"cpu"` |
| `Tensor.cuda()` | 返回 `self` |
| `Tensor.to("cuda")` / `.to(torch.device("cuda:0"))` | 目标设备改写为 `"cpu"`（`.to(dtype)` / `.to(tensor)` 语义保持不变） |
| `nn.Module.cuda()` | 返回 `self` |
| `torch.cuda.Event` | 用 `time.perf_counter()` 实现的 CPU 计时器 |
| `torch.cuda.empty_cache / set_device / synchronize / device / ...` | 空操作 |
| `torch.cuda.is_available()` | 如实返回 `False` |

**注意**：shim 只管"设备"，**不碰算法**。真正的算法实现（光栅器与 KNN）是本目录
里独立的、可验证的代码。

---

## 光栅器实现要点

逐行对照 `cuda_rasterizer/forward.cu`、`auxiliary.h`、`rasterizer_impl.cu` 复刻：

| 论文 / CUDA | 本实现 |
|---|---|
| `computeCov3D`：`Σ = MᵀM`，`M = S R` | `_compute_cov3d()`，批量化 3×3 |
| `computeCov2D`：EWA 仿射近似 `Σ' = J Wᵀ Σ W Jᵀ` + `+0.3` 低通 | 同上，`torch.bmm` |
| `ndc2Pix(v,S) = ((v+1)S − 1)/2` | 同式 |
| 视锥剔除 `p_view.z ≤ 0.2` | 同式（用 `torch.where` 保证可导） |
| 3σ 半径 `ceil(3√λmax)` | 同式 |
| 16×16 tile、`getRect` 包围盒 | 同式 |
| `key = (tile_id << 32) \| float_bits(depth)` radix sort | 用 `tile_id*N + depth_rank` 构造**等价的全序键**（见下） |
| 逐像素前向 alpha 合成 | `[T,P,K]` 稠密张量 + `cumsum(log1p(-α))` 的对数域透射率 |

### 相机矩阵约定（最容易错的地方）

CUDA 按列主序读取 `float*`，PyTorch 是行主序张量，因此正确形式是**右乘**：

```python
p_hom = torch.cat([means3D, ones], 1) @ projmatrix
```

### `means2D` 的直通梯度（densification 的生命线）

上游 `gaussian_model.add_densification_stats()` 直接读
`viewspace_point_tensor.grad`，而 `viewspace_point_tensor` 是一个**全零哑张量**。
CUDA 内核在反向时把 `dL/d(2D mean)` 写进它的 `.grad`。纯 autograd 实现必须显式
重建这条通路：

```python
means2d_used = point_image + (means2D[:, :2] - means2D[:, :2].detach())
```

前向值仍是解析投影，反向把梯度记到 `means2D`，**语义与 CUDA 的 `dL_dmeans2D`
完全一致**，且不会与 `dL/dmeans3D` 重复计数。

### 刻意的、已记录的三处偏差

1. **不实现 `test_T < 1e-4` 提前退出**——本实现把候选全部算完，比 CUDA **更精确**，
   差异远低于 8 bit 量化。
2. **反向由 autograd 给出**，而非手写 `backward.cu`；数学上等价，且经
   `torch.autograd.gradcheck` 与中心差分双重验证。
3. **每个 tile 的候选数上限 `K_MAX`**（默认 128，可用 `DSH_3DGS_KMAX` 覆盖），
   保留最近的若干个。透射率衰减到 1e-4 后，被丢弃的尾部不可见。
   另有 `DSH_3DGS_BUDGET`（默认 6e6）用于限制 `[T,256,K]` 稠密张量的元素总量，
   以便在只有约 2 GB 空闲内存的机器上可预测地运行。

### 与 CUDA 的已知**非**差异

上游 CUDA 内核的 tile 归属、3σ 半径（`ceil`）、逐像素剔除、alpha 上下限都是
**阶跃/饱和**的不可导操作，其 `backward.cu` 同样把它们当作前向算好的常数。
本实现行为一致，因此 `scales` 梯度在少数跨阶跃的坐标上会出现**中心差分不收敛**
的现象——这是算法固有性质，不是实现错误（详见
`notes/lessons-learned.md` 第四节，以及 `tests/test_cpu_rasterizer.py` 的 C2 段，
那里有可证伪的判据与对照组）。

---

## 验证结果

```
A. SH 与上游 utils/sh_utils.eval_sh 对比 ....... max|err| = 4.8e-07
B. 与独立 naive 标量参考实现对比 ............... max|err| = 2.4e-07
C. autograd vs 中心差分（5 组参数全部） ........ max|err| ≈ 1e-08，0 个坏点
C3. 严格 torch.autograd.gradcheck ............. PASS
D. dL/d(means2D) 直通路由 ..................... 非空、前两列非零、第三列为零
```

运行方式：

```powershell
$env:PYTHONPATH = "<repo>\cpu_port"
python cpu_port\tests\test_cpu_rasterizer.py
```

---

## 复现命令

```powershell
cd gaussian-splatting
$env:PYTHONPATH   = "..\cpu_port"
$env:TORCH_HOME   = "..\torch_home"      # LPIPS / metrics.py 需要
$env:OMP_NUM_THREADS = "12"

..\venv\Scripts\python.exe train.py  -s ..\data\truck -m ..\output\truck `
    --data_device cpu --disable_viewer -r 4 --iterations 30000

..\venv\Scripts\python.exe render.py -m ..\output\truck --iteration 30000
..\venv\Scripts\python.exe metrics.py -m ..\output\truck      # 需要 --eval 划分
```
