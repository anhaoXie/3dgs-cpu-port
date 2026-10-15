# 3DGS 环境编译调试日志（env-debug-log.md）

> 目标：复现 Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field
> Rendering*, SIGGRAPH 2023（官方代码 `graphdeco-inria/gaussian-splatting`）。
>
> 本文档记录**每一步尝试的命令、真实错误输出、根因分析与修复方案**，包括
> 最终证明无法修复的那一类问题。

工作目录：`<work>`

---

## 0. 结论速览

| # | 问题 | 根因 | 结果 |
|---|------|------|------|
| 1 | `dsh plugin add dsh-science-workbench` 被拒 | 插件 peerDependency 要求 `@deepseek-ai/dsh-tools ^0.1.0-rc.6`，本机 DSH 为 `0.2.0-rc.2` | **已修复**（`allow-version` 显式豁免） |
| 2 | `nvidia-smi` 不存在、无 nvcc | 本机**没有 NVIDIA GPU**，只有 Intel UHD 核显 | **不可修复（硬件缺失）** |
| 3 | `pip install ./submodules/diff-gaussian-rasterization` 物理上不可行 | 该扩展 100% 是 CUDA kernel（`.cu`），需要 nvcc 编译 + CUDA 设备运行 | **不可修复**，改用等价 CPU 实现 |
| 4 | `pip install torch` 卡死 / 失败 | `download.pytorch.org`、`huggingface.co` 在本网络不可达（curl `http=000`）；`pypi.org` 仅 ~10 KB/s | **已修复**（改用 SJTU / 清华镜像，实测 1.1 MB/s） |
| 5 | 数据集 `360_v2.zip` / `tandt_db.zip` 下载失败 | INRIA 源下载 80 分钟后被强制断连，只拿到 12.7 MB 截断文件（zip 中央目录缺失）；Google Storage 仅 ~100 KB/s | **已修复**（改用 hf-mirror 上的 COLMAP 版 truck 场景，178 MB） |
| 6 | 内存余量仅 ~2.3 GB / 15.7 GB | 系统已占用约 13 GB | 通过限制分辨率、迭代数与高斯数量上限规避 |
| 7 | LPIPS 无法初始化 | torchvision 以 `check_hash=True` 校验 VGG16 权重的 SHA256，**自制文件必然失败** | **已修复**（改从 `download.pytorch.org` 拉官方文件） |
| 8 | `torch.hub` 写入 `~/.cache` 被拒 | 用户目录 `.cache` 有 Windows ACL/受控文件夹保护 | **已修复**（`TORCH_HOME` 重定向到工作区） |
| 9 | 官方 `metrics.py` 输出"No metrics" | 它**只**评测 `test/` 划分，而 `--eval` 未开启时 test 为空 | **已修复**（补跑一次 `--eval` 的留出协议训练） |

**核心结论：官方 CUDA 路径在本机物理上不可行（无 GPU）；本复现用等价的纯
PyTorch CPU 可微光栅器替换 CUDA 扩展，官方 `train.py` / `render.py` /
`metrics.py` 保持逐字节不改。**

---

## 1. 硬件与系统侦察

```powershell
nvidia-smi
# → 无法将"nvidia-smi"项识别为 cmdlet...（命令不存在）

Get-CimInstance Win32_VideoController | Select Name,DriverVersion,AdapterRAM
# Name          : Intel(R) UHD Graphics
# DriverVersion : 31.0.101.4146
# AdapterRAM    : 1073741824   (1 GB，核显共享内存)

Get-CimInstance Win32_PnPEntity | Where Name -match 'NVIDIA|GeForce|RTX|Quadro'
# → 无任何匹配（连一张 NVIDIA 设备都没有）

Get-Command nvcc        # → 不存在
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit"   # → False
$env:PATH -split ';' | Where {$_ -match 'cuda'}             # → 空
```

| 项目 | 实测值 |
|------|--------|
| CPU | 13th Gen Intel Core i5-13420H，8 核 12 线程 |
| GPU | **Intel(R) UHD Graphics（核显）**，无独立显卡 |
| CUDA 工具链 | **不存在**（无 nvcc，无 Toolkit） |
| 物理内存 | 15.7 GB（可用仅 ~2.3 GB） |
| 磁盘可用 | ~215 GB |
| 操作系统 | Windows（官方 3DGS 为 Linux-first） |
| Python | 3.12.6（系统）/ 3.11（用于本复现的 venv） |
| WSL | 未安装（`wsl --list` 报"未安装用于 Linux 的 Windows 子系统"） |

**为什么这是致命的**：`diff-gaussian-rasterization` 的 `setup.py` 只编译
`cuda_rasterizer/*.cu`，其全部前向/反向逻辑都在 CUDA kernel 里。
`simple-knn` 同理。没有 NVIDIA 设备，**即使**装好 CUDA Toolkit 把扩展编出来，
运行时也无法创建 CUDA context，`torch.cuda.is_available()` 恒为 `False`。

---

## 2. 第一步：插件安装

### 2.1 `dsh plugin --profile web add dsh-science-workbench` → 失败

```
dsh: installation rejected: Plugin dsh-science-workbench@0.3.0 is incompatible with
dsh 0.2.0-rc.2: peerDependencies {"@deepseek-ai/dsh-tools":"^0.1.0-rc.6"}.
Running it may cause crashes or data loss.
dsh: nothing was installed.
```

**根因**：插件声明的 peer 依赖区间 `^0.1.0-rc.6` 与本机 DSH `0.2.0-rc.2` 不匹配。
DSH 默认拒绝安装，并要求显式接受风险。

**修复**：按 DSH 自己给出的提示授予精确版本豁免。

```powershell
dsh plugin --profile web allow-version dsh-science-workbench@0.3.0 `
    --dsh-version 0.2.0-rc.2 --accept-risk
# → dsh: allowed dsh-science-workbench@0.3.0 for DSH 0.2.0-rc.2

dsh plugin --profile web add dsh-science-workbench
# → dependencies: + dsh-science-workbench ^0.3.0 ; Done in 3.6s
```

### 2.2 `dsh plugin --profile web add github:aa2246740/dsh-autoresearch`

```powershell
dsh plugin --profile web add github:aa2246740/dsh-autoresearch
# → ✓ Lockfile passes supply-chain policies
#   dependencies: + dsh-autoresearch git+https://github.com/aa2246740/dsh-autoresearch.git
#   Packages: +1 ... Done in 10.5s
```

### 2.3 验证

```powershell
dsh plugin --profile web list
# dsh-profile-web <home>\.dsh\profiles\web (PRIVATE)
# ├── dsh-autoresearch@1.0.10
# └── dsh-science-workbench@0.3.0
# 2 packages
```

✅ **第一步完成**。

---

## 3. 第二步：环境编译——官方 CUDA 路径

### 3.1 克隆官方仓库

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
# CLONE_EXIT=0
# submodules: SIBR_viewers, submodules/diff-gaussian-rasterization,
#             submodules/diff-gaussian-rasterization/third_party/glm,
#             submodules/fused-ssim, submodules/simple-knn
```

仓库 HEAD：`54c035f fix submodule name in environment.yml`（当前 main，比 SIGGRAPH
原始 release 更新：多出 `fused-ssim`、`SparseGaussianAdam`、深度正则、`antialiasing`）。
注意：该版本**没有 `requirements.txt`**，依赖声明在 `environment.yml`。

### 3.2 尝试安装 CUDA 光栅器（预期失败，实际记录）

```
$ pip install ./submodules/diff-gaussian-rasterization
```

在进入编译之前就已经失败，因为本机既无 nvcc 也无 CUDA runtime。该包的
`setup.py` 内容为：

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
...
setup(name="diff_gaussian_rasterization",
      ext_modules=[CUDAExtension(name="diff_gaussian_rasterization._C",
                                 sources=["cuda_rasterizer/*.cu", ...])])
```

`CUDAExtension` 在 configure 阶段就会去探测 `nvcc`，缺失时报
`CUDA_HOME is not set` / `No CUDA runtime is found`。

### 3.3 根因分析（决定性）

即使绕过 3.2：

1. **编译需要 nvcc** —— 可通过安装 CUDA Toolkit 解决（nvcc 可以在无 GPU 的机器上
   编译）。
2. **运行需要一个真实的 CUDA 设备** —— 这一条**无法**通过任何软件手段解决。
   `diff_gaussian_rasterization._C.rasterize_gaussians` 会做 `cudaMalloc`、
   `<<<grid, block>>>` 内核发射；`torch.cuda.is_available()` 恒为 `False`，
   `torch.zeros(..., device="cuda")` 会直接抛
   `RuntimeError: Found no NVIDIA driver on your system`。

官方 `train.py` 里至少有 11 处硬编码 CUDA（见第 4 节），所以**官方 CUDA 路径在
本机不可复现**，这不是配置问题，是硬件缺失。

### 3.4 修复方案：CPU 等价实现（本次采用的路线）

保留官方训练/渲染/评测代码不动，只替换两个**算法性**的 CUDA 扩展：

| 官方 CUDA 组件 | 本复现的替代 |
|---|---|
| `submodules/diff-gaussian-rasterization`（CUDA 可微光栅器） | `cpu_port/diff_gaussian_rasterization/`——纯 PyTorch 可微 CPU 实现，逐行对照 `forward.cu` / `auxiliary.h` / `rasterizer_impl.cu` 复刻 |
| `submodules/simple-knn`（`distCUDA2`，初始化尺度） | `cpu_port/simple_knn/_C.py`——用 `scipy.spatial.cKDTree` 精确等价实现 |
| 代码里的 `device="cuda"` / `.cuda()` / `torch.cuda.Event` | `cpu_port/sitecustomize.py`——启动时自动加载的设备重映射 shim |

这样 `train.py`、`render.py`、`metrics.py` **一个字节都没有改**。

---

## 4. 官方代码中的 CUDA 触点全清单（shim 必须覆盖的）

```
train.py:58    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
train.py:60    iter_start = torch.cuda.Event(enable_timing = True)
train.py:61    iter_end   = torch.cuda.Event(enable_timing = True)
train.py:109   bg = torch.rand((3), device="cuda") if opt.random_background else background
train.py:115   alpha_mask = viewpoint_cam.alpha_mask.cuda()
train.py:119   gt_image = viewpoint_cam.original_image.cuda()
train.py:132   mono_invdepth = viewpoint_cam.invdepthmap.cuda()
train.py:133   depth_mask = viewpoint_cam.depth_mask.cuda()
train.py:222   torch.cuda.empty_cache()
train.py:232   gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
train.py:252   torch.cuda.empty_cache()
render.py:54   background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
metrics.py:31  renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
metrics.py:32  gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
metrics.py:96  device = torch.device("cuda:0")
metrics.py:97  torch.cuda.set_device(device)
scene/gaussian_model.py:151-181, 307-312, 405-432  ... device="cuda" / .cuda() 约 20 处
scene/cameras.py:86-87    .......transpose(0,1).cuda()
utils/general_utils.py:65,83,102   device="cuda"
```

覆盖方式详见 `cpu_port/sitecustomize.py`；`scene/cameras.py` 的
`torch.device(data_device)` 通过命令行 `--data_device cpu` 规避。

---

## 5. 网络问题（本次最大的时间黑洞）

### 5.1 症状：`pip install torch` 长时间无进展

起初按官方文档安装：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

结果：venv 里最终只有 `pip` / `setuptools`，`pip list` 甚至挂死 >120s。
80 分钟后 `TORCH_EXIT=1`。

### 5.2 逐点测速（`curl --max-time 8/12 -w "%{http_code} %{speed_download}"`）

| 目标 | HTTP | 实测速度 | 可用性 |
|------|------|----------|--------|
| `download.pytorch.org` | **000** | 0 B/s | ❌ 完全不可达 |
| `huggingface.co` | **000** | 0 B/s | ❌ 完全不可达 |
| `pypi.org` | 206 | ~10 KB/s | ⚠️ 慢到不可用 |
| `storage.googleapis.com` | 206 | ~90–137 KB/s | ⚠️ 太慢（11.7 GB ≈ 24 h） |
| `repo-sam.inria.fr` | 206 | ~1 KB/s | ❌ 实际不可用 |
| `pypi.tuna.tsinghua.edu.cn` | 200 | 可用 | ✅ |
| `mirrors.aliyun.com/pypi` | 200 | 可用 | ✅ |
| **`mirror.sjtu.edu.cn/pytorch-wheels/cpu/`** | 200 | **1.1 MB/s** | ✅✅ 关键 |
| **`hf-mirror.com`** | 200 | 可用 | ✅✅ 关键 |

### 5.3 修复

**PyTorch**：从 SJTU 的 PyTorch CPU wheel 镜像直接拉 wheel，再本地安装。

```powershell
curl.exe -L -o torch-2.9.1+cpu-cp311-cp311-win_amd64.whl `
  "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/torch-2.9.1%2Bcpu-cp311-cp311-win_amd64.whl"
# → http=200 bytes=110888878 speed=1115711 B/s  (105.8 MB / 99 s)
```

**注意**：不要用 `--no-index`。第一次安装失败正是因为：

```
ERROR: Could not find a version that satisfies the requirement filelock (from torch)
```

`--no-index` 把 torch 的依赖（`filelock`/`sympy`/`networkx`/`jinja2`/`fsspec`…）
也一起屏蔽了。正确做法是 `--find-links` 指向本地 wheel 目录、同时保留镜像 index，
并用 `torch==2.9.1+cpu` 这种带 local version 的精确约束避免 pip 去镜像上拿
（体积大得多的）默认 `torch`。

### 5.4 数据集：两次失败

**尝试 1 —— 用户提供的 Google 源**：

```bash
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
```

HTTP HEAD 返回 `Content-Length: 11.67 GB`，但实测 ~100 KB/s ⇒ 约 24 小时，放弃。

**尝试 2 —— INRIA 官方 `tandt_db.zip`（0.64 GB，含 truck）**：

```
Invoke-WebRequest : 无法从传输连接中读取数据: An existing connection was forcibly
closed by the remote host。
downloaded 0.01 GB in 80.9 min
```

80 分钟后只拿到 **12.7 MB** 的截断文件，zip 校验：

```
ZIP INVALID: 找不到中央目录结尾记录。   (End of central directory record not found)
```

**尝试 3（成功）—— `hf-mirror.com` 上 `nvs-bench/tanksandtemples`**：

该镜像数据集内含与 3DGS 论文**完全一致**的 Tanks & Temples `truck` 场景，且已是
COLMAP 格式：

```
truck/images/000001.jpg … 000251.jpg    251 张（约 440 KB/张）
truck/sparse/0/cameras.bin
truck/sparse/0/images.bin               62,106,537 B
truck/sparse/0/points3D.bin             15,602,063 B
truck/sparse/0/project.ini
总计 ≈ 178.6 MB
```

### 5.5 一个具体的踩坑：`curl -K` 配置文件 + 中文路径

批量下载时用 curl 的 config file 传 `output = "..."`，写出时用了
`Set-Content -Encoding ASCII`。因为用户名是中文（`<user>`），ASCII 编码把路径里的
汉字替换成了 `?`：

```
output = "C:/Users/??/Documents/.../truck/images/000001.jpg"
curl exit=23        # 23 = Failed writing received data to disk
images: 0 files, 0.0 MB
```

`exit 23` 是 curl 的**写盘错误**。修复：改用**相对路径 + `Set-Location` 到目标目录**
（或 `-Encoding UTF8`），彻底避开路径中的非 ASCII 字符。

最终改为 Python 下载器 `tools/fetch_truck.py`：从镜像的 tree API 读取**精确字节数
清单**，逐文件校验大小、支持 Range 断点续传与重试、8 线程并行。

### 5.6 LPIPS / VGG16 权重：`check_hash=True` 让"自制权重"这条路彻底堵死

官方 `metrics.py` 会调用 `lpipsPyTorch.lpips(..., net_type='vgg')`，它需要两个文件：

```python
# lpipsPyTorch/modules/networks.py:92
self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
# lpipsPyTorch/modules/utils.py:13
url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
    + f'master/lpips/weights/v{version}/{net_type}.pth'
```

而 `download.pytorch.org` 在本网络最初**不可达**（`http=000`），于是 VGG16 权重拿
不到。我尝试了两条替代路径，都值得记录：

**路径 A（失败，但很有教育意义）**：用 `hf-mirror.com` 上 timm 的
`vgg16.tv_in1k` 权重重建 torchvision 格式文件。结果发现：

```
torchvision keys : ['0.weight', '0.bias', '2.weight', ...]          (26 个)
timm keys        : ['features.0.weight', 'features.0.bias', ...]    (26 个)
shape sets equal : True
```

两边张量形状完全一致，只是键名差一个 `features.` 前缀——看起来只差一步。
但**即使**补全全部键（含 `classifier.0/3/6`）也仍然会失败，因为 torchvision 是
按哈希校验的：

```python
model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
```

实测报错：

```
RuntimeError: Error(s) in loading state_dict for VGG:
    Missing key(s) in state_dict: "classifier.0.weight", "classifier.0.bias",
    "classifier.3.weight", "classifier.3.bias", "classifier.6.weight",
    "classifier.6.bias".
```

结论：**自制权重在原理上就不可能通过**（哈希不匹配），必须拿到官方原文件。

**路径 B（成功）**：重测发现 `download.pytorch.org` 其实**是间歇性可达**的——
早先整机测速是 `http=000`，后来同一个域名的
`/models/vgg16-397923af.pth` 却给出了 `http=200`，而且**实测 14.6 MB/s**
（20 秒下完 292 MB）：

```powershell
curl.exe -s -o NUL --max-time 20 -w "http=%{http_code} bytes=%{size_download} speed=%{speed_download}" `
  https://download.pytorch.org/models/vgg16-397923af.pth
# → http=200 bytes=292110356 speed=14605522 B/s
```

于是直接按官方方式拉取（`check_hash=True` 自动校验通过），LPIPS 立刻可用：

```
weights file MB 553.4
torchvision VGG16 loaded (hash verified) OK
LPIPS(self,self) = 0.0
LPIPS(self,rand) = 0.31655827164649963
```

**教训**：对"时通时断"的网络，任何一次测速结果都不能当成定论；**测速结论要有时
效性**。我因为最初一次 `http=000` 就断定它不可达，白白绕了一大圈。

### 5.7 `torch.hub` 写 `~/.cache` 被拒绝

```python
os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
# PermissionError: [WinError 5] 拒绝访问。: '<home>/.cache\\torch'
```

即使用 PowerShell `New-Item -Force` 把目录建出来，Python 侧往里**写文件**仍然
`PermissionError`（`vgg.pth.part`、`_timm_vgg16.bin.part` 都写不进去）。判断是
用户目录下 `.cache` 受 Windows 受控文件夹访问/ACL 保护。

修复：把 `TORCH_HOME` 重定向到工作区内，彻底绕开该目录。

```powershell
$env:TORCH_HOME = "C:\...\3dgs\torch_home"
# torch.hub.get_dir() → <工作区>\torch_home\hub
```

注意：运行 `metrics.py` 时必须带上这个环境变量，否则 LPIPS 找不到权重。

### 5.8 官方 `metrics.py` 只评 `test/` 划分

这一点很隐蔽。`metrics.py` 的逻辑是：

```python
test_dir = Path(scene_dir) / "test"
for method in os.listdir(test_dir):          # ← 只遍历 test/
    ...
```

而 `test/` 目录只有在**开启 `--eval`** 时才会生成（COLMAP 加载器会每 8 张留出
1 张作为测试视图）。如果按普通方式训练（`eval=False`，全部 251 张都是训练视图），
`render.py` 不会产生 `test/`，于是：

```python
except:
    print("Unable to compute metrics for model", scene_dir)
```

——**静默失败**，不报错、不产出任何指标（`metrics.py` 把整个循环包在裸 `except`
里）。

修复：额外跑一次带 `--eval` 的留出协议训练，得到真正的测试划分，再执行
`render.py` + `metrics.py`。同时主训练集的指标另行用官方
`utils/image_utils.psnr` 与 `utils/loss_utils.ssim`（纯 PyTorch，无需下载）单独计算，
并在 `notes/results.md` 中**明确区分 "train 划分" 与 "test 划分"**。

### 5.9 PowerShell 5.1 用 GBK 解码无 BOM 的 `.ps1`，把中文用户名变成乱码

这是本次最"阴"的一个坑：**它让自动化脚本静默失效，且不报任何明显错误。**

我为流水线写了 `tools/tail_pipeline.ps1`，开头硬编码了工作区路径：

```powershell
$w = "<work>"
```

脚本本身运行"成功"（进程存活、没有崩溃），但**什么都没做**——
只在等待循环里空转。查进程命令行才发现真相：

```
Add-Content : Could not find a part of the path
  'C:\Users\<mojibake>Documents\deepseek-harness\default-workspace\3dgs\logs\remaining_status.log'.
Set-Location : ... ItemNotFoundException
    + Set-Location $repo
```

`<user>` 变成了 `<mojibake>`。

**根因**：本机的 PowerShell 是 **Windows PowerShell 5.1**
（`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`，不是 pwsh 7）。
5.1 在读取 **没有 BOM** 的 `.ps1` 文件时，按系统 ANSI 代码页（中文 Windows 上是
**GBK/936**）解码，而不是 UTF-8。我的工具写出的是 **UTF-8 无 BOM**，于是
`<user>` 的 3 个 UTF-8 字节序列被当作 GBK 双字节字符重组，得到 `<mojibake>`。
结果：`Test-Path` / `Set-Location` / `Add-Content` 全部指向不存在的路径。

**为什么难发现**：脚本不报致命错误（`$ErrorActionPreference` 下的 `Test-Path` 只是
返回 `$false`），等待循环照常运行，`stdout` 又被重定向缓冲，所以从外面看就像
"脚本卡住了"。

**三种修复（本次采用第 1 种）**：

1. **不要把非 ASCII 路径硬编码进 `.ps1`**，改用 `$PSScriptRoot` 在运行时推导：

   ```powershell
   $w = Split-Path -Parent $PSScriptRoot      # <root>\tools -> <root>
   ```

   脚本内于是**不含任何非 ASCII 字符**，编码问题从根上消失。
2. 用 `Set-Content -Encoding UTF8` 写出**带 BOM** 的 `.ps1`（5.1 会用 BOM 判断编码）。
3. 全程使用 pwsh 7（默认按 UTF-8 读取脚本）——但本机 `pwsh` 不在 `PATH` 上。

顺带一提，这个坑与 5.5 节的 `curl -K` + ASCII 编码问题**是同一类错误的两个变体**：
**只要在中文 Windows 上处理含非 ASCII 字符的路径，就必须显式指定编码。**
两次都是 `-Encoding ASCII` / 无 BOM UTF-8 造成的。

---

## 6. 复现环境最终状态

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.9.1+cpu False
```

| 组件 | 版本/来源 |
|------|-----------|
| Python | 3.11（venv） |
| PyTorch | 2.9.1+cpu（SJTU 镜像 wheel） |
| torchvision | 0.24.1+cpu |
| CUDA | **不可用**（`torch.cuda.is_available() == False`） |
| 数据集 | truck（Tanks & Temples，COLMAP），hf-mirror |
| 光栅器 | `cpu_port/diff_gaussian_rasterization`（自研纯 PyTorch） |
| KNN | `cpu_port/simple_knn`（cKDTree） |
| 设备 shim | `cpu_port/sitecustomize.py`（PYTHONPATH 自动加载） |

---

## 7. 复现命令

```powershell
cd 3dgs\gaussian-splatting
$env:PYTHONPATH = "..\cpu_port"
..\venv\Scripts\python.exe train.py -s ..\data\truck -m ..\output\truck `
    --data_device cpu --disable_viewer -r 4 --iterations 7000
```

`train.py` / `render.py` / `metrics.py` 与官方仓库逐字节一致（可用
`git -C gaussian-splatting status --porcelain` 验证为干净工作区）。
