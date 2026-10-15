# lessons-learned.md — 最难的一步是什么？为什么？

## 一句话答案

**最难的一步是：在一台没有 NVIDIA GPU 的机器上，写出一个"够正确"的纯 PyTorch
可微光栅器，让官方 `train.py` / `render.py` / `metrics.py` 一个字节都不改就能跑
起来并且真的收敛。**

难点不在"把公式抄一遍"（公式在论文和 `forward.cu` 里都有），而在于**三个隐性契
约**：相机矩阵的转置约定、`means2D` 的梯度路由、以及"离散化步骤不可导"这件事
在训练里到底要紧不要紧。

---

## 一、为什么"没有 GPU"本身不是最难的部分

发现没有 GPU 只花了 5 分钟：

```powershell
nvidia-smi                                     # 命令不存在
Get-CimInstance Win32_VideoController          # 只有 Intel(R) UHD Graphics
Get-Command nvcc                               # 不存在
```

这是一个**明确的、非黑即白的阻断**，一旦确认就没有任何"再试一下"的空间。
它的处理方式很干脆：接受它、绕开它、写清楚。真正消耗时间和脑力的，是它引出的
连锁后果。

反而是**网络环境**消耗了最多墙钟时间：`download.pytorch.org` 与
`huggingface.co` 直接不可达（curl `http=000`），`pypi.org` 只有 ~10 KB/s，
INRIA 官方的 `tandt_db.zip` 在下到 12.7 MB 时被强制断连（80 分钟白费）。
这部分很磨人，但本质上只是**试错 + 换源**，不需要创造性思考：
最终靠 `mirror.sjtu.edu.cn`（1.1 MB/s）和 `hf-mirror.com` 解决。

真正难的是下面三件事。

---

## 二、难点 1：相机矩阵的转置约定（差点让结果"看起来对但在漂"）

3DGS 的矩阵约定非常绕，因为 CUDA 内核按 **列主序** 的 `float*` 读写，而 PyTorch
是行主序张量：

```c
// auxiliary.h — transformPoint4x4
float4 transformed = {
    matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z  + matrix[12],   // x
    matrix[1]*p.x + matrix[5]*p.y + matrix[9]*p.z  + matrix[13],   // y
    matrix[2]*p.x + matrix[6]*p.y + matrix[10]*p.z + matrix[14],   // z
    matrix[3]*p.x + matrix[7]*p.y + matrix[11]*p.z + matrix[15] }; // w
```

把它翻译成对 `full_proj_transform` 张量（形状 `[4,4]`）的操作，正确形式是

```python
p_hom = torch.cat([means3D, ones], 1) @ projmatrix      # 右乘，不是左乘
```

**为什么这里最容易错**：`cameras.py` 里存的是
`getProjectionMatrix(...).transpose(0,1)`，然后
`full_proj_transform = world_view @ projection`。两处转置叠在一起，凭"直觉"
写出 `projmatrix @ p` 会得到一个**看起来能渲染出东西、但几何是错的**结果——
图像会出现、loss 也会下降，只是场景是镜像/错位的。这类 bug 不会崩，只会悄悄
毁掉最终结果。

我自己就一度在**测试脚本**里写错了：把 `projmatrix[2,3]=1.0`（w 行）写成了
`projmatrix[3,2]=1.0`，结果 `p_hom.w` 恒为 0，`1/(w+1e-7)` 爆炸，所有高斯被剔
除，渲染出一张纯背景图。更糟的是——**当时 gradcheck "通过"了**，因为常量图
像的数值雅可比与解析雅可比当然完全一致（都是 0）。这个"假绿"如果不揪出来，
会让我误以为梯度是对的。

**教训**：光过 gradcheck 不算验证。必须加**非退化断言**——我在测试里补了
`std(img) > 0.02`、`visible == 40/40`、"d img/d means3D ≠ 0"这类检查，否则
测试会在完全无效的状态下报成功。这是本次最值得记的一条工程经验。

---

## 三、难点 2：怎么让"未改动的官方代码"拿到它要的梯度

官方 `train.py` 靠这两行做致密化：

```python
gaussians.max_radii2D[visibility_filter] = torch.max(...)
gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
```

而 `add_densification_stats` 内部是：

```python
self.xyz_gradient_accum[update_filter] += torch.norm(
    viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
```

也就是说：**它直接读 `viewspace_point_tensor.grad`**。这个张量就是
`gaussian_renderer` 里那个 `screenspace_points`：

```python
screenspace_points = torch.zeros_like(pc.get_xyz, ..., device="cuda") + 0
```

它是一个**全零的哑张量**。在 CUDA 实现里，前向内核把投影后的 2D 坐标写进它，
反向内核把 `dL/d(2D mean)` 写进它的 `.grad`。

在纯 autograd 实现里，这个"哑张量"不会拿到梯度——因为我的前向是自己算投影的，
根本没用到它。如果 `grad` 是 `None`，`train.py` 会直接崩；如果我在最后接一个
`x * 0` 去骗出一个梯度，梯度值就是 0，致密化会**静默失效**（不崩，但永远不再
分裂高斯，最终图像糊成一团）。

**解法是直通（straight-through）恒等式**：

```python
means2d_used = point_image + (means2D[:, :2] - means2D[:, :2].detach())
```

- 前向值 = `point_image`（正确的解析投影），`means2D` 项贡献恰好为 0；
- 反向时 `d(loss)/d(means2D) = d(loss)/d(point_image)`，**正是 CUDA 内核
  `dL_dmeans2D` 的语义**；
- 而 `d(loss)/d(means3D)` 依然由 autograd 沿着"投影 → 光栅化"的正常链式法则
  给出，与 CUDA 的解析 `dL_dmean3D` 等价，**没有重复计数**（因为 `means2D` 是
  叶子，与 `means3D` 无依赖关系）。

**为什么这算难点**：它不是数学问题，是**接口语义考古**。你必须先读懂
`gaussian_model.py` 里那行 `viewspace_point_tensor.grad`，才能反推出
`diff_gaussian_rasterization` 这个"黑盒扩展"在梯度层面到底承诺了什么。官方
文档完全没写。

我用测试 D 专门守住这个契约：`means2D.grad` 必须非空、前两列非零、第三列为零。

---

## 四、难点 3：算法本身在若干点是"不可导"的，而你得判断它要紧不要紧

3DGS 的光栅化里有大量硬阈值：

| 位置 | 代码 | 性质 |
|---|---|---|
| 视锥剔除 | `p_view.z <= 0.2` | 阶跃 |
| 退化协方差 | `det == 0` | 阶跃 |
| 3σ 半径 | `ceil(3*sqrt(λmax))` | **阶跃（整数量化）** |
| tile 矩形 | `(int)((p ± r)/16)` | **阶跃** |
| 逐像素剔除 | `if (power > 0) continue` | 阶跃 |
| alpha 下限 | `if (alpha < 1/255) continue` | 阶跃 |
| alpha 上限 | `min(0.99, ...)` | 饱和 |
| 透射率截断 | `if (test_T < 1e-4) break` | 提前退出 |

这些都不是可导操作。官方 CUDA 的 `backward.cu` **完全不处理它们**——它把 tile
归属、半径、可见性当成前向算好的常数，直接对连续部分求导。

我的第一版 gradcheck 结果：

```
means3D    n_bad=  0/  15
scales     n_bad=  1/  18     <-- 一个元素对不上
rotations  n_bad=  0/  20
opacities  n_bad=  0/   5
shs        n_bad=  0/ 240
```

**"scales 有 1 个元素对不上"该怎么判？** 两种可能截然不同：
(a) 我的协方差/尺度梯度推错了； (b) 这个元素恰好跨过了 `ceil()` 或 tile 边界的
阶跃，中心差分本身失效。

如果误判成 (a)，我会去"修"一个本来正确的公式，越修越错。

我用了一个**可证伪的判据**：阶跃处的中心差分不收敛——它的值会随 `eps` 剧烈变化；
而光滑点的差分与 `eps` 无关。实测该元素：

```
analytic |d sum/d scale[15]| = 3.4348e+01
FD eps=1e-05                 = 7.7938e+01     (2.3x)
FD eps=1e-06                 = 4.7049e+02     (13.7x)
FD eps=1e-07                 = 3.4348e+01     (1.00x, 恰好等于解析值)
对照组 scale[8]              = 1.00x 离散度，FD 与解析完全一致
```

`eps=1e-6` 时差分被抬高了 13.7 倍（跨过了阶跃），`eps` 缩到 `1e-7` 时两侧都不再
跨阶跃，差分**收敛回解析值**。同时对照组元素的差分离散度是 1.00x。
这就证明：解析梯度在基点上是**对的**，不一致完全来自算法固有的离散性。

**为什么这算难点**：它要求你既能读懂数值现象，又能**抵抗"把测试改绿"的诱惑**。
最省事的做法是把 `scales` 从 gradcheck 里删掉——那就把一个真实的、有价值的
认识给抹掉了。正确的做法是把它变成一个**能解释、能复现、有对照组**的结论。

顺带一个有意思的发现：初始化时 `_rotation` 的梯度**精确为 0**。这不是 bug，
是数学必然——`q=(1,0,0,0)` 时 `R=I`，而 `Sigma = MᵀM` 对 `R` 的非对角元的一阶
导在 `R=I` 处恰好为零（因为此时 `R` 的非对角元为 0，而对角元的一阶导也为 0）。
官方 3DGS 同样从单位旋转初始化，靠浮点噪声 + Adam 的梯度归一化走出这个驻点。

---

## 五、如果重做一次，我会怎么改

1. **先做接口考古，再写代码。** 我一开始把 80% 的精力放在"抄 forward.cu 的公式"
   上，其实真正的风险在 `gaussian_renderer/__init__.py` 和
   `gaussian_model.py` 里那几行**看起来无关紧要**的赋值。应该先 grep 出所有
   `.grad` / `.grad` 消费者的用法，把契约写下来。
2. **不要只依赖 gradcheck。** gradcheck 在"函数退化成常量"时会给出**假阳性**。
   任何梯度测试都必须配一组非退化断言。
3. **测试的参考实现要和被测实现用不同的写法。** 我那份 naive reference 用
   Python 标量循环逐像素重写了 `preprocessCUDA` + `renderCUDA`，和被测的向量化
   tile 实现**没有共用一行代码**。首次对齐误差 2.4e-07——这份信心是
   "自己和自己比"永远给不了的。
4. **网络问题要尽早探测，不要等 80 分钟。** 一条 `curl -w "%{http_code}
   %{speed_download}"` 就能在 10 秒内判断一个源能不能用。我应该在
   `pip install` 之前就先测速。
5. **把离散性当成设计输入。** 与其指望 gradcheck 全绿，不如一开始就设计成
   "连续部分严格验证 + 离散部分显式说明"。
6. **中文 Windows 上，凡路径必显式指定编码。** 我在这上面栽了两次，而且是同一类
   错误的不同变体：
   - `Set-Content -Encoding ASCII` 把 `curl` 配置里的中文用户名写成 `?` →
     `curl exit 23`（写盘失败），255 个文件一个都没下下来；
   - Windows PowerShell 5.1 用 **GBK** 解码无 BOM 的 `.ps1`，把硬编码路径里的
     `<user>` 变成 `<mojibake>` → 自动化脚本**静默空转几十分钟**，不报致命错误。

   第二次更危险，因为它不崩：脚本"在运行"，只是什么都没做。**修复思路是同一个：
   不要在有非 ASCII 路径的环境里依赖默认编码**——要么显式写 BOM / UTF-8，要么
   干脆让脚本里不出现非 ASCII 字符（例如用 `$PSScriptRoot` 推导路径）。
7. **不要相信"脚本在跑"就等于"脚本在做正确的事"。** 我因为迟迟看不到输出，
   一度以为是 `extract_curves.py` 卡住（单独跑它 90 秒就完成）。真正的问题在
   更早的地方——路径根本没对上。**给自动化脚本加显式的进度落盘**
   （我最终改成每一步都 `Add-Content` 到一个状态文件），比盯着缓冲的 stdout
   猜测要有效得多。

---

## 六、结论

本项目里，"没有 GPU"是一个**容易发现、容易绕开**的硬约束；
"网络被墙"是一个**费时间但不需要智慧**的障碍；
而**"写一个能骗过官方未改动训练代码的可微光栅器"**才是真正的难点——它同时考验
对论文数学的理解、对他人代码接口语义的考古能力，以及在"测试不过"时**不把测试
改绿**的判断力。

最后一点，其实是最难的。
