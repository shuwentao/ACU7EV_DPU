# 人脸检测显示系统工程设计文档

> 目标平台：Xilinx ZynqMP（DPUCZDX8G_ISA1_B4096），Petalinux 2022.1 / linux-xlnx 5.16
> 参考实现：`meta-user/recipes-apps/drm-plane/files/drm_plane.c`（零拷贝 V4L2→KMS）
> 已有参考：`model/facedete_haar_affine/`（Qt + OpenCV Haar + 仿射对齐）
> DPU 模型：`model/face_mobileV3/FaceDetector_0_int.xmodel`、`gate320_E.xmodel`

---

## 0. 结论先行（Qt / GStreamer / 原 C 程序？）

**建议：继续用 C/C++ + KMS(DRM) + V4L2 + VART(DPU) 这条路线，把 `drm_plane.c` 扩展成多线程应用。不要换 Qt，也不换 GStreamer。**

| 方案 | 评价 | 结论 |
|------|------|------|
| **原 C/KMS + V4L2 + VART** | 已验证零拷贝采集 + 硬件叠层；DPU 用 VART 原生 C++ 接入最顺；5.16 下 GFX 原生小 fb 可硬件定位，免 CPU 合成。 | ✅ 推荐 |
| Qt（EGLFS/KMS 或 QLabel） | 现有 Qt 参考用 `QLabel` 是 **CPU 合成整帧 1080p**，把 drm_plane 想避开的卡顿又引回来；EGLFS 在定制 DP 驱动上更易踩坑。仅当需要按钮/菜单等 GUI 控件才值得。 | ❌ 显示层不换 |
| GStreamer（kmssink） | kmssink 不易同时做「src 裁剪 + 逐像素 alpha 叠加 + 任意 crtc 定位」的组合，且 appsink 接 DPU 多一次拷贝/延迟。 | ❌ 不划算 |

> Qt 参考里的**算法逻辑可白嫖**：双摄环形缓冲采集、`estimatePartialAffine` + `warpAffine` 的 112×112 对齐，直接搬进新 C++ 工程，只把 Haar 换成 DPU、把 QLabel 显示换成 KMS。

---

## 1. 硬件约束（已按 Petalinux 2022.1 / 5.16 内核核对）

两个 plane 行为**完全不同**，这是本设计的关键：

| Plane | 格式 | fb 尺寸限制 | 定位 | src 裁剪 |
|-------|------|-------------|------|----------|
| **VID（Video Plane，DPDMA，下层 id≈39）** | NV12 / YUYV / RGB（见下） | **必须 = CRTC 分辨率 (1920×1080)**，传原生 320×320 会 `EINVAL` | crtc_x/y 可定位 | ✅ 支持（fb 内裁出 320×320） |
| **GFX（Graphics Plane，上层 id≈40）** | ARGB8888 | **无 fb==crtc 强制**，原生 320×320 fb 允许 | crtc_x/y 任意（1081,0 没问题） | ✅ 支持 |

要点：
1. **VID 层强制 fb.width==crtc.w、fb.height==crtc.h**：你只能在「全屏 1920×1080 的大 fb」里用 `src_w/src_h` 裁出一块（如 1080×1080 或 320×320），**不能**分配一个原生 320×320 的 dumb fb 直接喂 VID。
2. **GFX 层没有这个 fb 尺寸限制**：原生 320×320 ARGB fb 直接 `SetPlane(crtc=(1081,0,320,320))` 稳定支持 —— 这是阶段 1 放 video1、阶段 2 画框的落点。
3. **硬件层序**：GFX(40) 压在 VID(39) 之上。所以 VID 放主视频、GFX 放 PiP/OSD，透明处自然透出下层视频，无需额外合成。
4. **摄像头 DMA 忽略 `data_offset`**：相机总从 buffer 偏移 0 写起。对 video0（落在 (0,0)）无影响；对 video1（走 GFX 原生小 fb，偏移 0 即整块小 buffer）也无影响。它只影响"零拷贝能否把相机图直接落到一个非 (0,0) 的大 buffer 落点"——本方案里 video1 不依赖此，故不构成约束。
5. 屏幕 1920×1080：`video0(1080) + video1(320) = 1400 < 1920`，右侧余黑边，合理。

---

## 2. 阶段 1：video0 左 (0,0,1080,1080) + video1 右 (1081,0,320,320)

### 2.1 图层分配（核心）
```
VID  (下层, 39): video0  → fb 1920x1080, src裁(0,0,1080,1080), crtc(0,0,1080,1080)
GFX  (上层, 40): video1  → fb 原生 320x320 ARGB,          crtc(1081,0,320,320)
                  （GFX 其余区域 alpha=0 透出 video0 / 黑底）
```

### 2.2 数据流
```
video0 (1080x1080) ──零拷贝 DMA 写进 VID 全屏fb左半──→ SetPlane(VID, src裁1080, crtc 0,0)
video1 (320x320)   ──采集→ 格式转 ARGB8888 ──→ 原生320x320 GFX fb ──→ SetPlane(GFX, crtc 1081,0)
```

- **video0 @ (0,0)（VID 层，零拷贝）**：
  - 建 **1920×1080** 的 VID fb（RGB888 或 NV12，见 §4 风险1）。
  - 相机零拷贝写：复用 `drm_plane.c` 的 DMABUF 技巧，`bytesperline = 整屏 pitch`、`data_offset = 0`，1080×1080 自然落在 fb 左侧。
  - `SetPlane`：`src=(0,0, 1080<<16,1080<<16)`，`crtc=(0,0,1080,1080)`。
- **video1 @ (1081,0)（GFX 层，原生小 fb）**：
  - 建**原生 320×320 ARGB8888** 的 GFX fb（无尺寸限制，直接合法）。
  - 相机采集 video1 到普通 buffer → **格式转换 BGR/RGB/NV12 → ARGB8888**（补 alpha=0xFF，必要时 BGR↔RGB 交换）。这是唯一额外 CPU 开销，但仅 320×320≈10 万像素，且是直写小 fb，**远轻于合成进全屏 buffer**。
  - `SetPlane`：`src=(0,0,320<<16,320<<16)`，`crtc=(1081,0,320,320)`。硬件定位，零合成。
- **CRTC 点亮**：`drmModeSetCrtc` 用任一块 fb（如 video0 的 VID fb 或一块黑 fb）设一次模式即可，不参与逐帧刷新。

### 2.3 线程模型
- 线程 A：`video0` 采集（零拷贝 DMABUF 轮换）→ 写 VID fb → 主循环 `SetPlane(VID)`。
- 线程 B：`video1` 采集 → ARGB 转换 → 写 GFX 320×320 fb → `SetPlane(GFX)`。
- 两路各自 `DIRTYFB` 触发刷新，互不阻塞。

> 阶段 1 不引入 DPU，先把双路布局（VID 主画面 + GFX 原生小 fb PiP）跑通。

---

## 3. 阶段 2：仅显示 video0 + DPU 检测 video1 + 在 video0 上画框（分层）

### 3.1 图层分配
```
VID  (下层, 39): video0  → 同阶段1（唯一屏幕内容）
GFX  (上层, 40): 检测框   → 原生 ARGB fb（覆盖 video0 区域, 如 1080x1080 @ crtc(0,0)），框线 alpha=0xFF，其余 alpha=0 透出 video0
video1           → 不显示，仅喂 DPU
```

### 3.2 数据流
```
video0 (1080x1080) ──零拷贝──→ VID 层（唯一显示内容）
video1 (320x320)  ──采集到独立buffer──→ DPU 线程
                                          │
        FaceDetector_0_int.xmodel ──bbox + 5 landmarks (YuNet, conf=0.3,k=10)
                                          │
                              仿射对齐 112x112 (warpAffine, 复用已有代码)
                                          │
                                  gate320_E.xmodel ──识别/比对
                                          │
                              坐标映射 H: video1(320)→video0(1080)
                                          │
                                  画框 → GFX(ARGB) 层
```

### 3.3 DPU 接入（VART，C++）
- 用 `xir::Graph` + `vart::Runner` 加载两个 `.xmodel`（注意 `readme.txt`：输入 fix=6、输出 fix=4，需做量化反缩放）。
- 输入预处理：把 320×320 的 BGR 按模型要求转 RGB/归一化（采集是 BGR24，DPU 输入一般要 RGB，注意交换）。
- `FaceDetector` 输出 YuNet 风格：bbox + 5 关键点 + score；阈值 0.3、top-k=10。
- 仿射对齐：直接用 `facedete_haar_affine/face_lookup.cpp` 里已写好的 `estimatePartialAffine`（相似变换）+ `warpAffine` 到 112×112 模板。
- `gate320_E` 做识别/比对（确认它是输出 embedding 还是直接分类，决定要不要本地人脸库比对）。

### 3.4 画框（GFX 层，硬件叠加）
框画在 **GFX/ARGB 层**（原生小 fb 或覆盖 video0 的 fb）：遍历矩形边框像素，写 `alpha=0xFF, RGB=框色`；边框外保持 alpha=0 透出 video0。这是硬件叠加，CPU 只写少量框线像素，零合成开销。

### 3.5 ⚠️ 首要澄清点：video1 与 video0 的空间对应关系
框要画在 video0 上，但检测发生在 video1。必须有 **video1(320×320) 坐标 → video0(1080×1080) 坐标的映射 H**。三种可能：
- **(A) 两路同一场景、不同相机/视角**：离线 `cv::findHomography` 标定单应矩阵 H，运行时 `box0 = H · box1`。
- **(B) video1 是 video0 的下采样/裁剪（同一传感器）**：简单缩放 `s = 1080/320` + 平移对齐。
- **(C) 两路完全无关**：框会错位，**阶段 2 无法成立**。

> 动手前先确认 (A)/(B)/(C)，这是阶段 2 能否正确的前提，比选型更重要。

---

## 4. 待确认 / 风险清单
1. **VID 层接受的像素格式**：5.16 的 Video Plane 原生偏 NV12/YUYV。`drm_dual_plane.c` 默认用 `DRM_FORMAT_RGB888` 赌 DPDMA 支持。若 AddFB2/SetPlane 报 `EINVAL`，说明 VID 只吃 YUV，需把 video0 改为 **NV12 采集 + 2-plane（Y+UV）DMABUF** 处理（不是单纯改宏），届时再补该分支。
2. **GFX 的 ARGB 转换开销**：相机不输出 ARGB，video1→ARGB 必有一道格式转换（轻量），阶段 2 不显示 video1 则无需转换。
3. **plane id 映射**：以板子启动时打印的 `planes:` 列表为准，确认 VID≈39 / GFX≈40 且 GFX 确为上層；关 `g_alpha_en` 的 `gfx_id` 必须对应真正的 GFX 层。
4. DPU 模型 input tensor 具体格式（RGB/BGR、量化 scale、是否减均值），需读 `FaceDetector_0_int.xmodel` 输入节点。
5. `gate320_E.xmodel` 是 embedding 还是分类输出？决定要不要本地人脸库 + 距离阈值。
6. 是否需要把人脸 ID/姓名也叠加（GFX 层画文字，需 tiny 字库或 Cairo）。

---

## 5. 复用清单
| 来自 | 复用内容 | 改动 |
|------|----------|------|
| `drm_plane.c` | 零拷贝采集、plane 枚举、双 layer 选择、`g_alpha_en=0`、DIRTYFB+SetPlane 主循环 | 改为 VID=主视频(1920×1080 fb+src裁)、GFX=原生小 fb PiP/OSD |
| `facedete_haar_affine/capture_video.*` | 双摄环形缓冲（mutex + imgIn/imgOut/imgUse） | 改成两路独立、去掉 QLabel |
| `facedete_haar_affine/face_lookup.cpp` | `estimatePartialAffine` + `warpAffine` 112×112 对齐 | Haar → VART DPU；加坐标映射 H |
| `model/face_mobileV3/*.xmodel` | DPU 检测 + 识别 | VART 加载 |

---

## 6. 推荐落地顺序
1. ✅ **阶段 1 已实现**：`app/drm_dual_plane.c` + `app/Makefile`。video0 走 VID（1920×1080 fb + src 裁 1080×1080，零拷贝落左半）、video1 走 GFX（原生 320×320 RGB888 + crtc 1081,0，零拷贝）。两路 `poll` 独立 `SetPlane`。**首次跑请重点看：VID 层是否接受 `DRM_FORMAT_RGB888`**（若 `drmModeAddFB2`/`SetPlane` 报 `EINVAL`，把 `VID_FMT` 改成 `DRM_FORMAT_NV12` 并让 video0 以 NV12 采集）。
2. ⬜ 接入 VART，在 video1 上跑通 `FaceDetector`，打印 bbox/landmarks（先不显示）。
3. ⬜ 接 `gate320_E` + 仿射对齐，跑通识别。
4. ⬜ 确认 video0/video1 映射 H，把 video1 改为不显示、GFX 复用为 ARGB 画框层（阶段 2）。
5. ⬜ 收尾：加退出/热键、性能统计（FPS）、可选人脸 ID 文字叠加。
