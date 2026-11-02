/*
 * face_detect_dpu.cpp —— 合并版：单进程同时采集 video0(显示) + video1(检测) 并本地画框
 *
 * 显示架构（依据同 BSP 的 drm-plane / drm-plane-720p 参考，已在本板实测可行）：
 *   - VID 层(下层, RGB888)：显示 video0 摄像头画面（本层保持干净，未来可改零拷贝 DMABUF）。
 *   - GFX 层(上层, ARGB8888, id=40)：叠加人脸框。关闭该层的 global alpha(g_alpha_en=0)
 *     以启用逐像素 alpha —— 框外 alpha=0 透出下层视频，框内 alpha=255 不透明。
 *   - 关键：每次 CPU 写完 dumb buffer 必须 drmModeDirtyFB 通知驱动刷新，否则不显示。
 *   - 两路 camera 在同一进程内几乎同时 STREAMON，锁定时序，规避卡顿/撕裂。
 *   - DPU 每帧推理（~11ms@30fps，不跳帧），本地去抖后画框；无跨进程共享内存。
 *
 * 用法：face_detect_dpu [模型目录/文件] [相机设备(video1, 默认 /dev/video1)]
 *   自动打开 /dev/video0 做显示。环境变量：FACE_THR=阈值；DBG=1 打印调试。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <libdrm/drm.h>
#include <libdrm/drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <linux/dma-buf.h>

#include <map>
#include <dirent.h>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_helper.hpp>
#include <vart/dpu/vitis_dpu_runner_factory.hpp>

/* ---------------- 显示：video0 经 VID 层(下层) + 人脸框经 GFX 层(上层) ---------------- */
#define SCREEN_W 1920
#define SCREEN_H 1080
#define CAM0_W   1080
#define CAM0_H   1080
#define NBUF_FB  4
#define BOX_THICK 3
#define V4L_FMT  V4L2_PIX_FMT_BGR24
#define DRM_FMT_VID  DRM_FORMAT_RGB888
#define DRM_FMT_GFX  DRM_FORMAT_ARGB8888

static int      drm_fd = -1;
static uint32_t g_crtc_id = 0, g_conn_id = 0;
static uint32_t g_vid_plane_id = 0;   /* 下层视频平面 (RGB888) */
static uint32_t g_gfx_plane_id = 0;   /* 上层叠加平面 (ARGB8888) */
static drmModeModeInfo g_mode;
static int      g_screen_w = 0, g_screen_h = 0;

/* video 帧缓冲（VID 层，RGB888，全屏 1920x1080） */
static uint32_t vid_fb[NBUF_FB], vid_handle[NBUF_FB];
static uint64_t vid_size[NBUF_FB];
static uint32_t vid_pitch[NBUF_FB];
static void*    vid_map[NBUF_FB] = {nullptr};
/* 框 帧缓冲（GFX 层，ARGB8888，全屏 1920x1080，默认全透明） */
static uint32_t gfx_fb[NBUF_FB], gfx_handle[NBUF_FB];
static uint64_t gfx_size[NBUF_FB];
static uint32_t gfx_pitch[NBUF_FB];
static void*    gfx_map[NBUF_FB] = {nullptr};

/* 最新检测结果（1080x1080 主相机坐标空间），显示分支读取后画框 */
static int    g_detected = 0;
static int32_t g_bx = 0, g_by = 0, g_bw = 0, g_bh = 0;

static int set_plane_prop(uint32_t plane_id, const char* name, uint64_t val)
{
    drmModeObjectPropertiesPtr props =
        drmModeObjectGetProperties(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE);
    if (!props) return -1;
    int ret = -1;
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyPtr p = drmModeGetProperty(drm_fd, props->props[i]);
        if (!p) continue;
        if (p->name && strcmp(p->name, name) == 0)
            ret = drmModeObjectSetProperty(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE, p->prop_id, val);
        drmModeFreeProperty(p);
        if (ret == 0) break;
    }
    drmModeFreeObjectProperties(props);
    return ret;
}

static int get_plane_type(uint32_t plane_id)
{
    drmModeObjectPropertiesPtr props =
        drmModeObjectGetProperties(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE);
    if (!props) return -1;
    int type = -1;
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyPtr p = drmModeGetProperty(drm_fd, props->props[i]);
        if (!p) continue;
        if (p->name && strcmp(p->name, "type") == 0)
            type = (int)props->prop_values[i];
        drmModeFreeProperty(p);
        if (type >= 0) break;
    }
    drmModeFreeObjectProperties(props);
    return type;
}

static int drm_init(void)
{
    drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) { perror("open drm"); return -1; }
    /* 抢回 DRM master：确保能看到并控制全部 plane（含 GFX/40）。
     * 若控制台 fbcon 或上次未退出的进程占着 master，这里以 root 身份接管。 */
    if (drmSetMaster(drm_fd) < 0)
        fprintf(stderr, "warn: drmSetMaster 失败(非 master，可能仍能枚举到全部 plane)\n");
    else
        printf("DRM master acquired\n");
    drmModeResPtr res = drmModeGetResources(drm_fd);
    if (!res) { fprintf(stderr, "drmModeGetResources failed\n"); return -1; }
    drmModeConnectorPtr conn = nullptr;
    for (int i = 0; i < res->count_connectors && !conn; i++) {
        drmModeConnectorPtr c = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) conn = c;
        else if (c) drmModeFreeConnector(c);
    }
    if (!conn) { fprintf(stderr, "no connected connector\n"); return -1; }
    if (conn->encoder_id) {
        drmModeEncoderPtr e = drmModeGetEncoder(drm_fd, conn->encoder_id);
        if (e) { if (e->crtc_id) g_crtc_id = e->crtc_id; drmModeFreeEncoder(e); }
    }
    if (!g_crtc_id && res->count_crtcs > 0) g_crtc_id = res->crtcs[0];
    if (!g_crtc_id) { fprintf(stderr, "no crtc\n"); return -1; }
    g_conn_id = conn->connector_id;
    memset(&g_mode, 0, sizeof(g_mode));
    for (int i = 0; i < conn->count_modes; i++)
        if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED) { g_mode = conn->modes[i]; break; }
    if (!g_mode.hdisplay) g_mode = conn->modes[0];
    g_screen_w = g_mode.hdisplay; g_screen_h = g_mode.vdisplay;
    printf("mode: %dx%d@%u crtc_id=%u conn_id=%u\n", g_screen_w, g_screen_h, g_mode.vrefresh, g_crtc_id, g_conn_id);

    uint32_t crtc_idx = 0;
    for (int i = 0; i < res->count_crtcs; i++)
        if (res->crtcs[i] == g_crtc_id) crtc_idx = (uint32_t)i;

    /* 本板 drmModeGetPlaneResources 不返回 GFX 平面(40)，但 40 确实存在且可用
     * （drm_dual_plane 39 40 已实测 SetPlane(40) 可显示）。故优先按已知可用 ID 选层，
     * 再用 drmModeGetPlane 逐个校验；都不成再回退全量枚举。可用 VID_PLANE/GFX_PLANE 覆盖。 */
    uint32_t def_vid = 39, def_gfx = 40;
    if (getenv("VID_PLANE")) def_vid = (uint32_t)atoi(getenv("VID_PLANE"));
    if (getenv("GFX_PLANE")) def_gfx = (uint32_t)atoi(getenv("GFX_PLANE"));

    auto probe = [&](uint32_t id, int* usable, int* has_rgb, int* has_argb, int* type) -> int {
        drmModePlanePtr p = drmModeGetPlane(drm_fd, id);
        if (!p) return -1;
        *usable = (p->possible_crtcs & (1u << crtc_idx)) ? 1 : 0;
        *has_rgb = 0; *has_argb = 0;
        for (uint32_t f = 0; f < p->count_formats; f++) {
            if (p->formats[f] == DRM_FORMAT_RGB888 || p->formats[f] == DRM_FORMAT_BGR888) *has_rgb = 1;
            if (p->formats[f] == DRM_FORMAT_ARGB8888 || p->formats[f] == DRM_FORMAT_ABGR8888) *has_argb = 1;
        }
        *type = get_plane_type(id);
        drmModeFreePlane(p);
        return 0;
    };
    auto typestr = [](int t){ return t==0?"overlay":(t==1?"primary":(t==2?"cursor":"?")); };

    uint32_t cand_vid = 0, cand_gfx = 0;
    {
        int uv=0,hr=0,ha=0,tv=0, ug=0,gr=0,ga=0,tg=0;
        if (probe(def_vid, &uv,&hr,&ha,&tv) == 0) {
            printf("  probe VID id=%-3u type=%-8s usable=%d rgb888=%d argb=%d\n", def_vid, typestr(tv), uv, hr, ha);
            if (uv && hr && !ha) cand_vid = def_vid;
        } else printf("  probe VID id=%-3u : 不存在\n", def_vid);
        if (probe(def_gfx, &ug,&gr,&ga,&tg) == 0) {
            printf("  probe GFX id=%-3u type=%-8s usable=%d rgb888=%d argb=%d\n", def_gfx, typestr(tg), ug, gr, ga);
            if (ug && ga) cand_gfx = def_gfx;
        } else printf("  probe GFX id=%-3u : 不存在\n", def_gfx);
    }

    /* 回退：全量枚举（部分驱动 GetPlaneResources 正常时可用） */
    if (!cand_vid || !cand_gfx) {
        drmModePlaneResPtr pl = drmModeGetPlaneResources(drm_fd);
        if (pl) {
            for (uint32_t i = 0; i < pl->count_planes; i++) {
                int u=0,r=0,a=0,t=0;
                if (probe(pl->planes[i], &u,&r,&a,&t) == 0) {
                    printf("  enum id=%-3u type=%-8s usable=%d rgb888=%d argb=%d\n", pl->planes[i], typestr(t), u, r, a);
                    if (u) {
                        if (a && !cand_gfx) cand_gfx = pl->planes[i];
                        if (r && !a && !cand_vid) cand_vid = pl->planes[i];
                    }
                }
            }
            drmModeFreePlaneResources(pl);
        }
    }

    if (!cand_vid || !cand_gfx) {
        fprintf(stderr, "ERROR: 无法通过已知 ID(%u/%u) 或枚举找到 VID(RGB) 与 GFX(ARGB) 两个平面。\n", def_vid, def_gfx);
        fprintf(stderr, "       可用 VID_PLANE / GFX_PLANE 环境变量显式指定平面 ID。\n");
        return -1;
    }
    g_vid_plane_id = cand_vid;
    g_gfx_plane_id = cand_gfx;
    printf("VID(下层视频) plane_id=%u  GFX(上层叠加) plane_id=%u\n", g_vid_plane_id, g_gfx_plane_id);
    return 0;
}

static int make_fbs(void)
{
    for (int i = 0; i < NBUF_FB; i++) {
        /* VID 帧缓冲：RGB888，全屏 1920x1080 */
        struct drm_mode_create_dumb creq = {0}; struct drm_mode_map_dumb mreq = {0};
        creq.width = (uint32_t)SCREEN_W; creq.height = (uint32_t)SCREEN_H; creq.bpp = 24;
        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) { perror("CREATE_DUMB vid"); return -1; }
        vid_handle[i] = creq.handle; vid_size[i] = creq.size; vid_pitch[i] = creq.pitch;
        uint32_t hh[4] = { vid_handle[i], 0, 0, 0 };
        uint32_t pp[4] = { vid_pitch[i], 0, 0, 0 };
        uint32_t oo[4] = { 0, 0, 0, 0 };
        if (drmModeAddFB2(drm_fd, (uint32_t)SCREEN_W, (uint32_t)SCREEN_H, DRM_FMT_VID, hh, pp, oo, &vid_fb[i], 0) < 0) { perror("AddFB2 vid"); return -1; }
        mreq.handle = vid_handle[i];
        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) { perror("MAP_DUMB vid"); return -1; }
        vid_map[i] = mmap(NULL, vid_size[i], PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, mreq.offset);
        if (vid_map[i] == MAP_FAILED) { perror("mmap vid"); return -1; }
        memset(vid_map[i], 0, vid_size[i]);

        /* GFX 帧缓冲：ARGB8888，全屏，初始全透明（alpha=0） */
        struct drm_mode_create_dumb creq2 = {0}; struct drm_mode_map_dumb mreq2 = {0};
        creq2.width = (uint32_t)SCREEN_W; creq2.height = (uint32_t)SCREEN_H; creq2.bpp = 32;
        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq2) < 0) { perror("CREATE_DUMB gfx"); return -1; }
        gfx_handle[i] = creq2.handle; gfx_size[i] = creq2.size; gfx_pitch[i] = creq2.pitch;
        uint32_t gh[4] = { gfx_handle[i], 0, 0, 0 };
        uint32_t gp[4] = { gfx_pitch[i], 0, 0, 0 };
        uint32_t go[4] = { 0, 0, 0, 0 };
        if (drmModeAddFB2(drm_fd, (uint32_t)SCREEN_W, (uint32_t)SCREEN_H, DRM_FMT_GFX, gh, gp, go, &gfx_fb[i], 0) < 0) { perror("AddFB2 gfx"); return -1; }
        mreq2.handle = gfx_handle[i];
        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq2) < 0) { perror("MAP_DUMB gfx"); return -1; }
        gfx_map[i] = mmap(NULL, gfx_size[i], PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, mreq2.offset);
        if (gfx_map[i] == MAP_FAILED) { perror("mmap gfx"); return -1; }
        memset(gfx_map[i], 0, gfx_size[i]);   /* 全透明 */
    }
    return 0;
}

/* video0 采集状态 */
struct V0 {
    int      fd = -1;
    void*    mmap_buf[8] = {nullptr};   /* 兜底 memcpy 用：video0 的 MMAP 源缓冲 */
    unsigned mmap_len[8] = {0};
    int      v4l_pitch = 0;             /* memcpy 用：相机缓冲每行字节数 */
    int      nbuf = 0;
    int      dmabuf_fd[8] = {-1};       /* 零拷贝：DP dumb buffer 导出的 dmabuf fd，供 V4L2 DMABUF 导入/QBUF */
    int      zero_copy = 0;             /* 1=零拷贝(相机DMA直写DP缓冲) 0=回退 memcpy */
};
static V0 g_v0;

/* 打开 video0（显示采集）。零拷贝走"DP dumb buffer 导出 dmabuf → V4L2 DMABUF 导入，
 * 相机 DMA 直写 DP 缓冲"；否则 MMAP 兜底（CPU memcpy）。VID_ZC=1 时尝试零拷贝。 */
static int v0_open(void)
{
    g_v0.fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
    if (g_v0.fd < 0) { perror("open /dev/video0"); return -1; }

    /* S_FMT：采集分辨率 = 全屏 video0(CAM0_W x CAM0_H)。
     * 零拷贝时 bytesperline 必须设为【整屏 dumb buffer 的 pitch】(vid_pitch)，
     * 让相机 DMA 按整屏行步进、把画面直接写进 DP 全屏缓冲的 (0,0) 处；
     * memcpy 兜底时设为 CAM0_W*3（相机写进自己的 1080 宽缓冲）。 */
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = (unsigned)CAM0_W; fmt.fmt.pix_mp.height = (unsigned)CAM0_H;
    fmt.fmt.pix_mp.pixelformat = V4L_FMT;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    unsigned want_pitch = (unsigned)(CAM0_W * 3);
    if (getenv("VID_ZC")) want_pitch = (unsigned)vid_pitch[0];   /* 整屏 pitch（零拷贝落点） */
    fmt.fmt.pix_mp.plane_fmt[0].bytesperline = want_pitch;
    if (ioctl(g_v0.fd, VIDIOC_S_FMT, &fmt) < 0) { perror("S_FMT v0"); return -1; }
    if ((int)fmt.fmt.pix_mp.plane_fmt[0].bytesperline != (int)want_pitch)
        fprintf(stderr, "warn: video0 bytesperline 被改为 %u(期望 %u)，画面可能错位/拉伸\n",
                fmt.fmt.pix_mp.plane_fmt[0].bytesperline, want_pitch);
    g_v0.v4l_pitch = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    if (g_v0.v4l_pitch == 0) g_v0.v4l_pitch = (unsigned)(CAM0_W * 3);

    g_v0.zero_copy = 0;
    if (getenv("VID_ZC")) {
        /* 零拷贝：把每块 DP dumb buffer 导出 dmabuf fd，交给 V4L2 用 DMABUF 导入，
         * 相机 DMA 直接写进 DP 缓冲（CPU 不参与）。失败则回退 MMAP memcpy。 */
        struct v4l2_requestbuffers req = {0};
        req.count = NBUF_FB; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; req.memory = V4L2_MEMORY_DMABUF;
        if (ioctl(g_v0.fd, VIDIOC_REQBUFS, &req) == 0) {
            int ok = 1;
            int nzc = (int)req.count;
            for (int i = 0; i < nzc; i++) {
                int dmafd = -1;
                if (drmPrimeHandleToFD(drm_fd, vid_handle[i], 0, &dmafd) < 0 || dmafd < 0) { ok = 0; break; }
                g_v0.dmabuf_fd[i] = dmafd;
                struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_DMABUF;
                buf.index = (unsigned)i; buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
                pl[0].data_offset = 0; pl[0].length = (unsigned)vid_size[i]; pl[0].m.fd = dmafd;
                if (ioctl(g_v0.fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF(DMABUF) v0"); ok = 0; break; }
            }
            if (ok) {
                enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                if (ioctl(g_v0.fd, VIDIOC_STREAMON, &t) == 0) g_v0.zero_copy = 1;
                else perror("STREAMON(DMABUF) v0");
            }
        }
        if (!g_v0.zero_copy)
            fprintf(stderr, "warn: VID_ZC 零拷贝初始化失败，回退 MMAP memcpy\n");
    }

    if (!g_v0.zero_copy) {
        /* 兜底 MMAP：video0 用 MMAP 缓冲，显示时 CPU memcpy 进 DP dumb buffer */
        struct v4l2_requestbuffers req = {0};
        req.count = NBUF_FB; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(g_v0.fd, VIDIOC_REQBUFS, &req) < 0) { perror("REQBUFS v0"); return -1; }
        g_v0.nbuf = (int)req.count;
        for (int i = 0; i < g_v0.nbuf; i++) {
            struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
            buf.index = (unsigned)i; buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
            if (ioctl(g_v0.fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("QUERYBUF v0"); return -1; }
            g_v0.mmap_len[i] = pl[0].length;
            g_v0.mmap_buf[i] = mmap(NULL, pl[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, g_v0.fd, pl[0].m.mem_offset);
            if (g_v0.mmap_buf[i] == MAP_FAILED) { perror("mmap v0"); return -1; }
            if (ioctl(g_v0.fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF v0"); return -1; }
        }
        enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(g_v0.fd, VIDIOC_STREAMON, &t) < 0) { perror("STREAMON v0"); return -1; }
    }

    printf("v4l2 /dev/video0: %dx%d [%s]\n", CAM0_W, CAM0_H,
           g_v0.zero_copy ? "DMABUF 零拷贝(相机DMA直写DP缓冲)" : "MMAP memcpy 兜底");
    return 0;
}

static void cap_qbuf_video0(unsigned idx); /* 前向声明：定义见下方 Cap 段 */

static int grab_video0(int* fi)
{
    struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = g_v0.zero_copy ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
    buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
    if (ioctl(g_v0.fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) perror("DQBUF v0");
        return -1;
    }
    *fi = (int)buf.index;
    return 0;   /* 还回采集队列由 show_frame 处理（零拷贝/兜底路径不同） */
}

/* 在 GFX 帧缓冲上画绿框（坐标已是 1080x1080 主相机空间，裁剪到 video0 区）。
 * ARGB8888 内存字节序为 B,G,R,A；绿色(R=255,G=255,B=0) 不透明(A=255) 写为 (0,255,255,255)。 */
static void draw_box_argb(uint8_t* base, int pitch, int x, int y, int w, int h)
{
    auto setpx = [&](int px, int py) {
        if (px < 0 || py < 0 || px >= CAM0_W || py >= CAM0_H) return;
        uint8_t* p = base + (size_t)py * pitch + (size_t)px * 4;
        p[0] = 0; p[1] = 255; p[2] = 255; p[3] = 255;
    };
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > CAM0_W) w = CAM0_W - x;
    if (y + h > CAM0_H) h = CAM0_H - y;
    if (w <= 0 || h <= 0) return;
    int t = BOX_THICK;
    for (int yy = y; yy < y + h; yy++)
        for (int xx = x; xx < x + w; xx++)
            if (xx < x + t || xx >= x + w - t || yy < y + t || yy >= y + h - t) setpx(xx, yy);
}

/* 零拷贝：把指定 DP dumb buffer（已作为 dmabuf 导入 V4L2）还回采集队列 */
static void zc_qbuf_video0(unsigned idx)
{
    struct v4l2_buffer qb = {0}; struct v4l2_plane qp[VIDEO_MAX_PLANES] = {0};
    qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; qb.memory = V4L2_MEMORY_DMABUF;
    qb.index = idx; qb.length = VIDEO_MAX_PLANES; qb.m.planes = qp;
    qp[0].data_offset = 0; qp[0].length = (unsigned)vid_size[idx]; qp[0].m.fd = g_v0.dmabuf_fd[idx];
    if (ioctl(g_v0.fd, VIDIOC_QBUF, &qb) < 0)
        fprintf(stderr, "warn: QBUF(DMABUF) v0 failed: %s\n", strerror(errno));
}

/* 显示一帧：video0 拷进 VID 层 + 框画进 GFX 层，均 DIRTYFB 后 SetPlane */
static int g_fi = 0;
static int g_disp_v0 = -1;   /* 零拷贝：当前显示在 VID 平面的缓冲索引（持有，下一帧替换时才还回相机，防撕裂） */
static void show_frame(int v0_idx)
{
    /* VID 层：零拷贝 = 相机已把帧 DMA 写进 DP dumb buffer，直接 SetPlane(vid_fb[fi])；
     *         兜底 = memcpy 相机 MMAP 缓冲进 vid_fb[g_fi]。两者随后 DIRTYFB + 还回采集队列。
     *         零拷贝用 vid_fb[fi]（按 DQBUF 索引轮换，强制驱动提交新帧，避免黑屏）。 */
    if (g_v0.zero_copy) {
        /* 持有当前显示缓冲防撕裂：先把【上一帧】正显示中的缓冲还回相机（它已不再显示），
         * 再把新帧上平面；本帧缓冲转入"显示中"状态被持有，直到被下一帧替换才还回。 */
        if (g_disp_v0 >= 0 && g_disp_v0 != v0_idx) zc_qbuf_video0((unsigned)g_disp_v0);
        drmModeDirtyFB(drm_fd, vid_fb[v0_idx], NULL, 0);
        if (drmModeSetPlane(drm_fd, g_vid_plane_id, g_crtc_id, vid_fb[v0_idx], 0,
                            0, 0, (uint32_t)SCREEN_W, (uint32_t)SCREEN_H,
                            0, 0, ((uint32_t)SCREEN_W) << 16, ((uint32_t)SCREEN_H) << 16) < 0
            && errno != EBUSY)
            fprintf(stderr, "warn: SetPlane(vid-zc) failed: %s\n", strerror(errno));
        g_disp_v0 = v0_idx;
    } else {
        const uint8_t* s = (const uint8_t*)g_v0.mmap_buf[v0_idx];
        int sp = g_v0.v4l_pitch;
        uint8_t* d = (uint8_t*)vid_map[g_fi];
        int dp = (int)vid_pitch[g_fi];
        for (int y = 0; y < CAM0_H; y++)
            memcpy(d + (size_t)y * dp, s + (size_t)y * sp, (size_t)(CAM0_W * 3));
        drmModeDirtyFB(drm_fd, vid_fb[g_fi], NULL, 0);
        if (drmModeSetPlane(drm_fd, g_vid_plane_id, g_crtc_id, vid_fb[g_fi], 0,
                            0, 0, (uint32_t)SCREEN_W, (uint32_t)SCREEN_H,
                            0, 0, ((uint32_t)SCREEN_W) << 16, ((uint32_t)SCREEN_H) << 16) < 0
            && errno != EBUSY)
            fprintf(stderr, "warn: SetPlane(vid) failed: %s\n", strerror(errno));
        cap_qbuf_video0((unsigned)v0_idx);   /* 兜底：还回 MMAP 缓冲 */
    }

    /* GFX 层：先整帧清透明，再画框（若检测到） */
    memset(gfx_map[g_fi], 0, gfx_size[g_fi]);
    if (g_detected) draw_box_argb((uint8_t*)gfx_map[g_fi], (int)gfx_pitch[g_fi], g_bx, g_by, g_bw, g_bh);
    drmModeDirtyFB(drm_fd, gfx_fb[g_fi], NULL, 0);
    if (drmModeSetPlane(drm_fd, g_gfx_plane_id, g_crtc_id, gfx_fb[g_fi], 0,
                        0, 0, (uint32_t)SCREEN_W, (uint32_t)SCREEN_H,
                        0, 0, ((uint32_t)SCREEN_W) << 16, ((uint32_t)SCREEN_H) << 16) < 0
        && errno != EBUSY)
        fprintf(stderr, "warn: SetPlane(gfx) failed: %s\n", strerror(errno));

    g_fi = (g_fi + 1) % NBUF_FB;
}

/* ---------------- 检测：VART DPU（沿用原逻辑） ---------------- */
static std::string read_kernel_from_meta(const std::string& xmodel_path) {
    std::string dir = xmodel_path.substr(0, xmodel_path.find_last_of('/'));
    std::string meta_path = dir + "/meta.json";
    FILE* f = fopen(meta_path.c_str(), "rb");
    if (!f) return "";
    std::string s; char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    size_t k = s.find("\"kernel\"");
    if (k == std::string::npos) return "";
    size_t a = s.find('[', k);
    if (a == std::string::npos) return "";
    size_t p = s.find('"', a);
    if (p == std::string::npos) return "";
    size_t q = s.find('"', p + 1);
    if (q == std::string::npos) return "";
    return s.substr(p + 1, q - p - 1);
}

static volatile int g_stop = 0;
static void on_sig(int) { g_stop = 1; }

/* ---------------- 摄像头（video1, MMAP, MPLANE, BGR24） ---------------- */
struct Cap {
    int fd = -1, w = 0, h = 0, nbuf = 0;
    void* mmap_buf[8] = {nullptr};
    unsigned mmap_len[8] = {0};
};

static int cap_open(Cap* c, const char* dev, int w, int h)
{
    c->w = w; c->h = h;
    c->fd = open(dev, O_RDWR | O_NONBLOCK);
    if (c->fd < 0) { perror("open cam"); return -1; }
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = (unsigned)w; fmt.fmt.pix_mp.height = (unsigned)h;
    fmt.fmt.pix_mp.pixelformat = V4L_FMT;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    if (ioctl(c->fd, VIDIOC_S_FMT, &fmt) < 0) { perror("S_FMT"); return -1; }
    if (fmt.fmt.pix_mp.pixelformat != V4L_FMT) {
        fprintf(stderr, "相机不支持 BGR24(实际 %.4s)\n", (char*)&fmt.fmt.pix_mp.pixelformat);
        return -1;
    }
    struct v4l2_requestbuffers req = {0};
    req.count = 4; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(c->fd, VIDIOC_REQBUFS, &req) < 0) { perror("REQBUFS"); return -1; }
    c->nbuf = (int)req.count;
    for (int i = 0; i < c->nbuf; i++) {
        struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
        buf.index = (unsigned)i; buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
        if (ioctl(c->fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("QUERYBUF"); return -1; }
        c->mmap_len[i] = pl[0].length;
        c->mmap_buf[i] = mmap(NULL, pl[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, pl[0].m.mem_offset);
        if (c->mmap_buf[i] == MAP_FAILED) { perror("mmap"); return -1; }
        if (ioctl(c->fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF"); return -1; }
    }
    enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(c->fd, VIDIOC_STREAMON, &t) < 0) { perror("STREAMON"); return -1; }
    return 0;
}

static int cap_grab(Cap* c, uint8_t** out)
{
    struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
    buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
    if (ioctl(c->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) perror("DQBUF");
        return -1;
    }
    *out = (uint8_t*)c->mmap_buf[buf.index];
    return (int)buf.index;
}

static void cap_qbuf_video0(unsigned idx)
{
    struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
    buf.index = idx; buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
    ioctl(g_v0.fd, VIDIOC_QBUF, &buf);
}

static void cap_qbuf(Cap* c, int idx)
{
    struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
    buf.index = (unsigned)idx; buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
    ioctl(c->fd, VIDIOC_QBUF, &buf);
}

static std::string find_xmodel(const std::string& dir)
{
    DIR* d = opendir(dir.c_str());
    if (!d) { fprintf(stderr, "find_xmodel: opendir(%s) 失败: %s\n", dir.c_str(), strerror(errno)); return std::string(); }
    struct dirent* e; std::string r;
    while ((e = readdir(d))) {
        std::string n = e->d_name;
        if (n.size() >= 6 && n.compare(n.size() - 6, 6, ".xmodel") == 0) { r = dir + "/" + n; break; }
    }
    closedir(d);
    return r;
}

static std::string resolve_xmodel(const std::string& arg)
{
    if (arg.size() >= 7 && arg.compare(arg.size() - 7, 7, ".xmodel") == 0) return arg;
    return find_xmodel(arg);
}

static int get_fix(const std::map<std::string,int>& m, const std::string& name)
{ auto it = m.find(name); return (it != m.end()) ? it->second : 0; }

static int fixpos_of(const xir::Tensor* t, const std::map<std::string,int>& m)
{
    if (t->has_attr("fix_point")) { try { return t->get_attr<int>("fix_point"); } catch (...) {} }
    auto it = m.find(t->get_name());
    return (it != m.end()) ? it->second : 0;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline int8_t quant_i8(float v, float scale)
{ int iv = (int)lroundf(v * scale); if (iv > 127) iv = 127; if (iv < -128) iv = -128; return (int8_t)iv; }

static void resize_bgr(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh)
{
    float fx = (float)sw / dw, fy = (float)sh / dh;
    for (int y = 0; y < dh; y++) {
        float sy = (y + 0.5f) * fy - 0.5f; if (sy < 0) sy = 0; if (sy > sh - 1) sy = sh - 1;
        int y0 = (int)sy, y1 = y0 + 1; if (y1 > sh - 1) y1 = sh - 1;
        float dy = sy - y0;
        for (int x = 0; x < dw; x++) {
            float sx = (x + 0.5f) * fx - 0.5f; if (sx < 0) sx = 0; if (sx > sw - 1) sx = sw - 1;
            int x0 = (int)sx, x1 = x0 + 1; if (x1 > sw - 1) x1 = sw - 1;
            float dx = sx - x0;
            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * sw + x0) * 3 + c], v01 = src[(y0 * sw + x1) * 3 + c];
                float v10 = src[(y1 * sw + x0) * 3 + c], v11 = src[(y1 * sw + x1) * 3 + c];
                float v = (v00 * (1 - dx) + v01 * dx) * (1 - dy) + (v10 * (1 - dx) + v11 * dx) * dy;
                dst[(y * dw + x) * 3 + c] = (uint8_t)(v + 0.5f);
            }
        }
    }
}

struct Det { float x1, y1, x2, y2, score; };

int main(int argc, char** argv)
{
    /* 用法守卫：合并版不接收平面 ID。正确用法：
     *   face_detect_dpu <模型目录或.xmodel> [相机设备(默认/dev/video1)] [阈值]
     * 若把旧 drm_dual_plane 的 "39 40" 当参数传进来（argv[1] 是纯数字），提示正确用法。 */
    auto is_digits = [](const char* s) {
        if (!s || !*s) return false;
        for (const char* p = s; *p; p++) if (*p < '0' || *p > '9') return false;
        return true;
    };
    if (argc < 2 || is_digits(argv[1])) {
        fprintf(stderr,
            "用法: face_detect_dpu <模型目录或.xmodel> [相机设备(默认/dev/video1)] [阈值]\n"
            "示例: face_detect_dpu /usr/share/vart/face_detector/gate320.xmodel /dev/video1\n"
            "注意: 平面 ID(如 39 40)由程序自动探测，不要再手动传。\n");
        return 2;
    }

    const char* model_dir = argv[1];
    const char* cam_dev  = (argc > 2) ? argv[2] : "/dev/video1";
    int dbg = getenv("DBG") ? 1 : 0;
    float score_thr = 0.40f;
    if (argc > 3) score_thr = (float)atof(argv[3]);
    else if (getenv("FACE_THR")) score_thr = (float)atof(getenv("FACE_THR"));

    /* 去抖迟滞（我主动加的稳定逻辑）可配：DET_OFF=1 直接按阈值开关；否则连续 N_DET 帧确认出现 / N_LOST 帧确认消失 */
    int det_off = getenv("DET_OFF") ? 1 : 0;
    int n_det  = getenv("DET_NDET")  ? atoi(getenv("DET_NDET"))  : 3;
    int n_lost = getenv("DET_NLOST") ? atoi(getenv("DET_NLOST")) : 5;

    fprintf(stderr, "[DBG-VER] face_detect_dpu MERGE_GFX (单进程: 双路同流 + GFX 叠层画框)\n");
    fprintf(stderr, "[DBG] argv: model_dir=%s cam_dev=%s dbg=%d score_thr=%.2f\n",
            model_dir, cam_dev, dbg, score_thr);

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* === 显示初始化 + 两路相机同时 STREAMON（在慢速 DPU 编译之前，确保双路同流锁定时序） === */
    if (drm_init() < 0) return 1;
    if (make_fbs() < 0) return 1;
    if (v0_open() < 0) return 1;                       /* video0 STREAMON */

    /* video1 在 runner 创建前以固定 320 开流（锁时序）；Msz 随后从 runner 推导 */
    Cap cap;
    if (cap_open(&cap, cam_dev, 320, 320) < 0) return 1; /* video1 STREAMON（与 video0 几乎同时） */
    fprintf(stderr, "[DBG] 双路相机已同时 STREAMON（video0 + video1@320）\n");

    /* === DPU runner 创建（慢速：编译 DPU kernel）；此时双路已在流，源已锁定 === */
    std::string xmodel = resolve_xmodel(model_dir);
    fprintf(stderr, "[DBG] 解析到 xmodel='%s'\n", xmodel.c_str());
    std::unique_ptr<xir::Graph> graph;
    std::map<std::string,int> fixmap;
    if (!xmodel.empty()) {
        try {
            graph = xir::Graph::deserialize(xmodel);
            for (auto t : graph->get_tensors())
                if (t->has_attr("fix_point")) fixmap[t->get_name()] = t->get_attr<int>("fix_point");
        } catch (...) { fprintf(stderr, "XIR 读取失败(忽略)\n"); }
    } else {
        fprintf(stderr, "错误: 未解析到 .xmodel, 请检查模型路径 %s\n", model_dir); return 1;
    }

    std::vector<std::string> knames;
    if (graph) {
        auto root = graph->get_root_subgraph();
        if (root) {
            auto children = root->get_children();
            knames.push_back(graph->get_name());
            knames.push_back(graph->get_name() + "_0");
            for (auto sub : children) {
                std::string n = sub->get_name();
                bool is_fix = n.find("quant") != std::string::npos
                           || n.find("fix") != std::string::npos
                           || n.find("input") != std::string::npos;
                if (n.find("subgraph_") != std::string::npos && !is_fix) knames.push_back(n);
            }
        }
    }
    std::string meta_kernel = read_kernel_from_meta(xmodel);
    if (!meta_kernel.empty()) knames.push_back(meta_kernel);

    std::unique_ptr<vart::Runner> runner_u;
    std::string used_kname;
    for (auto& kn : knames) {
        fprintf(stderr, "[DBG] try create_dpu_runner kernel='%s'\n", kn.c_str());
        try { runner_u = vart::dpu::DpuRunnerFactory::create_dpu_runner(xmodel, kn); }
        catch (...) { runner_u = nullptr; }
        if (!runner_u) continue;
        auto it_in = runner_u->get_input_tensors();
        if (!it_in.empty() && !it_in[0]->get_name().empty()) { used_kname = kn; break; }
        fprintf(stderr, "[DBG] kernel='%s' 输入名为空(非入口子图), 跳过\n", kn.c_str());
        runner_u = nullptr;
    }
    if (!runner_u) { fprintf(stderr, "create_dpu_runner 失败: %s\n", xmodel.c_str()); return 1; }
    fprintf(stderr, "[DBG] create_dpu_runner OK kernel='%s'\n", used_kname.c_str());

    vart::Runner* runner = runner_u.get();
    auto inputs  = runner->get_input_tensors();
    auto outputs = runner->get_output_tensors();
    auto in_bufs  = vart::alloc_cpu_flat_tensor_buffers(inputs);
    auto out_bufs = vart::alloc_cpu_flat_tensor_buffers(outputs);
    auto inshape = inputs[0]->get_dims();
    bool nhwc = (runner->get_tensor_format() == vart::Runner::TensorFormat::NHWC);
    int in_fix = fixpos_of(inputs[0], fixmap);

    /* 模型输入边长（正方形） */
    int Msz = 320;
    if (inshape.size() >= 4) {
        int a = (int)inshape[nhwc ? 1 : 2], b = (int)inshape[nhwc ? 2 : 3];
        Msz = (a == b) ? a : (a > b ? a : b);
    }
    fprintf(stderr, "[DBG] 模型输入 Msz=%d nhwc=%d in_fix=%d\n", Msz, (int)nhwc, in_fix);

    /* === 点亮 CRTC（在 GFX 透明 fb 上）+ 关 GFX 的 global alpha（启用逐像素 alpha） === */
    if (drmModeSetCrtc(drm_fd, g_crtc_id, gfx_fb[0], 0, 0, &g_conn_id, 1, &g_mode) < 0)
        fprintf(stderr, "warn: SetCrtc failed: %s\n", strerror(errno));
    else
        printf("crtc enabled: %dx%d@%u\n", g_screen_w, g_screen_h, g_mode.vrefresh);
    {
        int r0 = set_plane_prop(g_gfx_plane_id, "g_alpha_en", 0);
        printf("GFX g_alpha_en -> 0 : %s (逐像素 alpha 启用；框外透出视频)\n",
               r0 == 0 ? "ok" : "failed(框可能不显示!)");
    }

    std::vector<uint8_t> rsbuf((size_t)Msz * Msz * 3);
    int frame = 0;
    static int stable = 0, hi_run = 0, lo_run = 0;
    static int32_t last_bx = 0, last_by = 0, last_bw = 0, last_bh = 0;
    static int prev_face = 0;

    struct pollfd pfd[2];
    pfd[0].fd = g_v0.fd;       pfd[0].events = POLLIN;
    pfd[1].fd = cap.fd;        pfd[1].events = POLLIN;

    printf("running: detect+display (DPU 每帧, GFX 叠层画框, 去抖=%s)  [Ctrl+C stop]\n",
           det_off ? "关(DET_OFF)" : "开");
    while (!g_stop) {
        int n = poll(pfd, 2, 1000);
        if (n < 0) { if (errno == EINTR) continue; perror("poll"); break; }
        if (n == 0) continue;

        /* video1：抓帧 → 预处理+INT8 → 推理 → 解码 → 去抖 → 更新最新框 */
        if (pfd[1].revents & POLLIN) {
            uint8_t* raw = nullptr;
            int idx = cap_grab(&cap, &raw);
            if (idx < 0) { usleep(500); }
            else if (raw) {
                const uint8_t* bgr = raw;
                if (cap.w != Msz || cap.h != Msz) { resize_bgr(raw, cap.w, cap.h, rsbuf.data(), Msz, Msz); bgr = rsbuf.data(); }
                auto ind = in_bufs[0]->data();
                int8_t* idst = reinterpret_cast<int8_t*>(ind.first);
                float scale = (float)(1LL << in_fix);
                for (int y = 0; y < Msz; y++) {
                    for (int x = 0; x < Msz; x++) {
                        const uint8_t* p = bgr + ((size_t)y * Msz + x) * 3;
                        float fr = (p[2] / 255.0f - 0.5f) / 0.5f;
                        float fg = (p[1] / 255.0f - 0.5f) / 0.5f;
                        float fb = (p[0] / 255.0f - 0.5f) / 0.5f;
                        if (nhwc) {
                            size_t o = ((size_t)y * Msz + x) * 3;
                            idst[o + 0] = quant_i8(fr, scale);
                            idst[o + 1] = quant_i8(fg, scale);
                            idst[o + 2] = quant_i8(fb, scale);
                        } else {
                            idst[(0 * (size_t)Msz + y) * Msz + x] = quant_i8(fr, scale);
                            idst[(1 * (size_t)Msz + y) * Msz + x] = quant_i8(fg, scale);
                            idst[(2 * (size_t)Msz + y) * Msz + x] = quant_i8(fb, scale);
                        }
                    }
                }

                std::vector<vart::TensorBuffer*> in_ptr, out_ptr;
                for (auto& b : in_bufs) in_ptr.push_back(b.get());
                for (auto& b : out_bufs) out_ptr.push_back(b.get());
                auto job = runner->execute_async(in_ptr, out_ptr);
                runner->wait(job.first, -1);

                std::vector<Det> dets;
                for (size_t k = 0; k < outputs.size(); k++) {
                    auto oshape = outputs[k]->get_dims();
                    int Hd, Wd;
                    if (nhwc) { Hd = (int)oshape[1]; Wd = (int)oshape[2]; }
                    else      { Hd = (int)oshape[2]; Wd = (int)oshape[3]; }
                    int C = (nhwc ? (int)oshape[3] : (int)oshape[1]);
                    int stride = Msz / Hd;
                    int out_fix = fixpos_of(outputs[k], fixmap);
                    auto od = out_bufs[k]->data();
                    const int8_t* d8 = reinterpret_cast<const int8_t*>(od.first);
                    float s = (float)ldexp(1.0, -out_fix);
                    std::vector<float> tmpF((size_t)Hd * Wd * C);
                    for (size_t i = 0; i < tmpF.size(); i++) tmpF[i] = d8[i] * s;
                    const float* d = tmpF.data();
                    const float CONF = 0.05f;
                    for (int y = 0; y < Hd; y++) {
                        for (int x = 0; x < Wd; x++) {
                            float score = 0, l = 0, t = 0, r = 0, b = 0;
                            if (nhwc) {
                                size_t o = ((size_t)y * Wd + x) * C;
                                score = sigmoidf(d[o + 0]); l = d[o + 1]; t = d[o + 2]; r = d[o + 3]; b = d[o + 4];
                            } else {
                                score = sigmoidf(d[(0 * Hd + y) * Wd + x]);
                                if (score < CONF) continue;
                                l = d[(1 * Hd + y) * Wd + x]; t = d[(2 * Hd + y) * Wd + x];
                                r = d[(3 * Hd + y) * Wd + x]; b = d[(4 * Hd + y) * Wd + x];
                            }
                            if (score < CONF) continue;
                            float cx = (x + 0.5f) * stride, cy = (y + 0.5f) * stride;
                            Det det;
                            det.x1 = cx - l * stride; det.y1 = cy - t * stride;
                            det.x2 = cx + r * stride; det.y2 = cy + b * stride;
                            det.score = score;
                            if (det.x2 > det.x1 && det.y2 > det.y1) dets.push_back(det);
                        }
                    }
                }

                Det best = {0, 0, 0, 0, -1};
                for (auto& d : dets) if (d.score > best.score) best = d;

                double sc = 1080.0 / Msz;
                int bx = 0, by = 0, bw = 0, bh = 0;
                if (best.score >= 0) {
                    bx = (int)round(best.x1 * sc); by = (int)round(best.y1 * sc);
                    bw = (int)round((best.x2 - best.x1) * sc); bh = (int)round((best.y2 - best.y1) * sc);
                    if (bx < 0) { bw += bx; bx = 0; }
                    if (by < 0) { bh += by; by = 0; }
                }

                if (best.score >= 0) { last_bx = bx; last_by = by; last_bw = bw; last_bh = bh; }
                if (det_off) {
                    stable = (best.score >= score_thr) ? 1 : 0;   /* 关去抖：直接按阈值开关 */
                    hi_run = lo_run = 0;
                } else {
                    if (best.score >= score_thr) { hi_run++; lo_run = 0; }
                    else { lo_run++; hi_run = 0; }
                    if (!stable && hi_run >= n_det) stable = 1;
                    if (stable && lo_run >= n_lost) stable = 0;
                }
                int detected = stable;
                if (detected) { g_bx = last_bx; g_by = last_by; g_bw = last_bw; g_bh = last_bh; }
                g_detected = detected;

                int has_face = detected;
                if (has_face != prev_face) {
                    if (has_face)
                        printf(">>> FACE DETECTED frame=%d score=%.3f box=(%d,%d,%d,%d)\n",
                               frame, best.score, bx, by, bw, bh);
                    else
                        printf(">>> FACE LOST frame=%d\n", frame);
                    prev_face = has_face; fflush(stdout);
                }
                if (dbg && (frame % 30 == 0))
                    printf("[detect] frame=%d dets=%zu best_score=%.3f box=(%d,%d,%d,%d)\n",
                           frame, dets.size(), best.score, bx, by, bw, bh);
                frame++;
                cap_qbuf(&cap, idx);
            } else {
                cap_qbuf(&cap, idx);
            }
        }

        /* video0：显示（含画框）。若本帧处理期间相机积压了多帧，仅显示【最新】一帧，
         * 中间帧立即还回（零拷贝/兜底分别还回），避免队列堆积造成的延迟。
         * 注意：只有"上一帧已确认不是最新(last>=0)"时才还回，最后一帧留给 show_frame
         * 持有上平面——否则会把正在显示的缓冲又还回相机，导致撕裂 + 重复 QBUF(EINVAL)。 */
        if (pfd[0].revents & POLLIN) {
            int v0_idx = -1, last = -1;
            while (grab_video0(&v0_idx) == 0) {
                if (last >= 0) {   /* 上一帧已确认不是最新 → 还回 */
                    if (g_v0.zero_copy) zc_qbuf_video0((unsigned)last);
                    else cap_qbuf_video0((unsigned)last);
                }
                last = v0_idx;
            }
            if (last >= 0) show_frame(last);   /* 仅最新帧上平面（由 show_frame 持有到被替换） */
        }
    }

    printf("face_detect_dpu 退出\n");
    return 0;
}
