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
#include <algorithm>
#include <cctype>
#include <opencv2/opencv.hpp>   /* ARM CPU 跑 mobilefacenet 用 cv::dnn */
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
/* 录入目标区：屏幕中央约 30% 区域（人脸须置于此才允许录入；比原先 50% 更严、更明确） */
#define ENROLL_ROI_X0 ((int)(0.35f * CAM0_W))
#define ENROLL_ROI_Y0 ((int)(0.35f * CAM0_H))
#define ENROLL_ROI_W  ((int)(0.30f * CAM0_W))
#define ENROLL_ROI_H  ((int)(0.30f * CAM0_H))
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

/* 人脸识别（ARM CPU / OpenCV dnn 跑 mobilefacenet）：最新被识别帧绘到右端 (1081,0) */
static cv::dnn::Net g_rec_net;
static std::vector<std::pair<std::string, std::vector<float>>> g_gallery;  /* name + embedding */
static cv::Mat        g_rec_frame;   /* 最近一次被识别的 320x320 帧(BGR) */
static std::string    g_rec_label;
static float          g_rec_sim = 0;
static int            g_rec_valid = 0;
static std::string    g_rec_printed;  /* 已播报的识别标签（人脸丢失时由主循环清空，便于重现时重播） */
static int            g_out_C = 0;   /* 检测模型输出通道数（确认是否含关键点） */
static float          g_rec_sim_thr = 0.40f;  /* 识别相似度阈值(REC_SIM)：余弦相似度 >= 此值才认出某人 */
static std::string    g_gallery_path;  /* REC_GALLERY 路径，录入时追加写 */
static std::string    g_pending_name;  /* 终端输入的待录入姓名（非空即触发录入） */
static std::string    g_enroll_once;   /* --enroll 一次性录入模式的目标姓名 */
static int            g_enroll_arming = 0;   /* --enroll 交互模式：等待用户按 Enter 才开始录入 */
static std::string    g_enroll_arm_name;     /* 待开始录入的姓名 */
static int            g_enroll_n = 0;  /* 录入累计帧数（非连续，中心稳定即可） */
static std::vector<float> g_enroll_sum;/* 累计 embedding 之和（末了求平均） */
static int            g_enroll_cx = 0, g_enroll_cy = 0, g_enroll_refbw = 0; /* 中心稳定基准 */
static int            g_enroll_last = 0; /* 上一帧采集到的帧号（用于超时复位） */
static int            g_enrolled_ok = 0; /* 本次运行实际录入成功次数 */

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
    if (getenv("VID_PLANE")) def_vid = (uint32_t)atoi(getenv("VID_PLANE"));   /* 下层视频平面 ID（叠放顺序异常时覆盖） */
    if (getenv("GFX_PLANE")) def_gfx = (uint32_t)atoi(getenv("GFX_PLANE"));   /* 上层叠加/画框平面 ID */

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

/* 在 GFX 帧缓冲上画矩形（坐标已是 1080x1080 主相机空间）。
 * ARGB8888 内存字节序为 B,G,R,A。 */
static void draw_rect_argb(uint8_t* base, int pitch, int x, int y, int w, int h, uint8_t R, uint8_t G, uint8_t B, uint8_t A)
{
    auto setpx = [&](int px, int py) {
        if (px < 0 || py < 0 || px >= CAM0_W || py >= CAM0_H) return;
        uint8_t* p = base + (size_t)py * pitch + (size_t)px * 4;
        p[0] = B; p[1] = G; p[2] = R; p[3] = A;
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

/* 绿框：检测/识别到的人脸（真正绿色 R=0,G=255,B=0） */
static void draw_box_argb(uint8_t* base, int pitch, int x, int y, int w, int h)
{
    draw_rect_argb(base, pitch, x, y, w, h, 0, 255, 0, 255);
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
static void blit_bgr_to_gfx(uint8_t* gfx, int gpitch, const cv::Mat& img, int ox, int oy);  /* 前向声明：识别缩略图贴 GFX */
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
    /* 录入(含待开始)时画白色目标区，提示人脸须放屏幕中央 */
    if (!g_enroll_once.empty() || !g_pending_name.empty() || g_enroll_arming)
        draw_rect_argb((uint8_t*)gfx_map[g_fi], (int)gfx_pitch[g_fi],
                       ENROLL_ROI_X0, ENROLL_ROI_Y0, ENROLL_ROI_W, ENROLL_ROI_H, 255, 255, 255, 200);
    /* 右端缩略图已关闭：识别结果改由日志打印（见 do_recognition 的 [rec] 识别到:） */
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

struct Det { float x1, y1, x2, y2, score; float lmk[5][2]; int has_lmk; };

/* 关键点合理性校验：模型输出关键点时，要求 5 点都大致落在检测框内。
 * 误检/噪声框的关键点常常跑到框外或乱飞，借此过滤（真实人脸的 5 点必在框内）。 */
static bool lmk_in_box(const Det& d)
{
    if (!d.has_lmk) return false;
    float mx = (d.x2 - d.x1) * 0.6f, my = (d.y2 - d.y1) * 0.6f;
    for (int k = 0; k < 5; k++) {
        float lx = d.lmk[k][0], ly = d.lmk[k][1];
        if (lx < d.x1 - mx || lx > d.x2 + mx || ly < d.y1 - my || ly > d.y2 + my) return false;
    }
    return true;
}

/* ---------------- 人脸识别（ARM CPU / OpenCV dnn） ---------------- */
static float cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty()) return -1.0f;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na <= 0 || nb <= 0) return -1.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

/* gallery 文件：每行 "名字 f0 f1 ... fN"（embedding） */
static void load_gallery(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "[rec] 无 gallery 文件 %s，识别仅输出 Unknown\n", path.c_str()); return; }
    char line[16384];
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        char name[256] = {0}; int ni = 0;
        while (*p && !isspace((unsigned char)*p) && ni < 255) name[ni++] = *p++;
        name[ni] = 0;
        std::vector<float> emb; float v; int n = 0;
        while (*p) {
            if (sscanf(p, "%f%n", &v, &n) == 1) { emb.push_back(v); p += n; } else p++;
        }
        if (!emb.empty()) g_gallery.emplace_back(name, emb);
    }
    fclose(f);
    fprintf(stderr, "[rec] gallery 载入 %zu 条\n", g_gallery.size());
}

/* 把 BGR(cv::Mat) 以不透明方式贴到 GFX(ARGB) 缓冲的 (ox,oy) */
static void blit_bgr_to_gfx(uint8_t* gfx, int gpitch, const cv::Mat& img, int ox, int oy)
{
    int w = img.cols, h = img.rows;
    if (w <= 0 || h <= 0) return;
    for (int y = 0; y < h; y++) {
        const uint8_t* s = img.data + (size_t)y * img.step;
        uint8_t* d = gfx + ((size_t)(oy + y) * gpitch + (size_t)ox * 4);
        for (int x = 0; x < w; x++) {
            const uint8_t* p = s + (size_t)x * 3;
            d[x*4+0] = p[0]; d[x*4+1] = p[1]; d[x*4+2] = p[2]; d[x*4+3] = 255;  /* B,G,R,A */
        }
    }
}

/* 仿射对齐112x112 → mobilefacenet → 返回 L2 归一化 embedding（失败返回空） */
static std::vector<float> compute_embedding(const uint8_t* bgr320, int W, int H, const Det& best)
{
    if (g_rec_net.empty()) return {};
    cv::Mat bgr_mat(H, W, CV_8UC3, (void*)bgr320);
    cv::Mat face112;
    if (best.has_lmk) {
        /* 关键点归一为 [左眼,右眼,鼻,左嘴角,右嘴角]：最上两点为眼(按x排序)，最下两点为嘴(按x排序)，余下为鼻 */
        int order[5] = {0,1,2,3,4};
        std::sort(order, order + 5, [&](int a, int b){ return best.lmk[a][1] < best.lmk[b][1]; });
        if (best.lmk[order[0]][0] > best.lmk[order[1]][0]) std::swap(order[0], order[1]);
        if (best.lmk[order[3]][0] > best.lmk[order[4]][0]) std::swap(order[3], order[4]);
        float lm[5][2];
        for (int i = 0; i < 5; i++) { lm[i][0] = best.lmk[order[i]][0]; lm[i][1] = best.lmk[order[i]][1]; }

        static const float ref[5][2] = { {38.2946f,51.6963f},{73.5318f,51.6963f},{56.0252f,102.453f},{41.5493f,165.729f},{70.7299f,165.729f} };
        std::vector<cv::Point2f> src, dst;
        for (int i = 0; i < 5; i++) { src.emplace_back(lm[i][0], lm[i][1]); dst.emplace_back(ref[i][0], ref[i][1]); }
        cv::Mat M = cv::estimateAffinePartial2D(src, dst);
        if (M.empty()) return {};
        cv::warpAffine(bgr_mat, face112, M, cv::Size(112,112), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    } else {
        /* 无关键点：用检测框中心裁剪到 112x112（对齐精度低，但能正常录入/识别） */
        int cx = (int)((best.x1 + best.x2) * 0.5f);
        int cy = (int)((best.y1 + best.y2) * 0.5f);
        int side = (int)std::max(best.x2 - best.x1, best.y2 - best.y1);
        if (side <= 0) return {};
        int x0 = cx - side/2, y0 = cy - side/2;
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        int x1 = x0 + side, y1 = y0 + side;
        if (x1 > W) { x1 = W; x0 = x1 - side; }
        if (y1 > H) { y1 = H; y0 = y1 - side; }
        cv::Rect r(x0, y0, x1 - x0, y1 - y0);
        if (r.width <= 0 || r.height <= 0) return {};
        cv::resize(bgr_mat(r), face112, cv::Size(112,112), 0, 0, cv::INTER_LINEAR);
    }

    /* 预处理 (img-127.5)/128，转 RGB（mobilefacenet 训练域） */
    cv::Mat rgb; cv::cvtColor(face112, rgb, cv::COLOR_BGR2RGB);
    cv::Mat blob = cv::dnn::blobFromImage(rgb, 1.0f/128.0f, cv::Size(), cv::Scalar(127.5f,127.5f,127.5f), false, false);
    g_rec_net.setInput(blob);
    cv::Mat feat = g_rec_net.forward();
    cv::Mat fv32; feat.convertTo(fv32, CV_32F); fv32 = fv32.reshape(1, 1);
    cv::normalize(fv32, fv32, 1.0, 0.0, cv::NORM_L2);
    return std::vector<float>((float*)fv32.data, (float*)fv32.data + fv32.total());
}

/* 检测到脸且有5关键点：取embedding → 与gallery比对 → 存缩略图 */
static void do_recognition(const uint8_t* bgr320, int W, int H, const Det& best, int dbg)
{
    if (g_rec_net.empty()) return;
    std::vector<float> fv = compute_embedding(bgr320, W, H, best);
    if (fv.empty()) return;

    g_rec_label = "Unknown"; g_rec_sim = -1.0f;
    float best_sim = -1.0f; std::string best_name;
    for (auto& g : g_gallery) {
        float s = cosine(g.second, fv);
        if (s > best_sim) { best_sim = s; best_name = g.first; }
    }
    float sim_thr = g_rec_sim_thr;
    if (!g_gallery.empty() && best_sim >= sim_thr) { g_rec_label = best_name; g_rec_sim = best_sim; }

    /* 识别到已知人时打印一次（标签变化才打，避免每帧刷屏）；人脸确实丢失(detected==0)时由主循环清空 g_rec_printed，
     * 这样同一个人每次重新出现都会重新播报。 */
    if (g_rec_label != "Unknown") {
        if (g_rec_label != g_rec_printed)
            printf("[rec] 识别到: %s (sim=%.2f)\n", g_rec_label.c_str(), g_rec_sim);
        g_rec_printed = g_rec_label;
    } else {
        g_rec_printed.clear();
    }

    cv::Mat bgr_mat(H, W, CV_8UC3, (void*)bgr320);
    g_rec_frame = bgr_mat.clone();   /* 当前检测帧（缩略图已关闭，仅作调试保留） */
    g_rec_valid = 1;
    if (dbg) {
        float nn = 0; for (float v : fv) nn += v * v;
        printf("[rec] sim=%.3f label=%s feat_norm=%.3f\n", g_rec_sim, g_rec_label.c_str(), sqrtf(nn));
    }
}

/* 录入：把 name + embedding 追加写进 gallery 文件，并加入内存库（后续立即可识别） */
static void enroll_face(const std::string& name, const std::vector<float>& fv)
{
    if (name.empty() || fv.empty()) return;
    g_gallery.emplace_back(name, fv);
    FILE* f = fopen(g_gallery_path.c_str(), "a");
    if (!f) { fprintf(stderr, "[enroll] 无法写 gallery %s\n", g_gallery_path.c_str()); return; }
    fprintf(f, "%s", name.c_str());
    for (float v : fv) fprintf(f, " %.6g", v);
    fprintf(f, "\n");
    fclose(f);
    fprintf(stderr, "[enroll] 已录入 '%s' (dim=%zu)，当前库 %zu 人\n", name.c_str(), fv.size(), g_gallery.size());
    g_enrolled_ok++;
}

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

    /* 一次性录入模式: face_detect_dpu --enroll <姓名> <模型> [相机(默认/dev/video1)] */
    int enroll_mode = 0;
    std::string enroll_name;
    if (argc >= 3 && std::string(argv[1]) == "--enroll") {
        enroll_mode = 1; enroll_name = argv[2];
    }

    if ((!enroll_mode && (argc < 2 || is_digits(argv[1]))) || (enroll_mode && argc < 3)) {
        fprintf(stderr,
            "用法:\n"
            "  显示+识别 : face_detect_dpu <模型目录或.xmodel> [相机(默认/dev/video1)] [阈值]\n"
            "  录入人脸 : face_detect_dpu --enroll <姓名> <模型> [相机]\n"
            "示例:\n"
            "  face_detect_dpu /usr/share/vart/face_detector/gate320.xmodel /dev/video1\n"
            "  face_detect_dpu --enroll Zhang_San /usr/share/vart/face_detector/gate320.xmodel /dev/video1\n"
            "注意: 平面 ID(如 39 40)由程序自动探测，不要再手动传。\n");
        return 2;
    }

    const char* model_dir = enroll_mode ? argv[3] : argv[1];
    const char* cam_dev  = enroll_mode ? ((argc > 4) ? argv[4] : "/dev/video1")
                                      : ((argc > 2) ? argv[2] : "/dev/video1");
    if (enroll_mode) g_enroll_once = enroll_name;

    /* ============================================================================
     *  全部可配置参数
     * ----------------------------------------------------------------------------
     *  一、命令行参数
     *    显示+识别 : face_detect_dpu <模型(.xmodel或目录)> [相机(默认/dev/video1)] [检测阈值]
     *    录入人脸 : face_detect_dpu --enroll <姓名> <模型> [相机]
     *    注：非录入模式的第 3 个位置参数等价于 FACE_THR（环境变量优先于位置参数）。
     *
     *  二、环境变量（均可不重编直接覆盖；启动时"配置"区会打印生效值）
     *    变量        默认                              说明
     *    ----------- --------------------------------- -------------------------------
     *    FACE_THR    0.40     检测/绿框门槛；分数≥它才算"有人脸"，也决定要不要跑识别(mobilefacenet)
     *    ENROLL_THR  0.55     录入分数门槛；背景噪声≤0.45、真脸≥0.6，取中间：噪声被挡、真脸必过
     *    REC_SIM     0.40     识别余弦相似度决策阈值；≥它才认出某人，否则判 Unknown（调高更严）
     *    DET_OFF     未设置   去抖迟滞开关：设置(非空)=关(每帧直接判定)，未设置=开(连续N帧确认)
     *    DET_NDET    1        去抖开时，连续多少帧有脸才确认"出现"
     *    DET_NLOST   5        去抖开时，连续多少帧无脸才确认"消失"
     *    DET_SMOOTH  0.35     绿框时间平滑(EMA)系数：越大跟手越紧、越小越稳；0=关平滑(画原始框)
     *    REC_MODEL   /root/tmp/face_rec/deploy/mobilefacenet.onnx   识别模型(.onnx)
     *    REC_GALLERY /root/tmp/face_rec/gallery.txt                已知人库(录入时追加写)
     *    VID_PLANE   39        VID(下层视频) DRM 平面 ID；叠放顺序异常时改
     *    GFX_PLANE   40        GFX(上层叠加/画框) DRM 平面 ID
     *    VID_ZC      未设置    设置即启用 video0 零拷贝(DP dumb buffer → dmabuf)
     *    DBG         未设置    设置即打印每帧识别细节(sim / feat_norm)
     * ========================================================================== */
    int dbg = getenv("DBG") ? 1 : 0;
    float score_thr = 0.40f;
    if (!enroll_mode && argc > 3) score_thr = (float)atof(argv[3]);
    if (getenv("FACE_THR")) score_thr = (float)atof(getenv("FACE_THR"));   /* 环境变量始终优先 */
    /* 录入门槛：仅当检测分数 >= 此值才允许录入。背景噪声最高约 0.45，真实人脸一般 >=0.60，
     * 故默认取中间值 0.55（噪声被挡、真脸必过）。可用 ENROLL_THR 环境变量覆盖。 */
    float enroll_thr = getenv("ENROLL_THR") ? (float)atof(getenv("ENROLL_THR")) : 0.55f;
    g_rec_sim_thr = getenv("REC_SIM") ? (float)atof(getenv("REC_SIM")) : 0.40f;   /* 识别相似度阈值(全局，供打印与识别共用) */

    /* 去抖迟滞（我主动加的稳定逻辑）可配：DET_OFF=1 时去掉迟滞（仍受 score_thr 控制）；
     * 否则连续 N_DET 帧确认出现 / N_LOST 帧确认消失。注意看值，DET_OFF=0 才真正关。 */
    int det_off = (getenv("DET_OFF") && atoi(getenv("DET_OFF")) != 0) ? 1 : 0;
    int n_det  = getenv("DET_NDET")  ? atoi(getenv("DET_NDET"))  : 1;   /* 默认 1 帧即出框，避免"没绿框" */
    int n_lost = getenv("DET_NLOST") ? atoi(getenv("DET_NLOST")) : 5;
    /* 框时间平滑系数(EMA)：越大跟手越紧、越小越稳；0 表示关平滑（画原始框） */
    float smooth_a = getenv("DET_SMOOTH") ? (float)atof(getenv("DET_SMOOTH")) : 0.35f;

    /* 人脸识别模型（mobilefacenet.onnx）与 gallery 加载（ARM CPU / OpenCV dnn） */
    std::string rec_model = getenv("REC_MODEL") ? getenv("REC_MODEL") : "/root/tmp/face_rec/deploy/mobilefacenet.onnx";
    try { g_rec_net = cv::dnn::readNetFromONNX(rec_model); }
    catch (...) { g_rec_net = cv::dnn::Net(); }
    if (g_rec_net.empty())
        fprintf(stderr, "[rec] 警告: 未加载识别模型 %s（将只检测，不识别）\n", rec_model.c_str());
    else
        fprintf(stderr, "[rec] 识别模型已加载: %s\n", rec_model.c_str());
    g_gallery_path = getenv("REC_GALLERY") ? getenv("REC_GALLERY") : "/root/tmp/face_rec/gallery.txt";
    load_gallery(g_gallery_path);

    fprintf(stderr, "[DBG-VER] face_detect_dpu MERGE_GFX (单进程: 双路同流 + GFX 叠层画框)\n");
    if (enroll_mode)
        fprintf(stderr,
            "==== 配置(录入模式) ====\n"
            "  模型     : %s\n  相机     : %s\n  录入姓名 : %s\n"
            "  FACE_THR : %.2f\n  ENROLL_THR: %.2f (录入门槛)\n  REC_SIM  : %.2f (识别相似度阈值)\n"
            "  识别模型 : %s\n  gallery  : %s (%zu 人)\n"
            "========================\n",
            model_dir, cam_dev, enroll_name.c_str(),
            score_thr, enroll_thr, g_rec_sim_thr,
            rec_model.c_str(), g_gallery_path.c_str(), g_gallery.size());
    else
        fprintf(stderr,
            "==== 配置 ====\n"
            "  模型     : %s\n  相机     : %s\n  FACE_THR : %.2f\n  ENROLL_THR: %.2f (录入门槛)\n"
            "  DET_OFF  : %d (0=去抖开)\n  DET_NDET : %d  DET_NLOST: %d\n"
            "  DET_SMOOTH: %.2f (框平滑系数,0=关)\n"
            "  VID_ZC   : %d\n  DBG      : %d\n"
            "  识别模型 : %s%s\n  gallery  : %s (%zu 人)\n  REC_SIM  : %.2f (识别相似度阈值)\n"
            "  右端缩略图: 已关闭(识别结果打印到日志 [rec] 识别到:)\n"
            "====================\n",
            model_dir, cam_dev, score_thr, enroll_thr,
            det_off, n_det, n_lost,
            smooth_a,
            (getenv("VID_ZC") ? 1 : 0), dbg,
            rec_model.c_str(), g_rec_net.empty() ? " (未加载!)" : "",
            g_gallery_path.c_str(), g_gallery.size(),
            g_rec_sim_thr);

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* === 显示初始化 + 两路相机同时 STREAMON（在慢速 DPU 编译之前，确保双路同流锁定时序） ===
     * 录入模式也照常显示（便于对准人脸）；DRM 不可用时降级为无显示运行（仍可检测/录入）。 */
    if (drm_init() < 0)
        fprintf(stderr, "warn: DRM 初始化失败，将无显示运行（仍可检测/录入；如需画面请确认 root 与 /dev/dri/card0）\n");
    else if (make_fbs() < 0) {
        fprintf(stderr, "warn: make_fbs 失败，无显示运行\n"); drm_fd = -1;
    } else if (v0_open() < 0) {
        fprintf(stderr, "warn: video0 打开失败，无显示视频源（DRM 仍可用）\n");
    }
    if (drm_fd >= 0)
        fprintf(stderr, "[DBG] drm: fd=%d crtc=%u conn=%u vid_plane=%u gfx_plane=%u\n",
                drm_fd, g_crtc_id, g_conn_id, g_vid_plane_id, g_gfx_plane_id);

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

    /* === 点亮 CRTC（在 GFX 透明 fb 上）+ 关 GFX 的 global alpha（启用逐像素 alpha） ===
     * 仅在 CRTC 当前未激活时才 SetCrtc：避免与板子已有的显示管理(控制台/Weston)冲突或
     * 因缺少 DRM master 而失败。若 CRTC 已激活，跳过即可——下面 SetPlane 仍能叠加显示。 */
    if (drm_fd < 0) {
        fprintf(stderr, "ERROR: drm_fd 无效(%d)，DRM 未初始化成功，无法显示（检查是否 root 且 /dev/dri/card0 存在）\n", drm_fd);
    } else {
        drmModeCrtcPtr cur = drmModeGetCrtc(drm_fd, g_crtc_id);
        int already_on = (cur && cur->mode_valid) ? 1 : 0;
        if (cur) drmModeFreeCrtc(cur);
        if (!already_on) {
            if (drmModeSetCrtc(drm_fd, g_crtc_id, gfx_fb[0], 0, 0, &g_conn_id, 1, &g_mode) < 0)
                fprintf(stderr, "warn: SetCrtc failed: %s (显示可能已由系统接管，SetPlane 仍会叠加)\n", strerror(errno));
            else
                printf("crtc enabled: %dx%d@%u\n", g_screen_w, g_screen_h, g_mode.vrefresh);
        } else {
            printf("crtc 已激活(沿用系统显示)，跳过 SetCrtc\n");
        }
    }
    if (drm_fd >= 0) {
        int r0 = set_plane_prop(g_gfx_plane_id, "g_alpha_en", 0);
        printf("GFX g_alpha_en -> 0 : %s (逐像素 alpha 启用；框外透出视频)\n",
               r0 == 0 ? "ok" : "failed(框可能不显示!)");
    }

    std::vector<uint8_t> rsbuf((size_t)Msz * Msz * 3);
    int frame = 0;
    static int stable = 0, hi_run = 0, lo_run = 0;
    static int32_t last_bx = 0, last_by = 0, last_bw = 0, last_bh = 0;
    static int32_t sm_bx = 0, sm_by = 0, sm_bw = 0, sm_bh = 0, sm_init = 0;  /* 平滑后的框 */
    static int prev_face = 0;
    int g_quit = 0;

    struct pollfd pfd[3];
    pfd[0].fd = g_v0.fd;                        /* video0 显示源（未打开则为 -1，poll 忽略） */
    pfd[0].events = POLLIN;
    pfd[1].fd = cap.fd;                        pfd[1].events = POLLIN;
    int nfd = 2;
    if (enroll_mode) {
        if (isatty(0)) {
            g_enroll_arming = 1; g_enroll_arm_name = enroll_name;
            fprintf(stderr, "[enroll] 将人脸放入白色框内并居中，对准后按 Enter 开始录入 '%s'（输入 q 取消）...\n", enroll_name.c_str());
        } else {
            g_enroll_once = enroll_name;   /* 非交互(重定向)模式：保持原有自动开始 */
            fprintf(stderr, "[enroll] 非交互模式：检测到稳定人脸后自动录入 '%s'...  (Ctrl+C 取消)\n", enroll_name.c_str());
        }
    } else {
        printf("running: detect+display (DPU 每帧, GFX 叠层画框, 去抖=%s)  [Ctrl+C stop]\n",
               det_off ? "关(DET_OFF)" : "开");
        fprintf(stderr, "[enroll] 录入: 在终端输入姓名并回车，对准摄像头即可录入；输入 q 退出程序\n");
    }
    if (isatty(0)) { pfd[2].fd = 0; pfd[2].events = POLLIN; nfd = 3; }
    while (!g_stop) {
        int n = poll(pfd, nfd, 1000);
        if (n < 0) { if (errno == EINTR) continue; perror("poll"); break; }
        if (n == 0) continue;

        /* 终端录入：读姓名行 → 置 g_pending_name 触发录入；q 退出 */
        if (nfd == 3 && (pfd[2].revents & POLLIN)) {
            static char inbuf[1024]; static size_t inlen = 0;
            char tmp[256]; ssize_t r = read(0, tmp, sizeof(tmp));
            if (r > 0) {
                for (ssize_t i = 0; i < r; i++) {
                    if (tmp[i] == '\n' || tmp[i] == '\r') {
                        inbuf[inlen] = 0; inlen = 0;
                        std::string line(inbuf);
                        size_t a = line.find_first_not_of(" \t\r\n");
                        size_t b = line.find_last_not_of(" \t\r\n");
                        if (a != std::string::npos) line = line.substr(a, b - a + 1);
                        if (line == "q" || line == "quit" || line == "exit") {
                            if (g_enroll_arming) { g_enroll_arming = 0; g_enroll_arm_name.clear(); fprintf(stderr, "[enroll] 已取消录入\n"); }
                            else g_quit = 1;
                        } else if (g_enroll_arming) {
                            /* 按 Enter(任意输入) 即开始录入 */
                            g_enroll_once = g_enroll_arm_name; g_enroll_arming = 0;
                            g_enroll_n = 0; g_enroll_sum.clear();
                            fprintf(stderr, "[enroll] 开始录入 '%s'：保持脸在白色框内，采集中... (Ctrl+C 取消)\n", g_enroll_once.c_str());
                        } else if (!line.empty()) {
                            g_pending_name = line; g_enroll_n = 0; g_enroll_sum.clear();
                            fprintf(stderr, "[enroll] 对准摄像头，正在录入 '%s' (中心稳定即可，无需静止)...\n", line.c_str());
                        }
                    } else if (inlen < sizeof(inbuf) - 1) inbuf[inlen++] = tmp[i];
                }
            }
        }
        if (g_quit) break;

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
                    g_out_C = C;   /* 记录输出通道数，供识别分支判断是否有关键点 */
                    const float CONF = 0.05f;
                    auto ch = [&](int c, int yy, int xx) -> float {
                        return nhwc ? d[((size_t)yy*Wd + xx)*C + c]
                                    : d[((size_t)c*Hd + yy)*Wd + xx];
                    };
                    for (int y = 0; y < Hd; y++) {
                        for (int x = 0; x < Wd; x++) {
                            float score = sigmoidf(ch(0, y, x));
                            if (score < CONF) continue;
                            float l = ch(1, y, x), t = ch(2, y, x), r = ch(3, y, x), b = ch(4, y, x);
                            float cx = (x + 0.5f) * stride, cy = (y + 0.5f) * stride;
                            Det det;
                            det.x1 = cx - l * stride; det.y1 = cy - t * stride;
                            det.x2 = cx + r * stride; det.y2 = cy + b * stride;
                            det.score = score; det.has_lmk = 0;
                            if (C >= 15) {   /* score+4框+10点=15 通道：取 5 关键点 */
                                for (int kk = 0; kk < 5; kk++) {
                                    det.lmk[kk][0] = cx + ch(5 + 2*kk, y, x) * stride;
                                    det.lmk[kk][1] = cy + ch(5 + 2*kk + 1, y, x) * stride;
                                }
                                det.has_lmk = 1;
                            }
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

                /* 人脸识别：仅在确有把握的人脸(score>=阈值，与绿框同门槛)才跑，避免误检框被认成已知人 */
                if (best.score >= score_thr) do_recognition(bgr, (int)Msz, (int)Msz, best, dbg);

                /* 录入：name 已设定(--enroll 一次性 / 运行时终端输入)后开始采集。
                 * 不要求连续帧、不要求绝对静止：累计若干帧(中心漂移不大即可)，平均 embedding 后写入 gallery。
                 * 注意：录入必须有识别模型(mobilefacenet.onnx)才能提特征。 */
                if (!g_pending_name.empty() || !g_enroll_once.empty()) {
                    if (g_rec_net.empty()) {
                        if (!g_enroll_once.empty()) {
                            fprintf(stderr, "[enroll] 错误: 录入需要识别模型 %s，请先放置后重试（程序继续以检测模式运行）\n", rec_model.c_str());
                            g_enroll_once.clear();
                        } else {
                            static int w = 0; if (!w) { fprintf(stderr, "[enroll] 错误: 录入需要识别模型(mobilefacenet.onnx)，请先放置\n"); w = 1; }
                        }
                    } else if (best.score >= enroll_thr && best.x2 > best.x1 && best.y2 > best.y1) {
                        /* 用(平滑后的)框中心做稳定基准，比原始抖动框稳得多 */
                        int cx, cy, bw;
                        if (smooth_a > 0.0f && sm_init) { cx = sm_bx + sm_bw/2; cy = sm_by + sm_bh/2; bw = sm_bw; }
                        else { cx = (int)(((best.x1 + best.x2) * 0.5f) * sc); cy = (int)(((best.y1 + best.y2) * 0.5f) * sc); bw = (int)((best.x2 - best.x1) * sc); }
                        /* 误检过滤：人脸框尺寸须合理（太小=噪声/太远，太大=异常）；模型有关键点时要求 5 关键点落在框内 */
                        int bw_ok  = (bw >= (int)(0.12f * CAM0_W) && bw <= (int)(0.70f * CAM0_W));
                        int lmk_ok = (g_out_C < 15) ? 1 : (lmk_in_box(best) ? 1 : 0);
                        /* 人脸须在录入目标区(屏幕中央)内才采集；否则提示并把脸移过去，重置累计 */
                        int in_roi = (cx >= ENROLL_ROI_X0 && cx <= ENROLL_ROI_X0 + ENROLL_ROI_W &&
                                      cy >= ENROLL_ROI_Y0 && cy <= ENROLL_ROI_Y0 + ENROLL_ROI_H);
                        if (!bw_ok || !lmk_ok) {
                            static int bad_w = 0;
                            if (frame - bad_w > 30) { fprintf(stderr, "[enroll] 检测框不像人脸(尺寸/关键点异常)，已忽略；请正脸对准镜头\n"); bad_w = frame; }
                            g_enroll_n = 0; g_enroll_sum.clear();
                        } else if (!in_roi) {
                            static int roi_w = 0;
                            if (frame - roi_w > 30) { fprintf(stderr, "[enroll] 请把人脸移到屏幕中央白色框内再录入(当前位置偏了)\n"); roi_w = frame; }
                            g_enroll_n = 0; g_enroll_sum.clear();
                        } else {
                            /* 超时复位：>3s 没采到有效帧则重新开始累计（frame 约 30fps → 90帧） */
                            if (g_enroll_n > 0 && (frame - g_enroll_last) > 90) { g_enroll_n = 0; g_enroll_sum.clear(); }
                            if (g_enroll_n == 0) {       /* 首次/复位：建立中心基准 */
                                g_enroll_cx = cx; g_enroll_cy = cy; g_enroll_refbw = bw; g_enroll_sum.clear();
                            } else {                      /* 中心漂移过大 → 视为换脸/大幅移动，复位 */
                                int d = (int)sqrt((double)(cx-g_enroll_cx)*(cx-g_enroll_cx) + (double)(cy-g_enroll_cy)*(cy-g_enroll_cy));
                                int tol = (int)(0.4f * (g_enroll_refbw > 0 ? g_enroll_refbw : 200)) + 8;
                                if (d > tol) { g_enroll_n = 0; g_enroll_sum.clear(); g_enroll_cx = cx; g_enroll_cy = cy; g_enroll_refbw = bw; }
                            }
                            std::vector<float> efv = compute_embedding(bgr, (int)Msz, (int)Msz, best);
                            if (!efv.empty()) {
                                if (g_enroll_sum.empty()) g_enroll_sum = efv;
                                else for (size_t i = 0; i < efv.size(); i++) g_enroll_sum[i] += efv[i];
                                g_enroll_n++; g_enroll_last = frame;
                                fprintf(stderr, "[enroll] 采集中 %d/5 (中心稳定即可，无需静止)\n", g_enroll_n);
                                if (g_enroll_n >= 5) {
                                    for (float& v : g_enroll_sum) v /= g_enroll_n;
                                    std::string nm = g_enroll_once.empty() ? g_pending_name : g_enroll_once;
                                    enroll_face(nm, g_enroll_sum);
                                    fprintf(stderr, "[enroll] 已录入 '%s'，程序继续以检测+识别模式运行（Ctrl+C 退出；同人再出现将显示 label）\n", nm.c_str());
                                    g_pending_name.clear(); g_enroll_once.clear(); g_enroll_n = 0; g_enroll_sum.clear();
                                }
                            }
                        }
                    }
                    /* 无人脸帧：不累计也不复位（等待下一帧） */
                }
                if (g_out_C > 0 && g_out_C < 15) {
                    static int w = 0;
                    if (!w) { fprintf(stderr, "[rec] 检测模型输出通道=%d(<15)，未含5关键点，识别已跳过\n", g_out_C); w = 1; }
                }
                if (det_off) {
                    /* 关去抖：当帧直接判定，但仍受 score_thr 控制（FACE_THR 生效） */
                    stable = (best.score >= score_thr) ? 1 : 0;
                    hi_run = lo_run = 0;
                } else {
                    if (best.score >= score_thr) { hi_run++; lo_run = 0; }
                    else { lo_run++; hi_run = 0; }
                    if (!stable && hi_run >= n_det) stable = 1;
                    if (stable && lo_run >= n_lost) stable = 0;
                }
                int detected = stable;
                if (!detected) g_rec_printed.clear();   /* 人脸确实丢失：复位识别播报，下次同人重现会重新打印 */
                /* 框时间平滑：EMA + 每帧限幅。
                 * EMA 压小抖动；限幅防止"原始框突然远跳/误检"导致显示框大幅漂移或瞬移。 */
                if (best.score >= score_thr && smooth_a > 0.0f) {
                    if (!sm_init) { sm_bx = bx; sm_by = by; sm_bw = bw; sm_bh = bh; sm_init = 1; }
                    else {
                        float nbx = smooth_a * bx + (1.0f - smooth_a) * sm_bx;
                        float nby = smooth_a * by + (1.0f - smooth_a) * sm_by;
                        float nbw = smooth_a * bw + (1.0f - smooth_a) * sm_bw;
                        float nbh = smooth_a * bh + (1.0f - smooth_a) * sm_bh;
                        int base = sm_bw > 0 ? sm_bw : 200;
                        int maxmv = (int)(0.12f * base) + 4;   /* 每帧中心最多移动 ~12% 框宽 */
                        int maxgw = (int)(0.12f * base) + 4;   /* 每帧尺寸最多变化 ~12% 框宽 */
                        int d;
                        d = (int)nbx - sm_bx; if (d > maxmv) d = maxmv; else if (d < -maxmv) d = -maxmv; sm_bx += d;
                        d = (int)nby - sm_by; if (d > maxmv) d = maxmv; else if (d < -maxmv) d = -maxmv; sm_by += d;
                        d = (int)nbw - sm_bw; if (d > maxgw) d = maxgw; else if (d < -maxgw) d = -maxgw; sm_bw += d;
                        d = (int)nbh - sm_bh; if (d > maxgw) d = maxgw; else if (d < -maxgw) d = -maxgw; sm_bh += d;
                    }
                }
                if (detected) {
                    if (smooth_a > 0.0f && sm_init) { g_bx = sm_bx; g_by = sm_by; g_bw = sm_bw; g_bh = sm_bh; }
                    else { g_bx = last_bx; g_by = last_by; g_bw = last_bw; g_bh = last_bh; }
                } else {
                    sm_init = 0;   /* 完全消失后复位，下次出现直接对齐避免残影拖移 */
                }
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

    if (enroll_mode) {
        if (g_enrolled_ok > 0)
            fprintf(stderr, "录入完成：%s（已写入 %s，共 %d 人）\n", enroll_name.c_str(), g_gallery_path.c_str(), g_enrolled_ok);
        else
            fprintf(stderr, "未录入：未检测到稳定人脸，已取消（gallery 未改动）\n");
    }
    printf("face_detect_dpu 退出\n");
    return 0;
}
