/*
 * drm_dual_plane.c —— 双摄像头双图层显示（阶段 1）
 *
 * ZynqMP DP 硬约束（见内核 zynqmp_disp.c）：
 *   **两块显示图层（VID + GFX）的 src 裁剪尺寸必须完全相同。**
 * 因此两块层都设为整屏 1920x1080，各自在 fb 内摆放内容：
 *
 *   video0 (1080x1080) -> VID/下层(plane 39, RGB888 无 alpha)：
 *       整屏 fb 1920x1080，相机放左上 (0,0)，右 840 列留黑（透出下层背景）。
 *   video1 (320x320)  -> GFX/上层(plane 40, ARGB8888 带 alpha)：
 *       整屏 fb 1920x1080，相机放 (1081,0) 处 320x320 实心，其余像素 alpha=0（透明）。
 *   两块层 src 均 = 整屏 1920x1080 -> 满足硬件“两层同尺寸”约束。
 *
 * video0 默认 MMAP(CPU 拷贝，保证不撕裂)；环境变量 DP_DMABUF=1 可走 DMABUF 零拷贝（RGB888）；
 * video1 相机先零拷贝进 RGB888 暂存 fb（320x320），每帧转 ARGB 贴到整屏 fb 的 (1081,0)。
 * 任一路 DMABUF 导入失败自动回退 MMAP(CPU 拷贝)。
 *
 * 用法：./drm_dual_plane [vid_plane] [gfx_plane]
 * 阶段2：算法把人脸框（1080x1080 坐标）写入共享内存 /face_boxes，本程序在 video0 上画矩形（无文字）。
 *        共享内存路径可用环境变量 FACE_SHM 覆盖；协议见 face_boxes.h。
 */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <poll.h>

#include <libdrm/drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <linux/videodev2.h>
#include <linux/dma-buf.h>
#include <libdrm/drm.h>
#include "face_boxes.h"

#define DRM_DEV    "/dev/dri/card0"
#define V4L_DEV0   "/dev/video0"
#define V4L_DEV1   "/dev/video1"

#define SCREEN_W 1920
#define SCREEN_H 1080
#define CAM0_W 1080
#define CAM0_H 1080
#define CAM1_W 320
#define CAM1_H 320
#define CAP1_OFF_X 1081      /* 320x320 小窗在整屏 fb 中的左上 x */
#define CAP1_OFF_Y 0

#define V4L_FMT  V4L2_PIX_FMT_BGR24      /* 相机原样输出；本 DP 硬件按 B,G,R 解释 */
#define VID_FMT  DRM_FORMAT_RGB888       /* video0 的 VID 层 fb 格式（无 alpha）*/
#define GFX_FMT  DRM_FORMAT_ARGB8888     /* video1 的 GFX 层 fb 格式（带 alpha 透明）*/

#define NBUF_FB 8
#define BOX_THICK 3       /* 人脸框线宽（像素）*/

/* 阶段2：共享内存人脸框（算法侧写入，显示侧读取） */
static face_boxes_t *g_boxes = NULL;
static face_box_t g_prev[NBUF_FB][FACE_BOXES_MAX];
static int g_prev_count[NBUF_FB] = {0};
static int g_stage2 = 0;       /* 1=阶段2：GFX 整屏覆盖 video0 画框，关闭 320 小窗 */
static int g_gfx_fi = 0;       /* 阶段2：GFX fb 轮转索引 */

static volatile sig_atomic_t g_run = 1;
static void on_signal(int s) { (void)s; g_run = 0; }

static int drm_fd = -1;
static uint32_t g_crtc_id = 0, g_conn_id = 0;
static uint32_t g_vid_plane_id = 0;
static uint32_t g_gfx_plane_id = 0;
static int g_screen_w = 0, g_screen_h = 0;
static drmModeModeInfo g_mode;

static int set_plane_prop(uint32_t plane_id, const char *name, uint64_t val)
{
    drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE);
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

/* 自动选层：支持 ARGB8888 的 = GFX(上层, video1)；另一可用层 = VID(下层, video0) */
static int drm_init(int want_vid, int want_gfx)
{
    drmModeResPtr res;
    drmModeConnectorPtr conn = NULL;
    drmModePlaneResPtr planes;

    drm_fd = open(DRM_DEV, O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) { perror("open " DRM_DEV); return -1; }

    res = drmModeGetResources(drm_fd);
    if (!res) { fprintf(stderr, "drmModeGetResources failed\n"); return -1; }

    for (int i = 0; i < res->count_connectors && !conn; i++) {
        drmModeConnectorPtr c = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) conn = c;
        else if (c) drmModeFreeConnector(c);
    }
    if (!conn) { fprintf(stderr, "no connected connector\n"); return -1; }

    if (conn->encoder_id) {
        drmModeEncoderPtr enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
        if (enc) { if (enc->crtc_id) g_crtc_id = enc->crtc_id; drmModeFreeEncoder(enc); }
    }
    if (!g_crtc_id && res->count_crtcs > 0) g_crtc_id = res->crtcs[0];
    if (!g_crtc_id) { fprintf(stderr, "no crtc found\n"); return -1; }

    g_conn_id = conn->connector_id;
    memset(&g_mode, 0, sizeof(g_mode));
    for (int i = 0; i < conn->count_modes; i++)
        if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED) { g_mode = conn->modes[i]; break; }
    if (!g_mode.hdisplay) g_mode = conn->modes[0];
    g_screen_w = g_mode.hdisplay;
    g_screen_h = g_mode.vdisplay;
    printf("mode: %dx%d@%u  crtc_id=%u conn_id=%u\n", g_screen_w, g_screen_h, g_mode.vrefresh, g_crtc_id, g_conn_id);

    planes = drmModeGetPlaneResources(drm_fd);
    if (!planes) { fprintf(stderr, "drmModeGetPlaneResources failed\n"); return -1; }

    uint32_t crtc_idx = 0;
    for (int i = 0; i < res->count_crtcs; i++)
        if (res->crtcs[i] == g_crtc_id) crtc_idx = (uint32_t)i;

    uint32_t gfx_cand = 0, vid_cand = 0;
    printf("planes:\n");
    for (uint32_t i = 0; i < planes->count_planes; i++) {
        drmModePlanePtr p = drmModeGetPlane(drm_fd, planes->planes[i]);
        if (!p) continue;
        int usable = (p->possible_crtcs & (1u << crtc_idx)) ? 1 : 0;
        int has_argb = 0, has_rgb = 0;
        for (uint32_t f = 0; f < p->count_formats; f++) {
            if (p->formats[f] == DRM_FORMAT_ARGB8888) has_argb = 1;
            if (p->formats[f] == DRM_FORMAT_RGB888 || p->formats[f] == DRM_FORMAT_BGR888) has_rgb = 1;
        }
        const char *role = has_argb ? "GFX(上层,支持ARGB)" : (has_rgb ? "VID(下层,仅RGB/YUV)" : "?");
        printf("  id=%-3u usable=%d rgb888=%d argb8888=%d -> %s\n", p->plane_id, usable, has_rgb, has_argb, role);
        if (usable) {
            if (has_argb && !gfx_cand) gfx_cand = p->plane_id;
            else if (!vid_cand) vid_cand = p->plane_id;
        }
        drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(planes);

    if (!gfx_cand && !vid_cand) { fprintf(stderr, "no usable plane\n"); return -1; }
    if (!gfx_cand) gfx_cand = vid_cand;
    if (!vid_cand) vid_cand = gfx_cand;

    if (want_gfx > 0) g_gfx_plane_id = (uint32_t)want_gfx; else g_gfx_plane_id = gfx_cand;
    if (want_vid > 0) g_vid_plane_id = (uint32_t)want_vid; else g_vid_plane_id = vid_cand;

    printf("VID(lower,video0) plane_id=%u  GFX(upper,video1) plane_id=%u\n", g_vid_plane_id, g_gfx_plane_id);
    return 0;
}

typedef struct {
    int v4l_fd;
    const char *dev;
    int cam_w, cam_h;          /* 相机分辨率 */
    int disp_w, disp_h;        /* 显示 fb 尺寸（两块均为整屏 1920x1080）*/
    int off_x, off_y;          /* 相机内容在显示 fb 中的左上位置 */
    int plane_id;
    uint32_t disp_fmt;         /* 显示 fb 格式（VID=RGB888, GFX=ARGB8888）*/
    int staging;               /* 1=相机先进 RGB888 暂存 fb 再转（GFX 因格式不同）*/
    int is_gfx;                /* 1=该层为顶层 GFX，额外叠加人脸框 */

    /* 显示 fb */
    uint32_t fb_id[NBUF_FB];
    uint32_t bo_handle[NBUF_FB];
    uint64_t bo_size[NBUF_FB];
    uint32_t bo_pitch[NBUF_FB];
    void *bo_map[NBUF_FB];
    /* 暂存 fb（仅 GFX，RGB888，与相机同尺寸）*/
    uint32_t cap_fb_id[NBUF_FB];
    uint32_t cap_handle[NBUF_FB];
    uint64_t cap_size[NBUF_FB];
    uint32_t cap_pitch[NBUF_FB];
    void *cap_map[NBUF_FB];

    /* v4l2 采集 */
    int use_dmabuf;
    uint32_t vbuf_fd[NBUF_FB];
    void *mmap_buf[NBUF_FB];
    uint32_t mmap_size[NBUF_FB];
    uint32_t v4l_pitch;
} cam_ctx_t;

static cam_ctx_t g_cam0, g_cam1;

static int make_fbs(uint32_t *fb_id, uint32_t *handle, uint64_t *size, uint32_t *pitch,
                    void **map, int w, int h, int bpp, uint32_t fmt)
{
    for (int i = 0; i < NBUF_FB; i++) {
        struct drm_mode_create_dumb creq = {0};
        struct drm_mode_map_dumb mreq = {0};
        int ret;
        creq.width = (uint32_t)w; creq.height = (uint32_t)h; creq.bpp = (uint32_t)bpp;
        ret = drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq);
        if (ret < 0) { perror("CREATE_DUMB"); return -1; }
        handle[i] = creq.handle; size[i] = creq.size; pitch[i] = creq.pitch;
        uint32_t hh[4] = { handle[i], 0, 0, 0 };
        uint32_t pp[4] = { pitch[i], 0, 0, 0 };
        uint32_t oo[4] = { 0, 0, 0, 0 };
        ret = drmModeAddFB2(drm_fd, (uint32_t)w, (uint32_t)h, fmt, hh, pp, oo, &fb_id[i], 0);
        if (ret < 0) { fprintf(stderr, "AddFB2 failed: %s\n", strerror(errno)); return -1; }
        mreq.handle = handle[i];
        ret = drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq);
        if (ret < 0) { perror("MAP_DUMB"); return -1; }
        map[i] = mmap(NULL, size[i], PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, mreq.offset);
        if (map[i] == MAP_FAILED) { perror("mmap"); return -1; }
        memset(map[i], 0, size[i]);
    }
    return 0;
}

static int create_fbs(cam_ctx_t *c)
{
    for (int i = 0; i < NBUF_FB; i++) {
        c->bo_map[i] = MAP_FAILED; c->cap_map[i] = MAP_FAILED; c->mmap_buf[i] = MAP_FAILED;
        c->vbuf_fd[i] = -1;
    }
    int bpp = (c->disp_fmt == DRM_FORMAT_ARGB8888) ? 32 : 24;
    if (make_fbs(c->fb_id, c->bo_handle, c->bo_size, c->bo_pitch, c->bo_map,
                 c->disp_w, c->disp_h, bpp, c->disp_fmt) < 0) return -1;
    printf("fb[%s]: disp %ux%u fmt=0x%x pitch=%u\n", c->dev, c->disp_w, c->disp_h, c->disp_fmt, c->bo_pitch[0]);
    if (c->staging) {
        if (make_fbs(c->cap_fb_id, c->cap_handle, c->cap_size, c->cap_pitch, c->cap_map,
                     c->cam_w, c->cam_h, 24, DRM_FORMAT_RGB888) < 0) return -1;
        printf("fb[%s]: staging %ux%u (RGB888) pitch=%u\n", c->dev, c->cam_w, c->cam_h, c->cap_pitch[0]);
    }
    return 0;
}

/* 尝试 DMABUF 零拷贝；失败回退 MMAP(CPU 拷贝) */
static int v4l2_init(cam_ctx_t *c)
{
    struct v4l2_format fmt = {0};
    struct v4l2_requestbuffers req = {0};
    int ret, i;

    c->use_dmabuf = 1;
    c->v4l_fd = open(c->dev, O_RDWR | O_NONBLOCK);
    if (c->v4l_fd < 0) { perror(c->dev); return -1; }

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = (unsigned int)c->cam_w;
    fmt.fmt.pix_mp.height = (unsigned int)c->cam_h;
    fmt.fmt.pix_mp.pixelformat = V4L_FMT;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    /* 零拷贝时相机按整 fb/暂存 行步进写入 */
    fmt.fmt.pix_mp.plane_fmt[0].bytesperline = c->staging ? c->cap_pitch[0] : c->bo_pitch[0];
    ret = ioctl(c->v4l_fd, VIDIOC_S_FMT, &fmt);
    if (ret < 0) { perror("VIDIOC_S_FMT"); return -1; }
    c->v4l_pitch = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    if (c->v4l_pitch != (c->staging ? c->cap_pitch[0] : c->bo_pitch[0]))
        fprintf(stderr, "warn: %s bytesperline 驱动改为 %u\n", c->dev, c->v4l_pitch);

    req.count = NBUF_FB; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; req.memory = V4L2_MEMORY_DMABUF;
    if (getenv("DP_DMABUF") && ioctl(c->v4l_fd, VIDIOC_REQBUFS, &req) == 0) {
        for (i = 0; i < NBUF_FB; i++) {
            struct v4l2_buffer buf = {0};
            struct v4l2_plane planes[VIDEO_MAX_PLANES] = {0};
            int dmafd = -1;
            uint32_t h = c->staging ? c->cap_handle[i] : c->bo_handle[i];
            ret = drmPrimeHandleToFD(drm_fd, h, 0, &dmafd);
            if (ret < 0 || dmafd < 0) { perror("drmPrimeHandleToFD"); break; }
            c->vbuf_fd[i] = dmafd;
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_DMABUF;
            buf.index = (unsigned int)i; buf.length = VIDEO_MAX_PLANES; buf.m.planes = planes;
            planes[0].data_offset = 0;
            planes[0].length = (uint32_t)(c->staging ? c->cap_size[i] : c->bo_size[i]);
            planes[0].m.fd = dmafd;
            ret = ioctl(c->v4l_fd, VIDIOC_QBUF, &buf);
            if (ret < 0) { fprintf(stderr, "warn: %s QBUF(DMABUF) 失败(%s)，回退 MMAP\n", c->dev, strerror(errno)); break; }
        }
        if (ret == 0) {
            enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            if (ioctl(c->v4l_fd, VIDIOC_STREAMON, &t) < 0) { perror("STREAMON(DMABUF)"); return -1; }
            c->use_dmabuf = 1;
            printf("v4l2 %s: %ux%u -> plane %u @(%d,%d) [DMABUF 零拷贝]\n",
                   c->dev, c->cam_w, c->cam_h, c->plane_id, c->off_x, c->off_y);
            return 0;
        }
        for (i = 0; i < NBUF_FB; i++) if (c->vbuf_fd[i] >= 0) { close(c->vbuf_fd[i]); c->vbuf_fd[i] = -1; }
        enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        ioctl(c->v4l_fd, VIDIOC_STREAMOFF, &t);
        req.count = 0; ioctl(c->v4l_fd, VIDIOC_REQBUFS, &req);
    } else {
        fprintf(stderr, "warn: %s REQBUFS(DMABUF) 失败(%s)，回退 MMAP\n", c->dev, strerror(errno));
    }

    c->use_dmabuf = 0;
    req.count = NBUF_FB; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(c->v4l_fd, VIDIOC_REQBUFS, &req) < 0) { perror("REQBUFS(MMAP)"); return -1; }
    for (i = 0; i < NBUF_FB; i++) {
        struct v4l2_buffer buf = {0};
        struct v4l2_plane planes[VIDEO_MAX_PLANES] = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
        buf.index = (unsigned int)i; buf.length = VIDEO_MAX_PLANES; buf.m.planes = planes;
        if (ioctl(c->v4l_fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("QUERYBUF"); return -1; }
        if (c->v4l_pitch == 0) c->v4l_pitch = (uint32_t)(c->cam_w * 3);
        c->mmap_size[i] = planes[0].length;
        c->mmap_buf[i] = mmap(NULL, planes[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, c->v4l_fd, planes[0].m.mem_offset);
        if (c->mmap_buf[i] == MAP_FAILED) { perror("mmap v4l2"); return -1; }
        if (ioctl(c->v4l_fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF(MMAP)"); return -1; }
    }
    enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(c->v4l_fd, VIDIOC_STREAMON, &t) < 0) { perror("STREAMON(MMAP)"); return -1; }
    printf("v4l2 %s: %ux%u -> plane %u @(%d,%d) [MMAP CPU拷贝]\n",
           c->dev, c->cam_w, c->cam_h, c->plane_id, c->off_x, c->off_y);
    return 0;
}

/* 把相机原始 BGR 数据（MMAP 回退时）拷进“源缓冲”：
 *   VID: 直接拷进整屏显示 fb 的 (off_x,off_y)（RGB888，按原样，与零拷贝一致）
 *   GFX: 拷进 RGB888 暂存 fb
 */
static void copy_raw_to_src(cam_ctx_t *c, int fi)
{
    const uint8_t *s = (const uint8_t *)c->mmap_buf[fi];
    int sp = (int)c->v4l_pitch;
    int row = c->cam_w * 3;
    if (c->staging) {
        uint8_t *d = (uint8_t *)c->cap_map[fi];
        int dp = (int)c->cap_pitch[0];
        for (int y = 0; y < c->cam_h; y++)
            memcpy(d + (size_t)y * dp, s + (size_t)y * sp, (size_t)row);
    } else {
        uint8_t *d = (uint8_t *)c->bo_map[fi] + (size_t)c->off_y * c->bo_pitch[fi] + (size_t)c->off_x * 3;
        int dp = (int)c->bo_pitch[fi];
        for (int y = 0; y < c->cam_h; y++)
            memcpy(d + (size_t)y * dp, s + (size_t)y * sp, (size_t)row);
    }
}

/* GFX: 把暂存 RGB888(BGR 原样) 转 ARGB(B,G,R,255) 贴到整屏 fb 的 (off_x,off_y) */
static void bgr_to_argb_blit(cam_ctx_t *c, int fi)
{
    const uint8_t *s = (const uint8_t *)c->cap_map[fi];
    uint8_t *d = (uint8_t *)c->bo_map[fi];
    int sp = (int)c->cap_pitch[0];
    int dp = (int)c->bo_pitch[fi];
    int dx0 = c->off_x, dy0 = c->off_y;
    for (int y = 0; y < c->cam_h; y++) {
        const uint8_t *src = s + (size_t)y * sp;
        uint8_t *dst = d + ((size_t)(dy0 + y) * dp + (size_t)dx0 * 4);
        for (int x = 0; x < c->cam_w; x++) {
            uint8_t b = src[x*3+0], g = src[x*3+1], r = src[x*3+2];
            dst[x*4+0] = b; dst[x*4+1] = g; dst[x*4+2] = r; dst[x*4+3] = 0xFF;
        }
    }
}

/* 阶段2：人脸框渲染（在 GFX 整屏 ARGB fb 上，覆盖 video0 区域 0..1080） */
static void box_put_px(uint8_t *base, uint32_t pitch, int px, int py, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (px < 0 || py < 0 || px >= SCREEN_W || py >= SCREEN_H) return;
    uint8_t *p = base + (size_t)py * pitch + (size_t)px * 4;
    p[0] = b; p[1] = g; p[2] = r; p[3] = a;   /* DP 读 ARGB 为 B,G,R,A */
}

/* 清除一个框轮廓（alpha=0，透出下层） */
static void box_clear(uint8_t *base, uint32_t pitch, const face_box_t *bx)
{
    int x0 = bx->x, y0 = bx->y, w = bx->w, h = bx->h, t = BOX_THICK;
    for (int yy = y0; yy < y0 + h; yy++)
        for (int xx = x0; xx < x0 + w; xx++)
            if (xx < x0 + t || xx >= x0 + w - t || yy < y0 + t || yy >= y0 + h - t)
                box_put_px(base, pitch, xx, yy, 0, 0, 0, 0);
}

/* 画一个框轮廓（默认绿，可由 color 指定 0xRRGGBB） */
static void box_draw(uint8_t *base, uint32_t pitch, const face_box_t *bx)
{
    uint8_t r = 0, g = 255, b = 0, a = 255;
    if (bx->color) { r = (bx->color >> 16) & 0xff; g = (bx->color >> 8) & 0xff; b = bx->color & 0xff; }
    int x0 = bx->x, y0 = bx->y, w = bx->w, h = bx->h, t = BOX_THICK;
    for (int yy = y0; yy < y0 + h; yy++)
        for (int xx = x0; xx < x0 + w; xx++)
            if (xx < x0 + t || xx >= x0 + w - t || yy < y0 + t || yy >= y0 + h - t)
                box_put_px(base, pitch, xx, yy, r, g, b, a);
}

/* 每帧：擦掉本 buffer 上一帧的框，读共享内存画新框（seqlock 读，支持多 buffer 不串帧） */
static void render_boxes(cam_ctx_t *c, int fi)
{
    uint8_t *base = (uint8_t *)c->bo_map[fi];
    uint32_t pitch = c->bo_pitch[fi];
    for (int i = 0; i < g_prev_count[fi]; i++) box_clear(base, pitch, &g_prev[fi][i]);

    face_box_t cur[FACE_BOXES_MAX];
    int n = 0;
    if (g_boxes) {
        for (;;) {
            uint32_t s0 = g_boxes->seq;
            if (s0 & 1u) continue;                 /* 写入中，重试 */
            n = g_boxes->count;
            if (n > FACE_BOXES_MAX) n = FACE_BOXES_MAX;
            memcpy(cur, g_boxes->boxes, (size_t)n * sizeof(face_box_t));
            __sync_synchronize();
            uint32_t s1 = g_boxes->seq;
            if (s0 == s1) break;                   /* 稳定，读取有效 */
        }
    }
    for (int i = 0; i < n; i++) {
        /* 坐标在 1080x1080 主相机空间 -> 屏幕 video0 区(0..1080)，裁剪 */
        if (cur[i].x < 0) { cur[i].w += cur[i].x; cur[i].x = 0; }
        if (cur[i].y < 0) { cur[i].h += cur[i].y; cur[i].y = 0; }
        if (cur[i].x + cur[i].w > 1080) cur[i].w = 1080 - cur[i].x;
        if (cur[i].y + cur[i].h > 1080) cur[i].h = 1080 - cur[i].y;
        if (cur[i].w <= 0 || cur[i].h <= 0) continue;
        box_draw(base, pitch, &cur[i]);
    }
    g_prev_count[fi] = n;
    memcpy(g_prev[fi], cur, (size_t)n * sizeof(face_box_t));
}

/* 阶段2：GFX 层整屏覆盖 video0，每帧只重画人脸框（不依赖摄像头，自带 buffer 轮转） */
static void display_gfx_boxes(void)
{
    int fi = g_gfx_fi;
    render_boxes(&g_cam1, fi);
    drmModeSetPlane(drm_fd, g_cam1.plane_id, g_crtc_id, g_cam1.fb_id[fi], 0,
                    0, 0, (uint32_t)g_cam1.disp_w, (uint32_t)g_cam1.disp_h,
                    0, 0, ((uint32_t)g_cam1.disp_w) << 16, ((uint32_t)g_cam1.disp_h) << 16);
    g_gfx_fi = (fi + 1) % NBUF_FB;
}

static int shm_init(const char *path)
{
    int fd = shm_open(path, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); return -1; }
    if (ftruncate(fd, (off_t)sizeof(face_boxes_t)) < 0) { perror("ftruncate"); close(fd); return -1; }
    void *p = mmap(NULL, sizeof(face_boxes_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap shm"); return -1; }
    g_boxes = (face_boxes_t *)p;
    printf("shm boxes: %s (max %d, 坐标空间 1080x1080)\n", path, FACE_BOXES_MAX);
    return 0;
}

static void display_frame(cam_ctx_t *c)
{
    struct v4l2_buffer buf = {0};
    struct v4l2_plane planes[VIDEO_MAX_PLANES] = {0};
    int ret;

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = c->use_dmabuf ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
    buf.length = VIDEO_MAX_PLANES;
    buf.m.planes = planes;

    ret = ioctl(c->v4l_fd, VIDIOC_DQBUF, &buf);
    if (ret < 0) { if (errno != EAGAIN) perror("DQBUF"); return; }
    int fi = (int)buf.index;

    if (!c->use_dmabuf) copy_raw_to_src(c, fi);

    if (c->staging) bgr_to_argb_blit(c, fi);   /* GFX: 转 ARGB 贴到整屏 fb */
    if (c->is_gfx) render_boxes(c, fi);         /* 阶段2：人脸框（叠加在 video0 区）*/

    drmModeDirtyFB(drm_fd, c->fb_id[fi], NULL, 0);
    /* 两块层均整屏 1920x1080，满足“两层 src 同尺寸”硬约束 */
    ret = drmModeSetPlane(drm_fd, c->plane_id, g_crtc_id, c->fb_id[fi], 0,
                          0, 0, (uint32_t)c->disp_w, (uint32_t)c->disp_h,
                          0, 0, ((uint32_t)c->disp_w) << 16, ((uint32_t)c->disp_h) << 16);
    if (ret < 0 && errno != EBUSY)
        fprintf(stderr, "warn: SetPlane(%s) failed: %s\n", c->dev, strerror(errno));

    if (c->use_dmabuf) {
        planes[0].m.fd = c->vbuf_fd[fi];
        planes[0].data_offset = 0;
        planes[0].length = (uint32_t)(c->staging ? c->cap_size[fi] : c->bo_size[fi]);
    }
    if (ioctl(c->v4l_fd, VIDIOC_QBUF, &buf) < 0) perror("QBUF");
}

static void cleanup(void)
{
    for (cam_ctx_t *c = &g_cam0; c <= &g_cam1; c++) {
        if (c->v4l_fd >= 0) {
            enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            ioctl(c->v4l_fd, VIDIOC_STREAMOFF, &t);
            for (int i = 0; i < NBUF_FB; i++) {
                if (c->vbuf_fd[i] >= 0) close(c->vbuf_fd[i]);
                if (c->mmap_buf[i] != MAP_FAILED) munmap(c->mmap_buf[i], c->mmap_size[i]);
            }
            close(c->v4l_fd);
        }
        if (c->plane_id) drmModeSetPlane(drm_fd, c->plane_id, g_crtc_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        for (int i = 0; i < NBUF_FB; i++) {
            if (c->cap_fb_id[i]) { drmModeRmFB(drm_fd, c->cap_fb_id[i]); if (c->cap_map[i] != MAP_FAILED) munmap(c->cap_map[i], c->cap_size[i]); if (c->cap_handle[i]) { struct drm_mode_destroy_dumb d = { .handle = c->cap_handle[i] }; drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d); } }
            if (c->fb_id[i]) drmModeRmFB(drm_fd, c->fb_id[i]);
            if (c->bo_map[i] != MAP_FAILED) munmap(c->bo_map[i], c->bo_size[i]);
            if (c->bo_handle[i]) { struct drm_mode_destroy_dumb d = { .handle = c->bo_handle[i] }; drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d); }
        }
    }
    if (g_boxes) munmap(g_boxes, sizeof(face_boxes_t));
    if (drm_fd >= 0) close(drm_fd);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int want_vid = (argc > 1) ? atoi(argv[1]) : 0;
    int want_gfx = (argc > 2) ? atoi(argv[2]) : 0;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "-s")) g_stage2 = 1;
    if (getenv("STAGE2")) g_stage2 = 1;

    /* video0 -> VID 下层：整屏 fb，相机放 (0,0)，RGB888 直接零拷贝 */
    g_cam0 = (cam_ctx_t){ .dev = V4L_DEV0, .cam_w = CAM0_W, .cam_h = CAM0_H,
                          .disp_w = SCREEN_W, .disp_h = SCREEN_H, .off_x = 0, .off_y = 0,
                          .disp_fmt = VID_FMT, .staging = 0 };
    /* video1 -> GFX 上层：整屏 ARGB fb，相机放 (1081,0)，经暂存 RGB888 转 ARGB */
    g_cam1 = (cam_ctx_t){ .dev = V4L_DEV1, .cam_w = CAM1_W, .cam_h = CAM1_H,
                          .disp_w = SCREEN_W, .disp_h = SCREEN_H, .off_x = CAP1_OFF_X, .off_y = CAP1_OFF_Y,
                          .disp_fmt = GFX_FMT, .staging = 1, .is_gfx = 1 };
    if (g_stage2) {                 /* 阶段2：GFX 整屏覆盖 video0 画框，关闭 320 小窗，camera 交给检测程序 */
        g_cam1.staging = 0;
        g_cam1.off_x = 0; g_cam1.off_y = 0;
        g_cam1.disp_w = SCREEN_W; g_cam1.disp_h = SCREEN_H;
    }

    if (drm_init(want_vid, want_gfx) < 0) { cleanup(); return 1; }
    g_cam0.plane_id = g_vid_plane_id;
    g_cam1.plane_id = g_gfx_plane_id;

    if (create_fbs(&g_cam0) < 0) { cleanup(); return 1; }
    if (create_fbs(&g_cam1) < 0) { cleanup(); return 1; }
    if (v4l2_init(&g_cam0) < 0) { cleanup(); return 1; }
    if (!g_stage2 && v4l2_init(&g_cam1) < 0) { cleanup(); return 1; }

    /* 阶段2：打开人脸框共享内存（算法未启动时仅为 NULL，框不显示，不影响出图） */
    const char *shmp = getenv("FACE_SHM");
    shm_init(shmp ? shmp : FACE_BOXES_SHM);

    if (drmModeSetCrtc(drm_fd, g_crtc_id, g_cam0.fb_id[0], 0, 0, &g_conn_id, 1, &g_mode) < 0)
        fprintf(stderr, "warn: SetCrtc failed: %s\n", strerror(errno));
    else
        printf("crtc enabled: %dx%d@%u\n", g_screen_w, g_screen_h, g_mode.vrefresh);

    /* GFX 用每像素 alpha 控制透明（g_alpha_en=0）；alpha 值不影响逐像素 alpha */
    if (set_plane_prop(g_gfx_plane_id, "g_alpha_en", 0) == 0)
        printf("GFX g_alpha_en -> 0 (逐像素 alpha 生效)\n");
    else
        printf("GFX g_alpha_en 设置失败（无关紧要）\n");

    if (g_stage2)
        printf("display[stage2]: video0 1080x1080 VID | 人脸框(shm:%s) | 320 小窗关闭  [Ctrl+C stop]\n",
               shmp ? shmp : FACE_BOXES_SHM);
    else
        printf("display: video0 1080x1080 @(0,0) VID | video1 320x320 @(%d,0) GFX | 人脸框(shm:%s)  [Ctrl+C stop]\n",
               CAP1_OFF_X, shmp ? shmp : FACE_BOXES_SHM);

    struct pollfd pfd[2];
    pfd[0].fd = g_cam0.v4l_fd; pfd[0].events = POLLIN;
    pfd[1].fd = g_cam1.v4l_fd; pfd[1].events = POLLIN;
    unsigned long f0 = 0, f1 = 0;
    while (g_run) {
        int n = poll(pfd, 2, 1000);
        if (n < 0) { if (errno == EINTR) continue; perror("poll"); break; }
        if (n == 0) continue;
        if (pfd[0].revents & POLLIN) {
            display_frame(&g_cam0); f0++;
            if (g_stage2) display_gfx_boxes();   /* 阶段2：每帧重画人脸框 */
        }
        if (!g_stage2 && pfd[1].revents & POLLIN) { display_frame(&g_cam1); f1++; }
        if ((f0 + f1) % 120 == 0)
            printf("frames: video0=%lu(%s) video1=%lu(%s)\n", f0, g_cam0.use_dmabuf ? "dma" : "mmap",
                   f1, g_cam1.use_dmabuf ? "dma" : "mmap");
    }
    printf("total: video0=%lu video1=%lu\n", f0, f1);
    cleanup();
    return 0;
}
