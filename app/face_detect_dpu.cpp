/*
 * face_detect_dpu.cpp —— 阶段2 人脸检测（MobileNetV3 主干 + FCOS/anchor-free head）
 *
 * 流程：开 /dev/video1(320 BGR) → 预处理(RGB, (x/255-0.5)/0.5, NCHW)
 *       → VART 跑 FaceDetector_0_int.xmodel
 *       → FCOS 解码(cls/box/lm, 3 个 stride) + 置信过滤(0.3)
 *       → 取最高分框 → 映射回 1080x1080 → 写 /face_boxes 共享内存
 *
 * 解码与预处理严格对齐 /mnt/data1/face_bench（common/dataset.py, common/models/detector.py,
 * common/eval_api.py）：FCOS anchor-free，无 variance，无 NMS（只取最高分一张脸）。
 *
 * 模型需放在一个目录里，含 meta.json：
 *   { "target": "DPUCZDX8G_ISA1_B4096", "filename": "FaceDetector_0_int.xmodel" }
 * 板子需有 /usr/lib/dpu.xclbin 与 /etc/vart.conf（DPU 固件）。
 *
 * 用法：face_detect_dpu [模型目录] [相机设备]
 *   环境变量：FACE_SHM 覆盖共享内存名；DBG=1 打印模型信息/逐帧最高分（用于校准）。
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
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "face_boxes.h"
#include <map>
#include <dirent.h>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_helper.hpp>
#include <vart/dpu/vitis_dpu_runner_factory.hpp>

/* 从同目录 meta.json 读取 DPU kernel 名（Vitis AI 模型包约定：meta.json 的 kernel 字段） */
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

/* ---------------- 摄像头（MMAP, MPLANE, BGR24） ---------------- */
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
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_BGR24;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    if (ioctl(c->fd, VIDIOC_S_FMT, &fmt) < 0) { perror("S_FMT"); return -1; }
    if (fmt.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_BGR24) {
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
        /* O_NONBLOCK 下无帧就绪返回 EAGAIN/EWOULDBLOCK，属正常轮询，不刷屏 */
        if (errno != EAGAIN && errno != EWOULDBLOCK) perror("DQBUF");
        return -1;
    }
    *out = (uint8_t*)c->mmap_buf[buf.index];
    return (int)buf.index;
}

static void cap_qbuf(Cap* c, int idx)
{
    struct v4l2_buffer buf = {0}; struct v4l2_plane pl[VIDEO_MAX_PLANES] = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; buf.memory = V4L2_MEMORY_MMAP;
    buf.index = (unsigned)idx; buf.length = VIDEO_MAX_PLANES; buf.m.planes = pl;
    ioctl(c->fd, VIDIOC_QBUF, &buf);
}

/* ---------------- 工具 ---------------- */
/* 在模型目录里找第一个 .xmodel（用于 XIR 读取量化 fixpos） */
static std::string find_xmodel(const std::string& dir)
{
    DIR* d = opendir(dir.c_str());
    if (!d) { fprintf(stderr, "find_xmodel: opendir(%s) 失败: %s\n", dir.c_str(), strerror(errno)); return std::string(); }
    struct dirent* e;
    std::string r;
    while ((e = readdir(d))) {
        std::string n = e->d_name;
        if (n.size() >= 6 && n.compare(n.size() - 6, 6, ".xmodel") == 0) { r = dir + "/" + n; break; }
    }
    closedir(d);
    return r;
}

/* 解析模型路径：argv 可直接给 .xmodel 文件，也可给含 .xmodel 的目录。
 * 注意 create_dpu_runner(file_name,...) 与 XIR 都要求“文件”而非目录。 */
static std::string resolve_xmodel(const std::string& arg)
{
    fprintf(stderr, "[DBG] resolve_xmodel: arg='%s'\n", arg.c_str());
    if (arg.size() >= 7 && arg.compare(arg.size() - 7, 7, ".xmodel") == 0) {
        fprintf(stderr, "[DBG] resolve_xmodel: 以 .xmodel 结尾 -> 当文件直接返回\n");
        return arg;                       /* 直接给了 xmodel 文件 */
    }
    std::string r = find_xmodel(arg);
    fprintf(stderr, "[DBG] resolve_xmodel: 当目录处理, find_xmodel 返回 '%s'\n", r.c_str());
    return r;                            /* 否则当目录处理，找首个 .xmodel */
}

/* 按 tensor 名查量化 fixpos（从 XIR 解析得到，vitis::ai::Tensor 不含该信息） */
static int get_fix(const std::map<std::string,int>& m, const std::string& name)
{
    auto it = m.find(name);
    return (it != m.end()) ? it->second : 0;
}

/* 取 DPU tensor 的量化 fix_point：优先读 tensor 自身属性，回退到按名查表（更鲁棒） */
static int fixpos_of(const xir::Tensor* t, const std::map<std::string,int>& m)
{
    if (t->has_attr("fix_point")) {
        try { return t->get_attr<int>("fix_point"); } catch (...) {}
    }
    auto it = m.find(t->get_name());
    return (it != m.end()) ? it->second : 0;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static inline int8_t quant_i8(float v, float scale)
{
    int iv = (int)lroundf(v * scale);
    if (iv > 127) iv = 127; if (iv < -128) iv = -128;
    return (int8_t)iv;
}

/* 双线性缩放 BGR24（cam_w x cam_h -> Msz x Msz） */
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
    const char* model_dir = (argc > 1) ? argv[1] : "/usr/share/vart/face_detector";
    const char* cam_dev  = (argc > 2) ? argv[2] : "/dev/video1";
    const char* shm_path = getenv("FACE_SHM") ? getenv("FACE_SHM") : FACE_BOXES_SHM;
    int dbg = getenv("DBG") ? 1 : 0;
    /* 运行期判定阈值：优先命令行第4参数，其次环境变量 FACE_THR，默认 0.40。
     * 卡在白墙背景响应(~0.38)之上、人脸响应之下；调试时随时调，无需重编。 */
    float score_thr = 0.40f;
    if (argc > 4) score_thr = (float)atof(argv[4]);
    else if (getenv("FACE_THR")) score_thr = (float)atof(getenv("FACE_THR"));

    /* === 版本标记 + 参数回显：始终打印，用来确认板端跑的是不是这份新二进制 === */
    fprintf(stderr, "[DBG-VER] face_detect_dpu META_V10 (composite runner / 入口校验 / debounce N_DET=3 N_LOST=5)\n");
    fprintf(stderr, "[DBG] argv: model_dir=%s cam_dev=%s shm=%s dbg=%d score_thr=%.2f\n",
            model_dir, cam_dev, shm_path, dbg, score_thr);

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

    /* 加载模型：XIR 解析 .xmodel 拿量化 fixpos 与 DPU kernel 名；DpuRunner 加载 kernel */
    std::string model_dir_s(model_dir);
    std::string xmodel = resolve_xmodel(model_dir_s);
    fprintf(stderr, "[DBG] 解析到 xmodel='%s' (空表示未解析到)\n", xmodel.c_str());
    std::map<std::string,int> fixmap;
    std::unique_ptr<xir::Graph> graph;        /* 反序列化结果，仅用于取图名/量化信息 */
    if (!xmodel.empty()) {
        try {
            graph = xir::Graph::deserialize(xmodel);
            for (auto t : graph->get_tensors())
                if (t->has_attr("fix_point")) fixmap[t->get_name()] = t->get_attr<int>("fix_point");
            auto root = graph->get_root_subgraph();
            if (root) {
                fprintf(stderr, "[DBG] graph name='%s' root name='%s' is_leaf=%d 直接子图数=%zu\n",
                        graph->get_name().c_str(), root->get_name().c_str(),
                        (int)root->is_leaf(), root->get_children().size());
            }
        } catch (...) { fprintf(stderr, "XIR 读取 fixpos/kernel 失败(忽略)\n"); }
    } else {
        fprintf(stderr, "警告: 未在 %s 找到 .xmodel, fixpos 用 0\n", model_dir_s.c_str());
    }

    if (xmodel.empty()) { fprintf(stderr, "错误: 未解析到 .xmodel, 请检查模型路径 %s\n", model_dir_s.c_str()); return 1; }

    /* 取 DPU kernel 名：本模型（FaceDetector / gate320）被 DPUCZDX8G 切成多个子图，
     * 必须加载“整图 composite runner”（用图名），由 VART 内部串联所有子图并暴露真实图像输入；
     * 若只加载某个子图（如 meta.json 指向的 box head），runner 输入会变成中间 feature map
     * （tensor 名为空），alloc/取数据时直接 SIGSEGV。
     * 顺序：图名(整图) -> 图名_0 -> 各非 quant/fix/input 子图 -> meta.json 指定子图(兜底)。 */
    std::vector<std::string> knames;
    if (graph) {
        auto root = graph->get_root_subgraph();
        if (root) {
            auto children = root->get_children();
            fprintf(stderr, "[DBG] root 子图数=%zu\n", children.size());
            for (auto sub : children)
                fprintf(stderr, "[DBG]   子图 '%s'\n", sub->get_name().c_str());
            knames.push_back(graph->get_name());            // 整图 composite（优先）
            knames.push_back(graph->get_name() + "_0");
            for (auto sub : children) {                      // 兜底：各非 quant/fix/input 子图
                std::string n = sub->get_name();
                bool is_fix = n.find("quant") != std::string::npos
                           || n.find("fix") != std::string::npos
                           || n.find("input") != std::string::npos;
                if (n.find("subgraph_") != std::string::npos && !is_fix)
                    knames.push_back(n);
            }
        }
    }
    std::string meta_kernel = read_kernel_from_meta(xmodel);   // 最后兜底
    if (!meta_kernel.empty()) {
        fprintf(stderr, "[DBG] meta.json kernel='%s'\n", meta_kernel.c_str());
        knames.push_back(meta_kernel);
    }
    std::unique_ptr<vart::Runner> runner_u;
    std::string used_kname;
    for (auto& kn : knames) {
        fprintf(stderr, "[DBG] try create_dpu_runner kernel='%s'\n", kn.c_str());
        try {
            runner_u = vart::dpu::DpuRunnerFactory::create_dpu_runner(xmodel, kn);
        } catch (...) { runner_u = nullptr; }
        if (!runner_u) continue;
        /* 校验选中的是“入口” runner：输入 tensor 名非空（真实图像输入有名字，
         * 中间子图的输入名为空 -> 选错了，继续试下一个 kernel）。 */
        auto it_in = runner_u->get_input_tensors();
        if (!it_in.empty() && !it_in[0]->get_name().empty()) {
            used_kname = kn; break;
        }
        fprintf(stderr, "[DBG] kernel='%s' 输入名为空(非入口子图), 跳过\n", kn.c_str());
        runner_u = nullptr;
    }
    if (!runner_u) { fprintf(stderr, "create_dpu_runner 失败: %s (需 DPU 编译后的 xmodel + /run/media/mmcblk0p1/dpu.xclbin)\n", xmodel.c_str()); return 1; }
    fprintf(stderr, "[DBG] create_dpu_runner OK kernel='%s'\n", used_kname.c_str());

    /* 现代 vart::Runner 接口（工厂返回的就是 vart::Runner，切勿再 reinterpret_cast
     * 成弃用的 vitis::ai::DpuRunner，否则会调到错误的 vtable 直接 SIGSEGV）。 */
    vart::Runner* runner = runner_u.get();
    auto inputs  = runner->get_input_tensors();      // std::vector<const xir::Tensor*>
    fprintf(stderr, "[DBG] inputs size=%zu\n", inputs.size());
    auto outputs = runner->get_output_tensors();
    fprintf(stderr, "[DBG] outputs size=%zu\n", outputs.size());

    /* 用 xir::Tensor* 分配 host 平坦 buffer（标准 vart 路径，返回 vart::TensorBuffer） */
    auto in_bufs  = vart::alloc_cpu_flat_tensor_buffers(inputs);
    auto out_bufs = vart::alloc_cpu_flat_tensor_buffers(outputs);
    fprintf(stderr, "[DBG] bufs alloc ok\n");

    auto inshape = inputs[0]->get_dims();            // [N,C,H,W] 或 [N,H,W,C]
    fprintf(stderr, "[DBG] inshape ndim=%zu\n", inshape.size());
    bool nhwc = (runner->get_tensor_format() == vart::Runner::TensorFormat::NHWC);
    int Msz = nhwc ? (int)inshape[1] : (int)inshape[2];   // 正方形输入边长
    int in_fix = fixpos_of(inputs[0], fixmap);
    int in_nelem = inputs[0]->get_element_num();
    fprintf(stderr, "[DBG] pre-model-print Msz=%d nhwc=%d in_fix=%d in_nelem=%d\n", Msz, (int)nhwc, in_fix, in_nelem);
    if (dbg)
        printf("[model] kernel=%s input %dx%dx%dx%d fmt=%s fixpos=%d INT8\n",
               used_kname.c_str(), (int)inshape[0], (int)inshape[1], (int)inshape[2], (int)inshape[3],
               nhwc ? "NHWC" : "NCHW", in_fix);

    /* 摄像头 */
    Cap cap;
    if (cap_open(&cap, cam_dev, Msz, Msz) < 0) return 1;   // 直接按模型尺寸采集（320 或 640）
    int cam_w = cap.w, cam_h = cap.h;
    std::vector<uint8_t> rsbuf((size_t)Msz * Msz * 3);      // 缩放后 BGR

    /* 共享内存（算法侧写） */
    face_boxes_t* shm = face_boxes_open();
    if (!shm) { fprintf(stderr, "face_boxes_open 失败: %s\n", shm_path); return 1; }

    int frame = 0, skip = 0;
    while (!g_stop) {
        uint8_t* raw = nullptr;
        int idx = cap_grab(&cap, &raw);
        if (idx < 0) { usleep(1000); continue; }
        if (raw == nullptr) { cap_qbuf(&cap, idx); continue; }

        /* 预处理 + INT8 量化合并为单次写入：BGR→RGB → /255 → (x-0.5)/0.5 → 直接填 DPU 输入。
         * 不再经中间 float 缓冲（fin），省一次全帧拷贝（320 帧约省 1.2MB/帧带宽）。 */
        const uint8_t* bgr = raw;
        if (cam_w != Msz || cam_h != Msz) { resize_bgr(raw, cam_w, cam_h, rsbuf.data(), Msz, Msz); bgr = rsbuf.data(); }
        auto ind = in_bufs[0]->data();
        int8_t* idst = reinterpret_cast<int8_t*>(ind.first);
        float scale = (float)(1LL << in_fix);
        for (int y = 0; y < Msz; y++) {
            for (int x = 0; x < Msz; x++) {
                const uint8_t* p = bgr + ((size_t)y * Msz + x) * 3;
                float fr = (p[2] / 255.0f - 0.5f) / 0.5f;   // BGR→RGB
                float fg = (p[1] / 255.0f - 0.5f) / 0.5f;
                float fb = (p[0] / 255.0f - 0.5f) / 0.5f;
                if (nhwc) {                                          // HWC: R,G,B
                    size_t o = ((size_t)y * Msz + x) * 3;
                    idst[o + 0] = quant_i8(fr, scale);
                    idst[o + 1] = quant_i8(fg, scale);
                    idst[o + 2] = quant_i8(fb, scale);
                } else {                                             // CHW: R,G,B
                    idst[(0 * (size_t)Msz + y) * Msz + x] = quant_i8(fr, scale);
                    idst[(1 * (size_t)Msz + y) * Msz + x] = quant_i8(fg, scale);
                    idst[(2 * (size_t)Msz + y) * Msz + x] = quant_i8(fb, scale);
                }
            }
        }

        /* 推理 */
        std::vector<vart::TensorBuffer*> in_ptr, out_ptr;
        for (auto& b : in_bufs) in_ptr.push_back(b.get());
        for (auto& b : out_bufs) out_ptr.push_back(b.get());
        auto job = runner->execute_async(in_ptr, out_ptr);
        runner->wait(job.first, -1);

        /* 解码：遍历 3 个 head（按 H 推断 stride） */
        std::vector<Det> dets;
        for (size_t k = 0; k < outputs.size(); k++) {
            auto oshape = outputs[k]->get_dims();           // [N,15,H,W] 或 [N,H,W,15]
            int Hd, Wd;
            if (nhwc) { Hd = (int)oshape[1]; Wd = (int)oshape[2]; }
            else      { Hd = (int)oshape[2]; Wd = (int)oshape[3]; }
            int C = (nhwc ? (int)oshape[3] : (int)oshape[1]);  // 真实输出通道数（不再写死 15）
            int stride = Msz / Hd;
            int out_fix = fixpos_of(outputs[k], fixmap);
            auto od = out_bufs[k]->data();                  // pair<uint64_t,size_t>
            const int8_t* d8 = reinterpret_cast<const int8_t*>(od.first);
            float s = (float)ldexp(1.0, -out_fix);          // 2^-fix 反量化
            std::vector<float> tmpF((size_t)Hd * Wd * C);   // 按真实通道数分配，避免越界
            for (size_t i = 0; i < tmpF.size(); i++) tmpF[i] = d8[i] * s;
            const float* d = tmpF.data();
            const float CONF = 0.05f;   // 对齐 face_bench detector.py 的 conf_thr=0.05
            for (int y = 0; y < Hd; y++) {
                for (int x = 0; x < Wd; x++) {
                    float score = 0, l = 0, t = 0, r = 0, b = 0;
                    if (nhwc) {                             // HWC: 通道在最后一维
                        size_t o = ((size_t)y * Wd + x) * C;
                        score = sigmoidf(d[o + 0]); l = d[o + 1]; t = d[o + 2];
                        r = d[o + 3]; b = d[o + 4];
                    } else {                                // CHW: 通道在 1 维
                        score = sigmoidf(d[(0 * Hd + y) * Wd + x]);
                        if (score < CONF) continue;
                        l = d[(1 * Hd + y) * Wd + x];
                        t = d[(2 * Hd + y) * Wd + x];
                        r = d[(3 * Hd + y) * Wd + x];
                        b = d[(4 * Hd + y) * Wd + x];
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
            if (dbg && frame == 0)
                printf("[model] head%zu shape=[%zu,%zu,%zu,%zu] C=%d stride=%d fixpos=%d fmt=%s\n",
                       k, oshape[0], oshape[1], oshape[2], oshape[3],
                       (int)(nhwc ? oshape[3] : oshape[1]), stride, out_fix,
                       nhwc ? "NHWC" : "NCHW");
        }

        /* 取最高分一张脸 */
        Det best = {0, 0, 0, 0, -1};
        for (auto& d : dets) if (d.score > best.score) best = d;

        /* 映射回 1080x1080（模型空间 → ×(1080/Msz)） */
        double sc = 1080.0 / Msz;
        face_box_t box; memset(&box, 0, sizeof(box));
        if (best.score >= 0) {
            int x = (int)round(best.x1 * sc), y = (int)round(best.y1 * sc);
            int w = (int)round((best.x2 - best.x1) * sc), h = (int)round((best.y2 - best.y1) * sc);
            if (x < 0) { w += x; x = 0; }
            if (y < 0) { h += y; y = 0; }
            box.x = x; box.y = y; box.w = w; box.h = h; box.score = (int)(best.score * 100);
            box.color = 0;   // 默认绿
        }
        /* 时序去抖（debounce）：模型单帧置信分数噪声大、常在阈值附近抖（同一张脸位置稳定但分数上下跳），
         * 导致 DETECTED/LOST 逐帧闪。用连续帧计数做迟滞：连续 N_DET 帧过线才判"有人"，
         * 连续 N_LOST 帧不过线才判"无人"。框位置本身稳定，去抖后框也稳定；
         * 白墙 max~0.38 < 0.5 永不过线，本就不会误报。 */
        static int stable = 0;            // 去抖后的稳定状态（0=无人,1=有人）
        static int hi_run = 0, lo_run = 0;
        static face_box_t last_box = {0}; // 记住最近的有效框，去抖期间保持框稳定
        if (best.score >= 0) last_box = box;
        if (best.score >= score_thr) { hi_run++; lo_run = 0; }
        else { lo_run++; hi_run = 0; }
        const int N_DET = 3, N_LOST = 5;  // 可调：开/关所需的连续帧数
        if (!stable && hi_run >= N_DET) stable = 1;
        if (stable && lo_run >= N_LOST) stable = 0;
        int detected = stable;
        face_box_t out = {0};
        if (detected) out = last_box;     // 去抖期间始终画最近的有效框（位置稳定）
        face_boxes_write(shm, &out, detected);

        /* 明确提示：人脸出现/消失时打印一次（无需画框也能确认检测有效） */
        static int prev_face = 0;
        int has_face = detected;
        if (has_face != prev_face) {
            if (has_face)
                printf(">>> FACE DETECTED frame=%d score=%.3f box=(%d,%d,%d,%d)\n",
                       frame, best.score, box.x, box.y, box.w, box.h);
            else
                printf(">>> FACE LOST frame=%d\n", frame);
            prev_face = has_face;
            fflush(stdout);
        }

        if (dbg && (frame % 30 == 0))
            printf("[detect] frame=%d dets=%zu best_score=%.3f box=(%d,%d,%d,%d)\n",
                   frame, dets.size(), best.score, box.x, box.y, box.w, box.h);
        frame++;

        cap_qbuf(&cap, idx);
    }

    if (shm) face_boxes_write(shm, nullptr, 0);   // 退出清空
    printf("face_detect_dpu 退出\n");
    return 0;
}
