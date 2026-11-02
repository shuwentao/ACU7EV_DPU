#ifndef FACE_BOXES_H
#define FACE_BOXES_H

#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* 显示程序与算法侧共用的“人脸框”共享内存协议。
 * 坐标空间：1080x1080 主相机图（显示程序直接映射到屏幕 0..1080 的 video0 区）。
 * 同步：seqlock（写：奇数->写->偶数；读：seq 偶数且前后一致才算有效）。
 */

#define FACE_BOXES_MAX  16
#define FACE_BOXES_SHM  "/face_boxes"

typedef struct {
    int32_t  x, y, w, h;   /* 1080x1080 主相机坐标空间 */
    int32_t  score;        /* 0..100，仅作参考 */
    uint32_t color;        /* 0xRRGGBB，0 = 默认绿 */
} face_box_t;

typedef struct {
    uint32_t   seq;        /* seqlock：奇数=写入中，偶数=稳定 */
    int32_t    count;
    face_box_t boxes[FACE_BOXES_MAX];
} face_boxes_t;

/* 算法侧：打开/创建共享内存（返回 NULL 表示失败） */
static inline face_boxes_t *face_boxes_open(void)
{
    int fd = shm_open(FACE_BOXES_SHM, O_CREAT | O_RDWR, 0666);
    if (fd < 0) return NULL;
    if (ftruncate(fd, (off_t)sizeof(face_boxes_t)) < 0) { close(fd); return NULL; }
    void *p = mmap(NULL, sizeof(face_boxes_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return (face_boxes_t *)p;
}

/* 算法侧：写入（seqlock 写：++seq 奇数 -> 拷贝 -> ++seq 偶数） */
static inline void face_boxes_write(face_boxes_t *m, const face_box_t *b, int n)
{
    if (!m) return;
    if (n > FACE_BOXES_MAX) n = FACE_BOXES_MAX;
    m->seq++;                          /* 奇数：写入中 */
    m->count = n;
    memcpy(m->boxes, b, (size_t)n * sizeof(face_box_t));
    __sync_synchronize();
    m->seq++;                          /* 偶数：稳定 */
}

#endif /* FACE_BOXES_H */
