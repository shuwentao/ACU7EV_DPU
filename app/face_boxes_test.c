/*
 * face_boxes_test.c —— 阶段2 验证用：在共享内存里写一个来回运动的绿色人脸框。
 * 用法（板上）：
 *   gcc face_boxes_test.c -o face_boxes_test -lrt
 *   ./face_boxes_test &        # 后台跑，同时运行 drm_dual_plane
 * 画面左侧 video0 上应出现一个左右移动的绿色矩形框。
 * 停止后框仍保留，直到写入 count=0 才消失（算法侧无人脸时应写 count=0）。
 */
#include "face_boxes.h"
#include <stdio.h>

int main(void)
{
    face_boxes_t *m = face_boxes_open();
    if (!m) { perror("face_boxes_open"); return 1; }
    printf("writing test box to %s (Ctrl+C to stop)\n", FACE_BOXES_SHM);

    int t = 0;
    while (1) {
        int phase = t % 200;                       /* 0..199 三角波 */
        int off   = phase < 100 ? phase : (200 - phase);  /* 0..100..0 */
        int cx    = 100 + 8 * off;                 /* 100..900 */
        face_box_t b = {
            .x = cx - 100, .y = 440, .w = 200, .h = 200,
            .score = 99, .color = 0x00FF00,
        };
        face_boxes_write(m, &b, 1);
        t++;
        usleep(33000);                             /* ~30fps */
    }
    return 0;
}
