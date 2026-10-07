"""第②步：仿射对正（官方 YuNet ONNX + warpAffine），推理走 onnxruntime(CPU)。

为什么用 onnxruntime 而不是 cv2.FaceDetectorYN：
  - 板子自带 OpenCV 4.6.0 的 dnn 解析不了 2023mar.onnx（getLayerData 错）；
  - onnxruntime 有 aarch64 预编译 wheel，板子 `pip install onnxruntime` 即可，
    且 PC/板子同一套代码。检测+对正全在 CPU，DPU 只留给识别网。

依赖: onnxruntime + opencv-python（只要基础 cv2 做 imread/warpAffine，不需 contrib）
模型: face_detection_yunet_2023mar.onnx（固定 640×640 输入；自动读取模型输入尺寸）

用法:
  pip install onnxruntime opencv-python
  python align.py --model face_detection_yunet_2023mar.onnx --img test.jpg --out aligned.jpg
  python align.py --model ... --img_dir raw/ --out_dir aligned/
  # 按算力配置: 只处理分数最高的前 N 张脸(0=全部)
  python align.py --model ... --img_dir raw/ --out_dir aligned/ --max_faces 4
  python align.py --model ... --img 1.jpg --max_faces 0   # 全部脸
"""
import os
import argparse
import numpy as np
import cv2
import onnxruntime as ort


# 112×112 标准 5 点模板（左眼, 右眼, 鼻尖, 左嘴角, 右嘴角）
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


def load_session(model_path):
    so = ort.SessionOptions()
    # 开图优化即可; 不要强行把 intra_op 线程设成 cpu_count,
    # 核数多的机器反而会因线程争用暴慢(实测 180ms -> 33s)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, so,
                                providers=["CPUExecutionProvider"])
    # 读取模型输入尺寸（如 [1,3,640,640]）
    h = sess.get_inputs()[0].shape[2]
    w = sess.get_inputs()[0].shape[3]
    return sess, int(w), int(h)


def detect_yunet(sess, img_bgr, in_w, in_h, score_thr=0.3, nms_thr=0.4):
    """返回 list[dict]，每项为原图像素坐标的
       {'bbox':(x,y,w,h), 'score':float, 'kps':(5,2)}。
    YuNet ONNX 输出三档 stride(8,16,32) 的多头：
       cls_*(N,1) 分类, obj_*(N,1) objectness,
       bbox_*(N,4) 编码中心/wh, kps_*(N,10) 5 关键点。

    解码(已实测与 cv2.FaceDetectorYN 对齐):
       score = sqrt(clamp(cls,0,1) * clamp(obj,0,1))   # 仅 clamp, 不再套 sigmoid
       center=(grid+bbox_raw)*stride, wh=exp(bbox_raw)*stride
       kps=(grid+kps_raw)*stride
    输入预处理: BGR 原值 0~255(不交换通道/不减均值/不缩放), 见下方 blob 构造。
    """
    H, W = in_h, in_w
    blob = cv2.resize(img_bgr, (W, H))
    # 官方 YuNet 预处理: 保持 BGR、0~255 原始像素值, 不做通道交换/均值/缩放
    # (实测: 若换成 RGB 或做 (x-127.5)/128 归一化, obj 分支会塌成 0 -> 检不到脸)
    blob = blob.astype(np.float32)
    blob = blob.transpose(2, 0, 1)[None]              # NCHW
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    # 一次推理取全部输出(原来对每个输出各 run 一次, 12 次/帧, 极慢)
    outs = sess.run(out_names, {in_name: blob})
    raw = dict(zip(out_names, outs))

    sx = img_bgr.shape[1] / W
    sy = img_bgr.shape[0] / H
    dets = []
    n_total = 0          # 模型所有候选(各 stride 网格)数
    best_total = 0.0     # 全局最高分(用于诊断"未检测到"原因)
    for s in (8, 16, 32):
        cls = raw[f"cls_{s}"][0, :, 0]
        obj = raw[f"obj_{s}"][0, :, 0]
        bx = raw[f"bbox_{s}"][0]
        kp = raw[f"kps_{s}"][0]
        g = H // s
        for idx in range(g * g):
            # 官方解码: score = sqrt(clamp(cls,0,1) * clamp(obj,0,1)), 不再套 sigmoid
            c = min(max(float(cls[idx]), 0.0), 1.0)
            o = min(max(float(obj[idx]), 0.0), 1.0)
            score = float(np.sqrt(c * o))
            n_total += 1
            best_total = max(best_total, score)
            if score < score_thr:
                continue
            gy = idx // g
            gx = idx % g
            cx = (gx + bx[idx, 0]) * s
            cy = (gy + bx[idx, 1]) * s
            w = np.exp(bx[idx, 2]) * s
            h = np.exp(bx[idx, 3]) * s
            kps = np.array([[ (gx + kp[idx, 2 * p]) * s,
                              (gy + kp[idx, 2 * p + 1]) * s ]
                            for p in range(5)], dtype=np.float32)
            # 映射回原图
            dets.append({
                "bbox": ((cx - w / 2) * sx, (cy - h / 2) * sy, w * sx, h * sy),
                "score": float(score),
                "kps": kps * np.array([sx, sy]),
            })
    if not dets:
        return [], n_total, best_total
    # 高效 NMS(原 O(n^2) 纯 Python 循环在框多时会爆到几十秒)
    dets.sort(key=lambda d: d["score"], reverse=True)
    boxes = [tuple(map(float, d["bbox"])) for d in dets]
    scores = [float(d["score"]) for d in dets]
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)
    if hasattr(idxs, "flatten"):
        idxs = idxs.flatten()
    keep = [dets[i] for i in idxs]
    return keep, n_total, best_total


def align_one(img_bgr, det, target=112):
    """对单张脸做仿射对正 -> 112×112。"""
    kps = det["kps"]                         # (5,2) 顺序: re,le,nose,rmouth,lmouth
    # 按 x 位置重排为模板顺序（左眼 x 小、右眼 x 大；嘴角同理），避免左右翻转
    eyes = sorted(kps[:2], key=lambda p: p[0])
    mouth = sorted(kps[3:5], key=lambda p: p[0])
    src = np.array([eyes[0], eyes[1], kps[2], mouth[0], mouth[1]],
                   dtype=np.float32)
    M = cv2.estimateAffinePartial2D(src, TEMPLATE)[0]
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, (target, target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", default="")
    ap.add_argument("--img_dir", default="")
    ap.add_argument("--out", default="aligned.jpg")
    ap.add_argument("--out_dir", default="aligned")
    ap.add_argument("--target", type=int, default=112)
    ap.add_argument("--score", type=float, default=0.5,
                    help="人脸置信度阈值(默认0.5)")
    ap.add_argument("--nms", type=float, default=0.4,
                    help="IoU NMS 阈值(默认0.4)")
    ap.add_argument("--max_faces", type=int, default=0,
                    help="最多对正几张脸: 0=全部(按算力配置); "
                         "N=只取分数最高的前 N 张")
    args = ap.parse_args()

    sess, in_w, in_h = load_session(args.model)

    def process(path):
        img = cv2.imread(path)
        if img is None:
            print("[warn] 读不到", path)
            return
        dets, n_total, best_total = detect_yunet(
            sess, img, in_w, in_h, score_thr=args.score, nms_thr=args.nms)
        if not dets:
            print(f"[warn] 未检测到人脸: {path}  "
                  f"(模型共 {n_total} 候选, 最高分 {best_total:.3f} < 阈值 {args.score})")
            return
        # 按算力配置限制处理张数: max_faces<=0 表示全部(已按分数降序)
        dets = dets[:args.max_faces] if args.max_faces > 0 else dets
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.img_dir:
            os.makedirs(args.out_dir, exist_ok=True)
        n = 0
        for i, det in enumerate(dets):
            aligned = align_one(img, det, args.target)
            if aligned is None:
                print(f"[warn] 对正失败: {path} face{i}")
                continue
            if args.img_dir:
                out = os.path.join(args.out_dir, f"{stem}_face{i}.jpg")
            else:
                # 单图模式: 多脸加索引，单脸保持原文件名
                out = args.out if len(dets) == 1 else \
                    f"{os.path.splitext(args.out)[0]}_{i}{os.path.splitext(args.out)[1]}"
            cv2.imwrite(out, aligned)
            print(f"[ok] {path} face{i} -> {out}  score={det['score']:.3f}")
            n += 1
        if n == 0:
            print("[warn] 对正全失败:", path)

    if args.img_dir:
        for f in sorted(os.listdir(args.img_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                process(os.path.join(args.img_dir, f))
    elif args.img:
        process(args.img)
    else:
        print("需指定 --img 或 --img_dir")


if __name__ == "__main__":
    main()
