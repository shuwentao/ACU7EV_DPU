#!/usr/bin/env python3
"""
阶段1: V4L2 输入 -> 人脸检测(YuNet) -> GStreamer 显示
即一条 v4l2src -> appsink -> [Python检测+画框] -> appsrc -> sink 的流水线。

实测已修的两个坑:
  - detect_yunet 的 NMS 原是 O(n^2) 纯 Python 循环, 均匀/背景画面会爆到 33s,
    已换 cv2.dnn.NMSBoxes (33s -> 0.47s, 见 align.py)。
  - 实时源显示后端(autovideosink/xvimagesink)默认 sync=true 会丢迟到帧导致冻结,
    显式 sink 加 sync=false; appsrc 手动递增时间戳(不靠 do-timestamp)。

板子用 --sink kms (DRM, 无桌面); PC 用 --sink xv / auto。
"""
import os
import sys
import time
import argparse
import traceback
import queue
import threading
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GLib

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import load_session, detect_yunet


def draw_dets(frame, dets, max_faces=0):
    show = dets[:max_faces] if max_faces > 0 else dets
    for d in show:
        x, y, w, h = [int(v) for v in d["bbox"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{d['score']:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for px, py in d["kps"]:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)
    return frame


class Detector:
    def __init__(self, args):
        if args.passthrough:
            self.sess = self.in_w = self.in_h = None
        else:
            self.sess, self.in_w, self.in_h = load_session(args.model)
        self.args = args
        self.w, self.h, self.fps = args.width, args.height, args.fps
        self.disp_appsrc = None
        self.running = True
        self.frame_q = queue.Queue(maxsize=1)   # 只保留最新一帧
        self.lock = threading.Lock()
        self.frames = 0
        self.t0 = time.time()
        self.last_dets, self.last_best, self.last_ms = [], 0.0, 0.0
        self.det_runs = 0
        self.push_no = 0
        self.n_feed = 0
        self.n_get = 0

    def caps_str(self):
        return (f"video/x-raw,format=BGR,width={self.w},height={self.h},"
                f"framerate={self.fps}/1")

    # 采集流线程: 取帧入队, 不阻塞
    def on_new_sample(self, appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            st = sample.get_caps().get_structure(0)
            cw = st.get_value("width"); ch = st.get_value("height")
            data = buf.extract_dup(0, buf.get_size())
            frame = np.frombuffer(data, dtype=np.uint8).reshape(ch, cw, 3).copy()
            try:
                self.frame_q.put_nowait(frame); self.n_feed += 1
            except queue.Full:
                pass
            return Gst.FlowReturn.OK
        except Exception:
            print("[EXCEPTION in appsink cb]\n" + traceback.format_exc(), flush=True)
            return Gst.FlowReturn.OK

    # 单 worker 线程: 检测 + 画框 + 推显示
    def worker(self):
        print("[info] worker 启动", flush=True)
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.n_get += 1
            self.frames += 1
            if self.sess is not None:
                run = (self.frames % self.args.detect_interval == 0)
                if run:
                    try:
                        t = time.perf_counter()
                        dets, n_total, best = detect_yunet(
                            self.sess, frame, self.in_w, self.in_h,
                            score_thr=self.args.score, nms_thr=self.args.nms)
                        ms = (time.perf_counter() - t) * 1000
                        with self.lock:
                            self.last_dets, self.last_best, self.last_ms = dets, best, ms
                        self.det_runs += 1
                        if self.det_runs % 10 == 0:
                            print(f"[info] 检测已跑 {self.det_runs} 次, 最近 {ms:.1f}ms, "
                                  f"人脸 {len(dets)} (最高 {best:.2f})", flush=True)
                    except Exception:
                        print("[EXCEPTION in detect]\n" + traceback.format_exc(), flush=True)
                        time.sleep(0.5)
                        dets, best = [], 0.0
                else:
                    with self.lock:
                        dets, best = self.last_dets, self.last_best
                draw_dets(frame, dets, self.args.max_faces)
            # 推回显示: 手动递增时间戳 (sync=false 的 sink 不会丢迟到帧)
            try:
                outbuf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
                outbuf.fill(0, frame.tobytes())
                dur = Gst.SECOND // max(1, self.fps)
                outbuf.pts = self.push_no * dur
                outbuf.duration = dur
                self.push_no += 1
                if self.disp_appsrc is not None:
                    self.disp_appsrc.emit("push-buffer", outbuf)
            except Exception:
                print("[EXCEPTION in push]\n" + traceback.format_exc(), flush=True)
            if self.frames % 30 == 0:
                dt = time.time() - self.t0
                with self.lock:
                    ms = self.last_ms
                print(f"[info] 显示 {self.frames} 帧, {self.frames/dt:.1f} fps, "
                      f"单次检测={ms:.1f}ms | feed={self.n_feed},get={self.n_get}",
                      flush=True)


def on_bus(bus, msg, loop, tag):
    t = msg.type
    if t == Gst.MessageType.EOS:
        print(f"[{tag}] EOS"); loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"[{tag} ERROR] {err.message}")
        if dbg:
            print(f"[{tag} DEBUG] {dbg}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, _ = msg.parse_warning()
        print(f"[{tag} WARN] {err.message}")
    return True


def main():
    ap = argparse.ArgumentParser(description="V4L2->YuNet检测->GStreamer显示")
    ap.add_argument("--model", default="face_detection_yunet_2023mar.onnx")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--max_faces", type=int, default=0)
    ap.add_argument("--detect_interval", type=int, default=3,
                    help="每 N 帧跑一次检测(其余帧复用上次框)")
    ap.add_argument("--sink", default="xv", choices=["auto", "kms", "wayland", "xv"],
                    help="显示后端: kms=板子(DRM); xv/auto=PC")
    ap.add_argument("--passthrough", action="store_true", help="跳过检测原帧直传")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not args.passthrough and not os.path.exists(args.model):
        print(f"[fatal] 模型不存在: {args.model}")
        sys.exit(1)

    Gst.init(None)
    loop = GLib.MainLoop()
    d = Detector(args)

    sink_map = {"auto": "autovideosink", "kms": "kmssink sync=false",
                "wayland": "waylandsink sync=false", "xv": "xvimagesink sync=false"}
    disp = sink_map[args.sink]

    cap_desc = (
        f"v4l2src device={args.device} ! videoconvert ! videoscale ! "
        f"{d.caps_str()} ! appsink name=src emit-signals=true "
        f"max-buffers=1 drop=true"
    )
    disp_desc = (
        f"appsrc name=out format=time is-live=true block=false "
        f"caps={d.caps_str()} ! queue ! videoconvert ! {disp}"
    )
    cap_pipe = Gst.parse_launch(cap_desc)
    disp_pipe = Gst.parse_launch(disp_desc)
    appsink = cap_pipe.get_by_name("src")
    appsink.connect("new-sample", d.on_new_sample)
    d.disp_appsrc = disp_pipe.get_by_name("out")
    for tag, p in (("cap", cap_pipe), ("disp", disp_pipe)):
        bus = p.get_bus(); bus.add_signal_watch()
        bus.connect("message", on_bus, loop, tag)
    worker = threading.Thread(target=d.worker, daemon=True)
    worker.start()
    disp_pipe.set_state(Gst.State.PLAYING)
    cap_pipe.set_state(Gst.State.PLAYING)
    print(f"[info] PLAYING {args.width}x{args.height}@{args.fps} "
          f"score>={args.score} sink={args.sink} | Ctrl+C 停止")
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[info] 用户中断")
    finally:
        d.running = False
        cap_pipe.set_state(Gst.State.NULL)
        disp_pipe.set_state(Gst.State.NULL)
        worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
