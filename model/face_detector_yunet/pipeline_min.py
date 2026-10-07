#!/usr/bin/env python3
"""
最小验证: 相机采集 -> 直接显示, 无中间处理。
目的: 确认 PyGObject(Gst) 绑定可用 + 显示后端能上屏。

用法:
  python3 pipeline_min.py                       # autovideosink 自动选
  python3 pipeline_min.py --sink kms            # DRM 直出(无桌面, 推荐)
  python3 pipeline_min.py --sink wayland        # 需 weston 在跑
  python3 pipeline_min.py --device /dev/video1 --width 640 --height 480

验证前提(板子上):
  python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
  能打印版本即 python3-gi + typelib 已齐。
"""
import sys
import argparse
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GLib


def build_pipeline(args):
    # 显示后端: auto / kms / wayland / xv
    sink_map = {
        "auto": "autovideosink",
        "kms": "kmssink",
        "wayland": "waylandsink",
        "xv": "xvimagesink",
    }
    sink = sink_map.get(args.sink, "autovideosink")

    # v4l2src 采集 -> 转格式/缩放 -> 显示
    # 注意: kmssink 通常接受 NV12/I420, videoconvert 会按需转换
    desc = (
        f"v4l2src device={args.device} ! "
        f"videoconvert ! "
        f"videoscale ! "
        f"video/x-raw,width={args.width},height={args.height},framerate={args.fps}/1 ! "
        f"{sink}"
    )
    print("[pipeline]", desc)
    return Gst.parse_launch(desc)


def on_bus_message(bus, msg, loop):
    t = msg.type
    if t == Gst.MessageType.EOS:
        print("[info] 收到 EOS, 退出")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"[error] {err.message}")
        if dbg:
            print(f"[debug] {dbg}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = msg.parse_warning()
        print(f"[warn] {err.message}")
    return True


def main():
    ap = argparse.ArgumentParser(description="相机采集->显示 最小验证(无中间处理)")
    ap.add_argument("--device", default="/dev/video0", help="相机设备(默认/dev/video0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sink", default="auto",
                    choices=["auto", "kms", "wayland", "xv"],
                    help="显示后端(默认 auto)")
    args = ap.parse_args()

    Gst.init(None)
    loop = GLib.MainLoop()

    try:
        pipeline = build_pipeline(args)
    except Exception as e:
        print(f"[fatal] 管道构建失败: {e}")
        print("  多为缺少 GStreamer 元素(如 kmssink 在 -bad 插件)或 python3-gi 未装")
        sys.exit(1)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    pipeline.set_state(Gst.State.PLAYING)
    print("[info] PLAYING ... 按 Ctrl+C 停止")
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[info] 用户中断")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("[info] 已停止")


if __name__ == "__main__":
    main()
