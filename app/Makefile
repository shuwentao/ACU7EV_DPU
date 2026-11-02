# 交叉编译示例（宿主机交叉工具链）：
#   make CC=aarch64-linux-gnu-gcc
# 板上原生编译：
#   make CC=gcc
# 依赖：libdrm（提供 xf86drm.h / xf86drmMode.h / libdrm/drm_fourcc.h）

CC      ?= gcc
CFLAGS  ?= -O2 -Wall
LDFLAGS ?= -ldrm

TARGET  := drm_dual_plane

$(TARGET): drm_dual_plane.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
