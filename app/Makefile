# 交叉编译示例（宿主机交叉工具链）：
#   make CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
# 板上原生编译：
#   make CC=gcc CXX=g++
# 依赖：
#   libdrm（提供 xf86drm.h / xf86drmMode.h / libdrm/drm_fourcc.h）—— 两个目标都需要
#   VART / XIR（提供 vitis_ai_library / vart / xir 头与库）—— 仅 face_detect_dpu 需要

CC       ?= gcc
CXX      ?= g++
CFLAGS   ?= -O2 -Wall
CXXFLAGS ?= -O2 -Wall -std=c++17
LDFLAGS  ?= -ldrm

TARGETS  := drm_dual_plane face_detect_dpu

all: $(TARGETS)

drm_dual_plane: drm_dual_plane.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

face_detect_dpu: face_detect_dpu.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) \
	    -lvart-dpu-runner -lvitis_ai_library-runner_helper -lvitis_ai_library-math \
	    -lvart-runner -lvart-util -lvart-buffer-object -lvart-mem-manager \
	    -lvart-dpu-controller -lvart-xrt-device-handle -lxir -lxrt_coreutil -lrt

clean:
	rm -f $(TARGETS)

.PHONY: all clean
