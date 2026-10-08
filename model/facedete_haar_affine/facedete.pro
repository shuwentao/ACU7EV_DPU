# 工程A: Haar 检测 + 仿射对齐(不用 YuNet)
QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = facedete_haar_affine
TEMPLATE = app

SOURCES += main.cpp \
           mainwindow.cpp \
           capture_video.cpp \
           face_lookup.cpp

HEADERS  += mainwindow.h \
           capture_video.h \
           face_lookup.h

# 优先用 pkg-config 取 opencv 头文件/库(PC 与板子通用);
# 若环境没有 .pc 文件, 兜底到标准 Linux 路径(/usr/include/opencv4)。
CONFIG += link_pkgconfig
OPENCV_PC =
packagesExist(opencv4): OPENCV_PC = opencv4
else: packagesExist(opencv): OPENCV_PC = opencv

!isEmpty(OPENCV_PC) {
    PKGCONFIG += $$OPENCV_PC
} else {
    message("pkg-config 未找到 opencv, 使用兜底路径 /usr/include/opencv4")
    INCLUDEPATH += /usr/include/opencv4
    LIBS += -lopencv_core -lopencv_highgui -lopencv_videoio \
            -lopencv_imgproc -lopencv_imgcodecs -lopencv_face -lopencv_objdetect
}

target.path = /home/root
# Haar 级联模型随工程部署(板子 make install 用)
haar_files.path = /home/root/haar_train
haar_files.files = $$PWD/../haar_train/haarcascade_frontalface_alt.xml
INSTALLS += target haar_files

# PC 直接 run: 把 Haar 模型复制到构建目录(可执行文件同目录), 免得找不到
copy_haar.target = copy_haar
copy_haar.commands = $$QMAKE_COPY $$shell_quote($$PWD/../haar_train/haarcascade_frontalface_alt.xml) $$shell_quote($$OUT_PWD)
QMAKE_EXTRA_TARGETS += copy_haar
PRE_TARGETDEPS += copy_haar
