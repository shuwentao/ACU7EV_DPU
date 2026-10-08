#ifndef FACE_LOOKUP_H
#define FACE_LOOKUP_H
#include <QThread>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv;

#define FACE_MAX_NUM 10

class capture_video;

class face_lookup : public QThread {
    Q_OBJECT
public:
    face_lookup(QObject *parent = 0);
    void run();
    capture_video *pCaptureVideo;
    std::vector<Rect> headers;     // 人脸框(原图坐标)
    std::vector<Mat>  alignedFaces;// 对齐后的 112x112 脸
    pthread_mutex_t mutex;         // 与 mainwindow 共享(同原工程)
signals:
    void sendImage(cv::Mat*);
    void sendFlushFace();
    void sendAligned(cv::Mat*);    // 发送对齐结果给界面
private:
    int init();
    CascadeClassifier face_cascade;
};
#endif
