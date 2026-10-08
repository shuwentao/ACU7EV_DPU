#ifndef CAPTURE_VIDEO_H
#define CAPTURE_VIDEO_H
#include <QThread>
#include <opencv2/opencv.hpp>
using namespace cv;

#define IMG_SIZE_WIDTH   640
#define IMG_SIZE_HEIGHT  480
#define IMG_MAT_CACHE_NUM 3

class capture_video : public QThread {
    Q_OBJECT
public:
    capture_video(QObject *parent = 0);
    void run();
    Mat imgMat[IMG_MAT_CACHE_NUM];
    static Mat* getImgMat(capture_video *pthis);
    static void freeImgMat(capture_video *pthis);
private:
    pthread_mutex_t mutex;
    int imgIn, imgOut, imgUse;
    Mat* getFreeMat(void);
};
#endif
