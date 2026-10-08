#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QWidget>
#include <QLabel>
#include <opencv2/opencv.hpp>
using namespace cv;

#define FACE_MAX_NUM 10

class capture_video;
class face_lookup;

class MainWindow : public QWidget {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
public slots:
    void showImg(cv::Mat*);
    void flushFace();
    void showAligned(cv::Mat*);
private:
    QLabel *pLabelImg;
    QLabel *pAlignedLabel;
    QLabel *pFaceMarkLabel[FACE_MAX_NUM];
    capture_video *pCaptureVideo;
    face_lookup  *pFaceLookup;
    int lastFaceNum;
};
#endif
