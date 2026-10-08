#include "mainwindow.h"
#include "capture_video.h"
#include "face_lookup.h"
#include <QGridLayout>
#include <QHBoxLayout>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent)
{
    pCaptureVideo = new capture_video(this);
    pFaceLookup   = new face_lookup(this);
    pFaceLookup->pCaptureVideo = pCaptureVideo;

    QObject::connect(pFaceLookup, SIGNAL(sendImage(cv::Mat*)),    this, SLOT(showImg(cv::Mat*)));
    QObject::connect(pFaceLookup, SIGNAL(sendFlushFace()),        this, SLOT(flushFace()));
    QObject::connect(pFaceLookup, SIGNAL(sendAligned(cv::Mat*)),  this, SLOT(showAligned(cv::Mat*)));

    pCaptureVideo->start();
    pFaceLookup->start();

    pLabelImg    = new QLabel; pLabelImg->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    pAlignedLabel = new QLabel("aligned"); pAlignedLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    pAlignedLabel->setStyleSheet("border:1px solid gray;");

    QHBoxLayout *pH = new QHBoxLayout(this);
    pH->addWidget(pLabelImg);
    pH->addWidget(pAlignedLabel);

    for(int i = 0; i < FACE_MAX_NUM; i++)
    {
        pFaceMarkLabel[i] = new QLabel(pLabelImg);
        pFaceMarkLabel[i]->setStyleSheet("border:1px solid yellow;");
        pFaceMarkLabel[i]->hide();
    }
    lastFaceNum = 0;
    QDir().mkpath("aligned");
}

MainWindow::~MainWindow() {}

void MainWindow::showImg(cv::Mat *img)
{
    QImage tmp = QImage(img->data, img->cols, img->rows, QImage::Format_RGB888).rgbSwapped();
    pCaptureVideo->freeImgMat(pCaptureVideo);
    pLabelImg->setPixmap(QPixmap::fromImage(tmp));
}

void MainWindow::flushFace()
{
    int num = pFaceLookup->headers.size();
    if(num > FACE_MAX_NUM) num = FACE_MAX_NUM;
    for(int i = 0; i < num; i++)
    {
        cv::Rect r = pFaceLookup->headers[i];
        pFaceMarkLabel[i]->setGeometry(r.x, r.y, r.width, r.height);
        pFaceMarkLabel[i]->show();
    }
    pthread_mutex_unlock(&pFaceLookup->mutex);
    for(int i = num; i < lastFaceNum; i++)
        pFaceMarkLabel[i]->hide();
    lastFaceNum = num;
}

void MainWindow::showAligned(cv::Mat *img)
{
    if(!img || img->empty()) { pAlignedLabel->hide(); return; }
    QImage q = QImage(img->data, img->cols, img->rows, QImage::Format_RGB888).rgbSwapped();
    pAlignedLabel->setPixmap(QPixmap::fromImage(q));
    pAlignedLabel->setFixedSize(img->cols, img->rows);
}
