#include "face_lookup.h"
#include "capture_video.h"
#include <QDebug>
#include <QCoreApplication>
#include <QStringList>
#include <vector>

// 112x112 标准 5 点模板(左眼, 右眼, 鼻尖, 左嘴角, 右嘴角)
static const cv::Point2f TEMPLATE[5] = {
    cv::Point2f(38.2946f, 51.6963f),
    cv::Point2f(73.5318f, 51.6963f),
    cv::Point2f(56.0252f, 71.7366f),
    cv::Point2f(41.5493f, 92.3655f),
    cv::Point2f(70.7299f, 92.3655f)
};

face_lookup::face_lookup(QObject *parent) : QThread(parent)
{
    pthread_mutex_init(&mutex, NULL);
}

// 手动计算相似变换(4自由度: 旋转+各向同性缩放+平移), 等价于 opencv 的
// estimateAffinePartial2D, 但不依赖 video 模块(部分 opencv 构建不含该函数)。
// 用全部对应点最小二乘求最优 a,b(平移 tx,ty)。
static cv::Mat estimatePartialAffine(const std::vector<cv::Point2f>& src,
                                     const std::vector<cv::Point2f>& dst)
{
    const int n = (int)src.size();
    if(n < 2) return cv::Mat();
    cv::Point2f Sc(0,0), Dc(0,0);
    for(int i = 0; i < n; i++) { Sc += src[i]; Dc += dst[i]; }
    Sc *= 1.f / n; Dc *= 1.f / n;
    double Cxx = 0, num = 0, den = 0;
    for(int i = 0; i < n; i++) {
        double sx = src[i].x - Sc.x, sy = src[i].y - Sc.y;
        double dx = dst[i].x - Dc.x, dy = dst[i].y - Dc.y;
        Cxx += sx*sx + sy*sy;            // Σ|xs|²
        num += sx*dx + sy*dy;            // = a·Cxx
        den += sx*dy - sy*dx;            // = b·Cxx
    }
    if(Cxx < 1e-8) return cv::Mat();
    double a = num / Cxx;
    double b = den / Cxx;
    double tx = Dc.x - (a*Sc.x - b*Sc.y);
    double ty = Dc.y - (b*Sc.x + a*Sc.y);
    cv::Mat M = (cv::Mat_<float>(2,3) << (float)a, (float)-b, (float)tx,
                                          (float)b, (float) a, (float)ty);
    return M;
}

void face_lookup::run()
{
    cv::Mat *pImg;
    if(init() < 0) qDebug("cascade load fail");

    while(1)
    {
        // 自旋等一帧
        while(1)
        {
            pImg = capture_video::getImgMat(pCaptureVideo);
            if(pImg) break;
        }

        cv::Mat gray;
        cv::cvtColor(*pImg, gray, cv::COLOR_BGR2GRAY);
        sendImage(pImg);   // 每帧都显示原图

        if(face_cascade.empty())
        {
            headers.clear(); alignedFaces.clear();
            pthread_mutex_lock(&mutex); sendFlushFace(); sendAligned(nullptr);
            continue;
        }

        // 缩小图检测提速(1/2 分辨率, 耗时约 1/4)
        const double detectScale = 0.5;
        cv::Mat graySmall;
        cv::resize(gray, graySmall, cv::Size(), detectScale, detectScale, cv::INTER_LINEAR);

        std::vector<cv::Rect> rects;
        face_cascade.detectMultiScale(graySmall, rects, 1.1, 3, 0, cv::Size(20, 20));

        headers.clear();
        alignedFaces.clear();
        for(size_t i = 0; i < rects.size(); i++)
        {
            cv::Rect r((int)(rects[i].x / detectScale),
                       (int)(rects[i].y / detectScale),
                       (int)(rects[i].width / detectScale),
                       (int)(rects[i].height / detectScale));
            headers.push_back(r);

            // 无 landmark 模型: 由 bbox 比例合成 5 关键点(仅适用正面脸, 启发式)
            cv::Point2f src[5] = {
                cv::Point2f(r.x + 0.34f  * r.width,  r.y + 0.46f  * r.height), // 左眼
                cv::Point2f(r.x + 0.656f * r.width,  r.y + 0.46f  * r.height), // 右眼
                cv::Point2f(r.x + 0.50f  * r.width,  r.y + 0.64f  * r.height), // 鼻尖
                cv::Point2f(r.x + 0.37f  * r.width,  r.y + 0.825f * r.height), // 左嘴角
                cv::Point2f(r.x + 0.63f  * r.width,  r.y + 0.825f * r.height)  // 右嘴角
            };

            cv::Mat M = estimatePartialAffine(
                std::vector<cv::Point2f>(src, src + 5),
                std::vector<cv::Point2f>(TEMPLATE, TEMPLATE + 5));
            cv::Mat aligned;
            if(!M.empty())
                cv::warpAffine(*pImg, aligned, M, cv::Size(112, 112));
            alignedFaces.push_back(aligned);
        }

        // 显示/保存最大脸的对齐结果
        if(!alignedFaces.empty() && !alignedFaces[0].empty())
            cv::imwrite("aligned/live.jpg", alignedFaces[0]);

        pthread_mutex_lock(&mutex);
        sendFlushFace();
        if(!alignedFaces.empty()) sendAligned(&alignedFaces[0]);
        else                      sendAligned(nullptr);
    }
}

int face_lookup::init()
{
    // 优先在可执行文件同目录 / 上级 haar_train 找, PC 直接 run 也能加载;
    // 再回退相对路径(板子 make install 后)和 /home/root(板子部署路径)。
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList cascade_paths = {
        appDir + "/haarcascade_frontalface_alt.xml",
        appDir + "/../haar_train/haarcascade_frontalface_alt.xml",
        "haar_train/haarcascade_frontalface_alt.xml",
        "/home/root/haar_train/haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt.xml"
    };
    for (const QString& p : cascade_paths)
    {
        if (face_cascade.load(p.toLocal8Bit().constData()))
        {
            qDebug() << "cascade loaded:" << p;
            return 0;
        }
    }
    qDebug() << "cascade load FAIL, tried:" << cascade_paths;
    return -1;
}
