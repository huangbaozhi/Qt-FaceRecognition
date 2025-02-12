#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <QImage>
#include <QPixmap>

using namespace cv;
using namespace cv::face;


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void initUi();

private slots:
    // 打开摄像头，显示视频流
    void on_pushButton_clicked();

    // 打开摄像头并实时显示
    void startVideoStream();
    void updateFrame();  // 定时更新视频帧

    void trainFaceRecognizer();  // 训练人脸识别模型

    void processFrame();
    QImage matToQImage(const Mat &mat);
    void saveFaceToDatabase(const Mat &faceImg, const QString &name);
    void createDatabase();

private:
    Ui::MainWindow *ui;

    cv::VideoCapture cap;  // 用于视频捕捉
    cv::CascadeClassifier faceCascade;  // 人脸检测分类器
    cv::Ptr<cv::face::LBPHFaceRecognizer> model;  // 人脸识别模型

    std::vector<cv::Mat> images;  // 存储训练图像
    std::vector<int> labels;  // 存储图像对应的标签

    QTimer *timer;

};
#endif // MAINWINDOW_H
