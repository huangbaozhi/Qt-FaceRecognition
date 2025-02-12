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
    void updateFrameTest();  // 定时更新视频帧

    // 使用训练好的模型进行人脸识别
    void recognizeFace();

private:
    Ui::MainWindow *ui;

    cv::VideoCapture cap;  // 用于视频捕捉
    cv::CascadeClassifier faceCascade;  // 人脸检测分类器
    cv::Ptr<cv::face::LBPHFaceRecognizer> model;  // 人脸识别模型

    std::vector<cv::Mat> images;  // 存储训练图像
    std::vector<int> labels;  // 存储图像对应的标签

    //sqlite3* db;  // SQLite数据库连接

    // cv::VideoCapture cap;  // 摄像头对象
    // cv::CascadeClassifier faceCascade;  // 人脸检测分类器
    // cv::Ptr<cv::face::LBPHFaceRecognizer> model;  // 人脸识别模型
    // std::vector<cv::Mat> images;  // 训练图像
    // std::vector<int> labels;  // 训练标签

    void trainFaceRecognizer();  // 训练人脸识别模型

    void openDatabase();  // 打开数据库
    void storeFaceInDatabase(const std::string& name, const cv::Mat& faceDescriptor);  // 将人脸特征存储到数据库
    void loadFacesFromDatabase();  // 从数据库加载人脸数据进行比对
    void trainFaceRecognizer(const cv::Mat& face, int label);  // 训练人脸识别模型


    // VideoCapture capture;
    // //CascadeClassifier faceCascade_2;
    // Ptr<LBPHFaceRecognizer> faceRecognizer;
    QTimer *timer;

    void initOpenCV();
    void processFrame();
    QImage matToQImage(const Mat &mat);
    void saveFaceToDatabase(const Mat &faceImg, const QString &name);
    void createDatabase();

};
#endif // MAINWINDOW_H
