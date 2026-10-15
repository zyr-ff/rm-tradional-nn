#pragma once
#include <opencv2/core.hpp>      // Mat/Point 等
#include <opencv2/imgproc.hpp>   // findContours/warpPerspective/cvtColor
#include <opencv2/geometry.hpp>  // OpenCV5: minAreaRect/boundingRect/getPerspectiveTransform
#include <string>

// ============================================================
//  共用数据类型：灯条 / 装甲板
//  传统方案(D2)与神经网络方案(D3)共用，避免两套结构重复
// ============================================================

// 单根灯条（装甲板两侧的发光灯管）
struct LightBar {
    cv::Point2f top;         // 灯条上端点（图像 y 较小）
    cv::Point2f bottom;      // 灯条下端点
    cv::Point2f center;      // 中点
    double length = 0.0;     // 长度(px)，即 top-bottom 距离
    double width = 0.0;      // 宽度(px)，旋转矩形短边
    double tiltAngle = 0.0;  // 与竖直方向的偏角(度)
    bool isRed = true;       // 颜色：true=红 false=蓝
    cv::RotatedRect rect;    // 轮廓拟合出的旋转矩形（绘制/调试用）
};

// 装甲板
// ⚠️ keypoints 顺序与 v8n 检测网络约定一致（见 camera_config.h）：
//    0=左上  1=左下  2=右下  3=右上
//    PnP 的 3D 物体点顺序(armorObjectPoints)与此一一对应，不要改！
struct Armor {
    cv::Point2f keypoints[4];
    LightBar left, right;         // 左右两根灯条（绘制/调试用）
    std::string color = "unknown";// "red" / "blue"
    bool isLarge = false;         // true=大装甲(235x127) false=小装甲(140x125)
    std::string number = "unknown";      // 数字识别结果（label.txt 中的类别名）
    double numberConf = 0.0;             // 数字识别置信度 0~1
    cv::Point3d position;         // PnP 解算：装甲板中心在相机坐标系下的位置(米)
    bool positionValid = false;   // PnP 是否解算成功（作业要求的异常处理）
};
