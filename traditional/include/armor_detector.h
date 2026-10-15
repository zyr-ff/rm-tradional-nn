#pragma once
#include "armor_types.h"
#include <string>
#include <vector>

// ============================================================
//  传统方案装甲板检测器（D2）
//
//  原理（作业报告里可以直接引用这套流程描述）：
//   1. 灰度化 + 固定阈值二值化：灯条是高亮区域，亮度远高于环境
//   2. findContours 找所有亮区轮廓，minAreaRect 拟合旋转矩形
//   3. 用几何特征筛掉非灯条（长宽比、倾角）
//   4. 在轮廓内统计 R/B 通道均值判断灯条颜色
//   5. 两两配对灯条成装甲板：长度比 / 中心距 / 连线水平角 三重几何约束
//
//  参数参考开源项目 rm_vision (chenjunnn) 的默认值，可根据视频效果微调
// ============================================================
class ArmorDetector {
public:
    explicit ArmorDetector(int binaryThres = 85);

    // 主入口：输入一帧 BGR 图像，返回识别出的装甲板列表
    // color: 只保留该颜色的装甲板（"red" / "blue"）
    std::vector<Armor> detect(const cv::Mat& frame, const std::string& color = "red");

    // 重载：允许外部传入自定义二值图（例如蓝光场景用 B 通道阈值替代灰度阈值），
    // 后续找灯条/配对流程与上面完全一致。
    std::vector<Armor> detect(const cv::Mat& frame, const cv::Mat& binary,
                              const std::string& color = "red");

    std::vector<LightBar> lastLights;  // 调试用：本帧筛出的所有灯条

private:
    std::vector<LightBar> findLights(const cv::Mat& bgr, const cv::Mat& binary);
    std::vector<Armor> matchLights(const std::vector<LightBar>& lights);
    bool isLight(const LightBar& l) const;

    int binaryThres_;          // 二值化阈值

    // ---- 灯条筛选参数 ----
    double minRatio_ = 0.1;    // 短边/长边 下限（灯条细长）
    double maxRatio_ = 0.4;    // 短边/长边 上限
    double maxTiltAngle_ = 40.0;   // 灯条最大倾角(度)，倾斜过大的大概率不是灯条

    // ---- 灯条配对参数（单位：灯条平均长度）----
    double minLightRatio_ = 0.7;       // 两灯条长度比下限（应差不多长）
    double minSmallDist_ = 0.8;        // 小装甲：中心距范围
    double maxSmallDist_ = 3.2;
    double minLargeDist_ = 3.2;        // 大装甲：中心距范围（与小区间衔接）
    double maxLargeDist_ = 5.5;
    double maxPairAngle_ = 35.0;       // 两灯条连线与水平方向最大夹角(度)
};
