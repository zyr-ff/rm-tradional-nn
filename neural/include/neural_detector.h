#pragma once
#include "armor_types.h"
#include <opencv2/imgproc.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// ============================================================
//  神经网络装甲板检测器 (D3)  ——  基于 ONNX Runtime 直接推理
//
//  模型: Infantry-v8n-20260726 (深圳大学 RobotPilots 战队开源)
//  结构: YOLOv8n-Pose 主干 + 自定义装甲板检测头，一次推理同时给
//        出 4 个角点 + 颜色 + 装甲类别。
//
//  为什么不用 OpenCV 的 cv::dnn 加载？
//    fp16 模型 + YOLOv8-Pose 自定义头，OpenCV5 内置 ONNX 解析器
//    跑它会内部崩溃。ONNX Runtime 是官方引擎，完美支持，也是
//    真实 RM 工程的做法。
//
//  ★ 模型输出张量 [1, 21, 6300]（6300 = 480x640 下 anchor-free 预测网格
//    60*80 + 30*40 + 15*20）。每个候选(anchor)由 21 个通道描述：
//      [0..3 ] 颜色分支得分(已过 Sigmoid, 0~1): 0=蓝 1=红 2=白 3=紫
//      [4..12] 装甲类别分支得分(已过 Sigmoid, 0~1): label.txt
//              {1,2,3,4,5,outpost,guard,base,negative}
//      [13..20] 4 个关键点 (x,y) 坐标, 共 8 个数值, 单位 = 输入图(letterbox)
//               像素坐标, 顺序固定为 [左上, 左下, 右下, 右上]
//    ⚠️ 颜色/装甲分支已经在模型内部过了 Sigmoid，C++ 里【不要再 sigmoid】。
//    ⚠️ 角点坐标是“输入图(480x640)空间”的像素，需 (x-padX)/scale 还原原图。
// ============================================================

// 模型原始输出的一个候选框（不过滤颜色/类别），供 inspect() 与调试用
struct RawBox {
    float corners[8] = {0,0,0,0,0,0,0,0};  // x0,y0,x1,y1,x2,y2,x3,y3 (输入图像素) 顺序 左上,左下,右下,右上
    float colorScore = 0;  int colorCls = 0;   // 0=蓝 1=红 2=白 3=紫
    float armorScore = 0;  int armorCls = 0;   // 索引进 label.txt
    float score = 0;       // 综合置信 = colorScore * armorScore
};

class ArmorDetectorNN {
public:
    explicit ArmorDetectorNN(const std::string& modelPath,
                             const std::string& labelPath = "");

    // 模型是否成功加载
    bool ok() const { return ok_; }

    // 主入口：输入一帧 BGR 图像，返回识别出的装甲板列表（已含 keypoints/颜色/大小/数字）
    // color: 只保留该颜色的装甲板（"red" / "blue"）
    std::vector<Armor> detect(const cv::Mat& frame, const std::string& color = "red");

    // 诊断用：跑一次前向，打印 N / 头部信息 / Top 候选的颜色·类别·角点
    std::vector<RawBox> inspect(const cv::Mat& frame);

    int numClasses() const { return NUM_ARMOR; }   // 装甲类别数(9)，兼容旧调用
    int numAnchors() const { return nAnchors_; }    // 候选框总数 N（模型固定吐这么多）

private:
    // 1) letterbox 预处理：等比缩放+补灰边 -> 480x640 输入图
    cv::Mat letterbox(const cv::Mat& src, double& scale, int& padX, int& padY);

    // 2) 把输入图打包成 NCHW 的 float 张量（BGR->RGB + /255）
    std::vector<float> buildInput(const cv::Mat& sq);

    // 3) 跑一次前向，返回所有高分候选框（不过滤颜色）
    std::vector<RawBox> runNetwork(const cv::Mat& frame, double& scale, int& padX, int& padY);

    // 4) 把模型输出张量解析成候选框列表（按真实三头布局）
    std::vector<RawBox> parseOutput(const float* data, const std::vector<int64_t>& dims);

    // 辅助：角点顺序 [左上,左下,右下,右上] -> 是否大装甲（两灯条中心距法，与传统一致）
    static bool isLargeByCorners(const cv::Point2f kp[4]);

    static constexpr int NUM_COLOR  = 4;   // 颜色分支通道数
    static constexpr int NUM_ARMOR  = 9;   // 装甲类别分支通道数
    static constexpr int NUM_CORNER = 8;   // 4 个角点 * 2

    bool ok_ = false;
    Ort::Env env_;                                    // ORT 环境
    Ort::Session session_{nullptr};                   // 推理会话
    std::string inputName_, outputName_;              // 模型输入/输出节点名
    int   inputW_     = 640;                          // 模型输入宽
    int   inputH_     = 480;                          // 模型输入高（4:3，与相机分辨率一致）
    float confThres_  = 0.50f;                        // 颜色/装甲 置信度门槛
    float iouThres_   = 0.45f;                         // NMS 的 IOU 门槛
    int   nAnchors_   = 0;                            // 候选框总数 N

    std::vector<std::string> labels_;                 // label.txt（数字/类型名）
};
