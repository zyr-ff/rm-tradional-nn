#pragma once
#include "armor_types.h"
#include <opencv2/dnn.hpp>      // readNetFromONNX / blobFromImage
#include <string>
#include <vector>

// ============================================================
//  数字识别分类器（D2，传统/NN 两方案共用）
//
//  原理：
//   1. 透视变换：把斜着拍的装甲板"掰正"——按 4 个关键点把装甲板区域
//      投影成一张正对相机的标准尺寸图（小 32x28 / 大 54x28，灯条高 12px 居中）
//   2. 取中间 20x28 的数字 ROI（两侧留白是为了容纳大数字）
//   3. 大津法(OTSU)自动阈值二值化，得到干净的黑白数字
//   4. 输入 mlp.onnx 多层感知机做分类，softmax 取最大概率类别
//
//  预处理流程与模型训练一致（参考 rm_vision 的 NumberClassifier），
//  ⚠️ 尺寸/二值化方式不能随意改，否则识别率会暴跌
// ============================================================
class NumberClassifier {
public:
    // modelPath: mlp.onnx 路径  labelPath: label.txt 路径  threshold: 置信度门槛
    NumberClassifier(const std::string& modelPath, const std::string& labelPath,
                     double threshold = 0.6);

    bool ok() const { return ready_; }

    // 就地识别：为每个 armor 填 number / numberConf，
    // 并剔除低置信度、negative、以及"数字与装甲类型不符"的假装甲
    // filter=false 时只填数字不做剔除（误差评估等场景用）
    void classify(const cv::Mat& frame, std::vector<Armor>& armors, bool filter = true);

private:
    cv::Mat extractNumberImage(const cv::Mat& frame, const Armor& armor);  // 调试可视化用
    cv::dnn::Net net_;
    std::vector<std::string> labels_;  // 类别名，按 label.txt 行序 = 输出索引
    double threshold_;
    bool ready_ = false;
};
