#include "number_classifier.h"
#include <algorithm>
#include <fstream>
#include <iostream>

NumberClassifier::NumberClassifier(const std::string& modelPath, const std::string& labelPath,
                                   double threshold)
    : threshold_(threshold)
{
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
        ready_ = true;
    } catch (const cv::Exception& e) {
        std::cerr << "[NumberClassifier] 加载模型失败: " << e.what() << std::endl;
        return;
    }
    std::ifstream f(labelPath);
    if (!f.is_open()) {
        std::cerr << "[NumberClassifier] 打开标签文件失败: " << labelPath << std::endl;
        ready_ = false;
        return;
    }
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) labels_.push_back(line);
}

cv::Mat NumberClassifier::extractNumberImage(const cv::Mat& frame, const Armor& armor)
{
    // ---- 透视变换参数（与训练时一致，勿改）----
    const int lightLength = 12;           // 变换后灯条像素高度
    const int warpHeight = 28;            // 变换后图像高
    const int smallW = 32, largeW = 54;   // 小/大装甲变换后宽
    const cv::Size roiSize(20, 28);       // 数字 ROI 尺寸

    // 源四边形：左灯条下、左灯条上、右灯条上、右灯条下（沿灯条从下往上、再从上往下）
    cv::Point2f src[4] = {armor.left.bottom, armor.left.top, armor.right.top, armor.right.bottom};

    // 目标四边形：灯条垂直居中，宽度按装甲类型
    const int topY = (warpHeight - lightLength) / 2 - 1;   // = 7
    const int bottomY = topY + lightLength;                // = 19
    const int warpW = armor.isLarge ? largeW : smallW;
    cv::Point2f dst[4] = {
        cv::Point2f(0, bottomY), cv::Point2f(0, topY),
        cv::Point2f(warpW - 1, topY), cv::Point2f(warpW - 1, bottomY)};

    cv::Mat warped;
    cv::warpPerspective(frame, warped, cv::getPerspectiveTransform(src, dst),
                        cv::Size(warpW, warpHeight));

    // 取中间 20x28 的数字区域
    return warped(cv::Rect((warpW - roiSize.width) / 2, 0, roiSize.width, roiSize.height));
}

void NumberClassifier::classify(const cv::Mat& frame, std::vector<Armor>& armors, bool filter)
{
    if (!ready_) return;

    std::vector<Armor> kept;
    for (auto& armor : armors) {
        // 1) 透视变换拿数字 ROI
        cv::Mat num = extractNumberImage(frame, armor);

        // 2) 大津法二值化 -> 归一化到 0/1 浮点（训练时的输入分布）
        cv::cvtColor(num, num, cv::COLOR_BGR2GRAY);
        cv::threshold(num, num, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Mat f32;
        num.convertTo(f32, CV_32F, 1.0 / 255.0);

        // 3) 推理
        cv::Mat blob;
        cv::dnn::blobFromImage(f32, blob);
        net_.setInput(blob);
        cv::Mat out = net_.forward();  // [1, 类别数]

        // 4) softmax -> 取最大概率
        float maxProb = *std::max_element(out.begin<float>(), out.end<float>());
        cv::Mat prob;
        cv::exp(out - maxProb, prob);
        prob /= cv::sum(prob)[0];
        cv::Point classId;
        double conf = 0.0;
        cv::minMaxLoc(prob.reshape(1, 1), nullptr, &conf, nullptr, &classId);

        int idx = classId.x;
        armor.numberConf = conf;
        armor.number = (idx >= 0 && idx < (int)labels_.size()) ? labels_[idx] : "unknown";

        // 5) 过滤：置信度不足 / negative 类 / 数字与装甲类型不符 => 判为假装甲
        if (!filter) continue;  // 只填数字不剔除（评估模式）
        bool typeMismatch =
            armor.isLarge ? (armor.number == "outpost" || armor.number == "2" ||
                             armor.number == "guard")
                          : (armor.number == "1" || armor.number == "base");
        if (conf >= threshold_ && armor.number != "negative" && !typeMismatch)
            kept.push_back(armor);
    }
    if (filter) armors.swap(kept);
}
