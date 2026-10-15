#include "neural_detector.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/dnn.hpp>   // 仅用于 cv::dnn::NMSBoxes（去重工具，推理本身走 ORT）

// ============================================================
//  ArmorDetectorNN 实现（ONNX Runtime 直接推理）
//  流程：letterbox -> 打包 NCHW -> 前向 -> 解析三头 -> 颜色过滤 -> NMS -> 组装 Armor
// ============================================================

ArmorDetectorNN::ArmorDetectorNN(const std::string& modelPath, const std::string& labelPath) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "armor_nn");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        std::ifstream f(modelPath, std::ios::binary);
        if (!f) { std::cerr << "[NN] 无法打开模型文件: " << modelPath << std::endl; return; }
        std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
        if (blob.empty()) { std::cerr << "[NN] 模型文件为空: " << modelPath << std::endl; return; }
        session_ = Ort::Session(env_, blob.data(), blob.size(), opts);

        auto inNames  = session_.GetInputNames();
        auto outNames = session_.GetOutputNames();
        inputName_  = inNames.empty()  ? "" : inNames[0];
        outputName_ = outNames.empty() ? "" : outNames[0];
        auto outTI = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        std::vector<int64_t> od = outTI.GetShape();
        // 输出 [1, 21, N] 或 [1, N, 21]：取不等于 1 与 21 的那个维度作为候选数 N
        nAnchors_ = 0;
        for (int64_t d : od) if (d != 1 && d != 21) nAnchors_ = (int)d;
        ok_ = true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[NN] ONNX Runtime 错误: " << e.what() << std::endl;
        ok_ = false;
    }

    // 可选：加载 label.txt（装甲类别名）
    if (!labelPath.empty()) {
        std::ifstream lf(labelPath);
        std::string line;
        while (std::getline(lf, line))
            if (!line.empty()) labels_.push_back(line);
    }
}

// ---------- 1) letterbox：等比缩放 + 补灰边，得到 480x640 输入图 ----------
cv::Mat ArmorDetectorNN::letterbox(const cv::Mat& src, double& scale, int& padX, int& padY) {
    int w = src.cols, h = src.rows;
    scale = std::min((double)inputW_ / w, (double)inputH_ / h);
    int nw = cvRound(w * scale), nh = cvRound(h * scale);
    cv::Mat r;
    cv::resize(src, r, cv::Size(nw, nh));
    padX = (inputW_ - nw) / 2;
    padY = (inputH_ - nh) / 2;
    cv::Mat out;
    cv::copyMakeBorder(r, out, padY, inputH_ - nh - padY,
                       padX, inputW_ - nw - padX,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

// ---------- 2) 打包 NCHW（BGR->RGB + /255）----------
std::vector<float> ArmorDetectorNN::buildInput(const cv::Mat& sq) {
    int h = sq.rows, w = sq.cols;
    std::vector<float> v((size_t)3 * h * w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3b& p = sq.at<cv::Vec3b>(y, x);
            int base = y * w + x;
            v[0 * h * w + base] = p[2] / 255.0f;   // R
            v[1 * h * w + base] = p[1] / 255.0f;   // G
            v[2 * h * w + base] = p[0] / 255.0f;   // B
        }
    }
    return v;
}

// ---------- 4) 解析三头输出 ----------
std::vector<RawBox> ArmorDetectorNN::parseOutput(const float* data, const std::vector<int64_t>& dims) {
    int64_t A = dims.size() >= 2 ? dims[1] : 0;
    int64_t B = dims.size() >= 3 ? dims[2] : 0;
    bool coordFirst = (A < B);                 // [通道][候选] ?
    int chAxis = (int)(coordFirst ? A : B);    // 应为 21
    int n      = (int)(coordFirst ? B : A);    // 候选数 N
    nAnchors_  = n;

    std::vector<RawBox> boxes;
    if (chAxis != 21 || n <= 0) return boxes;

    // 取某通道 c、候选 i 的值
    auto get = [&](int c, int i) -> float {
        return coordFirst ? data[(int64_t)c * n + i] : data[(int64_t)i * chAxis + c];
    };

    for (int i = 0; i < n; ++i) {
        // 颜色分支 0..3（已过 sigmoid）
        float cMax = -1e9; int cCls = 0;
        for (int c = 0; c < NUM_COLOR; ++c) {
            float v = get(c, i);
            if (v > cMax) { cMax = v; cCls = c; }
        }
        // 装甲类别分支 4..12（已过 sigmoid）
        float aMax = -1e9; int aCls = 0;
        for (int c = 0; c < NUM_ARMOR; ++c) {
            float v = get(4 + c, i);
            if (v > aMax) { aMax = v; aCls = c; }
        }
        // 角点 13..20（输入图像素，未 sigmoid）
        RawBox rb;
        for (int k = 0; k < NUM_CORNER; ++k) rb.corners[k] = get(13 + k, i);
        rb.colorScore = cMax; rb.colorCls = cCls;
        rb.armorScore = aMax; rb.armorCls = aCls;
        rb.score = cMax * aMax;
        // 双重门槛：颜色与装甲都要够自信，否则是背景/误检
        if (cMax < confThres_ || aMax < confThres_) continue;
        boxes.push_back(rb);
    }
    return boxes;
}

// ---------- 3) 跑一次前向 ----------
std::vector<RawBox> ArmorDetectorNN::runNetwork(const cv::Mat& frame, double& scale, int& padX, int& padY) {
    cv::Mat sq = letterbox(frame, scale, padX, padY);
    std::vector<float> input = buildInput(sq);
    std::vector<int64_t> inShape = {1, 3, inputH_, inputW_};

    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inTensor = Ort::Value::CreateTensor<float>(
        mi, input.data(), input.size(), inShape.data(), inShape.size());

    const char* inNames[]  = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, inNames, &inTensor, 1, outNames, 1);

    float* data = outputs.front().GetTensorMutableData<float>();
auto outTI  = outputs.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> od = outTI.GetShape();
    return parseOutput(data, od);
}

// ---------- 由角点判断大小装甲（两灯条中心距法，与传统 matchLights 一致）----------
bool ArmorDetectorNN::isLargeByCorners(const cv::Point2f kp[4]) {
    // 左灯条 = 左上-左下, 右灯条 = 右上-右下
    cv::Point2f leftC  = (kp[0] + kp[1]) * 0.5f;
    cv::Point2f rightC = (kp[3] + kp[2]) * 0.5f;
    double barLen = std::max(cv::norm(kp[0] - kp[1]), 1e-6);
    double centerDist = cv::norm(leftC - rightC) / barLen;   // 以灯条长为单位
    return centerDist >= 3.2;   // 与传统 minLargeDist_ 一致
}

// ---------- 主入口：检测一帧（颜色过滤 + NMS + 组装 Armor）----------
std::vector<Armor> ArmorDetectorNN::detect(const cv::Mat& frame, const std::string& color) {
    if (!ok_) return {};
    double scale; int padX, padY;
    std::vector<RawBox> raw = runNetwork(frame, scale, padX, padY);

    // 目标颜色 -> 需要的颜色类别 id
    int wantColor = -1;
    if (color == "blue") wantColor = 0;
    else if (color == "red") wantColor = 1;
    // white(2)/purple(3) 仅当显式指定才保留

    // 过滤：目标颜色 + 角点合理（角点坐标需在输入图范围内）
    std::vector<RawBox> kept;
    for (const auto& b : raw) {
        if (wantColor >= 0 && b.colorCls != wantColor) continue;
        kept.push_back(b);
    }

    // 组装 Armor；先做坐标还原，再做 NMS
    std::vector<cv::Rect> rects;
    std::vector<float>    scores;
    std::vector<Armor>    armors;
    for (const auto& b : kept) {
        cv::Point2f kp[4];
        for (int k = 0; k < 4; ++k) {
            float x = (b.corners[2 * k]     - padX) / (float)scale;
            float y = (b.corners[2 * k + 1] - padY) / (float)scale;
            kp[k] = cv::Point2f(x, y);
        }
        // 角点顺序 = [左上, 左下, 右下, 右上]，与 keypoints 约定完全对齐
        Armor a;
        a.keypoints[0] = kp[0]; a.keypoints[1] = kp[1];
        a.keypoints[2] = kp[2]; a.keypoints[3] = kp[3];

        a.color = (b.colorCls == 1) ? "red" : (b.colorCls == 0) ? "blue"
                 : (b.colorCls == 2) ? "white" : "purple";
        a.isLarge = isLargeByCorners(kp);

        // 合成左右灯条（供数字识别 classify 做透视变换）
        bool isRed = (b.colorCls == 1);
        a.left.top = kp[0];  a.left.bottom = kp[1];
        a.left.center = (kp[0] + kp[1]) * 0.5f;
        a.left.length = cv::norm(kp[0] - kp[1]);  a.left.width = 6.0;  a.left.isRed = isRed;
        a.right.top = kp[3]; a.right.bottom = kp[2];
        a.right.center = (kp[3] + kp[2]) * 0.5f;
        a.right.length = cv::norm(kp[3] - kp[2]); a.right.width = 6.0; a.right.isRed = isRed;

        // 数字：优先用装甲头类别名（label.txt）
        a.number = (!labels_.empty() && b.armorCls < (int)labels_.size())
                       ? labels_[b.armorCls] : std::to_string(b.armorCls + 1);
        a.numberConf = b.armorScore;

        // NMS 用角点外接矩形
        cv::Rect r = cv::boundingRect(std::vector<cv::Point2f>(kp, kp + 4));
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        rects.push_back(r); scores.push_back(b.score); armors.push_back(a);
    }

    // NMS 去重
    std::vector<int> keep;
    if (!rects.empty())
        cv::dnn::NMSBoxes(rects, scores, confThres_, iouThres_, keep);
    std::vector<Armor> result;
    for (int idx : keep) result.push_back(armors[idx]);
    // 按置信度降序，方便取最佳
    std::sort(result.begin(), result.end(),
              [](const Armor& x, const Armor& y) { return x.numberConf > y.numberConf; });
    return result;
}

// ---------- 诊断：亲自看到模型输出 ----------
std::vector<RawBox> ArmorDetectorNN::inspect(const cv::Mat& frame) {
    double scale; int padX, padY;
    auto boxes = runNetwork(frame, scale, padX, padY);
    std::cout << "[NN] 模型输出: 候选框总数 N=" << nAnchors_ << "\n";
    std::cout << "[NN] 颜色分支(蓝/红/白/紫) 装甲分支(9类) 4角点(x,y像素,输入图空间)\n";
    const char* colName[4] = {"blue", "red", "white", "purple"};
    std::cout << "[NN] 通过双门槛(颜色&装甲>" << confThres_ << ")的候选共 "
              << boxes.size() << " 个，列出前 10 个:\n";
    int shown = 0;
    for (const auto& b : boxes) {
        if (shown++ >= 10) break;
        printf("    角点(in)= [%.1f,%.1f, %.1f,%.1f, %.1f,%.1f, %.1f,%.1f]  color=%s(%.2f) armor[%d]=%.2f\n",
               b.corners[0], b.corners[1], b.corners[2], b.corners[3],
               b.corners[4], b.corners[5], b.corners[6], b.corners[7],
               colName[b.colorCls], b.colorScore, b.armorCls, b.armorScore);
    }
    return boxes;
}
