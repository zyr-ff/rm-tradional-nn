// ============================================================
//  D3 神经网络检测器 · 端到端验证 (test_nn)
//  对仿真图跑 detect(blue) + PnP 解算，打印 4 角点 / 颜色 / 大小 / 数字 / 位姿。
//  编译：见 CMakeLists.txt 的 test_nn 目标
//  运行：./test_nn.exe   （默认跑 3 张仿真真值图 code/build/gt/img1~3.jpg）
// ============================================================
#include "neural_detector.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
static std::string toAnsiShortPath(const std::string& utf8Path) {
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, nullptr, 0);
    if (n <= 0) return utf8Path;
    std::wstring w(n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, &w[0], n);
    wchar_t shortBuf[MAX_PATH] = {0};
    if (!GetShortPathNameW(w.c_str(), shortBuf, MAX_PATH)) return utf8Path;
    int a = WideCharToMultiByte(CP_ACP, 0, shortBuf, -1, nullptr, 0, nullptr, nullptr);
    if (a <= 0) return utf8Path;
    std::string s(a - 1, '\0');
    WideCharToMultiByte(CP_ACP, 0, shortBuf, -1, &s[0], a, nullptr, nullptr);
    return s;
}
#endif

static std::string findFile(const std::vector<std::string>& c) {
    for (auto& p : c) { std::ifstream f(p); if (f.good()) return p; }
    return "";
}

// 仿真相机内参 (960x720, fx=fy=869.117, 无畸变) —— 与 camera_config.h getSimCameraMatrix 一致
static cv::Mat simK() {
    return (cv::Mat_<double>(3,3) << 869.1168824543144, 0, 480, 0, 869.1168824543144, 360, 0, 0, 1);
}
static std::vector<cv::Point3f> objPts(bool isLarge) {
    double hw = isLarge ? 0.1175 : 0.0700;
    double hh = isLarge ? 0.0635 : 0.0625;
    return {cv::Point3f(-hw,-hh,0), cv::Point3f(-hw,hh,0), cv::Point3f(hw,hh,0), cv::Point3f(hw,-hh,0)};
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001); SetConsoleCP(65001);
#endif
    std::string mp = findFile({
        "../models/Infantry-v8n-fp32.onnx", "models/Infantry-v8n-fp32.onnx",
        "code/models/Infantry-v8n-fp32.onnx"});
    std::string lp = findFile({
        "../models/label.txt", "models/label.txt", "code/models/label.txt"});
    ArmorDetectorNN net(mp, lp);
    if (!net.ok()) { std::cerr << "模型加载失败: " << mp << std::endl; return -1; }
    std::cout << "[NN] 模型已加载, 装甲类别数=" << net.numClasses()
              << " 候选总数 N=" << net.numAnchors() << std::endl;

    std::vector<std::string> imgs = {
        "code/build/gt/img1.jpg", "code/build/gt/img2.jpg", "code/build/gt/img3.jpg"};
    cv::Mat K = simK();

    for (auto& p : imgs) {
#ifdef _WIN32
        std::string open = toAnsiShortPath(p);
#else
        std::string open = p;
#endif
        cv::Mat img = cv::imread(open);
        if (img.empty()) { std::cerr << "跳过(无法读取): " << p << std::endl; continue; }
        std::cout << "\n========== " << p << " (" << img.cols << "x" << img.rows << ") ==========\n";

        std::vector<Armor> armors = net.detect(img, "blue");
        if (armors.empty()) { std::cout << "  [无检测]\n"; continue; }

        for (size_t k = 0; k < armors.size(); ++k) {
            const auto& a = armors[k];
            printf("  装甲#%zu 颜色=%s 大小=%s 数字=%s(%.2f)\n",
                   k, a.color.c_str(), a.isLarge?"large":"small", a.number.c_str(), a.numberConf);
            printf("     角点(原图px): TL(%.1f,%.1f) BL(%.1f,%.1f) BR(%.1f,%.1f) TR(%.1f,%.1f)\n",
                   a.keypoints[0].x, a.keypoints[0].y, a.keypoints[1].x, a.keypoints[1].y,
                   a.keypoints[2].x, a.keypoints[2].y, a.keypoints[3].x, a.keypoints[3].y);

            std::vector<cv::Point2f> ip(a.keypoints, a.keypoints + 4);
            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(objPts(a.isLarge), ip, K, cv::Mat(),
                                   rvec, tvec, false, cv::SOLVEPNP_IPPE);
            if (ok) {
                double z = tvec.at<double>(2);
                printf("     PnP: t=(%.3f, %.3f, %.3f)  -> z=%.3f m\n",
                       tvec.at<double>(0), tvec.at<double>(1), z, z);
            } else {
                printf("     PnP 失败\n");
            }
        }
    }
    return 0;
}
