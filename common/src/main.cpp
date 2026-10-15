#include <opencv2/highgui.hpp>   // imshow/waitKey
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>   // VideoCapture
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include "armor_detector.h"
#include "armor_types.h"
#include "camera_config.h"
#include "number_classifier.h"
#include "pnp_solver.h"

#ifdef _WIN32
#include <windows.h>   // 仅 Windows：修复中文输出乱码（终端代码页切到 UTF-8）
#endif

#include "neural_detector.h"

// ===== 检测器开关：把下面这行注释掉即可切回传统方法(D2) 做对比 =====
#define USE_NN

// 视频默认路径，指向素材文件夹；如换位置请改这里
const char* VIDEO_PATH =
    "C:/work/electric_competition/robot/homework/second-train/素材/2026-05-20_23-57-31.avi";

// 在多个候选路径里找第一个存在的文件（兼容从 build/ 或项目根运行）
static std::string findFile(const std::vector<std::string>& candidates)   //同级目录还是下级目录的关系
{
    for (const auto& p : candidates) {
        std::ifstream f(p);               //用路径打开文件
        if (f.good()) return p;           //能打开说明文件存在，返回这个路径
    }
    return "";                            //找不到返回空串
}

// 在 Windows 上, OpenCV 的视频 API 只接受窄字符串(ANSI/GBK), 无法直接用 UTF-8
// 中文路径打开. 这里把 UTF-8 路径转成 8.3 短路径(纯 ASCII), 这样无论原路径多长/含
// 中文都能正常打开. (Linux/macOS 下 UTF-8 原生支持, 直接透传.)
#ifdef _WIN32
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

// ---- 绘制辅助 ----
static cv::Scalar colorOf(const std::string& c) { return c == "red" ? cv::Scalar(0, 0, 255)
                                                                    : cv::Scalar(255, 100, 0); }    //看到是红的就画成红色，看到蓝色就画成橙色，不参与决策

static void drawArmor(cv::Mat& frame, const Armor& a, int frameIdx)
{
    cv::Scalar col = colorOf(a.color);
    // 装甲板四边形（按关键点顺序连线）
    for (int i = 0; i < 4; ++i) {
        cv::line(frame, a.keypoints[i], a.keypoints[(i + 1) % 4], col, 2);     //这里（i + 1）% 4 完全可以写成0，写这个是统一规范，多少个点都可以这么干
        cv::circle(frame, a.keypoints[i], 3, col, -1);
    }
    // 左右灯条
    cv::line(frame, a.left.top, a.left.bottom, cv::Scalar(0, 255, 0), 1);      //Scarlar(0, 0, 0)代表（蓝，绿，红）  
    cv::line(frame, a.right.top, a.right.bottom, cv::Scalar(0, 255, 0), 1);

    // 顶部：颜色 + 数字 + 置信度
    cv::Point2f topMid = (a.keypoints[0] + a.keypoints[3]) / 2.0f;
    std::string txt = a.color + " " + a.number + " " +
                      std::to_string((int)(a.numberConf * 100)) + "%";
    cv::putText(frame, txt, topMid + cv::Point2f(-30, -12), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(255, 255, 255), 1.5);
    // 底部：PnP 解算坐标（米）
    if (a.positionValid) {
        std::stringstream ss;
        ss << "(" << std::fixed << std::setprecision(3) << a.position.x << ", "
           << a.position.y << ", " << a.position.z << ") m";
        cv::Point2f botMid = (a.keypoints[1] + a.keypoints[2]) / 2.0f;
        cv::putText(frame, ss.str(), botMid + cv::Point2f(-50, 18), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 255), 1.2);
    } else {
        cv::Point2f botMid = (a.keypoints[1] + a.keypoints[2]) / 2.0f;
        cv::putText(frame, "PnP failed", botMid + cv::Point2f(-30, 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.2);
    }
    (void)frameIdx;
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    // 把控制台输入/输出代码页设为 UTF-8(65001)，避免中文乱码
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    // 1) 打印相机参数，确认头文件配置正确
    std::cout << "[相机内参 K]\n" << getCameraMatrix() << std::endl;          //自检
    std::cout << "[畸变系数 D]\n" << getDistortionCoeffs() << std::endl;

    // 2) 构建流水线各模块（模型路径：先找 build/ 上一级，再找当前目录）
    // ---- 检测器选择：神经网络(D3) 或 传统(D2) ----
    #ifdef USE_NN
        std::string nnModel = findFile({
            "../models/Infantry-v8n-fp32.onnx",
            "models/Infantry-v8n-fp32.onnx",
            "code/models/Infantry-v8n-fp32.onnx"});
        ArmorDetectorNN detector(nnModel);       // 神经网络检测：YOLO 直接出框
        if (!detector.ok())
            std::cerr << "[警告] NN 模型不可用，请检查模型路径" << std::endl;
    #else
        ArmorDetector detector;                  // 传统检测：灯条->配对
    #endif
    PnPSolver solver;                             // PnP 位姿解算
    std::string modelPath = findFile({"../models/mlp.onnx", "models/mlp.onnx"});
    std::string labelPath = findFile({"../models/label.txt", "models/label.txt"});
    NumberClassifier classifier(modelPath, labelPath);  // 数字识别(MLP)
    if (!classifier.ok())
        std::cerr << "[警告] 数字模型不可用，仅显示检测框" << std::endl;

    // 3) 打开视频 (支持命令行覆盖: argv[1]=输入视频, argv[2]=输出视频)
    std::string inVideo  = (argc >= 2) ? argv[1] : std::string(VIDEO_PATH);
    std::string outVideo = (argc >= 3) ? argv[2] : "output_annotated.avi";
#ifdef _WIN32
    std::string openPath = toAnsiShortPath(inVideo);   // 中文路径 -> 8.3 短路径
#else
    std::string openPath = inVideo;
#endif
    cv::VideoCapture cap(openPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << inVideo << std::endl;
        return -1;
    }

    // 3.1) 初始化视频写入器: 把标注后的帧导出为视频文件
    double fps = cap.get(cv::CAP_PROP_FPS);
    int vw = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int vh = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (fps <= 0) fps = 30.0;
    // MJPG 编码器无需系统额外解码器, .avi 容器兼容性最好
    cv::VideoWriter writer(outVideo, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(vw, vh));
    if (!writer.isOpened()) {
        std::cerr << "[警告] 无法创建输出视频: " << outVideo
                  << " (程序仍会显示窗口, 但不会导出文件)\n";
    } else {
        std::cout << "[导出] 标注视频写入: " << outVideo
                  << "  (" << vw << "x" << vh << " @" << fps << " fps)\n";
    }

    cv::Mat frame;
    int frameIdx = 0, solvedCnt = 0, failCnt = 0, noneCnt = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        ++frameIdx;

        // 4) 完整流水线：检测 -> 数字识别 -> PnP 解算
        std::vector<Armor> armors = detector.detect(frame, "red");
        if (!armors.empty()) classifier.classify(frame, armors);
        for (auto& a : armors) {
            if (solver.solve(a)) ++solvedCnt;
            else ++failCnt;  // PnP失败（少见）
        }
        if (armors.empty()) ++noneCnt;  // 没检测到装甲板（较为常见）

        // 5) 绘制
        for (const auto& a : armors) drawArmor(frame, a, frameIdx);
        std::string hud = "frame " + std::to_string(frameIdx) +
                          "  armors: " + std::to_string(armors.size());
        cv::putText(frame, hud, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    {0, 255, 0}, 2);
        cv::imshow("preview", frame);
        if (writer.isOpened()) writer.write(frame);   // 导出: 把当前标注帧写入视频文件
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();
    // 6) 汇总统计（写报告时可引用）
    std::cout << "==== 运行统计 ====\n"
              << "总帧数: " << frameIdx << "\n有装甲帧: " << frameIdx - noneCnt
              << "\n无装甲帧: " << noneCnt << "\nPnP 成功: " << solvedCnt
              << "\nPnP 失败: " << failCnt << std::endl;
    if (writer.isOpened())
        std::cout << "标注视频已导出: " << outVideo << std::endl;
    std::cout << "D3 神经网络流水线 OK" << std::endl;
    return 0;
}
