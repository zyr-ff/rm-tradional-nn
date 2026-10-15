// ============================================================
//  D4 误差评估工具（任务4）
//  把老师给的标注图喂进 D2 流水线(检测->识别->PnP)，
//  得到每块装甲板的 tvec，再与真值 tx/ty/tz 逐板相减，得 Δx/Δy/Δz。
//
//  编译: 见 CMakeLists.txt 中的 eval_gt 目标
//  运行: ./eval_gt.exe <图1.jpg> <图2.jpg> <图3.jpg> ...
//        会自动把每张图匹配到最接近的图1/图2/图3 真值集。
// ============================================================
#include <opencv2/opencv.hpp>
#include "armor_detector.h"
#include "armor_types.h"
#include "camera_config.h"
#include "number_classifier.h"
#include "pnp_solver.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>

// ---------- 真值库（来自 素材/图像/图1~3.md，相机光学系，单位 m）----------
struct GTSet {
    std::string name;                 // 图名
    std::vector<cv::Point3d> pts;     // 4 块板: armor_1..4 的中心 (tx,ty,tz)
    std::vector<bool> vis;            // 是否正面(相机可见); 只和可见板配对
};

static std::vector<GTSet> loadGT() {
    return {
        {"图1(~3.79m)", {{-0.174340, 0.169600, 3.794632},
                          {-0.142456, 0.176579, 4.088549},
                          { 0.150773, 0.162930, 4.055556},
                          { 0.119301, 0.183937, 3.762585}},
                        { true, false, false, true}},           // armor_1,4 正面
        {"图2(~1.47m)", {{ 0.285747, 0.070019, 1.384573},
                          { 0.009916, 0.097664, 1.281592},
                          {-0.094899, 0.047440, 1.553161},
                          { 0.180498, 0.047525, 1.660025}},
                        { true, true, false, false}},           // armor_1,2 正面
        {"图3(~1.65m)", {{-0.179673, 0.053321, 1.746903},
                          { 0.102393, 0.059418, 1.835529},
                          { 0.189233, 0.070265, 1.553395},
                          {-0.092414, 0.092079, 1.467008}},
                        { false, false, true, true}},           // armor_3,4 正面
    };
}

// 贪心最近匹配: 把检测到的板(det) 与某真值集(gt) 中"正面可见的"板按 **x/y 平面距离**
// 一一配对(只用 x,y, 不用 z), 返回平均配对距离, 配对明细写进 pairs。
// 说明: x/y 误差小、可靠, 用作"这是哪块板"的判定依据; z 是待测的量, 不能拿它去选配对,
// 否则 z 的系统性偏差会把板错配到背面/邻近板。vis[j]=false 的背面板不参与配对。
static double greedyMatch(const std::vector<cv::Point3d>& det,
                          const std::vector<cv::Point3d>& gt,
                          const std::vector<bool>& vis,
                          std::vector<std::tuple<int,int,double,cv::Point3d,cv::Point3d>>& pairs)
{
    const int n = (int)det.size(), m = (int)gt.size();
    std::vector<bool> usedD(n, false), usedG(m, false);
    double total = 0.0;
    int cnt = 0;
    while (true) {
        int bi = -1, bj = -1; double bd = 1e18;
        for (int i = 0; i < n; ++i) if (!usedD[i])
            for (int j = 0; j < m; ++j) if (!usedG[j] && vis[j]) {
                double dx = det[i].x - gt[j].x, dy = det[i].y - gt[j].y;  // 只用 x/y
                double d = std::sqrt(dx * dx + dy * dy);
                if (d < bd) { bd = d; bi = i; bj = j; }
            }
        if (bi < 0) break;
        pairs.emplace_back(bi, bj, bd, det[bi], gt[bj]);
        total += bd; ++cnt; usedD[bi] = usedG[bj] = true;
    }
    return cnt ? total / cnt : 1e18;
}

// 由文件名中的序号锁定真值集: gt/img2.jpg -> 2 -> 图2(索引1)。
// 作业里 img1/2/3 与 图1/2/3 一一对应, 这是先验已知、不应靠算法猜的;
// 否则 z 的系统性误差会让"三维距离最近"的自动匹配选错集(如 img2 误配图3)。
// 返回 -1 表示无法解析序号, 退化为原来的自动匹配(用于其它未知图)。
static int gtIndexForImage(const std::string& path) {
    std::string name = path;
    size_t p = name.find_last_of("/\\");
    if (p != std::string::npos) name = name.substr(p + 1);
    size_t d = name.find_last_of('.');
    if (d != std::string::npos) name = name.substr(0, d);
    int idx = -1;                                   // 从末尾往前找连续数字串(序号)
    for (int i = (int)name.size() - 1; i >= 0; --i) {
        if (name[i] < '0' || name[i] > '9') break;
        idx = i;
    }
    if (idx < 0) return -1;                         // 文件名里没有数字
    int n = std::stoi(name.substr(idx));
    return (n >= 1 && n <= 3) ? (n - 1) : -1;       // img1->0, img2->1, img3->2
}

// ---------- 画结果图: 把检测框/角点/标签/距离画到图上并保存 ----------
// 和 D2 main.cpp 的 drawArmor 思路一致，但把 imshow 换成 imwrite(存文件)。
// ⚠️ putText 不支持中文，标签一律用英文(颜色/数字/置信度/距离)。
static void drawResult(const cv::Mat& img, const std::vector<Armor>& armors,
                       const std::string& outPath)
{
    cv::Mat out = img.clone();                       // 复制一份画，不污染原图
    for (const auto& a : armors) {
        // 颜色: 红甲->红框, 蓝甲->橙框(沿用 D2 colorOf 约定)
        cv::Scalar col = (a.color == "red") ? cv::Scalar(0, 0, 255)
                                            : cv::Scalar(255, 100, 0);

        // 1) 装甲板四边形 + 4 个角点(0左上 1左下 2右下 3右上)
        for (int i = 0; i < 4; ++i) {
            cv::line(out, a.keypoints[i], a.keypoints[(i + 1) % 4], col, 2);
            cv::circle(out, a.keypoints[i], 3, col, -1);   // -1 = 实心圆点
        }

        // 2) 顶部标签: 颜色 + 数字 + 置信度  (a.color/a.number 与终端输出同源)
        cv::Point2f topMid = (a.keypoints[0] + a.keypoints[3]) / 2.0f;
        char tag[64];
        snprintf(tag, sizeof(tag), "%s %s %.2f",
                 a.color.c_str(), a.number.c_str(), a.numberConf);
        cv::putText(out, tag, topMid + cv::Point2f(-45, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 1.6);

        // 3) 底部标签: 相机到装甲板的直线距离 = tvec 的模长(沿光轴深度 z 不是这个数)
        if (a.positionValid) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f m", cv::norm(a.position));
            cv::Point2f botMid = (a.keypoints[1] + a.keypoints[2]) / 2.0f;
            cv::putText(out, buf, botMid + cv::Point2f(-25, 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1.6);
        } else {
            // PnP 失败的板，标个灰字提醒
            cv::Point2f botMid = (a.keypoints[1] + a.keypoints[2]) / 2.0f;
            cv::putText(out, "PnP failed", botMid + cv::Point2f(-25, 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(128, 128, 128), 1.6);
        }
    }
    cv::imwrite(outPath, out);
    std::cout << "  结果图已保存: " << outPath << "\n";
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    // 参数解析: --real 切换到真实相机内参(内录图用)，其余当作图片路径
    bool useReal = false;
    std::vector<std::string> imgs;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "--real") useReal = true;
        else imgs.push_back(s);
    }
    if (imgs.empty()) {
        std::cerr << "用法: eval_gt.exe [--real] <图1.jpg> [图2.jpg ...]" << std::endl;
        std::cerr << "  (默认用仿真相机内参; 加 --real 用真实相机内参, 用于内录图像)" << std::endl;
        return -1;
    }

    auto gts = loadGT();

    // 真实相机标定分辨率(内参按此标定); 内录图分辨率不同需按宽高比缩放
    const double REF_W = 1358.0, REF_H = 1058.0;

    // 流水线工具（PnPSolver 在循环内按每张图的分辨率构造）
    ArmorDetector detector;
    // 模型路径: 兼容从 build/ 或项目根运行
    auto findFile = [](const std::vector<std::string>& c) -> std::string {
        for (auto& p : c) { std::ifstream f(p); if (f.good()) return p; }
        return "";
    };
    std::string mp = findFile({"../models/mlp.onnx", "models/mlp.onnx"});
    std::string lp = findFile({"../models/label.txt", "models/label.txt"});
    NumberClassifier classifier(mp, lp);
    std::cout << (classifier.ok() ? "[数字模型] 已加载\n" : "[数字模型] 未加载, 仅做位置评估\n");

    for (size_t ii = 0; ii < imgs.size(); ++ii) {
        const std::string& path = imgs[ii];
        cv::Mat img = cv::imread(path);
        if (img.empty()) { std::cerr << "跳过(无法读取): " << path << std::endl; continue; }

        // 选内参: 仿真图用仿真内参(已匹配960x720); 内录图用真实相机内参按分辨率缩放
        cv::Mat K, D;
        std::string kdesc;
        if (useReal) {
            double sx = img.cols / REF_W, sy = img.rows / REF_H;
            K = getCameraMatrix().clone();
            K.at<double>(0, 0) *= sx;  // fx
            K.at<double>(0, 2) *= sx;  // cx
            K.at<double>(1, 1) *= sy;  // fy
            K.at<double>(1, 2) *= sy;  // cy
            D = getDistortionCoeffs();
            char buf[128];
            snprintf(buf, sizeof(buf), "真实相机内参(缩放 sx=%.3f sy=%.3f)", sx, sy);
            kdesc = buf;
        } else {
            K = getSimCameraMatrix();
            D = getSimDistortion();
            kdesc = "仿真内参 fx=fy=869.117 cx=480 cy=360, 无畸变";
        }
        PnPSolver solver(K, D);
        std::cout << "\n==============================\n"
                  << "图像: " << path << "  (" << img.cols << "x" << img.rows
                  << ", " << kdesc << ")\n";

        // ---- 诊断: 打印各预处理下的轮廓与矩形统计(找出卡在哪一关) ----
        {
            cv::Mat gray, ch[3], bin;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::split(img, ch);
            auto dump = [&](const cv::Mat& binary, const std::string& tag) {
                std::vector<std::vector<cv::Point>> cs;
                cv::findContours(binary, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                int passShape = 0;
                std::string info;
                int shown = 0;
                for (auto& c : cs) {
                    if (c.size() < 5) continue;
                    cv::RotatedRect r = cv::minAreaRect(c);
                    double len = std::max(r.size.width, r.size.height);
                    double wid = std::min(r.size.width, r.size.height);
                    double ratio = wid / std::max(len, 1e-6);
                    cv::Point2f p[4]; r.points(p);
                    std::sort(p, p + 4, [](cv::Point2f a, cv::Point2f b){ return a.y < b.y; });
                    cv::Point2f top = (p[0] + p[1]) / 2.f, bot = (p[2] + p[3]) / 2.f;
                    double tilt = std::atan2(std::abs(top.x - bot.x),
                                             std::abs(top.y - bot.y)) * 180.0 / CV_PI;
                    bool ok = (ratio > 0.1 && ratio < 0.4 && tilt < 40.0);
                    if (ok) ++passShape;
                    if (len > 8 && shown < 8) {
                        char buf[160];
                        snprintf(buf, sizeof(buf), "    %s len=%.1f wid=%.1f ratio=%.2f tilt=%.0f%s\n",
                                 tag.c_str(), len, wid, ratio, tilt, ok ? " [灯条OK]" : "");
                        info += buf;
                        ++shown;
                    }
                }
                std::cout << "  [诊断" << tag << "] 轮廓数=" << cs.size()
                          << " 过形状筛选=" << passShape << "\n" << info;
            };
            cv::threshold(gray, bin, 85, 255, cv::THRESH_BINARY);
            dump(bin, "gray85");
            cv::threshold(ch[0], bin, 150, 255, cv::THRESH_BINARY);
            dump(bin, "B150");
        }

        // 仿真图是蓝甲: 灰度阈值对蓝光失效(蓝光灰度低、灰地面更亮)。
        // 依次尝试: ① 灰度阈值 ② B通道阈值 ③ B-R差值(rm_vision 经典做法),
        // 取第一个能检出装甲板的方案。
        std::vector<Armor> armors;
        std::string usedPre;
        auto tryDetect = [&](const cv::Mat& binary, const std::string& tag) {
            if (!armors.empty()) return;
            ArmorDetector det1(85);
            armors = det1.detect(img, binary, "red");
            auto blue = det1.detect(img, binary, "blue");
            armors.insert(armors.end(), blue.begin(), blue.end());
            if (!armors.empty()) usedPre = tag;
        };
        {
            cv::Mat gray, b, ch[3], bin, diff;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::split(img, ch);
            b = ch[0];                                    // B 通道
            cv::subtract(b, ch[2], diff);                 // B - R (蓝光为正)
            for (int t : {85, 120, 150, 180}) {
                cv::threshold(gray, bin, t, 255, cv::THRESH_BINARY);
                tryDetect(bin, "灰度阈值" + std::to_string(t));
            }
            for (int t : {130, 150, 170, 190, 210}) {
                cv::threshold(b, bin, t, 255, cv::THRESH_BINARY);
                tryDetect(bin, "B通道阈值" + std::to_string(t));
            }
            for (int t : {30, 50, 70}) {
                cv::threshold(diff, bin, t, 255, cv::THRESH_BINARY);
                tryDetect(bin, "B-R差值" + std::to_string(t));
            }
        }

        std::cout << "\n==============================\n"
                  << "图像: " << path << "\n"
                  << "  预处理方案: " << (usedPre.empty() ? "(全部失败)" : usedPre)
                  << ", 检测到装甲板: " << armors.size() << " 块\n";
        if (!armors.empty()) classifier.classify(img, armors, /*filter=*/false);  // 评估模式: 只填数字不剔除
        std::cout << "  数字识别过滤后剩: " << armors.size() << " 块 (评估模式不剔除)\n";

        std::vector<cv::Point3d> det;
        int pnpFail = 0;
        for (auto& ar : armors) { if (solver.solve(ar)) det.push_back(ar.position); else ++pnpFail; }
        std::cout << "  PnP 成功: " << det.size() << " 块, 失败: " << pnpFail << " 块\n";

        // ---- 画结果图: 把检测框/角点/标签/距离画到图上并保存 ----
        {
            std::string inName = path;
            size_t pos = inName.find_last_of("/\\");
            if (pos != std::string::npos) inName = inName.substr(pos + 1);
            drawResult(img, armors, "result_" + inName);
        }

        if (det.empty()) { std::cout << "  (无有效检测, 跳过)\n"; continue; }

        // 内录图(真实相机)没有真值集，只做检测+位姿，不做误差比对
        if (useReal) {
            std::cout << "  内录图无真值, 跳过误差比对(只看检测框与距离)\n";
            for (size_t k = 0; k < det.size(); ++k)
                printf("  板%-2zu  tvec=(%+0.3f,%+0.3f,%+0.3f)  距离=%.3f m\n",
                       k, det[k].x, det[k].y, det[k].z, cv::norm(det[k]));
            continue;
        }

        // 优先按文件名序号锁定真值集(img1->图1, img2->图2, img3->图3),
        // 这是作业先验, 避免 z 误差导致自动匹配选错集(如 img2 误配图3)。
        int bestGt = -1; double bestMean = 1e18;
        std::vector<std::tuple<int,int,double,cv::Point3d,cv::Point3d>> bestPairs;
        int forced = gtIndexForImage(path);
        if (forced >= 0 && forced < (int)gts.size()) {
            bestGt = forced;
            bestMean = greedyMatch(det, gts[forced].pts, gts[forced].vis, bestPairs);
            std::cout << "  按文件名锁定真值集: " << gts[bestGt].name << "\n";
        } else {
            // 文件名无法解析序号时, 退化为原来的自动匹配(三维距离最近, 全板参与)
            std::vector<bool> allVis(gts[0].pts.size(), true);
            for (size_t g = 0; g < gts.size(); ++g) {
                std::vector<std::tuple<int,int,double,cv::Point3d,cv::Point3d>> pr;
                double mean = greedyMatch(det, gts[g].pts, allVis, pr);
                if (mean < bestMean) { bestMean = mean; bestGt = (int)g; bestPairs = pr; }
            }
            std::cout << "  自动匹配到真值集: " << gts[bestGt].name
                      << "  平均配对误差: " << bestMean << " m\n";
        }
        std::cout << "  --------------------------------------------------\n";
        std::cout << "  配对  检测tvec(x,y,z)            真值(x,y,z)              Δx      Δy      Δz      |Δ|\n";
        double sumAbs = 0;
        for (auto& p : bestPairs) {
            int di = std::get<0>(p), gi = std::get<1>(p);
            cv::Point3d detp = std::get<3>(p), gtp = std::get<4>(p);
            double dx = detp.x - gtp.x, dy = detp.y - gtp.y, dz = detp.z - gtp.z;
            sumAbs += std::abs(dx) + std::abs(dy) + std::abs(dz);
            double d3 = std::sqrt(dx * dx + dy * dy + dz * dz);   // 三维欧氏距离 |Δ|
            printf("  A%-2d  (%+.3f,%+.3f,%+.3f)  (%+.3f,%+.3f,%+.3f)  %+.3f  %+.3f  %+.3f  %.3f\n",
                   gi + 1, detp.x, detp.y, detp.z, gtp.x, gtp.y, gtp.z, dx, dy, dz, d3);
        }
        std::cout << "  --------------------------------------------------\n";
        std::cout << "  平均分量误差 |Δx|+|Δy|+|Δz|/3 = "
                  << (bestPairs.empty() ? 0 : sumAbs / bestPairs.size() / 3.0) << " m\n";
    }
    std::cout << "\n评估完成。" << std::endl;
    return 0;
}
