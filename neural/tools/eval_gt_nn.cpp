// ============================================================
//  D4 误差评估工具 · 神经网络版 (eval_gt_nn)
//  与 eval_gt.cpp 完全相同的真值库 / 匹配 / 指标，只把检测器换成
//  ArmorDetectorNN（D3 神经网络）。用于和传统方法做公平误差对比。
//
//  编译: 见 CMakeLists.txt 的 eval_gt_nn 目标
//  运行: ./eval_gt_nn.exe <图1.jpg> <图2.jpg> <图3.jpg>
// ============================================================
#include <opencv2/opencv.hpp>
#include "armor_types.h"
#include "camera_config.h"
#include "pnp_solver.h"
#include "neural_detector.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>

// ---------- 真值库（来自 素材/图像/图1~3.md，相机光学系，单位 m）----------
struct GTSet {
    std::string name;
    std::vector<cv::Point3d> pts;
    std::vector<bool> vis;
};
static std::vector<GTSet> loadGT() {
    return {
        {"图1(~3.79m)", {{-0.174340, 0.169600, 3.794632},
                          {-0.142456, 0.176579, 4.088549},
                          { 0.150773, 0.162930, 4.055556},
                          { 0.119301, 0.183937, 3.762585}},
                        { true, false, false, true}},
        {"图2(~1.47m)", {{ 0.285747, 0.070019, 1.384573},
                          { 0.009916, 0.097664, 1.281592},
                          {-0.094899, 0.047440, 1.553161},
                          { 0.180498, 0.047525, 1.660025}},
                        { true, true, false, false}},
        {"图3(~1.65m)", {{-0.179673, 0.053321, 1.746903},
                          { 0.102393, 0.059418, 1.835529},
                          { 0.189233, 0.070265, 1.553395},
                          {-0.092414, 0.092079, 1.467008}},
                        { false, false, true, true}},
    };
}

// 贪心最近匹配（只用 x/y，z 不参与选配对，避免系统性偏差错配）
// 方案B：detPos 为成功板位置(按 validIdx 顺序)，idxList 为这些板在 armors 中的
//       原始下标。返回的 bi 直接是原始下标，与 boxCenters/yawDegAll/zPredAll 对齐，
//       即使中途有 PnP 失败的板也不会错位。
static double greedyMatch(const std::vector<cv::Point3d>& detPos,
                          const std::vector<int>& idxList,
                          const std::vector<cv::Point3d>& gt,
                          const std::vector<bool>& vis,
                          std::vector<std::tuple<int,int,double,cv::Point3d,cv::Point3d>>& pairs)
{
    const int n = (int)detPos.size(), m = (int)gt.size();
    std::vector<bool> usedD(n, false), usedG(m, false);
    double total = 0.0; int cnt = 0;
    while (true) {
        int bi = -1, bj = -1; double bd = 1e18;
        for (int i = 0; i < n; ++i) if (!usedD[i])
            for (int j = 0; j < m; ++j) if (!usedG[j] && vis[j]) {
                double dx = detPos[i].x - gt[j].x, dy = detPos[i].y - gt[j].y;
                double d = std::sqrt(dx * dx + dy * dy);
                if (d < bd) { bd = d; bi = i; bj = j; }
            }
        if (bi < 0) break;
        pairs.emplace_back(idxList[bi], bj, bd, detPos[bi], gt[bj]);
        total += bd; ++cnt; usedD[bi] = usedG[bj] = true;
    }
    return cnt ? total / cnt : 1e18;
}

static int gtIndexForImage(const std::string& path) {
    std::string name = path;
    size_t p = name.find_last_of("/\\");
    if (p != std::string::npos) name = name.substr(p + 1);
    size_t d = name.find_last_of('.');
    if (d != std::string::npos) name = name.substr(0, d);
    int idx = -1;
    for (int i = (int)name.size() - 1; i >= 0; --i) {
        if (name[i] < '0' || name[i] > '9') break;
        idx = i;
    }
    if (idx < 0) return -1;
    int n = std::stoi(name.substr(idx));
    return (n >= 1 && n <= 3) ? (n - 1) : -1;
}

static void drawResult(const cv::Mat& img, const std::vector<Armor>& armors, const std::string& outPath) {
    cv::Mat out = img.clone();
    for (const auto& a : armors) {
        cv::Scalar col = (a.color == "red") ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 100, 0);
        for (int i = 0; i < 4; ++i) {
            cv::line(out, a.keypoints[i], a.keypoints[(i + 1) % 4], col, 2);
            cv::circle(out, a.keypoints[i], 3, col, -1);
        }
        cv::Point2f topMid = (a.keypoints[0] + a.keypoints[3]) / 2.0f;
        char tag[64];
        snprintf(tag, sizeof(tag), "%s %s %.2f", a.color.c_str(), a.number.c_str(), a.numberConf);
        cv::putText(out, tag, topMid + cv::Point2f(-45, -10), cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 1.6);
        if (a.positionValid) {
            char buf[32]; snprintf(buf, sizeof(buf), "%.2f m", cv::norm(a.position));
            cv::Point2f botMid = (a.keypoints[1] + a.keypoints[2]) / 2.0f;
            cv::putText(out, buf, botMid + cv::Point2f(-25, 18), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1.6);
        }
    }
    cv::imwrite(outPath, out);
    std::cout << "  结果图已保存: " << outPath << "\n";
}

// 由关键点+内参重新解一次 PnP 拿 rvec，估计装甲板 yaw（法向与光轴夹角, 度）
// 用途: 验证"斜视导致深度高估"—— 装甲板偏转 yaw 后像素宽度按 cos(yaw) 缩短,
//       正视模型的 PnP 会把深度读成 z/cos(yaw), 即 z 比值 ≈ 1/cos(yaw)。
static double estimateYawDeg(const Armor& a, const cv::Mat& K, const cv::Mat& D) {
    double hw = a.isLarge ? ARMOR_LARGE_HALF_W : ARMOR_SMALL_HALF_W;
    double hh = a.isLarge ? ARMOR_LARGE_HALF_H : ARMOR_SMALL_HALF_H;
    std::vector<cv::Point2f> ip(a.keypoints, a.keypoints + 4);
    cv::Mat rv, tv;
    if (!cv::solvePnP(armorObjectPoints(hw, hh), ip, K, D, rv, tv,
                      false, cv::SOLVEPNP_IPPE))
        return -1;
    cv::Mat R;
    cv::Rodrigues(rv, R);
    cv::Mat n = R * (cv::Mat_<double>(3, 1) << 0, 0, 1);   // 装甲板法向(相机系)
    double c = std::abs(n.at<double>(2));                  // 与光轴夹角的 cos
    return std::acos(std::min(1.0, c)) * 180.0 / CV_PI;
}

// 由关键点几何直接估计"宽度缩短"程度: 表观宽高比 / 真实宽高比 = cos(斜视角),
// 返回斜视深度高估预测值 1/cos。比 rvec 反解更直接(rvec 受 IPPE 倾斜简并干扰)。
static double predictZRatioByAspect(const Armor& a) {
    cv::Point2f leftC  = (a.keypoints[0] + a.keypoints[1]) * 0.5f;
    cv::Point2f rightC = (a.keypoints[3] + a.keypoints[2]) * 0.5f;
    double wApp = cv::norm(leftC - rightC);                       // 表观宽(两灯条中心距)
    double hApp = std::max(cv::norm(a.keypoints[0] - a.keypoints[1]), 1e-6);  // 表观高(灯条长)
    double trueRatio = a.isLarge ? (235.0 / 127.0) : (140.0 / 125.0);         // 真实宽高比
    double cosF = std::min(1.0, (wApp / hApp) / trueRatio);       // = cos(斜视角)
    return 1.0 / std::max(cosF, 1e-6);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001); SetConsoleCP(65001);
#endif
    bool useReal = false;
    std::vector<std::string> imgs;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "--real") useReal = true; else imgs.push_back(s);
    }
    if (imgs.empty()) {
        std::cerr << "用法: eval_gt_nn.exe [--real] <图1.jpg> [图2.jpg ...]" << std::endl;
        return -1;
    }

    auto findFile = [](const std::vector<std::string>& c) -> std::string {
        for (auto& p : c) { std::ifstream f(p); if (f.good()) return p; } return "";
    };
    std::string mp = findFile({"../models/Infantry-v8n-fp32.onnx", "models/Infantry-v8n-fp32.onnx", "code/models/Infantry-v8n-fp32.onnx"});
    std::string lp = findFile({"../models/label.txt", "models/label.txt", "code/models/label.txt"});
    ArmorDetectorNN net(mp, lp);
    if (!net.ok()) { std::cerr << "[NN] 模型加载失败: " << mp << std::endl; return -1; }
    std::cout << "[NN] 模型已加载, 装甲类别数=" << net.numClasses()
              << " 候选总数 N=" << net.numAnchors() << std::endl;

    auto gts = loadGT();
    const double REF_W = 1358.0, REF_H = 1058.0;

    // 全局统计（跨图累加）: 召回 + 平均分量误差
    int totVis = 0, totDet = 0, totMatched = 0;
    double sumDx = 0, sumDy = 0, sumDz = 0;

    for (size_t ii = 0; ii < imgs.size(); ++ii) {
        const std::string& path = imgs[ii];
        cv::Mat img = cv::imread(path);
        if (img.empty()) { std::cerr << "跳过(无法读取): " << path << std::endl; continue; }

        cv::Mat K, D; std::string kdesc;
        if (useReal) {
            double sx = img.cols / REF_W, sy = img.rows / REF_H;
            K = getCameraMatrix().clone();
            K.at<double>(0,0)*=sx; K.at<double>(0,2)*=sx; K.at<double>(1,1)*=sy; K.at<double>(1,2)*=sy;
            D = getDistortionCoeffs();
            char buf[128]; snprintf(buf,sizeof(buf),"真实相机内参(缩放 sx=%.3f sy=%.3f)",sx,sy); kdesc=buf;
        } else {
            K = getSimCameraMatrix(); D = getSimDistortion();
            kdesc = "仿真内参 fx=fy=869.117 cx=480 cy=360, 无畸变";
        }
        PnPSolver solver(K, D);
        std::cout << "\n==============================\n图像: " << path
                  << "  (" << img.cols << "x" << img.rows << ", " << kdesc << ")\n";

        // NN 检测（蓝+红合并，和传统方法评估口径一致）
        std::vector<Armor> armors;
        auto blue = net.detect(img, "blue");
        auto red  = net.detect(img, "red");
        armors.insert(armors.end(), blue.begin(), blue.end());
        armors.insert(armors.end(), red.begin(), red.end());
        std::cout << "  NN 检测到装甲板: " << armors.size() << " 块\n";
        std::vector<cv::Point2f> boxCenters;               // 每个检测的框中心(像素)
        for (size_t k = 0; k < armors.size(); ++k) {
            const auto& a = armors[k];
            cv::Point2f c(0, 0);
            float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
            for (int i = 0; i < 4; ++i) {
                c += a.keypoints[i];
                umin = std::min(umin, a.keypoints[i].x); umax = std::max(umax, a.keypoints[i].x);
                vmin = std::min(vmin, a.keypoints[i].y); vmax = std::max(vmax, a.keypoints[i].y);
            }
            c /= 4.f;
            boxCenters.push_back(c);
            printf("    装甲#%zu %s %s(%.2f) [%s甲] 中心(%.1f,%.1f)px 框%.1fx%.1fpx\n", k,
                   a.color.c_str(), a.number.c_str(), a.numberConf,
                   a.isLarge ? "大" : "小", c.x, c.y, umax - umin, vmax - vmin);
        }

        // 方案B：PnP 结果由 solver.solve 直接写回 armors[k].position/positionValid，
        // 不另建压缩下标数组；所有诊断数组与 armors 等长，避免失败板造成下标错位。
        std::vector<double> yawDegAll(armors.size(), -1.0), zPredAll(armors.size(), -1.0);
        int pnpFail = 0;
        for (size_t k = 0; k < armors.size(); ++k) {
            if (solver.solve(armors[k])) {
                yawDegAll[k] = estimateYawDeg(armors[k], K, D);
                zPredAll[k] = predictZRatioByAspect(armors[k]);
            } else ++pnpFail;
        }
        std::cout << "  PnP 成功: " << ((int)armors.size() - pnpFail) << " 块, 失败: " << pnpFail << " 块\n";

        {   std::string inName = path; size_t pos = inName.find_last_of("/\\");
            if (pos != std::string::npos) inName = inName.substr(pos + 1);
            drawResult(img, armors, "nn_result_" + inName); }

        // 收集 PnP 成功板的位置与原始下标，供配对使用（失败板直接跳过，下标不被压缩）
        std::vector<cv::Point3d> detPos; std::vector<int> validIdx;
        for (size_t k = 0; k < armors.size(); ++k)
            if (armors[k].positionValid) { validIdx.push_back((int)k); detPos.push_back(armors[k].position); }
        if (validIdx.empty()) { std::cout << "  (无有效检测, 跳过)\n"; continue; }

        if (useReal) {
            for (size_t k = 0; k < armors.size(); ++k) {
                if (!armors[k].positionValid) continue;
                const auto& t = armors[k].position;
                printf("  板%-2zu tvec=(%+0.3f,%+0.3f,%+0.3f) 距离=%.3f m\n",
                       k, t.x, t.y, t.z, cv::norm(t));
            }
            continue;
        }

        int bestGt = -1; double bestMean = 1e18;
        std::vector<std::tuple<int,int,double,cv::Point3d,cv::Point3d>> bestPairs;
        int forced = gtIndexForImage(path);
        if (forced >= 0 && forced < (int)gts.size()) {
            bestGt = forced;
            bestMean = greedyMatch(detPos, validIdx, gts[forced].pts, gts[forced].vis, bestPairs);
            std::cout << "  按文件名锁定真值集: " << gts[bestGt].name << "\n";
        } else {
            std::vector<bool> allVis(gts[0].pts.size(), true);
            for (size_t g = 0; g < gts.size(); ++g) {
                std::vector<std::tuple<int,int,double,cv::Point3d,cv::Point3d>> pr;
                double mean = greedyMatch(detPos, validIdx, gts[g].pts, allVis, pr);
                if (mean < bestMean) { bestMean = mean; bestGt = (int)g; bestPairs = pr; }
            }
            std::cout << "  自动匹配到真值集: " << gts[bestGt].name << " 平均配对误差: " << bestMean << " m\n";
        }
        std::cout << "  --------------------------------------------------\n";
        std::cout << "  配对  检测tvec(x,y,z)            真值(x,y,z)              Δx      Δy      Δz      |Δ|\n";
        double sumAbs = 0;
        for (auto& p : bestPairs) {
            int di = std::get<0>(p), gi = std::get<1>(p);
            cv::Point3d detp = std::get<3>(p), gtp = std::get<4>(p);
            double dx = detp.x - gtp.x, dy = detp.y - gtp.y, dz = detp.z - gtp.z;
            sumAbs += std::abs(dx) + std::abs(dy) + std::abs(dz);
            double d3 = std::sqrt(dx*dx+dy*dy+dz*dz);
            printf("  A%-2d  (%+.3f,%+.3f,%+.3f)  (%+.3f,%+.3f,%+.3f)  %+.3f  %+.3f  %+.3f  %.3f\n",
                   gi+1, detp.x,detp.y,detp.z, gtp.x,gtp.y,gtp.z, dx,dy,dz, d3);
            // 图像平面精度: 检测框中心 vs 真值中心的投影(单位 px)。
            // 作用: 把"检测准不准"和"深度准不准"分开——框中心偏差小说明检测本身准,
            //       误差主要来自 PnP 深度; x/y 的米级误差里有相当部分是 z 误差沿射线放大。
            if (di < (int)boxCenters.size() && gtp.z > 1e-6) {
                double gu = K.at<double>(0,0)*gtp.x/gtp.z + K.at<double>(0,2);
                double gv = K.at<double>(1,1)*gtp.y/gtp.z + K.at<double>(1,2);
                printf("       框中心(%.1f,%.1f)px  真值投影(%.1f,%.1f)px  偏差(%+.1f,%+.1f)px\n",
                       boxCenters[di].x, boxCenters[di].y, gu, gv,
                       boxCenters[di].x - gu, boxCenters[di].y - gv);
            }
            // 斜视深度高估验证: z 比值 = z检测/z真值, 对比 关键点宽度缩短 的预测
            double yw = (di < (int)yawDegAll.size()) ? yawDegAll[di] : -1;
            double zp = (di < (int)zPredAll.size())   ? zPredAll[di]   : -1;
            if (yw >= 0 && gtp.z > 1e-6) {
                double zRatio = detp.z / gtp.z;
                printf("       yaw(PnP反解)=%.1f°  宽度缩短预测z比=%.2f  实际z比=%.2f\n",
                       yw, zp, zRatio);
            }
        }
        std::cout << "  --------------------------------------------------\n";
        std::cout << "  平均分量误差 |Δx|+|Δy|+|Δz|/3 = "
                  << (bestPairs.empty()?0:sumAbs/bestPairs.size()/3.0) << " m\n";

        // 全局统计: 召回 + 误差累加
        int nVis = 0;
        for (bool v : gts[bestGt].vis) nVis += v ? 1 : 0;
        totVis += nVis; totDet += (int)validIdx.size(); totMatched += (int)bestPairs.size();
        for (auto& p : bestPairs) {
            cv::Point3d d = std::get<3>(p), g = std::get<4>(p);
            sumDx += std::abs(d.x - g.x); sumDy += std::abs(d.y - g.y); sumDz += std::abs(d.z - g.z);
        }
    }
    std::cout << "\n========== NN 总体统计 ==========\n";
    std::cout << "  召回: 检出 " << totDet << " / 可见 " << totVis
              << " (有效配对 " << totMatched << ")\n";
    if (totMatched > 0)
        printf("  平均 |Δx|=%.3f  |Δy|=%.3f  |Δz|=%.3f m\n",
               sumDx / totMatched, sumDy / totMatched, sumDz / totMatched);
    std::cout << "NN 评估完成。" << std::endl;
    return 0;
}
