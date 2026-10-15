#include "armor_detector.h"
#include <algorithm>
#include <cmath>

ArmorDetector::ArmorDetector(int binaryThres) : binaryThres_(binaryThres) {}

std::vector<Armor> ArmorDetector::detect(const cv::Mat& frame, const std::string& color)
{
    // 1) 灰度 + 阈值二值化：灯条高亮 => 亮区
    cv::Mat gray, binary;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, binaryThres_, 255, cv::THRESH_BINARY);

    // 2) 找灯条
    lastLights = findLights(frame, binary);

    // 3) 配对成装甲板（按目标颜色过滤）
    std::vector<LightBar> colored;
    for (const auto& l : lastLights)
        if ((color == "red" && l.isRed) || (color == "blue" && !l.isRed))
            colored.push_back(l);
    return matchLights(colored);
}

// 重载：外部给二值图（蓝光等灰度阈值失效的场景用），其余流程相同
std::vector<Armor> ArmorDetector::detect(const cv::Mat& frame, const cv::Mat& binary,
                                         const std::string& color)
{
    lastLights = findLights(frame, binary);

    std::vector<LightBar> colored;
    for (const auto& l : lastLights)
        if ((color == "red" && l.isRed) || (color == "blue" && !l.isRed))
            colored.push_back(l);
    return matchLights(colored);
}

std::vector<LightBar> ArmorDetector::findLights(const cv::Mat& bgr, const cv::Mat& binary)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> lights;
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;  // 点太少，拟合旋转矩形不稳定

        cv::RotatedRect rrect = cv::minAreaRect(contour);

        // 由旋转矩形还原灯条几何信息
        LightBar l;
        l.rect = rrect;
        cv::Point2f pts[4];
        rrect.points(pts);
        // 按 y 排序：上面两点取均值为 top，下面两点取均值为 bottom
        std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
        l.top = (pts[0] + pts[1]) / 2.0f;
        l.bottom = (pts[2] + pts[3]) / 2.0f;
        l.center = (l.top + l.bottom) / 2.0f;
        l.length = cv::norm(l.top - l.bottom);
        l.width = std::min(rrect.size.width, rrect.size.height);
        l.tiltAngle = std::atan2(std::abs(l.top.x - l.bottom.x),
                                 std::abs(l.top.y - l.bottom.y)) * 180.0 / CV_PI;
        if (!isLight(l)) continue;

        // 颜色判定：轮廓内 R 通道均值 vs B 通道均值，谁大就是什么颜色
        cv::Rect bbox = rrect.boundingRect() & cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (bbox.width <= 0 || bbox.height <= 0) continue;
        cv::Mat roi = bgr(bbox);
        cv::Mat mask(bbox.size(), CV_8UC1, cv::Scalar(0));
        std::vector<cv::Point> shifted;
        for (const auto& p : contour) shifted.push_back(p - bbox.tl());  // 轮廓平移到 ROI 局部坐标
        cv::fillPoly(mask, shifted, cv::Scalar(255));
        cv::Scalar meanBGR = cv::mean(roi, mask);
        l.isRed = meanBGR[2] > meanBGR[0];  // [2]=R [0]=B

        lights.push_back(l);
    }
    return lights;
}

bool ArmorDetector::isLight(const LightBar& l) const
{
    // 灯条是细长竖直的高亮区域：短/长比例受限，倾角受限
    double ratio = l.width / std::max(l.length, 1e-6);
    return (ratio > minRatio_ && ratio < maxRatio_ && l.tiltAngle < maxTiltAngle_);
}

std::vector<Armor> ArmorDetector::matchLights(const std::vector<LightBar>& lights)
{
    // 按 x 排序，保证 light_1 在左、light_2 在右
    std::vector<LightBar> sorted = lights;
    std::sort(sorted.begin(), sorted.end(),
              [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });

    std::vector<Armor> armors;
    for (auto it1 = sorted.begin(); it1 != sorted.end(); ++it1) {
        for (auto it2 = it1 + 1; it2 != sorted.end(); ++it2) {
            const LightBar& L = *it1;
            const LightBar& R = *it2;

            // 约束1：两根灯条长度应接近
            double lightRatio = std::min(L.length, R.length) / std::max(L.length, R.length);
            if (lightRatio < minLightRatio_) continue;

            // 约束2：中心距（以灯条平均长度为单位）落在小/大装甲区间内
            double avgLen = (L.length + R.length) / 2.0;
            double centerDist = cv::norm(L.center - R.center) / avgLen;
            bool distOk = (centerDist >= minSmallDist_ && centerDist < maxSmallDist_) ||
                          (centerDist >= minLargeDist_ && centerDist < maxLargeDist_);
            if (!distOk) continue;

            // 约束3：两灯条连线应近似水平（装甲板不会斜得太离谱）
            cv::Point2f diff = L.center - R.center;
            double angle = std::abs(std::atan(diff.y / (std::abs(diff.x) + 1e-6))) * 180.0 / CV_PI;
            if (angle > maxPairAngle_) continue;

            // 排除"两灯条之间还夹着别的灯条"的假配对：
            // 若有第三根灯条的端点/中心落在这对灯条的包围盒里，则跳过
            std::vector<cv::Point2f> quad{L.top, L.bottom, R.top, R.bottom};
            cv::Rect bbox = cv::boundingRect(quad);
            bool contained = false;
            for (const auto& t : sorted) {
                if (&t == &L || &t == &R) continue;
                if (bbox.contains(t.top) || bbox.contains(t.bottom) || bbox.contains(t.center)) {
                    contained = true;
                    break;
                }
            }
            if (contained) continue;

            // 通过全部约束 => 组装装甲板
            Armor a;
            a.left = L;
            a.right = R;
            a.keypoints[0] = L.top;      // 0 左上
            a.keypoints[1] = L.bottom;   // 1 左下
            a.keypoints[2] = R.bottom;   // 2 右下
            a.keypoints[3] = R.top;      // 3 右上
            a.color = L.isRed ? "red" : "blue";
            a.isLarge = centerDist > minLargeDist_;  // 中心距大 => 大装甲
            armors.push_back(a);
        }
    }
    return armors;
}
