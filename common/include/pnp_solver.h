#pragma once
#include "armor_types.h"
// OpenCV5: solvePnP 及 SOLVEPNP_IPPE 已从 calib3d 移到 geometry/3d.hpp
#include <opencv2/geometry.hpp>

// ============================================================
//  PnP 位姿解算器（D2，传统/NN 两方案共用）
//
//  原理：
//   已知 4 个关键点在装甲板平面上的 3D 物理坐标（原点=装甲板中心，
//   单位米，见 camera_config.h 的 armorObjectPoints），以及它们在
//   图像中的 2D 像素坐标，用 solvePnP 解出装甲板相对相机的位姿
//   (rvec, tvec)。tvec 就是装甲板中心在相机坐标系下的位置。
//
//   - 小/大装甲用不同的 3D 半宽半高（140x125 / 235x127 mm）
//   - IPPE 算法专为"平面 4 点"设计，速度快且有两个候选解，
//     OpenCV 自动返回重投影误差更小的那个
//   - 畸变系数已在 solvePnP 内部处理（作业要求：正确处理畸变）
// ============================================================
class PnPSolver {
public:
    PnPSolver();  // 默认使用 camera_config.h 的标定内参

    // 重载：注入自定义内参（例如图像分辨率与标定分辨率不同时，需按比例缩放 fx/fy/cx/cy）
    PnPSolver(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

    // 就地解算：成功则填 armor.position（米）并置 positionValid=true
    // 返回 false 表示解算失败（关键点退化等），调用方需跳过该装甲
    bool solve(Armor& armor) const;

    const cv::Mat& cameraMatrix() const { return cameraMatrix_; }
    const cv::Mat& distCoeffs() const { return distCoeffs_; }

private:
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
};
