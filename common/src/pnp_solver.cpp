#include "pnp_solver.h"
#include "camera_config.h"

PnPSolver::PnPSolver()
{
    // 内参与畸变来自老师标定结果（camera_config.h）
    cameraMatrix_ = getCameraMatrix();
    distCoeffs_ = getDistortionCoeffs();
}

PnPSolver::PnPSolver(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
    : cameraMatrix_(cameraMatrix.clone()), distCoeffs_(distCoeffs.clone()) {}

bool PnPSolver::solve(Armor& armor) const
{
    // 按 isLarge 选 3D 物体点（单位：米，顺序与 keypoints 一一对应）
    double halfW = armor.isLarge ? ARMOR_LARGE_HALF_W : ARMOR_SMALL_HALF_W;
    double halfH = armor.isLarge ? ARMOR_LARGE_HALF_H : ARMOR_SMALL_HALF_H;
    std::vector<cv::Point3f> objectPoints = armorObjectPoints(halfW, halfH);

    std::vector<cv::Point2f> imagePoints(armor.keypoints, armor.keypoints + 4);

    cv::Mat rvec, tvec;
    // IPPE：精确平面四点解法；若退化（共线等）会抛异常/返回 false
    try {
        if (!cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, distCoeffs_,
                          rvec, tvec, false, cv::SOLVEPNP_IPPE))
            return false;
    } catch (const cv::Exception&) {
        return false;
    }

    // tvec = 装甲板中心在相机坐标系下的位置
    // 约定（见 camera_config.h）：x 向右、y 向下、z 沿光轴向前，单位米
    armor.position = cv::Point3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    armor.positionValid = true;
    return true;
}
