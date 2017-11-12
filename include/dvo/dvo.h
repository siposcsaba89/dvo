#pragma once
#include <napalm/napalm.h>
#include <Eigen/Eigen> 

struct DVOParams
{
    napalm::Context * ctx = nullptr;
    Eigen::Matrix3f K; /// if camera is calibrated, the undistorted camera matrix
};

class DVO
{
public:
    DVO(const DVOParams & params = DVOParams());
    ~DVO();
    Eigen::Matrix4f track(const napalm::Img * img, const napalm::Img * mask = nullptr);
    Eigen::Matrix4f track(const napalm::Img * img, const napalm::Img * depth, const napalm::Img * mask = nullptr);
    Eigen::Matrix4f track(const napalm::Img * img_left, const napalm::Img * img_left_mask,
        const napalm::Img * img_right, const napalm::Img * img_right_mask);

    
private:
    DVOParams m_params;
};

