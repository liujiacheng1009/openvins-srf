#pragma once
#include <opencv2/core.hpp>

#ifdef VIO_ENABLE_VISUALIZATION

cv::Mat DrawCorners(cv::Mat image, std::vector<cv::Point2f> const &corners, cv::Scalar const &color,
                    int marker_type);

cv::Mat DrawCorners(cv::Mat image, std::vector<cv::Point2f> const &corners,
                    std::vector<float> const &depths, cv::Scalar const &color, int marker_type);

cv::Mat DrawCorners(cv::Mat image, std::vector<cv::Point2f> const &corners,
                    std::vector<float> const &depths, std::vector<float> const &errs,
                    cv::Scalar const &color, int marker_type);

cv::Mat DrawFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> const &vec_pt1,
                            std::vector<cv::Point2f> const &vec_pt2,
                            std::vector<uchar> const &vec_status, bool use_horizontal_layout);

cv::Mat DrawFeatTrackResult(cv::Mat img1, cv::Mat img2,
                            std::vector<cv::Point2f> const &feats_on_img1,
                            std::vector<cv::Point2f> const &feats_on_img2,
                            std::vector<uchar> const &stats_on_img2,
                            std::vector<cv::Point2f> const &feats_on_img2_pred,
                            bool use_horizontal_layout);

cv::Mat DrawPnpResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point3f> const &p3d_on_img1,
                      std::vector<cv::Point2f> const &feats_on_img1,
                      std::vector<int> const &inlier_idxs);

void ShowCorners(cv::Mat image, std::vector<cv::Point2f> const &corners, std::string win_name);

void ShowFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> const &vec_pt1,
                         std::vector<cv::Point2f> const &vec_pt2,
                         std::vector<uchar> const &vec_status, std::string const &win_name,
                         bool use_horizon_layout);

void ShowFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> const &feats_on_img1,
                         std::vector<cv::Point2f> const &feats_on_img2,
                         std::vector<uchar> const &stats_on_img2,
                         std::vector<cv::Point2f> const &feats_on_img2_pred,
                         std::string const &win_name, bool use_horizon_layout, float scale = 1.0);

void ShowFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> const &feats_on_img1,
                         std::vector<cv::KeyPoint> const &feats_on_img2,
                         std::vector<uchar> const &stats_on_img2,
                         std::vector<cv::KeyPoint> const &feats_on_img2_pred,
                         std::string const &win_name, bool use_horizon_layout, float scale = 1.0);

void ShowPnpResult(cv::Mat img1, std::vector<cv::Point3f> const &p3d_on_img1,
                   std::vector<cv::Point2f> const &feats_on_img1,
                   std::vector<int> const &inlier_idxs, std::string const &win_name);

void ShowImage(std::string const &win_name, cv::Mat const &img);

#else

#define DrawCorners(...)
#define DrawFeatTrackResult(...)
#define DrawPnpResult(...)
#define ShowCorners(...)
#define ShowFeatTrackResult(...)
#define ShowPnpResult(...)
#define ShowImage(...)

#endif
