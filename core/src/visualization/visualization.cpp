#include "visualization.h"

#ifdef VIO_ENABLE_VISUALIZATION

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat DrawCorners(cv::Mat image, std::vector<cv::Point2f> const &corners, cv::Scalar const &color,
                    int marker_type)
{
    constexpr int kMarkerSize = 15;
    constexpr int kMarkerThickness = 2;
    cv::Mat canvas;
    cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
    for (auto &&corner : corners)
    {
        cv::drawMarker(canvas, corner, color, marker_type, kMarkerSize, kMarkerThickness);
    }

    return canvas;
}

cv::Mat DrawCorners(cv::Mat image, std::vector<cv::Point2f> const &corners,
                    std::vector<float> const &depths, cv::Scalar const &color, int marker_type)
{
    constexpr int kMarkerSize = 15;
    constexpr int kMarkerThickness = 2;
    cv::Mat canvas;
    cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);

    char text[16];
    for (size_t fi = 0; fi < corners.size(); ++fi)
    {
        cv::drawMarker(canvas, corners[fi], color, marker_type, kMarkerSize, kMarkerThickness);
        snprintf(text, sizeof(text), "%0.3f", depths[fi]);
        cv::putText(canvas, text, corners[fi], cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    return canvas;
}

cv::Mat DrawCorners(cv::Mat image, std::vector<cv::Point2f> const &corners,
                    std::vector<float> const &depths, std::vector<float> const &errs,
                    cv::Scalar const &color, int marker_type)
{
    constexpr int kMarkerSize = 15;
    constexpr int kMarkerThickness = 2;
    cv::Mat canvas;
    cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);

    char text[16];
    for (size_t fi = 0; fi < corners.size(); ++fi)
    {
        cv::drawMarker(canvas, corners[fi], color, marker_type, kMarkerSize, kMarkerThickness);
        snprintf(text, sizeof(text), "%0.3f", depths[fi]);
        cv::putText(canvas, text, corners[fi], cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
        snprintf(text, sizeof(text), "%f", errs[fi]);
        cv::putText(canvas, text, corners[fi] + cv::Point2f(0, 15), cv::FONT_HERSHEY_PLAIN, 1.0,
                    cv::Scalar(0, 128, 255), 2);
    }
    return canvas;
}

cv::Mat DrawFeatTrackResult(cv::Mat img1, cv::Mat img2,
                            std::vector<cv::Point2f> const &feats_on_img1,
                            std::vector<cv::Point2f> const &feats_on_img2,
                            std::vector<uchar> const &stats_on_img2,
                            std::vector<cv::Point2f> const &feats_on_img2_pred,
                            bool use_horizontal_layout)
{
    const int img_width = img1.cols;
    const int img_height = img1.rows;
    const int thickness = 2;
    const int marker_size = 13;

    const cv::Scalar color_img1(255, 0, 0);
    const cv::Scalar color_img2_track(0, 255, 0);
    const cv::Scalar color_img2_lost(0, 0, 255);
    const cv::Scalar color_img2_pred(128, 128, 0);

    cv::Mat canvas;
    if (use_horizontal_layout)
    {
        cv::hconcat(img1, img2, canvas);
    }
    else
    {
        cv::vconcat(img1, img2, canvas);
    }

    cv::Point2f img2_offset =
        use_horizontal_layout ? cv::Point2f(img_width, 0) : cv::Point2f(0, img_height);

    if (canvas.channels() == 1)
    {
        cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < feats_on_img1.size(); ++i)
    {
        cv::Point2f const &feat_on_img1 = feats_on_img1[i];
        cv::drawMarker(canvas, feat_on_img1, color_img1, cv::MARKER_CROSS, marker_size, thickness,
                       cv::LINE_8);
    }

    for (size_t i = 0; i < feats_on_img2_pred.size(); ++i)
    {
        cv::Point2f const &feat_on_img2_pred = feats_on_img2_pred[i];

        cv::drawMarker(canvas, feat_on_img2_pred + img2_offset, color_img2_pred,
                       cv::MARKER_TILTED_CROSS, marker_size, thickness, cv::LINE_8);
    }

    size_t N = std::min(feats_on_img2.size(), stats_on_img2.size());
    int track_success_num = 0;
    for (size_t i = 0; i < N; ++i)
    {
        cv::Point2f const &feat_on_img2 = feats_on_img2[i];
        uchar const &stat_on_img2 = stats_on_img2[i];

        if (stat_on_img2)
        {
            cv::drawMarker(canvas, feat_on_img2 + img2_offset, color_img2_track, cv::MARKER_CROSS,
                           marker_size, thickness, cv::LINE_8);
            ++track_success_num;
        }
        else
        {
            cv::drawMarker(canvas, feat_on_img2 + img2_offset, color_img2_lost, cv::MARKER_CROSS,
                           marker_size, thickness, cv::LINE_8);
        }
    }

    char txt[32];
    snprintf(txt, sizeof(txt), "[%d/%lu]", track_success_num, stats_on_img2.size());
    cv::putText(canvas, txt, cv::Point2f(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255),
                2);

    cv::putText(canvas, "track", cv::Point2f(10, 40), cv::FONT_HERSHEY_PLAIN, 1.0, color_img2_track);
    cv::putText(canvas, "pred", cv::Point2f(10, 60), cv::FONT_HERSHEY_PLAIN, 1.0, color_img2_pred);

    return canvas;
}

cv::Mat DrawFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> const &vec_pt1,
                            std::vector<cv::Point2f> const &vec_pt2,
                            std::vector<uchar> const &vec_status, bool use_horizontal_layout)
{
    int w = img1.cols;
    int h = img1.rows;
    char txt[32];
    cv::Scalar color_match(255, 0, 0);
    cv::Scalar color_lost(0, 0, 255);
    cv::Mat show_img;
    bool draw_feat_idx = (!vec_status.empty()) && (vec_status.size() == vec_pt1.size());
    if (use_horizontal_layout)
    {
        show_img = cv::Mat(h, 2 * w, CV_8UC1);
        img1.copyTo(show_img(cv::Rect(0, 0, w, h)));
        img2.copyTo(show_img(cv::Rect(w, 0, w, h)));
        cv::cvtColor(show_img, show_img, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < vec_pt1.size(); ++i)
        {
            cv::circle(show_img, vec_pt1[i], 3, CV_RGB(255, 0, 0));
            cv::circle(show_img, cv::Point2f(vec_pt2[i].x + w, vec_pt2[i].y), 3, CV_RGB(255, 0, 0));
            snprintf(txt, sizeof(txt), "%lu", i);
            if (draw_feat_idx)
            {
                if (vec_status[i])
                {
                    cv::putText(show_img, std::string(txt), vec_pt1[i] + cv::Point2f(5, 5),
                                cv::FONT_HERSHEY_PLAIN, 0.8, color_match);
                    cv::putText(show_img, std::string(txt),
                                cv::Point2f(vec_pt2[i].x + w, vec_pt2[i].y) + cv::Point2f(5, 5),
                                cv::FONT_HERSHEY_PLAIN, 0.8, color_match);
                }
                else
                {
                    cv::putText(show_img, std::string(txt), vec_pt1[i] + cv::Point2f(5, 5),
                                cv::FONT_HERSHEY_PLAIN, 0.8, color_lost);
                    cv::putText(show_img, std::string(txt),
                                cv::Point2f(vec_pt2[i].x + w, vec_pt2[i].y) + cv::Point2f(5, 5),
                                cv::FONT_HERSHEY_PLAIN, 0.8, color_lost);
                }
            }
            else
                cv::line(show_img, vec_pt1[i], cv::Point2f(vec_pt2[i].x + w, vec_pt2[i].y),
                         CV_RGB(0, 255, 0));
        }
    }
    else
    {
        show_img = cv::Mat(2 * h, w, CV_8UC1);
        img1.copyTo(show_img(cv::Rect(0, 0, w, h)));
        img2.copyTo(show_img(cv::Rect(0, h, w, h)));
        cv::cvtColor(show_img, show_img, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < vec_pt1.size(); ++i)
        {
            cv::circle(show_img, vec_pt1[i], 3, CV_RGB(255, 0, 0));
            cv::circle(show_img, cv::Point2f(vec_pt2[i].x, vec_pt2[i].y + h), 3, CV_RGB(255, 0, 0));
            if (vec_status.size() > 0)
            {
                if (vec_status[i])
                    cv::line(show_img, vec_pt1[i], cv::Point2f(vec_pt2[i].x, vec_pt2[i].y + h), color_match);
                else
                    cv::line(show_img, vec_pt1[i], cv::Point2f(vec_pt2[i].x, vec_pt2[i].y + h), color_lost);
            }
            else
                cv::line(show_img, vec_pt1[i], cv::Point2f(vec_pt2[i].x, vec_pt2[i].y + h),
                         CV_RGB(0, 255, 0));
        }
    }

    return show_img;
}

cv::Mat DrawPnpResult(cv::Mat &img1, std::vector<cv::Point3f> const &p3d_on_img1,
                      std::vector<cv::Point2f> const &feats_on_img1,
                      std::vector<int> const &inlier_idxs)
{
    cv::Mat canvas;
    if (img1.channels() == 1)
    {
        cv::cvtColor(img1, canvas, cv::COLOR_GRAY2BGR);
    }
    else
    {
        canvas = img1.clone();
    }

    if (p3d_on_img1.empty())
    {
        return canvas;
    }

    std::vector<uchar> stats_on_img1(p3d_on_img1.size(), 0);
    for (auto const &idx : inlier_idxs)
    {
        stats_on_img1[idx] = 1;
    }

    char txt[32];
    constexpr int thickness = 2;
    snprintf(txt, sizeof(txt), "[%lu/%lu]", inlier_idxs.size(), p3d_on_img1.size());
    cv::putText(canvas, txt, cv::Point2f(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255),
                thickness);

    for (size_t i = 0; i < p3d_on_img1.size(); ++i)
    {
        if (stats_on_img1[i])
        {
            cv::circle(canvas, feats_on_img1[i], 3, CV_RGB(0, 255, 0), 2);
        }
        else
        {
            cv::circle(canvas, feats_on_img1[i], 3, CV_RGB(255, 0, 0), 2);
        }

        snprintf(txt, sizeof(txt), "%lu:%0.1f", i, p3d_on_img1[i].z);
        cv::putText(canvas, txt, feats_on_img1[i], cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255),
                    thickness);
    }

    return canvas;
}

void ShowFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> const &vec_pt1,
                         std::vector<cv::Point2f> const &vec_pt2,
                         std::vector<uchar> const &vec_status, std::string const &win_name,
                         bool use_horizon_layout)
{
    cv::Mat canvas =
        DrawFeatTrackResult(img1, img2, vec_pt1, vec_pt2, vec_status, use_horizon_layout);
    cv::imshow(win_name.c_str(), canvas);
    cv::waitKey(1);
}

void ShowFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> const &feats_on_img1,
                         std::vector<cv::Point2f> const &feats_on_img2,
                         std::vector<uchar> const &stats_on_img2,
                         std::vector<cv::Point2f> const &feats_on_img2_pred,
                         std::string const &win_name, bool use_horizon_layout, float scale)
{
    cv::Mat canvas = DrawFeatTrackResult(img1, img2, feats_on_img1, feats_on_img2, stats_on_img2,
                                         feats_on_img2_pred, use_horizon_layout);

    cv::Mat canvas_rz;
    if (fabs(scale - 1.0) > 0.1)
    {
        cv::resize(canvas, canvas_rz, cv::Size(), scale, scale);
    }
    else
    {
        canvas_rz = canvas;
    }
    cv::imshow(win_name.c_str(), canvas_rz);
    cv::waitKey(1);
}

void ShowFeatTrackResult(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> const &feats_on_img1,
                         std::vector<cv::KeyPoint> const &feats_on_img2,
                         std::vector<uchar> const &stats_on_img2,
                         std::vector<cv::KeyPoint> const &feats_on_img2_pred,
                         std::string const &win_name, bool use_horizon_layout, float scale)
{
    auto cvkpts2cvpt2f = [](std::vector<cv::KeyPoint> const &kpts, std::vector<cv::Point2f> &pts)
    { std::transform(kpts.begin(), kpts.end(), std::back_inserter(pts),
                     [](const cv::KeyPoint &keypoint)
                     {
                         return keypoint.pt;
                     }); };
    std::vector<cv::Point2f> feats_on_img1_pts, feats_on_img2_pts, feats_on_img2_pred_pts;
    cvkpts2cvpt2f(feats_on_img1, feats_on_img1_pts);
    cvkpts2cvpt2f(feats_on_img2, feats_on_img2_pts);
    cvkpts2cvpt2f(feats_on_img2_pred, feats_on_img2_pred_pts);
    ShowFeatTrackResult(img1, img2, feats_on_img1_pts, feats_on_img2_pts, stats_on_img2, feats_on_img2_pred_pts, win_name, use_horizon_layout, scale);
}

void ShowCorners(cv::Mat image, const std::vector<cv::Point2f> &corners, std::string win_name)
{
    cv::Mat canvas =
        DrawCorners(image, corners, cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_CROSS);
    cv::imshow(win_name.c_str(), canvas);
    cv::waitKey(1);
}

void ShowPnpResult(cv::Mat img1, std::vector<cv::Point3f> const &p3d_on_img1,
                   std::vector<cv::Point2f> const &feats_on_img1,
                   std::vector<int> const &inlier_idxs, std::string const &win_name)
{
    cv::Mat canvas = DrawPnpResult(img1, p3d_on_img1, feats_on_img1, inlier_idxs);
    cv::imshow(win_name.c_str(), canvas);
    cv::waitKey(1);
}

void ShowImage(std::string const &win_name, cv::Mat const &canvas)
{
    cv::imshow(win_name.c_str(), canvas);
    cv::waitKey(1);
}
#endif
