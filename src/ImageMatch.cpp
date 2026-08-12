#include "ImageMatch.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <omp.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <unistd.h>

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>

using namespace cv::xfeatures2d;

using namespace cv;
using namespace std;

namespace
{
    bool g_use_robust_scoreb = false;
    bool g_use_affine_patch_score = false;

    enum class MatchMode
    {
        CensusGrad = 0,   // 方案A：Census + Gradient
        GrayGrad   = 1    // 方案B：ZNCC(gray) + ZNCC(gradient)
    };

    inline bool patchInside(const Mat& img, int r, int c, int half)
    {
        return (r >= half && c >= half && r < img.rows - half && c < img.cols - half);
    }

    inline int popcount64(uint64_t x)
    {
        return static_cast<int>(__builtin_popcountll(x));
    }

    static void computeGradientMag(const Mat& img, Mat& gradMag)
    {
        Mat gx, gy;
        Sobel(img, gx, CV_32F, 1, 0, 3);
        Sobel(img, gy, CV_32F, 0, 1, 3);
        gradMag = abs(gx) + abs(gy);   // 比 sqrt(gx^2+gy^2) 更快
    }

    struct LocalGridPrediction
    {
        int row = 0;
        int col = 0;
        int mode = 0;  // 1=surface branch; center may be robust local median fallback
        bool surface_rejected = false;
    };

    static double weightedMedian(std::vector<std::pair<double, double> > samples)
    {
        if (samples.empty()) return 0.0;
        std::sort(samples.begin(), samples.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first;
            });
        double total_w = 0.0;
        for (const auto& s : samples) total_w += std::max(0.0, s.second);
        if (total_w <= 0.0) return samples[samples.size() / 2].first;
        double acc = 0.0;
        for (const auto& s : samples) {
            acc += std::max(0.0, s.second);
            if (acc >= 0.5 * total_w) return s.first;
        }
        return samples.back().first;
    }

    static bool searchWindowTouchesImage(int pred_r, int pred_c,
                                         int rows, int cols,
                                         int ser_range)
    {
        return !(pred_r + ser_range < 0 || pred_r - ser_range >= rows ||
                 pred_c + ser_range < 0 || pred_c - ser_range >= cols);
    }

    static bool shouldSearchCurrentRccd(int aff_r, int aff_c,
                                        int local_r, int local_c,
                                        int rows, int cols,
                                        int ser_range)
    {
        return searchWindowTouchesImage(aff_r, aff_c, rows, cols, ser_range) ||
               searchWindowTouchesImage(local_r, local_c, rows, cols, ser_range);
    }

    static LocalGridPrediction selectReliableGridPrediction(
        int left_r, int left_c,
        int aff_r, int aff_c,
        int surface_r, int surface_c,
        int right_rows, int right_cols,
        int ser_range,
        int affine_max_dev,
        const std::vector<int>& localmatch,
        const std::vector<double>& weights)
    {
        LocalGridPrediction pred;
        pred.row = surface_r;
        pred.col = surface_c;
        pred.mode = 1;

        std::vector<std::pair<double, double> > dx_samples;
        std::vector<std::pair<double, double> > dy_samples;
        const int n = static_cast<int>(localmatch.size() / 5);
        dx_samples.reserve(n);
        dy_samples.reserve(n);
        for (int i = 0; i < n; ++i) {
            const double w = (i < static_cast<int>(weights.size())) ? weights[i] : 1.0;
            dx_samples.push_back(std::make_pair(
                static_cast<double>(localmatch[5 * i + 3] - localmatch[5 * i + 0]), w));
            dy_samples.push_back(std::make_pair(
                static_cast<double>(localmatch[5 * i + 4] - localmatch[5 * i + 1]), w));
        }

        const int median_r = left_r + static_cast<int>(weightedMedian(dx_samples) + 0.5);
        const int median_c = left_c + static_cast<int>(weightedMedian(dy_samples) + 0.5);
        const bool surface_in = searchWindowTouchesImage(surface_r, surface_c, right_rows, right_cols, ser_range);
        const bool median_in = searchWindowTouchesImage(median_r, median_c, right_rows, right_cols, ser_range);
        const double surf_dr = static_cast<double>(surface_r - aff_r);
        const double surf_dc = static_cast<double>(surface_c - aff_c);
        const double med_dr = static_cast<double>(median_r - aff_r);
        const double med_dc = static_cast<double>(median_c - aff_c);
        const double surface_dev = sqrt(surf_dr * surf_dr + surf_dc * surf_dc);
        const double median_dev = sqrt(med_dr * med_dr + med_dc * med_dc);
        const double dev_limit = std::max(static_cast<double>(affine_max_dev) * 3.0,
                                          static_cast<double>(ser_range) * 4.0);

        if (!surface_in ||
            (surface_dev > dev_limit && median_in &&
             median_dev <= static_cast<double>(affine_max_dev) &&
             median_dev < surface_dev * 0.75)) {
            pred.row = median_r;
            pred.col = median_c;
            pred.mode = 1;
            pred.surface_rejected = true;
        }
        return pred;
    }

    static LocalGridPrediction selectReliableFeaturePrediction(
        int left_r, int left_c,
        int aff_r, int aff_c,
        int local_r, int local_c,
        int right_rows, int right_cols,
        int local_ser_range,
        int affine_ser_range,
        int affine_max_dev,
        const std::vector<int>& localmatch,
        const std::vector<double>& weights)
    {
        LocalGridPrediction pred;
        pred.row = local_r;
        pred.col = local_c;
        pred.mode = -1;

        std::vector<std::pair<double, double> > dx_samples;
        std::vector<std::pair<double, double> > dy_samples;
        const int n = static_cast<int>(localmatch.size() / 5);
        dx_samples.reserve(n);
        dy_samples.reserve(n);
        for (int i = 0; i < n; ++i) {
            const double w = (i < static_cast<int>(weights.size())) ? weights[i] : 1.0;
            dx_samples.push_back(std::make_pair(
                static_cast<double>(localmatch[5 * i + 3] - localmatch[5 * i + 0]), w));
            dy_samples.push_back(std::make_pair(
                static_cast<double>(localmatch[5 * i + 4] - localmatch[5 * i + 1]), w));
        }

        const int median_r = left_r + static_cast<int>(weightedMedian(dx_samples) + 0.5);
        const int median_c = left_c + static_cast<int>(weightedMedian(dy_samples) + 0.5);
        const bool local_in = searchWindowTouchesImage(local_r, local_c, right_rows, right_cols, local_ser_range);
        const bool median_in = searchWindowTouchesImage(median_r, median_c, right_rows, right_cols, local_ser_range);
        const bool affine_in = searchWindowTouchesImage(aff_r, aff_c, right_rows, right_cols, affine_ser_range);
        const double local_dr = static_cast<double>(local_r - aff_r);
        const double local_dc = static_cast<double>(local_c - aff_c);
        const double med_dr = static_cast<double>(median_r - aff_r);
        const double med_dc = static_cast<double>(median_c - aff_c);
        const double local_dev = sqrt(local_dr * local_dr + local_dc * local_dc);
        const double median_dev = sqrt(med_dr * med_dr + med_dc * med_dc);
        const double dev_limit = std::max(static_cast<double>(affine_max_dev) * 3.0,
                                          static_cast<double>(local_ser_range) * 4.0);

        if (affine_in && !local_in && !median_in) {
            pred.row = aff_r;
            pred.col = aff_c;
            pred.mode = 0;
            pred.surface_rejected = true;
        } else if ((!local_in && median_in) ||
            (local_dev > dev_limit && median_in &&
             (median_dev <= static_cast<double>(affine_max_dev) || median_dev < local_dev * 0.75)) ||
            (!local_in && !affine_in)) {
            pred.row = median_r;
            pred.col = median_c;
            pred.surface_rejected = true;
        }
        return pred;
    }

    static float patchZNCC_U8(const Mat& img1, int r1, int c1,
                              const Mat& img2, int r2, int c2,
                              int half)
    {
        const int win = 2 * half + 1;
        const int N = win * win;

        double mean1 = 0.0, mean2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const uchar* p1 = img1.ptr<uchar>(r1 + dr);
            const uchar* p2 = img2.ptr<uchar>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                mean1 += p1[c1 + dc];
                mean2 += p2[c2 + dc];
            }
        }
        mean1 /= N;
        mean2 /= N;

        double num = 0.0, den1 = 0.0, den2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const uchar* p1 = img1.ptr<uchar>(r1 + dr);
            const uchar* p2 = img2.ptr<uchar>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                const double a = static_cast<double>(p1[c1 + dc]) - mean1;
                const double b = static_cast<double>(p2[c2 + dc]) - mean2;
                num  += a * b;
                den1 += a * a;
                den2 += b * b;
            }
        }

        const double denom = sqrt(std::max(den1 * den2, 1e-12));
        if (denom < 1e-12) return -1.0f;
        return static_cast<float>(num / denom);   // [-1, 1]
    }

    static float patchZNCC_F32(const Mat& img1, int r1, int c1,
                               const Mat& img2, int r2, int c2,
                               int half)
    {
        const int win = 2 * half + 1;
        const int N = win * win;

        double mean1 = 0.0, mean2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const float* p1 = img1.ptr<float>(r1 + dr);
            const float* p2 = img2.ptr<float>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                mean1 += p1[c1 + dc];
                mean2 += p2[c2 + dc];
            }
        }
        mean1 /= N;
        mean2 /= N;

        double num = 0.0, den1 = 0.0, den2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const float* p1 = img1.ptr<float>(r1 + dr);
            const float* p2 = img2.ptr<float>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                const double a = static_cast<double>(p1[c1 + dc]) - mean1;
                const double b = static_cast<double>(p2[c2 + dc]) - mean2;
                num  += a * b;
                den1 += a * a;
                den2 += b * b;
            }
        }

        const double denom = sqrt(std::max(den1 * den2, 1e-12));
        if (denom < 1e-12) return -1.0f;
        return static_cast<float>(num / denom);   // [-1, 1]
    }

    static float patchWeightedZNCC_U8(const Mat& img1, int r1, int c1,
                                      const Mat& img2, int r2, int c2,
                                      int half)
    {
        const float sigma2 = std::max(4.0f, 0.16f * static_cast<float>(half * half));
        double sw = 0.0, mean1 = 0.0, mean2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const uchar* p1 = img1.ptr<uchar>(r1 + dr);
            const uchar* p2 = img2.ptr<uchar>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                const float dist2 = static_cast<float>(dr * dr + dc * dc);
                const double w = 1.0 / (1.0 + dist2 / sigma2);
                mean1 += w * static_cast<double>(p1[c1 + dc]);
                mean2 += w * static_cast<double>(p2[c2 + dc]);
                sw += w;
            }
        }
        if (sw <= 1e-12) return -1.0f;
        mean1 /= sw;
        mean2 /= sw;

        double num = 0.0, den1 = 0.0, den2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const uchar* p1 = img1.ptr<uchar>(r1 + dr);
            const uchar* p2 = img2.ptr<uchar>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                const float dist2 = static_cast<float>(dr * dr + dc * dc);
                const double w = 1.0 / (1.0 + dist2 / sigma2);
                const double a = static_cast<double>(p1[c1 + dc]) - mean1;
                const double b = static_cast<double>(p2[c2 + dc]) - mean2;
                num  += w * a * b;
                den1 += w * a * a;
                den2 += w * b * b;
            }
        }

        const double denom = sqrt(std::max(den1 * den2, 1e-12));
        if (denom < 1e-12) return -1.0f;
        return static_cast<float>(num / denom);
    }

    static float patchWeightedZNCC_F32(const Mat& img1, int r1, int c1,
                                       const Mat& img2, int r2, int c2,
                                       int half)
    {
        const float sigma2 = std::max(4.0f, 0.16f * static_cast<float>(half * half));
        double sw = 0.0, mean1 = 0.0, mean2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const float* p1 = img1.ptr<float>(r1 + dr);
            const float* p2 = img2.ptr<float>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                const float dist2 = static_cast<float>(dr * dr + dc * dc);
                const double w = 1.0 / (1.0 + dist2 / sigma2);
                mean1 += w * static_cast<double>(p1[c1 + dc]);
                mean2 += w * static_cast<double>(p2[c2 + dc]);
                sw += w;
            }
        }
        if (sw <= 1e-12) return -1.0f;
        mean1 /= sw;
        mean2 /= sw;

        double num = 0.0, den1 = 0.0, den2 = 0.0;
        for (int dr = -half; dr <= half; ++dr)
        {
            const float* p1 = img1.ptr<float>(r1 + dr);
            const float* p2 = img2.ptr<float>(r2 + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                const float dist2 = static_cast<float>(dr * dr + dc * dc);
                const double w = 1.0 / (1.0 + dist2 / sigma2);
                const double a = static_cast<double>(p1[c1 + dc]) - mean1;
                const double b = static_cast<double>(p2[c2 + dc]) - mean2;
                num  += w * a * b;
                den1 += w * a * a;
                den2 += w * b * b;
            }
        }

        const double denom = sqrt(std::max(den1 * den2, 1e-12));
        if (denom < 1e-12) return -1.0f;
        return static_cast<float>(num / denom);
    }

    static bool sampleBilinearU8(const Mat& img, float r, float c, double* value)
    {
        const int r0 = static_cast<int>(std::floor(r));
        const int c0 = static_cast<int>(std::floor(c));
        if (r0 < 0 || c0 < 0 || r0 + 1 >= img.rows || c0 + 1 >= img.cols) return false;
        const float fr = r - static_cast<float>(r0);
        const float fc = c - static_cast<float>(c0);
        const uchar* p0 = img.ptr<uchar>(r0);
        const uchar* p1 = img.ptr<uchar>(r0 + 1);
        *value = (1.0f - fr) * ((1.0f - fc) * p0[c0] + fc * p0[c0 + 1]) +
                 fr * ((1.0f - fc) * p1[c0] + fc * p1[c0 + 1]);
        return true;
    }

    static bool sampleBilinearF32(const Mat& img, float r, float c, double* value)
    {
        const int r0 = static_cast<int>(std::floor(r));
        const int c0 = static_cast<int>(std::floor(c));
        if (r0 < 0 || c0 < 0 || r0 + 1 >= img.rows || c0 + 1 >= img.cols) return false;
        const float fr = r - static_cast<float>(r0);
        const float fc = c - static_cast<float>(c0);
        const float* p0 = img.ptr<float>(r0);
        const float* p1 = img.ptr<float>(r0 + 1);
        *value = (1.0f - fr) * ((1.0f - fc) * p0[c0] + fc * p0[c0 + 1]) +
                 fr * ((1.0f - fc) * p1[c0] + fc * p1[c0 + 1]);
        return true;
    }

    template <typename LeftAt, typename RightSample>
    static float patchAffineZNCC(const Mat& img1, const Mat& img2,
                                 int r1, int c1, int r2, int c2, int half,
                                 const float* local_affine,
                                 bool weighted,
                                 LeftAt left_at,
                                 RightSample right_sample)
    {
        const float sigma2 = std::max(4.0f, 0.16f * static_cast<float>(half * half));
        double sw = 0.0, mean1 = 0.0, mean2 = 0.0;
        for (int dr = -half; dr <= half; ++dr) {
            for (int dc = -half; dc <= half; ++dc) {
                const float rr = static_cast<float>(r2) + local_affine[0] * dr + local_affine[1] * dc;
                const float cc = static_cast<float>(c2) + local_affine[3] * dr + local_affine[4] * dc;
                double rv = 0.0;
                if (!right_sample(img2, rr, cc, &rv)) return -1.0f;
                const float dist2 = static_cast<float>(dr * dr + dc * dc);
                const double w = weighted ? 1.0 / (1.0 + dist2 / sigma2) : 1.0;
                mean1 += w * left_at(img1, r1 + dr, c1 + dc);
                mean2 += w * rv;
                sw += w;
            }
        }
        if (sw <= 1e-12) return -1.0f;
        mean1 /= sw;
        mean2 /= sw;

        double num = 0.0, den1 = 0.0, den2 = 0.0;
        for (int dr = -half; dr <= half; ++dr) {
            for (int dc = -half; dc <= half; ++dc) {
                const float rr = static_cast<float>(r2) + local_affine[0] * dr + local_affine[1] * dc;
                const float cc = static_cast<float>(c2) + local_affine[3] * dr + local_affine[4] * dc;
                double rv = 0.0;
                if (!right_sample(img2, rr, cc, &rv)) return -1.0f;
                const float dist2 = static_cast<float>(dr * dr + dc * dc);
                const double w = weighted ? 1.0 / (1.0 + dist2 / sigma2) : 1.0;
                const double a = left_at(img1, r1 + dr, c1 + dc) - mean1;
                const double b = rv - mean2;
                num += w * a * b;
                den1 += w * a * a;
                den2 += w * b * b;
            }
        }
        const double denom = sqrt(std::max(den1 * den2, 1e-12));
        if (denom < 1e-12) return -1.0f;
        return static_cast<float>(num / denom);
    }

    static float computeScoreBAffinePatch(const Mat& img1, const Mat& img2,
                                          const Mat& grad1, const Mat& grad2,
                                          int r1, int c1, int r2, int c2, int half,
                                          const float* local_affine)
    {
        auto left_u8 = [](const Mat& img, int r, int c) -> double {
            return static_cast<double>(img.at<uchar>(r, c));
        };
        auto left_f32 = [](const Mat& img, int r, int c) -> double {
            return static_cast<double>(img.at<float>(r, c));
        };
        const float s_gray = patchAffineZNCC(img1, img2, r1, c1, r2, c2, half,
            local_affine, g_use_robust_scoreb, left_u8, sampleBilinearU8);
        const float s_grad = patchAffineZNCC(grad1, grad2, r1, c1, r2, c2, half,
            local_affine, g_use_robust_scoreb, left_f32, sampleBilinearF32);
        if (!g_use_robust_scoreb) {
            return 0.90f * s_gray + 0.10f * s_grad;
        }

        const float s_gray_full = patchAffineZNCC(img1, img2, r1, c1, r2, c2, half,
            local_affine, false, left_u8, sampleBilinearU8);
        return 0.82f * s_gray + 0.13f * s_grad + 0.05f * s_gray_full;
    }

    static void computeCensusSignature(const Mat& img, int r, int c, int half, vector<uint64_t>& desc)
    {
        const int bits = (2 * half + 1) * (2 * half + 1) - 1;
        const int words = (bits + 63) >> 6;
        desc.assign(words, 0ull);

        const uchar center = img.at<uchar>(r, c);
        int bitIdx = 0;

        for (int dr = -half; dr <= half; ++dr)
        {
            const uchar* p = img.ptr<uchar>(r + dr);
            for (int dc = -half; dc <= half; ++dc)
            {
                if (dr == 0 && dc == 0) continue;
                if (p[c + dc] < center)
                {
                    desc[bitIdx >> 6] |= (1ull << (bitIdx & 63));
                }
                ++bitIdx;
            }
        }
    }

    static float censusSimilarity(const vector<uint64_t>& d1, const vector<uint64_t>& d2, int totalBits)
    {
        int hd = 0;
        const int n = static_cast<int>(d1.size());
        for (int i = 0; i < n; ++i)
        {
            hd += popcount64(d1[i] ^ d2[i]);
        }
        const float sim01 = 1.0f - static_cast<float>(hd) / static_cast<float>(std::max(totalBits, 1)); // [0,1]
        return 2.0f * sim01 - 1.0f; // 映射到 [-1,1]，便于与ZNCC统一
    }

    static float computeScoreA(const Mat& img1, const Mat& img2,
                               const Mat& grad1, const Mat& grad2,
                               int r1, int c1, int r2, int c2, int half,
                               const vector<uint64_t>& desc1, vector<uint64_t>& desc2_buf)
    {
        computeCensusSignature(img2, r2, c2, half, desc2_buf);
        const int totalBits = (2 * half + 1) * (2 * half + 1) - 1;

        const float s_census = censusSimilarity(desc1, desc2_buf, totalBits);       // [-1,1]
        const float s_grad   = patchZNCC_F32(grad1, r1, c1, grad2, r2, c2, half);   // [-1,1]

        // 主项靠 Census，梯度项作结构补充
        return 0.80f * s_census + 0.20f * s_grad;
    }

    static float computeScoreB(const Mat& img1, const Mat& img2,
                               const Mat& grad1, const Mat& grad2,
                               int r1, int c1, int r2, int c2, int half)
    {
        if (!g_use_robust_scoreb) {
            const float s_gray = patchZNCC_U8 (img1,  r1, c1, img2,  r2, c2, half);
            const float s_grad = patchZNCC_F32(grad1, r1, c1, grad2, r2, c2, half);
            return 0.90f * s_gray + 0.10f * s_grad;
        }
        const float s_gray_w = patchWeightedZNCC_U8 (img1,  r1, c1, img2,  r2, c2, half);  // [-1,1]
        const float s_grad_w = patchWeightedZNCC_F32(grad1, r1, c1, grad2, r2, c2, half);  // [-1,1]
        const float s_gray   = patchZNCC_U8         (img1,  r1, c1, img2,  r2, c2, half);  // [-1,1]

        // 中心加权支持窗口降低边角局部形变/遮挡对整窗相关性的破坏；
        // 少量全窗口灰度项保留对大块错误匹配的惩罚。
        return 0.82f * s_gray_w + 0.13f * s_grad_w + 0.05f * s_gray;
    }

    // limit_grid4/5 使用的窗口相关：先预计算左窗口 sum/sumsq，再只扫右窗口
    static void accumLeftPatchSums(const Mat& img, int r0, int c0, int half,
                                   float& sum1, float& s11, float& s1)
    {
        sum1 = s11 = s1 = 0.f;
        const int cols = img.cols;
        for (int k = -half; k <= half; ++k) {
            const uchar* row = img.data + (r0 + k) * cols;
            for (int m = -half; m <= half; ++m) {
                const float L = static_cast<float>(row[c0 + m]);
                sum1 += L;
                s11 += L * L;
                s1 += L;
            }
        }
    }

    static float patchCCWithPreLeft(const Mat& img1, const Mat& img2,
                                    int r1, int c1, int r2, int c2, int half,
                                    float sum1, float s11, float s1, float N)
    {
        if (sum1 == 0.f) return -1.f;
        float s12 = 0.f, s22 = 0.f, s2 = 0.f, sum2 = 0.f;
        const int cols1 = img1.cols;
        const int cols2 = img2.cols;
        for (int k = -half; k <= half; ++k) {
            const uchar* rowL = img1.data + (r1 + k) * cols1;
            const uchar* rowR = img2.data + (r2 + k) * cols2;
            for (int m = -half; m <= half; ++m) {
                const float L = static_cast<float>(rowL[c1 + m]);
                const float R = static_cast<float>(rowR[c2 + m]);
                sum2 += R;
                s22 += R * R;
                s2 += R;
                s12 += L * R;
            }
        }
        if (sum2 == 0.f) return -1.f;
        const float s12n = s12 / sum1 / sum2;
        const float s22n = s22 / sum2 / sum2;
        const float s11n = s11 / sum1 / sum1;
        const float s1n = s1 / sum1;
        const float s2n = s2 / sum2;
        const float den = (s11n - s1n * s1n / N) * (s22n - s2n * s2n / N);
        if (!(den > 0.f)) return -1.f;
        return (s12n - s1n * s2n / N) / std::sqrt(den);
    }

    static float gridPatchScore(const Mat& img1, const Mat& img2,
                                const Mat& grad1, const Mat& grad2,
                                int r1, int c1, int r2, int c2, int half,
                                float sum1, float s11, float s1, float N,
                                const float* fs)
    {
        if (!g_use_affine_patch_score) {
            return patchCCWithPreLeft(img1, img2, r1, c1, r2, c2, half, sum1, s11, s1, N);
        }
        float local_affine[6] = {fs[0], fs[1], 0.0f, fs[3], fs[4], 0.0f};
        return computeScoreBAffinePatch(img1, img2, grad1, grad2, r1, c1, r2, c2, half, local_affine);
    }

    // 仿射四角是否完全落在影像外（与 limit_match1 原判定一致）
    static bool fsFullyOutside(const float* fs, int rows1, int cols1, int rows2, int cols2)
    {
        const float c00 = fs[3] * 0 + fs[4] * 0 + fs[5];
        const float c01 = fs[3] * 0 + fs[4] * cols1 + fs[5];
        const float c10 = fs[3] * rows1 + fs[4] * 0 + fs[5];
        const float c11 = fs[3] * rows1 + fs[4] * cols1 + fs[5];
        const bool all_neg = (c00 < 0 && c01 < 0 && c10 < 0 && c11 < 0);
        const bool all_hi = (c00 > cols2 && c01 > cols2 && c10 > cols2 && c11 > cols2);
        return all_neg || all_hi;
    }

    // 右特征点空间分箱，避免 O(N1*N2) 全扫
    struct RightPointGrid {
        int bin = 64;
        int r0 = 0, c0 = 0, nr = 0, nc = 0;
        std::vector<std::vector<int>> cells;

        void build(const std::vector<int>& yr, const std::vector<int>& xc, int bin_size)
        {
            bin = std::max(16, bin_size);
            cells.clear();
            nr = nc = 0;
            if (yr.empty()) return;
            int rmin = yr[0], rmax = yr[0], cmin = xc[0], cmax = xc[0];
            for (size_t i = 1; i < yr.size(); ++i) {
                rmin = std::min(rmin, yr[i]);
                rmax = std::max(rmax, yr[i]);
                cmin = std::min(cmin, xc[i]);
                cmax = std::max(cmax, xc[i]);
            }
            r0 = rmin;
            c0 = cmin;
            nr = (rmax - rmin) / bin + 1;
            nc = (cmax - cmin) / bin + 1;
            cells.assign(static_cast<size_t>(nr * nc), {});
            for (int j = 0; j < (int)yr.size(); ++j) {
                const int br = (yr[j] - r0) / bin;
                const int bc = (xc[j] - c0) / bin;
                if (br < 0 || bc < 0 || br >= nr || bc >= nc) continue;
                cells[static_cast<size_t>(br * nc + bc)].push_back(j);
            }
        }

        void query(int pr, int pc, int half_r, int half_c, std::vector<int>& out) const
        {
            out.clear();
            if (cells.empty()) return;
            const int br0 = std::max(0, (pr - half_r - r0) / bin);
            const int br1 = std::min(nr - 1, (pr + half_r - r0) / bin);
            const int bc0 = std::max(0, (pc - half_c - c0) / bin);
            const int bc1 = std::min(nc - 1, (pc + half_c - c0) / bin);
            for (int br = br0; br <= br1; ++br) {
                for (int bc = bc0; bc <= bc1; ++bc) {
                    const auto& cell = cells[static_cast<size_t>(br * nc + bc)];
                    out.insert(out.end(), cell.begin(), cell.end());
                }
            }
        }
    };

    static int intra_CCD_match_core(char* imagepath1, char* imagepath2,
                                    int w_size, int ser_range, float threshold,
                                    int batch_size_r, int batch_size_c,
                                    int CCD_id, char* outpointxt_1,
                                    MatchMode mode)
    {
        Mat img_1 = imread(imagepath1, 0);
        Mat img_2 = imread(imagepath2, 0);
        if ((!img_1.data) || (!img_2.data))
        {
            std::cout << " --(!) Error reading images " << std::endl;
            return -1;
        }

        if ((w_size & 1) == 0)
        {
            std::cout << "Warning: w_size should be odd. Auto change to w_size+1." << std::endl;
            w_size += 1;
        }
        const int half = w_size / 2;

        Mat grad_1, grad_2;
        computeGradientMag(img_1, grad_1);
        computeGradientMag(img_2, grad_2);

        const int rows1 = img_1.rows;
        const int cols1 = img_1.cols;

        // 先收集格网点，再按点并行匹配
        std::vector<float> KeyPoint_x1, KeyPoint_y1;
        for (int i = 0; (i + 1) * batch_size_r <= rows1; i++)
        {
            for (int j = 0; (j + 1) * batch_size_c <= 48; j++)
            {
                const int r = i * batch_size_r + batch_size_r / 2;
                const int c = j * batch_size_c + batch_size_c / 2 + cols1 - 48;
                if (!patchInside(img_1, r, c, half)) continue;
                KeyPoint_x1.push_back(static_cast<float>(r));
                KeyPoint_y1.push_back(static_cast<float>(c));
            }
        }

        const int n_pts = static_cast<int>(KeyPoint_x1.size());
        std::vector<float> matchedx(n_pts, 0.f), matchedy(n_pts, 0.f), C_match(n_pts, -1.f);
        // 粗搜 stride=2，精化填洞（与 limit_grid_global 一致）
        const int coarse_stride = 2;
        const double t0 = omp_get_wtime();
        // 若外层已在 parallel（CCD 对并行），则内层不再嵌套，避免过订阅
        const int use_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
        printf("[intra_CCD] OpenMP points=%d w=%d ser=%d stride=%d threads=%d\n",
            n_pts, w_size, ser_range, coarse_stride, use_threads);
        fflush(stdout);

        #pragma omp parallel for schedule(dynamic, 8) if(!omp_in_parallel())
        for (int pi = 0; pi < n_pts; ++pi)
        {
            const int r = static_cast<int>(KeyPoint_x1[pi]);
            const int c = static_cast<int>(KeyPoint_y1[pi]);
            vector<uint64_t> desc1, desc2_buf;
            if (mode == MatchMode::CensusGrad)
            {
                computeCensusSignature(img_1, r, c, half, desc1);
            }

            float maxScore = -1e9f;
            float best_rr = static_cast<float>(r);
            float best_cc = static_cast<float>(c);

            for (int ii = -ser_range; ii < ser_range; ii += coarse_stride)
            {
                for (int jj = -ser_range; jj < ser_range; jj += coarse_stride)
                {
                    const int rr = r + ii;
                    const int cc = c - cols1 + 48 + jj;
                    if (!patchInside(img_2, rr, cc, half)) continue;

                    float score = -1.0f;
                    if (mode == MatchMode::CensusGrad)
                    {
                        score = computeScoreA(img_1, img_2, grad_1, grad_2,
                                              r, c, rr, cc, half,
                                              desc1, desc2_buf);
                    }
                    else
                    {
                        score = computeScoreB(img_1, img_2, grad_1, grad_2,
                                              r, c, rr, cc, half);
                    }
                    if (score > maxScore)
                    {
                        maxScore = score;
                        best_rr = static_cast<float>(rr);
                        best_cc = static_cast<float>(cc);
                    }
                }
            }

            // stride 空洞填补 + 子像素加权
            float s_weight = 0.0f;
            int s_r = static_cast<int>(best_rr);
            int s_c = static_cast<int>(best_cc);
            float mx = 0.0f, my = 0.0f;
            float best_local = maxScore;
            const int w_s = 2 * coarse_stride;
            for (int ii = -w_s; ii <= w_s; ii++)
            {
                for (int jj = -w_s; jj <= w_s; jj++)
                {
                    const int rr = s_r + ii;
                    const int cc = s_c + jj;
                    if (!patchInside(img_2, rr, cc, half)) continue;

                    float score = -1.0f;
                    if (mode == MatchMode::CensusGrad)
                    {
                        score = computeScoreA(img_1, img_2, grad_1, grad_2,
                                              r, c, rr, cc, half,
                                              desc1, desc2_buf);
                    }
                    else
                    {
                        score = computeScoreB(img_1, img_2, grad_1, grad_2,
                                              r, c, rr, cc, half);
                    }
                    if (score > best_local) best_local = score;

                    double sim01 = 0.5 * (static_cast<double>(score) + 1.0);
                    sim01 = std::max(0.0, std::min(1.0, sim01));
                    double Es = std::sqrt(static_cast<double>(ii * ii + jj * jj)) / static_cast<double>(std::max(1, w_s));
                    double E  = (1.0 - sim01) + 1e-8 * Es;
                    float weight = static_cast<float>(std::exp(-30.0 * E));
                    s_weight += weight;
                    mx += weight * static_cast<float>(rr);
                    my += weight * static_cast<float>(cc);
                }
            }

            if (s_weight > 0.0f)
            {
                matchedx[pi] = mx / s_weight;
                matchedy[pi] = my / s_weight;
            }
            else
            {
                matchedx[pi] = best_rr;
                matchedy[pi] = best_cc;
            }
            C_match[pi] = best_local;
        }

        printf("[intra_CCD] match done elapsed=%.1fs\n", omp_get_wtime() - t0);
        fflush(stdout);

        FILE* fp4 = fopen(outpointxt_1, "w");
        if (!fp4)
        {
            std::cout << " --(!) Error opening output file " << std::endl;
            return -1;
        }
        for (int i = 0; i < n_pts; i++)
        {
            if (C_match[i] >= threshold)
            {
                fprintf(fp4, "%d %f %f %d %f %f %f\n",
                        CCD_id,
                        KeyPoint_x1[i], KeyPoint_y1[i],
                        CCD_id + 1,
                        matchedx[i], matchedy[i],
                        C_match[i]);
            }
        }
        fclose(fp4);
        return 0;
    }
} // anonymous namespace

namespace {

struct ParsedImagePath {
	std::string root_path;
	std::string seq_id;
	int level;
	int img_id;
};

struct GlobalControlMatch {
	float src_row;
	float src_col;
	float dst_row;
	float dst_col;
};

struct GridPointRecord {
	int bj;
	float row;
	float col;
	int imgID;
	float mrow;
	float mcol;
	float score;
};

struct DisparityStats {
	bool valid;
	float pred_dst_row;
	float pred_dst_col;
	float sigma_row;
	float sigma_col;
};

struct CandidateSearchResult {
	bool valid;
	int imgID;
	float row;
	float col;
	float score;
};

void append_oriented_controls(const std::vector<GlobalControlMatch>& input_controls,
	bool reverse_direction,
	std::vector<GlobalControlMatch>& output_controls) {
	for(int ii=0; ii<(int)input_controls.size(); ++ii){
		GlobalControlMatch ctrl;
		if(reverse_direction){
			ctrl.src_row = input_controls[ii].dst_row;
			ctrl.src_col = input_controls[ii].dst_col;
			ctrl.dst_row = input_controls[ii].src_row;
			ctrl.dst_col = input_controls[ii].src_col;
		}
		else{
			ctrl = input_controls[ii];
		}
		output_controls.push_back(ctrl);
	}
}

bool parse_image_path_info(const char* imagepath, ParsedImagePath& info) {
	const std::string path(imagepath);
	const std::string downsample_tag = "/downsample/";
	const size_t down_pos = path.rfind(downsample_tag);
	if(down_pos == std::string::npos){
		return false;
	}
	const size_t seq_sep = path.rfind('/', down_pos - 1);
	if(seq_sep == std::string::npos){
		return false;
	}
	info.root_path = path.substr(0, seq_sep);
	info.seq_id = path.substr(seq_sep + 1, down_pos - seq_sep - 1);

	const size_t level_begin = down_pos + downsample_tag.size();
	const size_t level_end = path.find('/', level_begin);
	if(level_end == std::string::npos){
		return false;
	}
	info.level = atoi(path.substr(level_begin, level_end - level_begin).c_str());

	const std::string filename = path.substr(level_end + 1);
	const std::string prefix = info.seq_id + "_RED";
	if(filename.find(prefix) != 0){
		return false;
	}
	const size_t img_begin = prefix.size();
	const size_t img_end = filename.find('.', img_begin);
	info.img_id = atoi(filename.substr(img_begin, img_end - img_begin).c_str());
	return true;
}

bool load_mosaic_coefficients_global(const std::string& root_path, const std::string& seq_id, int level, std::vector<int>& mosaic_c) {
	char mosaictxt[512];
	sprintf(mosaictxt, "%s/%s/downsample/%d/mosaic.txt", root_path.c_str(), seq_id.c_str(), level);
	FILE* fp = fopen(mosaictxt, "r");
	if(fp == NULL){
		return false;
	}

	mosaic_c.clear();
	int CCD_id, beginR, endR, beginC, endC;
	while(fscanf(fp, "%d %d %d %d %d\n", &CCD_id, &beginR, &endR, &beginC, &endC) == 5){
		mosaic_c.push_back(beginR);
		mosaic_c.push_back(endR);
		mosaic_c.push_back(beginC);
		mosaic_c.push_back(endC);
	}
	fclose(fp);
	return !mosaic_c.empty();
}

int load_global_feature_controls(const std::string& root_path,
	const std::string& control_seq,
	const std::string& other_seq,
	int level,
	const std::vector<int>& control_mosaic,
	const std::vector<int>& other_mosaic,
	std::vector<GlobalControlMatch>& controls) {
	controls.clear();
	const int control_ccd_num = (int)control_mosaic.size() / 4;
	const int other_ccd_num = (int)other_mosaic.size() / 4;
	for(int j=0; j<control_ccd_num; ++j){
		if(control_mosaic[j*4+1] <= control_mosaic[j*4+0] ||
		   control_mosaic[j*4+3] <= control_mosaic[j*4+2]){
			continue;
		}
		char matchtxt[512];
		sprintf(matchtxt, "%s/%s/downsample/%d/%s_RED%d_match.txt",
			root_path.c_str(), control_seq.c_str(), level, control_seq.c_str(), j);
		FILE* fp = fopen(matchtxt, "r");
		if(fp == NULL){
			continue;
		}

		int bj, imgID;
		float row, col, mrow, mcol, mscore;
		while(true){
			if(fscanf(fp, "%d ", &bj) != 1){
				break;
			}
			if(bj == 1){
				if(fscanf(fp, "%f %f %d %f %f %f\n", &row, &col, &imgID, &mrow, &mcol, &mscore) != 6){
					break;
				}
				if(imgID < 0 || imgID >= other_ccd_num){
					continue;
				}
				GlobalControlMatch ctrl;
				ctrl.src_row = row + control_mosaic[j*4+0];
				ctrl.src_col = col + control_mosaic[j*4+2];
				ctrl.dst_row = mrow + other_mosaic[imgID*4+0];
				ctrl.dst_col = mcol + other_mosaic[imgID*4+2];
				controls.push_back(ctrl);
			}
			else{
				if(fscanf(fp, "%f %f\n", &row, &col) != 2){
					break;
				}
			}
		}
		fclose(fp);
	}
	return (int)controls.size();
}

bool estimate_global_prediction(float src_global_row,
	float src_global_col,
	const std::vector<GlobalControlMatch>& controls,
	float* pred_global_row,
	float* pred_global_col) {
	struct LocalConstraint {
		double dist;
		double d_row;
		double d_col;
		double disp_row;
		double disp_col;
		double weight;
	};

	std::vector<LocalConstraint> local_controls;
	local_controls.reserve(controls.size());
	for(int ii=0; ii<(int)controls.size(); ++ii){
		const float ctrl_src_row = controls[ii].src_row;
		const float ctrl_src_col = controls[ii].src_col;
		const float ctrl_dst_row = controls[ii].dst_row;
		const float ctrl_dst_col = controls[ii].dst_col;

		const double d_row = double(ctrl_src_row) - double(src_global_row);
		const double d_col = double(ctrl_src_col) - double(src_global_col);
		const double dist = sqrt(d_row*d_row + d_col*d_col);
		if(dist > 1500.0){
			continue;
		}

		LocalConstraint lc;
		lc.dist = dist;
		lc.d_row = d_row;
		lc.d_col = d_col;
		lc.disp_row = double(ctrl_dst_row) - double(ctrl_src_row);
		lc.disp_col = double(ctrl_dst_col) - double(ctrl_src_col);
		lc.weight = exp(-dist / 400.0);
		local_controls.push_back(lc);
	}
	if(local_controls.size() < 3){
		return false;
	}

	std::sort(local_controls.begin(), local_controls.end(),
		[](const LocalConstraint& a, const LocalConstraint& b){ return a.dist < b.dist; });
	if(local_controls.size() > 64){
		local_controls.resize(64);
	}

	double disp_row = 0.0;
	double disp_col = 0.0;
	double sum_weight = 0.0;
	for(int ii=0; ii<(int)local_controls.size(); ++ii){
		disp_row += local_controls[ii].disp_row * local_controls[ii].weight;
		disp_col += local_controls[ii].disp_col * local_controls[ii].weight;
		sum_weight += local_controls[ii].weight;
	}
	if(sum_weight <= 0.0){
		return false;
	}

	if(local_controls.size() >= 6){
		const int n = (int)local_controls.size();
		MatrixXf A = MatrixXf::Zero(n, 6);
		VectorXf Lx = VectorXf::Zero(n);
		VectorXf Ly = VectorXf::Zero(n);
		MatrixXf P = MatrixXf::Zero(n, n);
		for(int ii=0; ii<n; ++ii){
			A(ii,0) = float(local_controls[ii].d_row * local_controls[ii].d_row);
			A(ii,1) = float(local_controls[ii].d_row * local_controls[ii].d_col);
			A(ii,2) = float(local_controls[ii].d_col * local_controls[ii].d_col);
			A(ii,3) = float(local_controls[ii].d_row);
			A(ii,4) = float(local_controls[ii].d_col);
			A(ii,5) = 1.0f;
			Lx(ii) = float(local_controls[ii].disp_row);
			Ly(ii) = float(local_controls[ii].disp_col);
			P(ii,ii) = float(std::max(local_controls[ii].weight, 1e-6));
		}

		const MatrixXf normal = A.transpose() * P * A;
		const VectorXf rhs_x = A.transpose() * P * Lx;
		const VectorXf rhs_y = A.transpose() * P * Ly;
		const VectorXf coef_x = normal.completeOrthogonalDecomposition().solve(rhs_x);
		const VectorXf coef_y = normal.completeOrthogonalDecomposition().solve(rhs_y);
		if(coef_x.allFinite() && coef_y.allFinite()){
			disp_row = coef_x(5);
			disp_col = coef_y(5);
		}
		else{
			disp_row /= sum_weight;
			disp_col /= sum_weight;
		}
	}
	else{
		disp_row /= sum_weight;
		disp_col /= sum_weight;
	}

	*pred_global_row = float(src_global_row + disp_row);
	*pred_global_col = float(src_global_col + disp_col);
	return true;
}

std::string build_ccd_image_path(const ParsedImagePath& info, int img_id) {
	char imagepath[512];
	sprintf(imagepath, "%s/%s/downsample/%d/%s_RED%d.tif",
		info.root_path.c_str(), info.seq_id.c_str(), info.level, info.seq_id.c_str(), img_id);
	return std::string(imagepath);
}

void append_controls_from_source_match_file(const ParsedImagePath& src_info,
	const std::vector<int>& src_mosaic,
	const std::vector<int>& dst_mosaic,
	int src_ccd_id,
	bool use_grid_controls,
	std::vector<GlobalControlMatch>& controls) {
	char matchtxt[512];
	sprintf(matchtxt, "%s/%s/downsample/%d/%s_RED%d_%s.txt",
		src_info.root_path.c_str(), src_info.seq_id.c_str(), src_info.level,
		src_info.seq_id.c_str(), src_ccd_id, use_grid_controls ? "grid" : "match");
	FILE* fp = fopen(matchtxt, "r");
	if(fp == NULL){
		return;
	}

	const int dst_ccd_num = (int)dst_mosaic.size() / 4;
	int bj, imgID;
	float row, col, mrow, mcol, mscore;
	while(true){
		if(fscanf(fp, "%d ", &bj) != 1){
			break;
		}
		if(bj == 1){
			if(fscanf(fp, "%f %f %d %f %f %f\n", &row, &col, &imgID, &mrow, &mcol, &mscore) != 6){
				break;
			}
			if(imgID < 0 || imgID >= dst_ccd_num){
				continue;
			}
			GlobalControlMatch ctrl;
			ctrl.src_row = row + src_mosaic[src_ccd_id*4+0];
			ctrl.src_col = col + src_mosaic[src_ccd_id*4+2];
			ctrl.dst_row = mrow + dst_mosaic[imgID*4+0];
			ctrl.dst_col = mcol + dst_mosaic[imgID*4+2];
			controls.push_back(ctrl);
		}
		else{
			if(fscanf(fp, "%f %f\n", &row, &col) != 2){
				break;
			}
		}
	}
	fclose(fp);
}

void append_controls_from_reversed_target_matches(const ParsedImagePath& dst_info,
	const std::vector<int>& src_mosaic,
	const std::vector<int>& dst_mosaic,
	const std::vector<int>& src_ccd_ids,
	std::vector<GlobalControlMatch>& controls) {
	const int dst_ccd_num = (int)dst_mosaic.size() / 4;
	const int src_ccd_num = (int)src_mosaic.size() / 4;
	for(int dst_ccd_id=0; dst_ccd_id<dst_ccd_num; ++dst_ccd_id){
		char matchtxt[512];
		sprintf(matchtxt, "%s/%s/downsample/%d/%s_RED%d_match.txt",
			dst_info.root_path.c_str(), dst_info.seq_id.c_str(), dst_info.level, dst_info.seq_id.c_str(), dst_ccd_id);
		FILE* fp = fopen(matchtxt, "r");
		if(fp == NULL){
			continue;
		}

		int bj, imgID;
		float row, col, mrow, mcol, mscore;
		while(true){
			if(fscanf(fp, "%d ", &bj) != 1){
				break;
			}
			if(bj == 1){
				if(fscanf(fp, "%f %f %d %f %f %f\n", &row, &col, &imgID, &mrow, &mcol, &mscore) != 6){
					break;
				}
				if(imgID < 0 || imgID >= src_ccd_num){
					continue;
				}
				if(std::find(src_ccd_ids.begin(), src_ccd_ids.end(), imgID) == src_ccd_ids.end()){
					continue;
				}
				GlobalControlMatch ctrl;
				ctrl.src_row = mrow + src_mosaic[imgID*4+0];
				ctrl.src_col = mcol + src_mosaic[imgID*4+2];
				ctrl.dst_row = row + dst_mosaic[dst_ccd_id*4+0];
				ctrl.dst_col = col + dst_mosaic[dst_ccd_id*4+2];
				controls.push_back(ctrl);
			}
			else{
				if(fscanf(fp, "%f %f\n", &row, &col) != 2){
					break;
				}
			}
		}
		fclose(fp);
	}
}

int load_neighbor_pair_controls(const ParsedImagePath& src_info,
	const ParsedImagePath& dst_info,
	const std::vector<int>& src_mosaic,
	const std::vector<int>& dst_mosaic,
	bool use_grid_controls,
	std::vector<GlobalControlMatch>& controls) {
	controls.clear();
	const int src_ccd_num = (int)src_mosaic.size() / 4;
	if(src_ccd_num <= 0){
		return 0;
	}

	std::vector<int> src_ccd_ids;
	for(int offset=-1; offset<=2; ++offset){
		const int ccd_id = src_info.img_id + offset;
		if(ccd_id < 0 || ccd_id >= src_ccd_num){
			continue;
		}
		if(std::find(src_ccd_ids.begin(), src_ccd_ids.end(), ccd_id) == src_ccd_ids.end()){
			src_ccd_ids.push_back(ccd_id);
		}
	}

	for(int ii=0; ii<(int)src_ccd_ids.size(); ++ii){
		append_controls_from_source_match_file(src_info, src_mosaic, dst_mosaic, src_ccd_ids[ii], use_grid_controls, controls);
	}
	if(!controls.empty()){
		return (int)controls.size();
	}

	append_controls_from_reversed_target_matches(dst_info, src_mosaic, dst_mosaic, src_ccd_ids, controls);
	return (int)controls.size();
}

bool estimate_local_disparity(float src_row,
	float src_col,
	const std::vector<GlobalControlMatch>& controls,
	DisparityStats& stats) {
	struct LocalControlSample {
		double dist;
		double disp_row;
		double disp_col;
		double weight;
	};

	stats.valid = false;
	stats.pred_dst_row = 0.0f;
	stats.pred_dst_col = 0.0f;
	stats.sigma_row = 0.0f;
	stats.sigma_col = 0.0f;

	if(controls.size() < 3){
		return false;
	}

	std::vector<LocalControlSample> samples;
	samples.reserve(controls.size());
	for(int ii=0; ii<(int)controls.size(); ++ii){
		const double dr = double(controls[ii].src_row) - double(src_row);
		const double dc = double(controls[ii].src_col) - double(src_col);
		LocalControlSample sample;
		sample.dist = sqrt(dr*dr + dc*dc);
		sample.disp_row = double(controls[ii].dst_row) - double(controls[ii].src_row);
		sample.disp_col = double(controls[ii].dst_col) - double(controls[ii].src_col);
		sample.weight = 1.0;
		samples.push_back(sample);
	}

	std::sort(samples.begin(), samples.end(),
		[](const LocalControlSample& a, const LocalControlSample& b){ return a.dist < b.dist; });

	const double neighbor_radius = 200.0;
	const int min_neighbor_count = 5;
	std::vector<LocalControlSample> local_controls;
	for(int ii=0; ii<(int)samples.size(); ++ii){
		if(samples[ii].dist <= neighbor_radius || (int)local_controls.size() < min_neighbor_count){
			local_controls.push_back(samples[ii]);
		}
		if((int)local_controls.size() >= 24){
			break;
		}
	}
	if((int)local_controls.size() < 3){
		return false;
	}

	const double sigma_space = 100.0;
	double mean_row = 0.0;
	double mean_col = 0.0;
	double sum_weight = 0.0;
	for(int ii=0; ii<(int)local_controls.size(); ++ii){
		const double dist = local_controls[ii].dist;
		local_controls[ii].weight = exp(-(dist*dist)/(2.0*sigma_space*sigma_space));
		if(local_controls[ii].weight < 1e-6){
			local_controls[ii].weight = 1e-6;
		}
		mean_row += local_controls[ii].disp_row * local_controls[ii].weight;
		mean_col += local_controls[ii].disp_col * local_controls[ii].weight;
		sum_weight += local_controls[ii].weight;
	}
	if(sum_weight <= 0.0){
		return false;
	}
	mean_row /= sum_weight;
	mean_col /= sum_weight;

	double var_row = 0.0;
	double var_col = 0.0;
	for(int ii=0; ii<(int)local_controls.size(); ++ii){
		var_row += (local_controls[ii].disp_row - mean_row) * (local_controls[ii].disp_row - mean_row) * local_controls[ii].weight;
		var_col += (local_controls[ii].disp_col - mean_col) * (local_controls[ii].disp_col - mean_col) * local_controls[ii].weight;
	}
	var_row /= sum_weight;
	var_col /= sum_weight;

	stats.valid = true;
	stats.pred_dst_row = float(double(src_row) + mean_row);
	stats.pred_dst_col = float(double(src_col) + mean_col);
	stats.sigma_row = float(std::max(sqrt(std::max(var_row, 1.0)), 8.0));
	stats.sigma_col = float(std::max(sqrt(std::max(var_col, 1.0)), 8.0));
	return true;
}

// mosaic 占位（范围外 CCD 写 0,0,0,0）视为无效，避免尝试打开不存在的 RED*.tif
static bool mosaic_ccd_valid(const std::vector<int>& mosaic, int ii) {
	if(ii < 0 || ii * 4 + 3 >= (int)mosaic.size()) return false;
	return mosaic[ii*4+1] > mosaic[ii*4+0] && mosaic[ii*4+3] > mosaic[ii*4+2];
}

void collect_candidate_ccds(float pred_row,
	float pred_col,
	int search_row,
	int search_col,
	const std::vector<int>& dst_mosaic,
	std::vector<int>& candidates) {
	candidates.clear();
	const int dst_ccd_num = (int)dst_mosaic.size() / 4;
	for(int ii=0; ii<dst_ccd_num; ++ii){
		if(!mosaic_ccd_valid(dst_mosaic, ii)) continue;
		if(pred_row >= dst_mosaic[ii*4+0] && pred_row <= dst_mosaic[ii*4+1] &&
		   pred_col >= dst_mosaic[ii*4+2] && pred_col <= dst_mosaic[ii*4+3]){
			candidates.push_back(ii);
		}
	}
	if(!candidates.empty()){
		return;
	}

	for(int ii=0; ii<dst_ccd_num; ++ii){
		if(!mosaic_ccd_valid(dst_mosaic, ii)) continue;
		if(pred_row + search_row >= dst_mosaic[ii*4+0] && pred_row - search_row <= dst_mosaic[ii*4+1] &&
		   pred_col + search_col >= dst_mosaic[ii*4+2] && pred_col - search_col <= dst_mosaic[ii*4+3]){
			candidates.push_back(ii);
		}
	}
	if(!candidates.empty()){
		return;
	}

	double best_dist = 1e30;
	int best_ccd = -1;
	for(int ii=0; ii<dst_ccd_num; ++ii){
		if(!mosaic_ccd_valid(dst_mosaic, ii)) continue;
		double dr = 0.0;
		double dc = 0.0;
		if(pred_row < dst_mosaic[ii*4+0]) dr = dst_mosaic[ii*4+0] - pred_row;
		else if(pred_row > dst_mosaic[ii*4+1]) dr = pred_row - dst_mosaic[ii*4+1];
		if(pred_col < dst_mosaic[ii*4+2]) dc = dst_mosaic[ii*4+2] - pred_col;
		else if(pred_col > dst_mosaic[ii*4+3]) dc = pred_col - dst_mosaic[ii*4+3];
		double dist = sqrt(dr*dr + dc*dc);
		if(dist < best_dist){
			best_dist = dist;
			best_ccd = ii;
		}
	}
	if(best_ccd >= 0){
		candidates.push_back(best_ccd);
	}
}

bool ensure_cached_image(const ParsedImagePath& info, int img_id, std::vector<Mat>& image_cache) {
	if(img_id < 0 || img_id >= (int)image_cache.size()){
		return false;
	}
	if(image_cache[img_id].data != NULL){
		return true;
	}
	#pragma omp critical(ensure_cached_image)
	{
		if(image_cache[img_id].data == NULL){
			const std::string imagepath = build_ccd_image_path(info, img_id);
			image_cache[img_id] = imread(imagepath, 0);
		}
	}
	return image_cache[img_id].data != NULL;
}

bool ensure_cached_image_and_grad(const ParsedImagePath& info, int img_id,
	std::vector<Mat>& image_cache, std::vector<Mat>& grad_cache) {
	if(!ensure_cached_image(info, img_id, image_cache)){
		return false;
	}
	if(img_id < 0 || img_id >= (int)grad_cache.size()){
		return false;
	}
	if(grad_cache[img_id].data == NULL){
		#pragma omp critical(ensure_cached_grad)
		{
			if(grad_cache[img_id].data == NULL){
				computeGradientMag(image_cache[img_id], grad_cache[img_id]);
			}
		}
	}
	return grad_cache[img_id].data != NULL;
}

bool compute_patch_ncc(const Mat& img_1,
	const Mat& img_2,
	int src_row,
	int src_col,
	int dst_row,
	int dst_col,
	int patch_size,
	float* cc_value) {
	const int half_size = patch_size / 2;
	const float N = float(patch_size * patch_size);
	float s12 = 0.0f;
	float s11 = 0.0f;
	float s22 = 0.0f;
	float s1 = 0.0f;
	float s2 = 0.0f;
	int sum1 = 0;
	int sum2 = 0;

	for(int kk=-half_size; kk<=half_size; ++kk){
		for(int mm=-half_size; mm<=half_size; ++mm){
			const int r1 = src_row + kk;
			const int c1 = src_col + mm;
			const int r2 = dst_row + kk;
			const int c2 = dst_col + mm;
			const int g1 = img_1.data[r1*img_1.cols + c1];
			const int g2 = img_2.data[r2*img_2.cols + c2];
			sum1 += g1;
			sum2 += g2;
			s12 += float(g1 * g2);
			s22 += float(g2 * g2);
			s11 += float(g1 * g1);
			s1 += float(g1);
			s2 += float(g2);
		}
	}

	if(sum1 == 0 || sum2 == 0){
		return false;
	}

	s12 = s12 / float(sum1) / float(sum2);
	s22 = s22 / float(sum2) / float(sum2);
	s11 = s11 / float(sum1) / float(sum1);
	s1 = s1 / float(sum1);
	s2 = s2 / float(sum2);

	const float denom_left = s11 - s1*s1/N;
	const float denom_right = s22 - s2*s2/N;
	if(denom_left <= 1e-6f || denom_right <= 1e-6f){
		return false;
	}

	const float cc = (s12 - s1*s2/N) / sqrt(denom_left * denom_right);
	if(!std::isfinite(cc)){
		return false;
	}
	*cc_value = cc;
	return true;
}

	bool search_best_match_in_candidate(const Mat& img_1,
	const Mat& img_2,
	const Mat& grad_1,
	const Mat& grad_2,
	float src_row_f,
	float src_col_f,
	int center_row,
	int center_col,
	int search_row,
	int search_col,
	int base_window_size,
	CandidateSearchResult& result) {
	const int src_row = int(src_row_f + 0.5f);
	const int src_col = int(src_col_f + 0.5f);
	const int stride = 2;
	const int patch_size = int(double(base_window_size) * sqrt(double(stride)));
	const int half_patch = patch_size / 2;

	if(src_row < half_patch + 1 || src_col < half_patch + 1 ||
	   src_row > img_1.rows - half_patch - 1 || src_col > img_1.cols - half_patch - 1){
		return false;
	}
	if(grad_1.empty() || grad_2.empty() ||
	   grad_1.size() != img_1.size() || grad_2.size() != img_2.size()){
		return false;
	}

	// 与 intra_CCD_match_B 一致：ScoreB = 0.9*ZNCC(gray) + 0.1*ZNCC(grad)
	float best_score = -2.0f;
	int best_row = -1;
	int best_col = -1;
	for(int dr=-search_row; dr<=search_row; dr+=stride){
		for(int dc=-search_col; dc<=search_col; dc+=stride){
			const int dst_row = center_row + dr;
			const int dst_col = center_col + dc;
			if(dst_row < half_patch + 1 || dst_col < half_patch + 1 ||
			   dst_row > img_2.rows - half_patch - 1 || dst_col > img_2.cols - half_patch - 1){
				continue;
			}
			const float score = computeScoreB(img_1, img_2, grad_1, grad_2,
				src_row, src_col, dst_row, dst_col, half_patch);
			if(score > best_score){
				best_score = score;
				best_row = dst_row;
				best_col = dst_col;
			}
		}
	}
	if(best_row < 0 || best_col < 0){
		return false;
	}

	const int refine_patch = 19;
	const int refine_half = refine_patch / 2;
	for(int dr=-4*stride; dr<=4*stride; ++dr){
		for(int dc=-4*stride; dc<=4*stride; ++dc){
			const int dst_row = best_row + dr;
			const int dst_col = best_col + dc;
			if(dst_row < refine_half + 1 || dst_col < refine_half + 1 ||
			   dst_row > img_2.rows - refine_half - 1 || dst_col > img_2.cols - refine_half - 1){
				continue;
			}
			const float score = computeScoreB(img_1, img_2, grad_1, grad_2,
				src_row, src_col, dst_row, dst_col, refine_half);
			if(score > best_score){
				best_score = score;
				best_row = dst_row;
				best_col = dst_col;
			}
		}
	}

	result.valid = true;
	result.row = float(best_row);
	result.col = float(best_col);
		result.score = best_score;
		return true;
	}

	void setScoreBOptions(bool use_robust_scoreb, bool use_affine_patch_score)
	{
		g_use_robust_scoreb = use_robust_scoreb;
		g_use_affine_patch_score = use_affine_patch_score;
	}

	}

void ImageMatch::SetScoreBOptions(bool use_robust_scoreb, bool use_affine_patch_score)
{
	setScoreBOptions(use_robust_scoreb, use_affine_patch_score);
}

int ImageMatch::Forstner(uchar* pImg,int rows,int cols,int base, std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y)
{
	//分配内存存储每点的两个方向梯度（Robert梯度）
	float *gu = new float[rows*cols];
	float *gv = new float[rows*cols];
	memset(gu, 0, sizeof(float)*rows*cols);
	memset(gv, 0, sizeof(float)*rows*cols);

	//逐点计算梯度
	for(int ii=1;ii<rows-1;ii++){
		for(int jj=1;jj<cols-1;jj++){
			int m = ii*cols+jj;
			gu[ii*cols+jj]=pImg[(ii+1)*cols+(jj+1)]-pImg[ii*cols+jj];
			gv[ii*cols+jj]=pImg[ii*cols+(jj+1)]-pImg[(ii+1)*cols+jj];
		} 
	}

	//分配内存并计算存储各点灰度协方差矩阵参数（以5*5窗口为例）
	float *Sguu = new float[rows*cols];
	float *Sgvv = new float[rows*cols];
	float *Sguv = new float[rows*cols];
	memset(Sguu, 0, sizeof(float)*rows*cols);
	memset(Sgvv, 0, sizeof(float)*rows*cols);
	memset(Sguv, 0, sizeof(float)*rows*cols);


	int dim=7;

	//最后一行列无梯度，所以-dim/2-1。
	for(int ii=dim/2;ii<rows-dim/2-1;ii++){
		for(int jj=dim/2;jj<cols-dim/2-1;jj++){
			int m = ii*cols+jj;

			//5*5窗口内计算（ii，jj）点灰度协方差参数
			Sguu[m]=Sgvv[m]=Sguv[m]=0;
			for(int kk=-dim/2;kk<=dim/2;kk++){
				for(int ll=-dim/2;ll<=dim/2;ll++){
					Sguu[m] += (gu[(ii+kk)*cols+(jj+ll)])*(gu[(ii+kk)*cols+(jj+ll)]);
					Sgvv[m] += (gv[(ii+kk)*cols+(jj+ll)])*(gv[(ii+kk)*cols+(jj+ll)]);
					Sguv[m] += gu[(ii+kk)*cols+(jj+ll)]*gv[(ii+kk)*cols+(jj+ll)];
					//Sguu[m] += guu(pImg,ii+kk,jj+ll,cols)*guu(pImg,ii+kk,jj+ll,cols);
					//Sgvv[m] += gvv(pImg,ii+kk,jj+ll,cols)*gvv(pImg,ii+kk,jj+ll,cols);
					//Sguv[m] += guu(pImg,ii+kk,jj+ll,cols)*gvv(pImg,ii+kk,jj+ll,cols);;
				}
			}
		} 
	}

	//分配存储空间并计算兴趣值q和w
	float *q = new float[rows*cols];
	float *w = new float[rows*cols];
	memset(q, 0, rows*cols);
	memset(w, 0, rows*cols);

	int count=0;                         //用于计数
	float weva=0;                       //平均权值
	for(int ii=2;ii<rows-3;ii++){
		for(int jj=2;jj<cols-3;jj++){
			int m = ii*cols+jj;
			if(Sguu[m]+Sgvv[m]!=0)
			{
				//Sguu[m]=0.0000001;
				//规避分母为0
				w[m] = (Sguu[m]*Sgvv[m]-Sguv[m]*Sguv[m])/(Sguu[m]+Sgvv[m]);
				q[m] = 4*(Sguu[m]*Sgvv[m]-Sguv[m]*Sguv[m])/((Sguu[m]+Sgvv[m])*(Sguu[m]+Sgvv[m]));
				weva += w[m];
				count=count+1;
			} 
		} 
	}

	weva=weva/count;
	//printf("平均权值：%lf 待选点总数：%d\n",weva,count);

	//给定阈值Tq,Tw选取待选点
	//一般Tq=0.5~0.75
	//Tw=fw(w为权平均值,f=0.5~1.5)或Tw=5*wc(wc为权中值）
	float Tq=0.93;
	float Tw=2*weva;

	Mat image1;
	image1.create(rows, cols, CV_8UC1);
	unsigned char *pImg1 = image1.data;
	memset(pImg1, 255, rows*cols);

	for(int ii=2;ii<rows-3;ii++){
		for(int jj=2;jj<cols-3;jj++){
			int m = ii*cols+jj;
			if(w[m]>Tw && q[m]>Tq){
				int n=w[m];

				//9*9窗口内搜索区域最大值(仅以w为参考)
				int w_s = 7;
				if(ii>=w_s && ii<rows-w_s && jj>=w_s && jj<cols-w_s){
					for(int a=-w_s;a<=w_s;a++){
						for(int b=-w_s;b<=w_s;b++){
							if(n<w[(ii+a)*cols+jj+b]){
								n=w[(ii+a)*cols+jj+b];
							}
						}
					}
					//判断是否区域极值
					if(w[ii*cols+jj]>=n){
						KeyPoint_x.push_back(base+ii);
						KeyPoint_y.push_back(jj);
						//pImg[ii*cols+jj]=0;
						//circle(image2,cv::Point(jj,ii),3,CV_RGB(255,0,0),0.5,8,0); 
					}
				}
			} 
		}	
	}
	//delete []gu;
	//delete []gv;
	delete []pImg;
	delete []Sguu;
	delete []Sgvv;
	delete []Sguv;
	delete []q;
	delete []w;
	//imshow("fig2",image2);
	//waitKey(0);
	return 0;
}
void ImageMatch::GaussianKernel(float sigma,int dim, float* kernel){
	float m= 1.0/(2.0 * CV_PI * sigma * sigma);  

	int c=dim/2;
	float count=0;
	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			float p=m*exp(-((i-c)*(i-c)+(j-c)*(j-c))/(2.0 * sigma * sigma)); 
			kernel[i*dim+j] = p;
			count += p;
		}
	}
	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			kernel[i*dim+j] = kernel[i*dim+j]/count;
		}
	}
}
void ImageMatch::Gaussian_Juanji(float* InArray, float* OutArray, int rows, int cols, int dim, float sigma){
	float* kernel = new float[dim*dim];
	GaussianKernel(sigma,dim,kernel);
	int c=dim/2;
	for(int i=c;i<rows-c;i++){
		for(int j=c;j<cols-c;j++){
			float m=0;
			for(int k=0;k<dim;k++){
				for(int l=0;l<dim;l++){
					m += kernel[k*dim+l]*float(InArray[(i-c+k)*cols+j-c+l]);
				}
			}
			OutArray[i*cols+j]=m;
		}
	}
}
int ImageMatch::HarrisCorner(uchar* pImg,int rows,int cols,int Juanji_dim,float Juanji_sigma,int Localmax_dim,float threshold,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y){
	int channels = 1;

	//定义数组存储梯度
	float *Grx  = new float[rows*cols];
	float *Gry  = new float[rows*cols];
	float *Grxx = new float[rows*cols];
	float *Gryy = new float[rows*cols];
	float *Grxy = new float[rows*cols];
	memset(Grx,  0, rows*cols*sizeof(float));
	memset(Gry,  0, rows*cols*sizeof(float));
	memset(Grxx, 0, rows*cols*sizeof(float));
	memset(Gryy, 0, rows*cols*sizeof(float));
	memset(Grxy, 0, rows*cols*sizeof(float));

	//遍历影像计算各点x，y方向的梯度
	for(int ii=1;ii<rows-1;ii++){
		for(int jj=1;jj<cols-1;jj++){
			Grx[ii*cols+jj]  = pImg[ii*cols+jj+1]-pImg[ii*cols+jj-1];
			Gry[ii*cols+jj]  = pImg[(ii+1)*cols+jj]-pImg[(ii-1)*cols+jj];
			Grxx[ii*cols+jj] = Grx[ii*cols+jj]*Grx[ii*cols+jj];
			Gryy[ii*cols+jj] = Gry[ii*cols+jj]*Gry[ii*cols+jj];
			Grxy[ii*cols+jj] = Grx[ii*cols+jj]*Gry[ii*cols+jj];
		} 
	}

	//存储M矩阵中元素{A,B;B,C}
	float *Grxx1 = new float[rows*cols];      //A
	float *Gryy1 = new float[rows*cols];      //C
	float *Grxy1 = new float[rows*cols];      //B
	memset(Grxx1, 0, rows*cols*sizeof(float));
	memset(Gryy1, 0, rows*cols*sizeof(float));
	memset(Grxy1, 0, rows*cols*sizeof(float));

	//高斯滤波处理梯度
	Gaussian_Juanji(Grxx,Grxx1,rows,cols,Juanji_dim,Juanji_sigma);
	Gaussian_Juanji(Gryy,Gryy1,rows,cols,Juanji_dim,Juanji_sigma);
	Gaussian_Juanji(Grxy,Grxy1,rows,cols,Juanji_dim,Juanji_sigma);

	delete []Grx;delete []Gry;delete []Grxx;delete []Gryy;delete []Grxy;

	//定义数组存储每点角点响应值
	float *I = new float[rows*cols];
	memset(I,0,sizeof(float)*rows*cols);           

	//计算角点响应值
	for(int ii=0;ii<rows;ii++){
		for(int jj=0;jj<cols;jj++){
			I[ii*cols+jj] = pow(Grxx1[ii*cols+jj]*Gryy1[ii*cols+jj]-Grxy1[ii*cols+jj]*Grxy1[ii*cols+jj],2)-0.04*pow(Grxx1[ii*cols+jj]+Gryy1[ii*cols+jj],2);
		}
	}

	delete []Grxx1;delete []Gryy1;delete []Grxy1;

	//局部区域取极大值（取9*9窗口）
	//int dim=9;
	for(int ii=Localmax_dim/2;ii<rows-Localmax_dim/2;ii+=Localmax_dim){
		for(int jj=Localmax_dim/2;jj<cols-Localmax_dim/2;jj+=Localmax_dim){
			float localmax=I[ii*cols+jj];
			int xx,yy;
			xx=ii;yy=jj;
			//搜索区域最大值
			for(int a=-Localmax_dim/2;a<=Localmax_dim/2;a++){
				for(int b=-Localmax_dim/2;b<=Localmax_dim/2;b++){
					if(I[(ii+a)*cols+jj+b]>localmax && I[(ii+a)*cols+jj+b]>=threshold){
						localmax=I[(ii+a)*cols+jj+b];
						xx=ii+a;
						yy=jj+b;
					}
				}
			}
			KeyPoint_x.push_back(xx);
			KeyPoint_y.push_back(yy);
		}	
	}
	delete []I;
	return 0;
}
int ImageMatch::Feature_Detection(char* imagepath,int ch,int thresh,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y){
	if(ch==1){//Sift
		Mat img_1 = imread(imagepath, 0);
		if( !img_1.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}

		cv::Ptr<cv::SIFT> Detector = cv::SIFT::create();

		std::vector<cv::KeyPoint> keypoints_1;
		Detector->detect(img_1, keypoints_1);

		for(int i=0;i<keypoints_1.size();i++){
			KeyPoint_x.push_back(static_cast<int>(keypoints_1[i].pt.x));
			KeyPoint_y.push_back(static_cast<int>(keypoints_1[i].pt.y));
		}
	}
	else if(ch==2){//Surf
		Mat image = imread(imagepath, 0);
		if( !image.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			Mat img_1;
			img_1.create(rows/size,cols,CV_8UC1);
			uchar* pImg1 = img_1.data;
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;

			int minHessian = 300;
			cv::Ptr<cv::xfeatures2d::SURF> Detector = cv::xfeatures2d::SURF::create(minHessian);

			std::vector<KeyPoint> keypoints_1;
			Detector->detect( img_1, keypoints_1 );
			for(int i=0;i<keypoints_1.size();i++){
				KeyPoint_x.push_back(keypoints_1[i].pt.y+base);
				KeyPoint_y.push_back(keypoints_1[i].pt.x);
			}
		}
	}
	else if(ch==3){//opencvHarris
		Mat image = imread(imagepath, 0);
		if( !image.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			Mat img_1;
			img_1.create(rows/size,cols,CV_8UC1);
			uchar* pImg1 = img_1.data;
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;


			//进行Harris角点检测找出角点
			Mat cornerStrength;
			cornerHarris(img_1, cornerStrength, 2, 3, 0.01);

			//对灰度图进行阈值操作，得到二值图并显示
			Mat harrisCorner;
			threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);

			for(int i=0;i<harrisCorner.rows;i++){
				for(int j=0;j<harrisCorner.cols;j++){
					if(harrisCorner.data[i*harrisCorner.cols+j]>0){
						KeyPoint_x.push_back(i);
						KeyPoint_y.push_back(j);
					}
				}
			}
		}
	}
	else if(ch==4){
		//读取影像并获取行列信息
		Mat image = imread(imagepath,0); 
		//Mat image2 = imread(imageName,1); 
		if(image.empty()) 
		{
			printf( "Could not open or find the image. \n");
			return -1;
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			uchar* pImg1 = new uchar[cols*rows/size*sizeof(uchar)];
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;
			Forstner(pImg1,rows/size,cols,base,KeyPoint_x,KeyPoint_y);
			delete []pImg1;
		}
	}
	else if(ch==5){//Surf
		Mat image = imread(imagepath, 0);
		if( !image.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			Mat img_1;
			img_1.create(rows/size,cols,CV_8UC1);
			uchar* pImg1 = img_1.data;
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;

			int minHessian = 600;
			cv::Ptr<cv::xfeatures2d::SURF> Detector = cv::xfeatures2d::SURF::create(minHessian);

			std::vector<cv::KeyPoint> keypoints_1;
			Detector->detect(img_1, keypoints_1);
			for(int i=0;i<keypoints_1.size();i++){
				KeyPoint_x.push_back(static_cast<int>(keypoints_1[i].pt.y+base));
				KeyPoint_y.push_back(static_cast<int>(keypoints_1[i].pt.x));
			}
		}
	}
	else if(ch==6){//myHarris
		//读取影像并获取行列信息
		Mat image = imread(imagepath,0); 
		//Mat image2 = imread(imageName,1); 
		if(image.empty()) 
		{
			printf( "Could not open or find the image. \n");
			return -1;
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=40;
		for(int i=0;i<size;i++){
			uchar* pImg1 = new uchar[cols*rows/size*sizeof(uchar)];
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			std::vector<int> KeyPoint_xx,KeyPoint_yy;
			int base=rows/size*i;
			HarrisCorner(pImg1,rows/size,cols,9,0.5,thresh,150,KeyPoint_xx,KeyPoint_yy);

			if(KeyPoint_xx.size()>0){
				for(int k=0;k<KeyPoint_xx.size();k++){
					KeyPoint_x.push_back(KeyPoint_xx[k]+base);
					KeyPoint_y.push_back(KeyPoint_yy[k]);
				}
			}

			vector<int>().swap(KeyPoint_xx);
			vector<int>().swap(KeyPoint_yy);
			delete []pImg1;
		}
	}
	else
	{
		printf("输入的特征点提取方式无效！");
		return 0;
	}
	return 0;
}


void ImageMatch::RANSAC_fs2(std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,float sigma,int iter,int* match, float* fs_c){
	// 兼容旧接口：转调轻量实现（按值传参仍保留，避免改动所有调用点）
	RANSAC_fs2_new(KeyPoint_x1, KeyPoint_y1, KeyPoint_x2, KeyPoint_y2, sigma, iter, match, fs_c);
}

// 轻量仿射 RANSAC：每轮只解 3 点模型 + O(N) 残差，不做 2N×6 大矩阵
void ImageMatch::RANSAC_fs2_new(const std::vector<int>& KeyPoint_x1, const std::vector<int>& KeyPoint_y1,
	const std::vector<int>& KeyPoint_x2, const std::vector<int>& KeyPoint_y2,
	float sigma, int iter, int* match, float* fs_c)
{
	const int number = static_cast<int>(KeyPoint_x1.size());
	if (number < 3) {
		printf("匹配点对数小于3！\n");
		return;
	}
	if (iter < 1) iter = 1;
	if (match == nullptr || fs_c == nullptr) {
		printf("RANSAC_fs2_new: null match/fs_c\n");
		return;
	}

	const float sigma2 = sigma * sigma;
	float best_fs[6] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
	int best_count = 0;
	std::vector<char> best_mask(static_cast<size_t>(number), 0);
	std::vector<char> cur_mask(static_cast<size_t>(number), 0);

	const int progress_every = std::max(1, iter / 20);
	printf("[RANSAC_fs2_new] points=%d iter=%d sigma=%.3f\n", number, iter, sigma);
	fflush(stdout);
	srand(static_cast<unsigned>(time(nullptr)));

	auto solve_affine_3 = [](float x0, float y0, float x1, float y1, float x2, float y2,
		float u0, float v0, float u1, float v1, float u2, float v2, float out_fs[6]) -> bool
	{
		Matrix3f A;
		A << x0, y0, 1.f,
		     x1, y1, 1.f,
		     x2, y2, 1.f;
		Eigen::FullPivLU<Matrix3f> lu(A);
		if (lu.rank() < 3) return false;
		Vector3f Lx(u0, u1, u2);
		Vector3f Ly(v0, v1, v2);
		const Vector3f ax = lu.solve(Lx);
		const Vector3f ay = lu.solve(Ly);
		out_fs[0] = ax(0); out_fs[1] = ax(1); out_fs[2] = ax(2);
		out_fs[3] = ay(0); out_fs[4] = ay(1); out_fs[5] = ay(2);
		return true;
	};

	for (int it = 0; it < iter; ++it) {
		int i0 = rand() % number;
		int i1 = rand() % number;
		int i2 = rand() % number;
		int guard = 0;
		while ((i1 == i0 || i2 == i0 || i2 == i1) && guard++ < 32) {
			i1 = rand() % number;
			i2 = rand() % number;
		}
		if (i0 == i1 || i0 == i2 || i1 == i2) continue;

		float trial_fs[6];
		if (!solve_affine_3(
			static_cast<float>(KeyPoint_x1[i0]), static_cast<float>(KeyPoint_y1[i0]),
			static_cast<float>(KeyPoint_x1[i1]), static_cast<float>(KeyPoint_y1[i1]),
			static_cast<float>(KeyPoint_x1[i2]), static_cast<float>(KeyPoint_y1[i2]),
			static_cast<float>(KeyPoint_x2[i0]), static_cast<float>(KeyPoint_y2[i0]),
			static_cast<float>(KeyPoint_x2[i1]), static_cast<float>(KeyPoint_y2[i1]),
			static_cast<float>(KeyPoint_x2[i2]), static_cast<float>(KeyPoint_y2[i2]),
			trial_fs)) {
			continue;
		}

		int total = 0;
		for (int j = 0; j < number; ++j) {
			const float x = static_cast<float>(KeyPoint_x1[j]);
			const float y = static_cast<float>(KeyPoint_y1[j]);
			const float px = trial_fs[0] * x + trial_fs[1] * y + trial_fs[2];
			const float py = trial_fs[3] * x + trial_fs[4] * y + trial_fs[5];
			const float dx = px - static_cast<float>(KeyPoint_x2[j]);
			const float dy = py - static_cast<float>(KeyPoint_y2[j]);
			const bool inl = (dx * dx + dy * dy) < sigma2;
			cur_mask[static_cast<size_t>(j)] = inl ? 1 : 0;
			if (inl) ++total;
		}

		if (total > best_count) {
			best_count = total;
			best_mask = cur_mask;
			for (int k = 0; k < 6; ++k) best_fs[k] = trial_fs[k];
		}

		if ((it + 1) % progress_every == 0 || it + 1 == iter) {
			printf("[RANSAC_fs2_new] %d/%d (%.0f%%) best_inliers=%d/%d\n",
				it + 1, iter, 100.0 * (it + 1) / iter, best_count, number);
			fflush(stdout);
		}
	}

	if (best_count < 3) {
		printf("[RANSAC_fs2_new] 内点不足 (%d)，保留输入 fs_c\n", best_count);
		for (int j = 0; j < number; ++j) match[j] = -1;
		return;
	}

	// 内点最小二乘：累加 6×6 法方程，避免 2N×6 大矩阵
	MatrixXf AtA = MatrixXf::Zero(6, 6);
	VectorXf AtL = VectorXf::Zero(6);
	for (int j = 0; j < number; ++j) {
		if (!best_mask[static_cast<size_t>(j)]) {
			match[j] = -1;
			continue;
		}
		match[j] = j;
		const float x = static_cast<float>(KeyPoint_x1[j]);
		const float y = static_cast<float>(KeyPoint_y1[j]);
		const float u = static_cast<float>(KeyPoint_x2[j]);
		const float v = static_cast<float>(KeyPoint_y2[j]);
		// 行：[x y 1 0 0 0] -> u ; [0 0 0 x y 1] -> v
		AtA(0, 0) += x * x; AtA(0, 1) += x * y; AtA(0, 2) += x;
		AtA(1, 0) += x * y; AtA(1, 1) += y * y; AtA(1, 2) += y;
		AtA(2, 0) += x;     AtA(2, 1) += y;     AtA(2, 2) += 1.f;
		AtA(3, 3) += x * x; AtA(3, 4) += x * y; AtA(3, 5) += x;
		AtA(4, 3) += x * y; AtA(4, 4) += y * y; AtA(4, 5) += y;
		AtA(5, 3) += x;     AtA(5, 4) += y;     AtA(5, 5) += 1.f;
		AtL(0) += x * u; AtL(1) += y * u; AtL(2) += u;
		AtL(3) += x * v; AtL(4) += y * v; AtL(5) += v;
	}

	Eigen::FullPivLU<MatrixXf> lu(AtA);
	if (lu.isInvertible()) {
		const VectorXf x_ = lu.solve(AtL);
		for (int k = 0; k < 6; ++k) fs_c[k] = x_(k);
	} else {
		for (int k = 0; k < 6; ++k) fs_c[k] = best_fs[k];
	}

	printf("[RANSAC_fs2_new] done inliers=%d/%d fs=[%.4f %.4f %.4f %.4f %.4f %.4f]\n",
		best_count, number, fs_c[0], fs_c[1], fs_c[2], fs_c[3], fs_c[4], fs_c[5]);
	fflush(stdout);
}
void ImageMatch::RANSAC_plane(std::vector<int> matchpoint,float sigma,int iter, float* plane_c){
	//仿射变换系数求解
	int number = matchpoint.size()/3;  //总点数
	//int iter = 100;//number*20;

	if(number<3){
		printf("匹配点对数小于3！\n");
		return;
	}
	int pretotal=0;     //符合拟合模型的数据的个数
	VectorXf best(3);
	VectorXf mask0(number);
	VectorXf mask(number);
	for(int i=0;i<iter;i++){
		//随机选择三组点
		int *loc = new int[3];
		srand((unsigned)time(NULL));
		while(1){
			loc[0]=rand()%number;
			if(loc[0]>=0 && loc[0]<number) break;
		}
		while(loc[0]>=0 && loc[0]<number)   
		{       
			loc[1]=rand()%number;        
			if(loc[1]!=loc[0] && loc[1]>=0 && loc[1]<number) break;    
		}   
		while(loc[0]>=0 && loc[0]<number && loc[1]>=0 && loc[1]<number)
		{       
			loc[2]=rand()%number;        
			if((loc[2]!=loc[1]) && (loc[2]!=loc[0]) && loc[2]>=0 && loc[2]<number) break;  
		}

		float* sample = new float[3*3];
		for(int i=0;i<3;i++){
			sample[i*3+0]=matchpoint[3*loc[i]+0];
			sample[i*3+1]=matchpoint[3*loc[i]+1];
			sample[i*3+2]=matchpoint[3*loc[i]+2];
		}

		//拟合模型：Z=AX+BY+C V=Ax-L
		float *A = new float[3*3];
		float *L = new float[3*1];
		memset(A,0,sizeof(float)*9);
		memset(L,0,sizeof(float)*3);
		for(int j=0;j<3;j++){
			A[3*j+0]=sample[3*j+0];
			A[3*j+1]=sample[3*j+1];
			A[3*j+2]=1;

			L[j]=sample[3*j+2];
		}
		delete []sample;

		MatrixXf A_ = (Map<MatrixXf>(A,3,3)).transpose(); //列优先
		MatrixXf L_ = Map<MatrixXf>(L,3,1);
		VectorXf x = (A_.transpose()*A_).inverse()*A_.transpose()*L_;
		//cout << x << endl;
		delete []A;delete []L;

		MatrixXf AA=MatrixXf::Zero(number,3);
		MatrixXf LL=MatrixXf::Zero(number,1);

		for(int j=0;j<number;j++){
			VectorXf temp1(3);
			temp1 << matchpoint[j*3+0],matchpoint[j*3+1],1;
			AA.row(j)=temp1;

			LL(j)=matchpoint[j*3+2];

		}

		VectorXf v;
		v = AA*x-LL;

		VectorXf V(number);
		for(int j=0;j<number;j++){
			V(j)=sqrt(v(j)*v(j));  //求每个数据到拟合关系的残差
			//printf("%f\n",V(j));
		}

		int total=0;
		for(int i=0;i<number;i++){
			if(V(i)<sigma){
				total++;
				mask0(i)=1;
			}
			else{
				mask0(i)=0;
			}
		}

		if (total>pretotal){           //找到符合拟合数据最多的拟合关系
			pretotal=total;
			best=x;          //找到最好的拟合[a0,a1,a2,b0,b1,b2]
			mask=mask0;
		}
	}

	//找到符合最佳拟合的数据
	int count0=0;
	int count=0;
	std::vector<int> Localmatch;
	for(int i=0;i<number;i++){
		if(mask(i)==0){
			count0++;
		}
		else{
			Localmatch.push_back(matchpoint[3*i+0]);Localmatch.push_back(matchpoint[3*i+1]);
			Localmatch.push_back(matchpoint[3*i+2]);
			count=count+1;
		}
	}
	//更新fs_c
	int n=pretotal;
	float *A = new float[n*3];
	float *L = new float[n*1];
	memset(A,0,sizeof(float)*n*3);
	memset(L,0,sizeof(float)*n*1);
	for(int j=0;j<n;j++){
		A[3*j+0]=Localmatch[3*j+0];
		A[3*j+1]=Localmatch[3*j+1];
		A[3*j+2]=1;

		L[j]=Localmatch[3*j+2];
	}

	MatrixXf A_ = (Map<MatrixXf>(A,3,n)).transpose();
	MatrixXf L_ = Map<MatrixXf>(L,n,1);
	VectorXf x_ = (A_.transpose()*A_).inverse()*A_.transpose()*L_;

	for(int i=0;i<3;i++){
		plane_c[i]=x_(i);
	}
	cout << x_ << endl;
}
void ImageMatch::drawMatch(Mat imgL0,Mat imgR0,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf){
	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	std::cout<< "imgL rows: " << imgL.rows << " cols: " << imgL.cols << std::endl;

	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	std::cout<< "img rows: " << img.rows << " cols: " << img.cols << std::endl;

	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		if(Match[i]!=-1){
			int r=int(float(rand())/RAND_MAX+0.5)*255;
			int g=int(float(rand())/RAND_MAX+0.5)*255;
			int b=int(float(rand())/RAND_MAX+0.5)*255;
			//printf("%d %d\n",kpL[i].row,kpL[i].col);
			circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),3,CV_RGB(r,g,b),2,8,0);
			// std::cout<< "KeyPoint_y2[Match[i]]: " << KeyPoint_y2[Match[i]] << " KeyPoint_x2[Match[i]]: " << KeyPoint_x2[Match[i]] << std::endl;
			circle(img,cv::Point(KeyPoint_y2[Match[i]]/sf+imgL.cols,KeyPoint_x2[Match[i]]/sf),3,CV_RGB(r,g,b),2,8,0);
			// std::cout<< "KeyPoint_y1[i]: " << KeyPoint_y1[i] << " KeyPoint_x1[i]: " << KeyPoint_x1[i] << std::endl;
			line(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cv::Point(KeyPoint_y2[Match[i]]/sf+imgL.cols,KeyPoint_x2[Match[i]]/sf),CV_RGB(r,g,b),2,8,0);
			//line(img,cv::Point(0,0),cv::Point(200,200),CV_RGB(0,0,255),0.5,8,0);
			// std::cout<< "drawMatch: " << i << " KeyPoint_y1[i]: " << KeyPoint_y1[i] << " KeyPoint_x1[i]: " << KeyPoint_x1[i] 
			// 	<< " KeyPoint_y2[Match[i]]: " << KeyPoint_y2[Match[i]] << " KeyPoint_x2[Match[i]]: " << KeyPoint_x2[Match[i]] << std::endl;
		}
	}
	std::cout<< "drawMatch finished!" << std::endl;

	//imshow("Match",img);
	// 根据当前时间生成文件名后缀
	time_t now = time(0);
	struct tm* tstruct = localtime(&now);
	char filename[128];
	strftime(filename, sizeof(filename), "../out/match_%Y%m%d_%H%M%S.jpg", tstruct);
	imwrite(filename, img);
	//waitKey(0);
}
void ImageMatch::drawMatch1(Mat imgL0,Mat imgR0,std::vector<int> matched,int sf){
	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = matched.size()/4;
	for(int i=0;i<P_numL;i++){
		int row1= matched[4*i];
		int col1 = matched[4*i+1];
		int row2 = matched[4*i+2];
		int col2 = matched[4*i+3];
		int r=int(float(rand())/RAND_MAX+0.5)*255;
		int g=int(float(rand())/RAND_MAX+0.5)*255;
		int b=int(float(rand())/RAND_MAX+0.5)*255;
		//printf("%d %d\n",kpL[i].row,kpL[i].col);
		circle(img,cv::Point(col1/sf,row1/sf),3,CV_RGB(r,g,b),0.5,8,0);
		circle(img,cv::Point(col2/sf+imgL.cols,row2/sf),3,CV_RGB(r,g,b),0.5,8,0);
		line(img,cv::Point(col1/sf,row1/sf),cv::Point(col2/sf+imgL.cols,row2/sf),CV_RGB(r,g,b),0.5,8,0);
		//line(img,cv::Point(0,0),cv::Point(200,200),CV_RGB(0,0,255),0.5,8,0);

	}

	//imshow("Match",img);
	imwrite("../out/match.jpg",img);
	//waitKey(0);
}
void ImageMatch::drawMatch2(char* imagepath1,char* imagepath2, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int sf){
	Mat imgL0 = imread(imagepath1, 0);
	Mat imgR0 = imread(imagepath2, 0);
	if((!imgL0.data)||(!imgR0.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		if(i%2==0){
			int r=int(float(rand())/RAND_MAX+0.5)*255;
			int g=int(float(rand())/RAND_MAX+0.5)*255;
			int b=int(float(rand())/RAND_MAX+0.5)*255;
			//printf("%d %d\n",kpL[i].row,kpL[i].col);
			circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(r,g,b),0.5,8,0);
			circle(img,cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(r,g,b),0.5,8,0);
			line(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),CV_RGB(r,g,b),2,8,0);
		}
	}

	//imshow("Match",img);
	imwrite("../out/match.tif",img);
	//waitKey(0);
}
namespace {
	// 与 draw_fenfu_match0_inliers.py 默认一致
	constexpr int kFenfuMaxDraw = 8000;
	constexpr int kFenfuRadius = 3;
	constexpr int kFenfuSeed = 0;

	std::vector<size_t> fenfu_sample_idxs(size_t n, int max_draw, std::mt19937& rng) {
		std::vector<size_t> idxs(n);
		for (size_t i = 0; i < n; ++i) idxs[i] = i;
		if (max_draw > 0 && static_cast<int>(n) > max_draw) {
			std::shuffle(idxs.begin(), idxs.end(), rng);
			idxs.resize(static_cast<size_t>(max_draw));
		}
		return idxs;
	}

	bool fenfu_load_side_by_side(
		char* imagepath1, char* imagepath2, int sf, Mat& img_out, int& cols_l)
	{
		Mat imgL0 = imread(imagepath1, 0);
		Mat imgR0 = imread(imagepath2, 0);
		if ((!imgL0.data) || (!imgR0.data)) {
			std::cout << " --(!) Error reading images " << std::endl;
			return false;
		}
		Mat imgL, imgR;
		imgL.create(imgL0.rows / sf, imgL0.cols / sf, CV_8UC1);
		imgR.create(imgR0.rows / sf, imgR0.cols / sf, CV_8UC1);
		for (int i = 0; i < imgL.rows; i++) {
			for (int j = 0; j < imgL.cols; j++) {
				imgL.data[i * imgL.cols + j] = imgL0.data[sf * i * imgL0.cols + sf * j];
			}
		}
		for (int i = 0; i < imgR.rows; i++) {
			for (int j = 0; j < imgR.cols; j++) {
				imgR.data[i * imgR.cols + j] = imgR0.data[sf * i * imgR0.cols + sf * j];
			}
		}
		const int rows = imgL.rows > imgR.rows ? imgL.rows : imgR.rows;
		const int cols = imgL.cols + imgR.cols;
		cols_l = imgL.cols;
		img_out.create(rows, cols, CV_8UC3);
		img_out.setTo(Scalar(0, 0, 0));
		for (int i = 0; i < imgL.rows; i++) {
			for (int j = 0; j < imgL.cols; j++) {
				const uchar v = imgL.data[i * imgL.cols + j];
				img_out.data[3 * (i * cols + j) + 0] = v;
				img_out.data[3 * (i * cols + j) + 1] = v;
				img_out.data[3 * (i * cols + j) + 2] = v;
			}
		}
		for (int i = 0; i < imgR.rows; i++) {
			for (int j = 0; j < imgR.cols; j++) {
				const uchar v = imgR.data[i * imgR.cols + j];
				img_out.data[3 * (i * cols + imgL.cols + j) + 0] = v;
				img_out.data[3 * (i * cols + imgL.cols + j) + 1] = v;
				img_out.data[3 * (i * cols + imgL.cols + j) + 2] = v;
			}
		}
		return true;
	}
} // namespace

void ImageMatch::drawMatch3(char* imagepath1,char* imagepath2,char* outpath,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf){
	static const std::vector<int> empty;
	drawMatch3(imagepath1, imagepath2, outpath,
		KeyPoint_x1, KeyPoint_y1, KeyPoint_x2, KeyPoint_y2, Match, sf,
		empty, empty, empty, empty, false);
}

void ImageMatch::drawMatch3(char* imagepath1,char* imagepath2,char* outpath,
	std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,
	std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf,
	const std::vector<int>& unmatched_x1,const std::vector<int>& unmatched_y1,
	const std::vector<int>& unmatched_x2,const std::vector<int>& unmatched_y2,
	bool draw_lines)
{
	Mat img;
	int cols_l = 0;
	if (!fenfu_load_side_by_side(imagepath1, imagepath2, sf, img, cols_l)) {
		return;
	}

	// BGR: unmatched 黄, outlier 红, inlier 绿（与 Python 默认一致）
	const Scalar color_unmatched = CV_RGB(255, 255, 0);
	const Scalar color_outlier = CV_RGB(255, 0, 0);
	const Scalar color_inlier = CV_RGB(0, 255, 0);
	std::mt19937 rng(static_cast<unsigned>(kFenfuSeed));

	const size_t n_un_l = std::min(unmatched_x1.size(), unmatched_y1.size());
	const size_t n_un_r = std::min(unmatched_x2.size(), unmatched_y2.size());
	for (size_t i : fenfu_sample_idxs(n_un_l, kFenfuMaxDraw, rng)) {
		circle(img, cv::Point(unmatched_y1[i] / sf, unmatched_x1[i] / sf),
			kFenfuRadius, color_unmatched, 1, LINE_AA);
	}
	for (size_t i : fenfu_sample_idxs(n_un_r, kFenfuMaxDraw, rng)) {
		circle(img, cv::Point(unmatched_y2[i] / sf + cols_l, unmatched_x2[i] / sf),
			kFenfuRadius, color_unmatched, 1, LINE_AA);
	}

	const int P_numL = static_cast<int>(KeyPoint_x1.size());
	std::vector<int> inlier_idxs;
	std::vector<int> outlier_idxs;
	inlier_idxs.reserve(P_numL);
	outlier_idxs.reserve(P_numL);
	for (int i = 0; i < P_numL; i++) {
		if (Match[i] != -1) inlier_idxs.push_back(i);
		else outlier_idxs.push_back(i);
	}

	auto draw_pair = [&](int i, const Scalar& color, bool with_line) {
		const cv::Point p1(KeyPoint_y1[i] / sf, KeyPoint_x1[i] / sf);
		const cv::Point p2(KeyPoint_y2[i] / sf + cols_l, KeyPoint_x2[i] / sf);
		circle(img, p1, kFenfuRadius, color, 1, LINE_AA);
		circle(img, p2, kFenfuRadius, color, 1, LINE_AA);
		if (with_line) {
			line(img, p1, p2, color, 1, LINE_AA);
		}
	};

	{
		std::vector<size_t> s = fenfu_sample_idxs(outlier_idxs.size(), kFenfuMaxDraw, rng);
		for (size_t k : s) draw_pair(outlier_idxs[k], color_outlier, false);
	}
	{
		std::vector<size_t> s = fenfu_sample_idxs(inlier_idxs.size(), kFenfuMaxDraw, rng);
		for (size_t k : s) draw_pair(inlier_idxs[k], color_inlier, draw_lines);
	}

	std::cout << "fenfu_draw: inliers=" << inlier_idxs.size()
		<< ", outliers=" << outlier_idxs.size()
		<< ", unmatched_L=" << n_un_l << ", unmatched_R=" << n_un_r
		<< ", draw_lines=" << (draw_lines ? 1 : 0) << std::endl;

	imwrite(outpath, img);
}

void ImageMatch::drawFenfuMatchFromFiles(char* imagepath1,char* imagepath2,char* outpath,
	char* filepath,char* xulie_ID1,char* xulie_ID2,int layer,
	int ccd_begin,int ccd_end,int CCD_num,
	const int* mosaic_c1,const int* mosaic_c2,bool draw_lines)
{
	const double sfr = std::pow(2.0, static_cast<double>(layer)) / std::pow(2.0, 4.0);
	std::vector<int> inl_x1, inl_y1, inl_x2, inl_y2;
	std::vector<int> out_x1, out_y1, out_x2, out_y2;
	std::vector<int> un_x1, un_y1, un_x2, un_y2;

	auto pt_key = [](float row, float col, int img_id, float mrow, float mcol) {
		return std::make_tuple(
			static_cast<int>(std::lround(row * 10.f)),
			static_cast<int>(std::lround(col * 10.f)),
			img_id,
			static_cast<int>(std::lround(mrow * 10.f)),
			static_cast<int>(std::lround(mcol * 10.f)));
	};

	for (int j = ccd_begin; j < ccd_end; j++) {
		char fea_l[256], fea_r[256], match_path[256];
		sprintf(fea_l, "%s/%s/downsample/%d/%s_RED%d.txt", filepath, xulie_ID1, layer, xulie_ID1, j);
		sprintf(fea_r, "%s/%s/downsample/%d/%s_RED%d.txt", filepath, xulie_ID2, layer, xulie_ID2, j);
		sprintf(match_path, "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath, xulie_ID1, layer, xulie_ID1, j);

		std::set<std::tuple<int,int,int,int,int>> inlier_keys;
		FILE* fpm = fopen(match_path, "r");
		if (fpm) {
			int bj, imgID;
			float row, col, mrow, mcol, m_score;
			while (fscanf(fpm, "%d %f %f %d %f %f %f\n", &bj, &row, &col, &imgID, &mrow, &mcol, &m_score) == 7) {
				if (bj != 1 || imgID < 0 || imgID >= CCD_num) continue;
				inlier_keys.insert(pt_key(row, col, imgID, mrow, mcol));
				inl_x1.push_back(static_cast<int>(std::lround((row + mosaic_c1[j * 4 + 0]) * sfr)));
				inl_y1.push_back(static_cast<int>(std::lround((col + mosaic_c1[j * 4 + 2]) * sfr)));
				inl_x2.push_back(static_cast<int>(std::lround((mrow + mosaic_c2[imgID * 4 + 0]) * sfr)));
				inl_y2.push_back(static_cast<int>(std::lround((mcol + mosaic_c2[imgID * 4 + 2]) * sfr)));
			}
			fclose(fpm);
		}

		FILE* fpl = fopen(fea_l, "r");
		if (fpl) {
			int bj, imgID;
			float row, col, mrow, mcol, m_score;
			while (!feof(fpl)) {
				if (fscanf(fpl, "%d ", &bj) != 1) break;
				if (bj == 1) {
					if (fscanf(fpl, "%f %f %d %f %f %f\n", &row, &col, &imgID, &mrow, &mcol, &m_score) != 6) break;
					if (imgID < 0 || imgID >= CCD_num) continue;
					if (inlier_keys.count(pt_key(row, col, imgID, mrow, mcol))) continue;
					out_x1.push_back(static_cast<int>(std::lround((row + mosaic_c1[j * 4 + 0]) * sfr)));
					out_y1.push_back(static_cast<int>(std::lround((col + mosaic_c1[j * 4 + 2]) * sfr)));
					out_x2.push_back(static_cast<int>(std::lround((mrow + mosaic_c2[imgID * 4 + 0]) * sfr)));
					out_y2.push_back(static_cast<int>(std::lround((mcol + mosaic_c2[imgID * 4 + 2]) * sfr)));
				} else {
					if (fscanf(fpl, "%f %f\n", &row, &col) != 2) break;
					un_x1.push_back(static_cast<int>(std::lround((row + mosaic_c1[j * 4 + 0]) * sfr)));
					un_y1.push_back(static_cast<int>(std::lround((col + mosaic_c1[j * 4 + 2]) * sfr)));
				}
			}
			fclose(fpl);
		}

		FILE* fpr = fopen(fea_r, "r");
		if (fpr) {
			int bj, imgID;
			float row, col, mrow, mcol, m_score;
			while (!feof(fpr)) {
				if (fscanf(fpr, "%d ", &bj) != 1) break;
				if (bj == 1) {
					if (fscanf(fpr, "%f %f %d %f %f %f\n", &row, &col, &imgID, &mrow, &mcol, &m_score) != 6) break;
				} else {
					if (fscanf(fpr, "%f %f\n", &row, &col) != 2) break;
					un_x2.push_back(static_cast<int>(std::lround((row + mosaic_c2[j * 4 + 0]) * sfr)));
					un_y2.push_back(static_cast<int>(std::lround((col + mosaic_c2[j * 4 + 2]) * sfr)));
				}
			}
			fclose(fpr);
		}
	}

	// 拼成 Match 数组：先 outliers 再 inliers（drawMatch3 按 Match 分类）
	std::vector<int> kx1 = out_x1, ky1 = out_y1, kx2 = out_x2, ky2 = out_y2;
	const int n_out = static_cast<int>(out_x1.size());
	const int n_inl = static_cast<int>(inl_x1.size());
	kx1.insert(kx1.end(), inl_x1.begin(), inl_x1.end());
	ky1.insert(ky1.end(), inl_y1.begin(), inl_y1.end());
	kx2.insert(kx2.end(), inl_x2.begin(), inl_x2.end());
	ky2.insert(ky2.end(), inl_y2.begin(), inl_y2.end());
	std::vector<int> match(kx1.size(), -1);
	for (int i = 0; i < n_inl; i++) {
		match[n_out + i] = i;
	}

	std::cout << "drawFenfuMatchFromFiles layer=" << layer
		<< " inliers=" << n_inl << " outliers=" << n_out
		<< " unmatched_L=" << un_x1.size() << " unmatched_R=" << un_x2.size() << std::endl;

	drawMatch3(imagepath1, imagepath2, outpath, kx1, ky1, kx2, ky2, match.data(), 1,
		un_x1, un_y1, un_x2, un_y2, draw_lines);
}

void ImageMatch::drawMatch4(char* imagepath1,char* imagepath2,char* outpath,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int count1,int sf){
	Mat imgL0 = imread(imagepath1, 0);
	Mat imgR0 = imread(imagepath2, 0);
	if((!imgL0.data)||(!imgR0.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		if(Match[i]!=-1 && i%1==0 && i<count1){
			int r=255;//int(float(rand())/RAND_MAX+0.5)*255;
			int g=255;//int(float(rand())/RAND_MAX+0.5)*255;
			int b=0;//int(float(rand())/RAND_MAX+0.5)*255;
			circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(r,g,b),2,8,0);
			circle(img,cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(r,g,b),2,8,0);
			// line(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),CV_RGB(r,g,b),2,8,0);
		}
		//else if(i>=count1 && i%5==0){
		//	int r=255;//int(float(rand())/RAND_MAX+0.5)*255;
		//	int g=0;//int(float(rand())/RAND_MAX+0.5)*255;
		//	int b=0;//int(float(rand())/RAND_MAX+0.5)*255;
		//	circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(r,g,b),2,8,0);
		//	circle(img,cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(r,g,b),2,8,0);
		//}
		else{
			//circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(255,0,0),0.5,8,0);
			//circle(img,cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(255,0,0),0.5,8,0);
			//line(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cv::Point(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),CV_RGB(255,0,0),0.5,8,0);
		}
	}

	//imshow("Match",img);
	imwrite(outpath,img);
	//waitKey(0);
}


void ImageMatch::draw_fea(char* imagepath1, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,int sf,const char* outpath){
	Mat imgL0 = imread(imagepath1, 0);
	if((!imgL0.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat imgL;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}

	int rows= imgL.rows;
	int cols= imgL.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*cols+j)];
		}
	}
	
	int ls=390;
	int ws=18;
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		int r=KeyPoint_x1[i];
		int c=KeyPoint_y1[i];
		int R=255;//int(float(rand())/RAND_MAX+0.5)*255;
		int G=255;//int(float(rand())/RAND_MAX+0.5)*255;
		int B=0;//int(float(rand())/RAND_MAX+0.5)*255;
		for(int j=-ls;j<=ls;j++){
			for(int k=-ws;k<=ws;k++){
				if(r+j>=0 && r+j<rows && c+k>=0 && c+k<cols){
					img.data[3*((r+j)*cols+c+k)+0]=B;
					img.data[3*((r+j)*cols+c+k)+1]=G;
					img.data[3*((r+j)*cols+c+k)+2]=R;
				}
			}
		}
		for(int j=-ws;j<=ws;j++){
			for(int k=-ls;k<=ls;k++){
				if(r+j>=0 && r+j<rows && c+k>=0 && c+k<cols){
					img.data[3*((r+j)*cols+c+k)+0]=B;
					img.data[3*((r+j)*cols+c+k)+1]=G;
					img.data[3*((r+j)*cols+c+k)+2]=R;
				}
			}
		}
		//cv::Point p1=cv::Point((KeyPoint_y1[i]-1000)/sf,KeyPoint_x1[i]/sf);
		//cv::Point p2=cv::Point((KeyPoint_y1[i]+5)/sf,KeyPoint_x1[i]/sf);
		//line(img,p1,p2,CV_RGB(r,g,b),1,8,0);
		//circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),50,CV_RGB(r,g,b),1,8,0);
	}

	imwrite(outpath,img);
	cout<<P_numL<<endl;
}

void ImageMatch::draw_res(char* imagepath1, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<double> resx,std::vector<double> resy,int sf,const char* outpath){
	Mat imgL0 = imread(imagepath1, 0);
	if((!imgL0.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat imgL;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}

	int rows= imgL.rows;
	int cols= imgL.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*cols+j)];
		}
	}
	
	cout<<sf<<endl;
	double sf1=1000;
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		int r=KeyPoint_x1[i];
		int c=KeyPoint_y1[i];
		int R=255;//int(float(rand())/RAND_MAX+0.5)*255;
		int G=255;//int(float(rand())/RAND_MAX+0.5)*255;
		int B=0;//int(float(rand())/RAND_MAX+0.5)*255;
		Point p1=cv::Point(int(double(KeyPoint_y1[i])/double(sf)+0.5),int(double(KeyPoint_x1[i])/double(sf)+0.5));
		//cv::Point p3=cv::Point(int(double(KeyPoint_y1[i])/double(sf)+0.5)+10,int(double(KeyPoint_x1[i])/double(sf)+0.5)+10);
		Point p2=cv::Point(int((double(KeyPoint_y1[i])+sf1*resy[i])/double(sf)+0.5),int((double(KeyPoint_x1[i])+sf1*resx[i])/double(sf)+0.5));
		line(img,p1,p2,CV_RGB(R,G,B),2,8,0);
		circle(img,p1,6,CV_RGB(R,G,B),-1,8,0);
		/*for(int j=-ls;j<=ls;j++){
			for(int k=-ws;k<=ws;k++){
				if(r+j>=0 && r+j<rows && c+k>=0 && c+k<cols){
					img.data[3*((r+j)*cols+c+k)+0]=B;
					img.data[3*((r+j)*cols+c+k)+1]=G;
					img.data[3*((r+j)*cols+c+k)+2]=R;
				}
			}
		}
		for(int j=-ws;j<=ws;j++){
			for(int k=-ls;k<=ls;k++){
				if(r+j>=0 && r+j<rows && c+k>=0 && c+k<cols){
					img.data[3*((r+j)*cols+c+k)+0]=B;
					img.data[3*((r+j)*cols+c+k)+1]=G;
					img.data[3*((r+j)*cols+c+k)+2]=R;
				}
			}
		}*/
		//cv::Point p1=cv::Point((KeyPoint_y1[i]-1000)/sf,KeyPoint_x1[i]/sf);
		//cv::Point p2=cv::Point((KeyPoint_y1[i]+5)/sf,KeyPoint_x1[i]/sf);
		//line(img,p1,p2,CV_RGB(r,g,b),1,8,0);
		//circle(img,cv::Point(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),50,CV_RGB(r,g,b),1,8,0);
	}

	imwrite(outpath,img);
	cout<<P_numL<<endl;
}

int ImageMatch::CC_match2(char* imagepath1,char* imagepath2,int w_size,float threshold,int ch,float* fs_c,int RCCD_id, char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	const int half = w_size / 2;
	Mat grad_1, grad_2;
	computeGradientMag(img_1, grad_1);
	computeGradientMag(img_2, grad_2);

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout<<"Error opening feature point file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);

	FILE *fp2=fopen(featurepointxt_2,"r"); 
	if(fp2==NULL){
		std::cout<<"Error opening feature point file: "<<featurepointxt_2<<std::endl;
		return -1;
	}
	while(!feof(fp2)){
		fscanf(fp2,"%d ",&bj);
		if(bj==1){
			fscanf(fp2,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp2,"%d %d\n",&row,&col);
		}
		KeyPoint_x2.push_back(row);
		KeyPoint_y2.push_back(col);
	}
	fclose(fp2);

	//Feature_Detection(imagepath1,ch,KeyPoint_x1,KeyPoint_y1);
	//Feature_Detection(imagepath2,ch,KeyPoint_x2,KeyPoint_y2);

	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	int count=0;
	#pragma omp parallel for schedule(dynamic, 50)
	for(int i=0;i<(int)KeyPoint_x1.size();i++){
		float maxCC=-1000;
		for(int j=0;j<KeyPoint_x2.size();j++){
				if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size &&
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size)
			{
				const float score = computeScoreB(img_1, img_2, grad_1, grad_2,
				                                  KeyPoint_x1[i], KeyPoint_y1[i],
				                                  KeyPoint_x2[j], KeyPoint_y2[j],
				                                  half);
				if(score>maxCC && score>=threshold){
					maxCC=score;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
	}

	std::cout<<"CC match finished!"<<std::endl;

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		std::cout<<"Error opening feature point file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	std::cout<<"Write match points to "<<outpointxt_1<<std::endl;
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
			//matched[i]=-1;
			count--;  
		}
	}
	fclose(fp3);
	fclose(fp4);

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	//int * match = new int[KeyPoint_x11.size()];
	//memset(match,-1,sizeof(int)*KeyPoint_x11.size());
	//float* fs_c = new float[6];
	//RANSAC_fs(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,5,match,fs_c);  //剔除粗差

	/*
	FILE *fp=fopen(matchpointxt,"w");
	int countMP=0;
	for(int i=0;i<KeyPoint_x11.size();i++){
		if(match[i]!=-1){
			fprintf(fp,"%d %d %d %d %d\n",countMP,KeyPoint_x11[i],KeyPoint_y11[i],KeyPoint_x22[i],KeyPoint_y22[i]);
			countMP++;
		}
	}
	*/

	std::cout<<"Matched points number: "<<count<<std::endl;
	std::cout<<"Average shift: "<<dr<<" "<<dc<<std::endl;

	// int sf = 1;
	// drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	// //drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);
	// std::cout<<"Draw match points finished!"<<std::endl;

	delete [] matched;
	delete [] C_match;
	vector<int>().swap(KeyPoint_x1);
	vector<int>().swap(KeyPoint_y1);
	vector<int>().swap(KeyPoint_x2);
	vector<int>().swap(KeyPoint_y2);
	vector<int>().swap(KeyPoint_x11);
	vector<int>().swap(KeyPoint_y11);
	vector<int>().swap(KeyPoint_x22);
	vector<int>().swap(KeyPoint_y22);
	return 0;
}
int ImageMatch::limit_match1(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int ch,float* fs_c,int RCCD_id, char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	const int half = w_size / 2;
	Mat grad_1, grad_2;
	computeGradientMag(img_1, grad_1);
	computeGradientMag(img_2, grad_2);

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%d %d %d %d %d %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
			fprintf(fp11,"%d %d\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	FILE *fp2=fopen(featurepointxt_2,"r"); 
	if(fp2==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_2<<std::endl;
		return -1;
	}
	while(!feof(fp2)){
		fscanf(fp2,"%d ",&bj);
		if(bj==1){
			fscanf(fp2,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp2,"%d %d\n",&row,&col);
		}
		KeyPoint_x2.push_back(row);
		KeyPoint_y2.push_back(col);
	}
	fclose(fp2);

	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	int count=0;
	if(fsFullyOutside(fs_c, rows1, cols1, rows2, cols2)){
		delete[] matched;
		delete[] C_match;
		printf("[limit_match1] RCCD=%d skip (no overlap) L=%zu R=%zu\n",
			RCCD_id, KeyPoint_x1.size(), KeyPoint_x2.size());
		fflush(stdout);
		return 0;
	}
	const int nL = (int)KeyPoint_x1.size();
	const int nR = (int)KeyPoint_x2.size();
	const double t0 = omp_get_wtime();
	printf("[limit_match1] start RCCD=%d L=%d R=%d w=%d ser=%d thr=%.2f threads=%d\n",
		RCCD_id, nL, nR, w_size, ser_range, threshold, omp_get_max_threads());
	fflush(stdout);

	RightPointGrid grid;
	grid.build(KeyPoint_x2, KeyPoint_y2, std::max(32, ser_range));
	const int half_r = 2 * ser_range;
	const int half_c = ser_range;
	int done_pts = 0;
	const int progress_step = std::max(1, nL / 10);

	#pragma omp parallel for schedule(dynamic, 50)
	for(int i=0;i<nL;i++){
		float maxCC=-1000;
		const float pr = fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2];
		const float pc = fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5];
		std::vector<int> cands;
		grid.query((int)(pr + 0.5f), (int)(pc + 0.5f), half_r, half_c, cands);
		for(int t=0;t<(int)cands.size();t++){
			const int j = cands[t];
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size&&
					abs(KeyPoint_x2[j]-pr)<2*ser_range &&
				abs(KeyPoint_y2[j]-pc)<ser_range)
			{
				const float score = computeScoreB(img_1, img_2, grad_1, grad_2,
				                                  KeyPoint_x1[i], KeyPoint_y1[i],
				                                  KeyPoint_x2[j], KeyPoint_y2[j],
				                                  half);
				if(score>maxCC && score>=threshold){
					maxCC=score;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
		int cur;
		#pragma omp atomic capture
		cur = ++done_pts;
		if (cur % progress_step == 0 || cur == nL) {
			#pragma omp critical(limit_match1_progress)
			{
				printf("[limit_match1] %d/%d (%.0f%%) elapsed=%.1fs\n",
					cur, nL, 100.0 * cur / std::max(1, nL), omp_get_wtime() - t0);
				fflush(stdout);
			}
		}
	}

	int n_ok = 0;
	for (int i = 0; i < nL; ++i) if (C_match[i] >= threshold) ++n_ok;
	printf("[limit_match1] done RCCD=%d matched=%d/%d elapsed=%.1fs\n",
		RCCD_id, n_ok, nL, omp_get_wtime() - t0);
	fflush(stdout);

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
			//matched[i]=-1;
			count--;  
		}
	}
	fclose(fp3);
	fclose(fp4);

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	// int sf = 2;
	// drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	// //drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);

	delete [] matched;
	delete [] C_match;
	vector<int>().swap(KeyPoint_x1);
	vector<int>().swap(KeyPoint_y1);
	vector<int>().swap(KeyPoint_x2);
	vector<int>().swap(KeyPoint_y2);
	vector<int>().swap(KeyPoint_x11);
	vector<int>().swap(KeyPoint_y11);
	vector<int>().swap(KeyPoint_x22);
	vector<int>().swap(KeyPoint_y22);

	return 0;
}

LocalAffineField::LocalAffineField() {
	for (int i = 0; i < 6; ++i) global_fs[i] = (i == 0 || i == 4) ? 1.f : 0.f;
}

void LocalAffineField::set_global(const float* fs) {
	for (int i = 0; i < 6; ++i) global_fs[i] = fs[i];
}

void LocalAffineField::clear() {
	n_tr = n_tc = 0;
	tile_fs.clear();
	tile_valid.clear();
}

void LocalAffineField::upsample(float scale) {
	if (scale == 1.f) return;
	global_fs[2] *= scale;
	global_fs[5] *= scale;
	for (size_t t = 0; t < tile_valid.size(); ++t) {
		if (!tile_valid[t]) continue;
		tile_fs[t * 6 + 2] *= scale;
		tile_fs[t * 6 + 5] *= scale;
	}
	mosaic_rows = static_cast<int>(mosaic_rows * scale + 0.5f);
	mosaic_cols = static_cast<int>(mosaic_cols * scale + 0.5f);
	tile_h = std::max(1, static_cast<int>(tile_h * scale + 0.5f));
	tile_w = std::max(1, static_cast<int>(tile_w * scale + 0.5f));
}

bool LocalAffineField::copy_affine_at(float r, float c, float* fs_out) const {
	if (n_tr <= 0 || n_tc <= 0 || tile_fs.empty()) {
		for (int i = 0; i < 6; ++i) fs_out[i] = global_fs[i];
		return false;
	}
	int tr = static_cast<int>(r) / std::max(tile_h, 1);
	int tc = static_cast<int>(c) / std::max(tile_w, 1);
	if (tr < 0) tr = 0;
	if (tc < 0) tc = 0;
	if (tr >= n_tr) tr = n_tr - 1;
	if (tc >= n_tc) tc = n_tc - 1;

	// 若本块无效，向邻块扩展搜索
	const int rad_max = std::max(n_tr, n_tc);
	for (int rad = 0; rad <= rad_max; ++rad) {
		for (int dtr = -rad; dtr <= rad; ++dtr) {
			for (int dtc = -rad; dtc <= rad; ++dtc) {
				if (rad > 0 && std::max(std::abs(dtr), std::abs(dtc)) != rad) continue;
				int ttr = tr + dtr;
				int ttc = tc + dtc;
				if (ttr < 0 || ttc < 0 || ttr >= n_tr || ttc >= n_tc) continue;
				const int idx = ttr * n_tc + ttc;
				if (!tile_valid[idx]) continue;
				for (int i = 0; i < 6; ++i) fs_out[i] = tile_fs[idx * 6 + i];
				return true;
			}
		}
	}
	for (int i = 0; i < 6; ++i) fs_out[i] = global_fs[i];
	return false;
}

bool LocalAffineField::predict(float r, float c, float* pred_r, float* pred_c) const {
	float fs[6];
	copy_affine_at(r, c, fs);
	*pred_r = fs[0] * r + fs[1] * c + fs[2];
	*pred_c = fs[3] * r + fs[4] * c + fs[5];
	return true;
}

void ImageMatch::build_local_affine_field(
	const std::vector<int>& x1, const std::vector<int>& y1,
	const std::vector<int>& x2, const std::vector<int>& y2,
	int mosaic_rows, int mosaic_cols, int n_tiles_r, int n_tiles_c,
	float sigma, int ransac_iters, int min_pts, float* global_fs,
	LocalAffineField& field)
{
	field.clear();
	field.set_global(global_fs);
	field.mosaic_rows = std::max(mosaic_rows, 1);
	field.mosaic_cols = std::max(mosaic_cols, 1);
	field.n_tr = std::max(n_tiles_r, 1);
	field.n_tc = std::max(n_tiles_c, 1);
	field.tile_h = (field.mosaic_rows + field.n_tr - 1) / field.n_tr;
	field.tile_w = (field.mosaic_cols + field.n_tc - 1) / field.n_tc;
	field.tile_fs.assign(static_cast<size_t>(field.n_tr * field.n_tc * 6), 0.f);
	field.tile_valid.assign(static_cast<size_t>(field.n_tr * field.n_tc), 0);

	const int n = static_cast<int>(x1.size());
	if (n < 3) return;

	for (int tr = 0; tr < field.n_tr; ++tr) {
		for (int tc = 0; tc < field.n_tc; ++tc) {
			const int r0 = tr * field.tile_h;
			const int r1 = std::min(field.mosaic_rows, (tr + 1) * field.tile_h);
			const int c0 = tc * field.tile_w;
			const int c1 = std::min(field.mosaic_cols, (tc + 1) * field.tile_w);
			// 带半块重叠，避免块边界断层
			const int margin_r = field.tile_h / 2;
			const int margin_c = field.tile_w / 2;
			const int rr0 = std::max(0, r0 - margin_r);
			const int rr1 = std::min(field.mosaic_rows, r1 + margin_r);
			const int cc0 = std::max(0, c0 - margin_c);
			const int cc1 = std::min(field.mosaic_cols, c1 + margin_c);

			std::vector<int> lx1, ly1, lx2, ly2;
			lx1.reserve(64); ly1.reserve(64); lx2.reserve(64); ly2.reserve(64);
			for (int i = 0; i < n; ++i) {
				if (x1[i] >= rr0 && x1[i] < rr1 && y1[i] >= cc0 && y1[i] < cc1) {
					lx1.push_back(x1[i]); ly1.push_back(y1[i]);
					lx2.push_back(x2[i]); ly2.push_back(y2[i]);
				}
			}
			const int idx = tr * field.n_tc + tc;
			if (static_cast<int>(lx1.size()) < min_pts) {
				for (int i = 0; i < 6; ++i) field.tile_fs[idx * 6 + i] = global_fs[i];
				field.tile_valid[idx] = 0;
				continue;
			}
			std::vector<int> match(lx1.size(), -1);
			float loc_fs[6];
			for (int i = 0; i < 6; ++i) loc_fs[i] = global_fs[i];
			RANSAC_fs2(lx1, ly1, lx2, ly2, sigma, std::max(200, ransac_iters / 20), match.data(), loc_fs);
			int nin = 0;
			for (size_t i = 0; i < match.size(); ++i) if (match[i] != -1) ++nin;
			if (nin >= std::max(3, min_pts / 2)) {
				for (int i = 0; i < 6; ++i) field.tile_fs[idx * 6 + i] = loc_fs[i];
				field.tile_valid[idx] = 1;
			} else {
				for (int i = 0; i < 6; ++i) field.tile_fs[idx * 6 + i] = global_fs[i];
				field.tile_valid[idx] = 0;
			}
		}
	}

	int n_valid = 0;
	for (size_t i = 0; i < field.tile_valid.size(); ++i) if (field.tile_valid[i]) ++n_valid;
	std::cout << "LocalAffineField: " << n_valid << "/" << (field.n_tr * field.n_tc)
	          << " tiles valid, tile=" << field.tile_h << "x" << field.tile_w << std::endl;
}

void ImageMatch::mark_local_inliers(
	const std::vector<int>& x1, const std::vector<int>& y1,
	const std::vector<int>& x2, const std::vector<int>& y2,
	const LocalAffineField& field, float sigma, int* match)
{
	const int n = static_cast<int>(x1.size());
	int nin = 0;
	for (int i = 0; i < n; ++i) {
		float pr = 0.f, pc = 0.f;
		field.predict(static_cast<float>(x1[i]), static_cast<float>(y1[i]), &pr, &pc);
		const float vr = pr - static_cast<float>(x2[i]);
		const float vc = pc - static_cast<float>(y2[i]);
		if (std::sqrt(vr * vr + vc * vc) <= sigma) {
			match[i] = i;
			++nin;
		} else {
			match[i] = -1;
		}
	}
	std::cout << "Local inliers: " << nin << "/" << n << " (sigma=" << sigma << ")" << std::endl;
}

int ImageMatch::limit_match1_field(char* imagepath1, char* imagepath2, int w_size, int ser_range, float threshold, int ch, float* fs_c, int RCCD_id,
	char* featurepointxt_1, char* featurepointxt_2, char* outpointxt_1,
	const LocalAffineField* field, int off_r1, int off_c1, int off_r2, int off_c2)
{
	if (field == nullptr || field->n_tr <= 0) {
		return limit_match1(imagepath1, imagepath2, w_size, ser_range, threshold, ch, fs_c, RCCD_id,
			featurepointxt_1, featurepointxt_2, outpointxt_1);
	}

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if ((!img_1.data) || (!img_2.data)) {
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}
	const int rows1 = img_1.rows, cols1 = img_1.cols;
	const int rows2 = img_2.rows, cols2 = img_2.cols;
	const int half = w_size / 2;
	Mat grad_1, grad_2;
	computeGradientMag(img_1, grad_1);
	computeGradientMag(img_2, grad_2);

	std::vector<int> KeyPoint_x1, KeyPoint_y1, KeyPoint_x2, KeyPoint_y2, Biaoji;
	int bj, row, col, imgID, mrow, mcol;
	float mscore;

	FILE* fp1 = fopen(featurepointxt_1, "r");
	if (!fp1) { std::cout << "Error opening file: " << featurepointxt_1 << std::endl; return -1; }
	FILE* fp11 = fopen(outpointxt_1, "w");
	while (!feof(fp1)) {
		fscanf(fp1, "%d ", &bj);
		fprintf(fp11, "%d ", bj);
		if (bj == 1) {
			fscanf(fp1, "%d %d %d %d %d %f\n", &row, &col, &imgID, &mrow, &mcol, &mscore);
			fprintf(fp11, "%d %d %d %d %d %f\n", row, col, imgID, mrow, mcol, mscore);
		} else {
			fscanf(fp1, "%d %d\n", &row, &col);
			fprintf(fp11, "%d %d\n", row, col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	FILE* fp2 = fopen(featurepointxt_2, "r");
	if (!fp2) { std::cout << "Error opening file: " << featurepointxt_2 << std::endl; return -1; }
	while (!feof(fp2)) {
		fscanf(fp2, "%d ", &bj);
		if (bj == 1) {
			fscanf(fp2, "%d %d %d %d %d %f\n", &row, &col, &imgID, &mrow, &mcol, &mscore);
		} else {
			fscanf(fp2, "%d %d\n", &row, &col);
		}
		KeyPoint_x2.push_back(row);
		KeyPoint_y2.push_back(col);
	}
	fclose(fp2);

	int* matched = new int[KeyPoint_x1.size()];
	memset(matched, -1, sizeof(int) * KeyPoint_x1.size());
	float* C_match = new float[KeyPoint_x1.size()];
	memset(C_match, 0, sizeof(float) * KeyPoint_x1.size());

	const float ser_r = static_cast<float>(2 * ser_range);
	const float ser_c = static_cast<float>(ser_range);
	const int nL = (int)KeyPoint_x1.size();
	const double t0 = omp_get_wtime();
	printf("[limit_match1_field] start RCCD=%d L=%d R=%zu w=%d ser=%d\n",
		RCCD_id, nL, KeyPoint_x2.size(), w_size, ser_range);
	fflush(stdout);
	RightPointGrid grid;
	grid.build(KeyPoint_x2, KeyPoint_y2, std::max(32, ser_range));

	#pragma omp parallel for schedule(dynamic, 50)
	for (int i = 0; i < nL; ++i) {
		float maxCC = -1000.f;
		const float mr = static_cast<float>(KeyPoint_x1[i] + off_r1);
		const float mc = static_cast<float>(KeyPoint_y1[i] + off_c1);
		float local_fs[6];
		const bool have_local_affine = field->copy_affine_at(mr, mc, local_fs);
		float pred_mr = 0.f, pred_mc = 0.f;
		field->predict(mr, mc, &pred_mr, &pred_mc);
		float pred_r = pred_mr - static_cast<float>(off_r2);
		float pred_c = pred_mc - static_cast<float>(off_c2);
		// 与全局仿射混合：若局部预测离谱则退回 fs_c
		const float g_r = fs_c[0] * KeyPoint_x1[i] + fs_c[1] * KeyPoint_y1[i] + fs_c[2];
		const float g_c = fs_c[3] * KeyPoint_x1[i] + fs_c[4] * KeyPoint_y1[i] + fs_c[5];
		if (std::fabs(pred_r - g_r) > 4.f * ser_range || std::fabs(pred_c - g_c) > 4.f * ser_range) {
			pred_r = g_r;
			pred_c = g_c;
		}

		std::vector<int> cands;
		grid.query((int)(pred_r + 0.5f), (int)(pred_c + 0.5f), 2 * ser_range, ser_range, cands);
		for (int t = 0; t < (int)cands.size(); ++t) {
			const int j = cands[t];
			if (KeyPoint_x1[i] < w_size || KeyPoint_y1[i] < w_size ||
				KeyPoint_x2[j] < w_size || KeyPoint_y2[j] < w_size ||
				KeyPoint_x1[i] > rows1 - w_size || KeyPoint_y1[i] > cols1 - w_size ||
				KeyPoint_x2[j] > rows2 - w_size || KeyPoint_y2[j] > cols2 - w_size) {
				continue;
			}
			if (std::fabs(KeyPoint_x2[j] - pred_r) >= ser_r ||
				std::fabs(KeyPoint_y2[j] - pred_c) >= ser_c) {
				continue;
			}
			const float score = (g_use_affine_patch_score && have_local_affine)
				? computeScoreBAffinePatch(img_1, img_2, grad_1, grad_2,
					KeyPoint_x1[i], KeyPoint_y1[i], KeyPoint_x2[j], KeyPoint_y2[j], half, local_fs)
				: computeScoreB(img_1, img_2, grad_1, grad_2,
					KeyPoint_x1[i], KeyPoint_y1[i], KeyPoint_x2[j], KeyPoint_y2[j], half);
			if (score > maxCC && score >= threshold) {
				maxCC = score;
				matched[i] = j;
				C_match[i] = maxCC;
			}
		}
	}
	{
		int n_ok = 0;
		for (int i = 0; i < nL; ++i) if (C_match[i] >= threshold) ++n_ok;
		printf("[limit_match1_field] done RCCD=%d matched=%d/%d elapsed=%.1fs\n",
			RCCD_id, n_ok, nL, omp_get_wtime() - t0);
		fflush(stdout);
	}

	int count = (int)KeyPoint_x1.size();
	int dr = 0, dc = 0;
	FILE* fp3 = fopen(featurepointxt_1, "r");
	if (!fp3) { delete[] matched; delete[] C_match; return -1; }
	FILE* fp4 = fopen(outpointxt_1, "w");
	for (int i = 0; i < (int)KeyPoint_x1.size(); ++i) {
		if (C_match[i] >= threshold) {
			dr += KeyPoint_x2[matched[i]] - KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]] - KeyPoint_y1[i];
			if (Biaoji[i] == 1) {
				fscanf(fp3, "%d %d %d %d %d %d %f\n", &bj, &row, &col, &imgID, &mrow, &mcol, &mscore);
				if (mscore < C_match[i]) {
					fprintf(fp4, "%d %d %d %d %d %d %f\n", bj, row, col, RCCD_id, KeyPoint_x2[matched[i]], KeyPoint_y2[matched[i]], C_match[i]);
				} else {
					fprintf(fp4, "%d %d %d %d %d %d %f\n", bj, row, col, imgID, mrow, mcol, mscore);
				}
			} else {
				fscanf(fp3, "%d %d %d\n", &bj, &row, &col);
				fprintf(fp4, "%d %d %d %d %d %d %f\n", 1, row, col, RCCD_id, KeyPoint_x2[matched[i]], KeyPoint_y2[matched[i]], C_match[i]);
			}
		} else {
			if (Biaoji[i] == 1) {
				fscanf(fp3, "%d %d %d %d %d %d %f\n", &bj, &row, &col, &imgID, &mrow, &mcol, &mscore);
				fprintf(fp4, "%d %d %d %d %d %d %f\n", bj, row, col, imgID, mrow, mcol, mscore);
			} else {
				fscanf(fp3, "%d %d %d\n", &bj, &row, &col);
				fprintf(fp4, "%d %d %d\n", bj, row, col);
			}
			--count;
		}
	}
	fclose(fp3);
	fclose(fp4);
	(void)ch; (void)dr; (void)dc;
	delete[] matched;
	delete[] C_match;
	return 0;
}

int ImageMatch::densify_match_from_seeds(char* imagepath1, char* imagepath2,
	int w_size, int ser_range, float threshold, int batch_r, int batch_c, int knn,
	char* seed_match_txt, int target_right_ccd, char* out_append_txt)
{
	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if ((!img_1.data) || (!img_2.data)) {
		std::cout << " --(!) Error reading images for densify" << std::endl;
		return -1;
	}
	if ((w_size & 1) == 0) w_size += 1;
	const int half = w_size / 2;
	Mat grad_1, grad_2;
	computeGradientMag(img_1, grad_1);
	computeGradientMag(img_2, grad_2);

	struct Seed { float r, c, mr, mc, score; };
	std::vector<Seed> seeds;
	FILE* fp = fopen(seed_match_txt, "r");
	if (!fp) {
		std::cout << "densify: cannot open seeds " << seed_match_txt << std::endl;
		return -1;
	}
	int bj, imgID;
	float row, col, mrow, mcol, mscore;
	while (fscanf(fp, "%d %f %f %d %f %f %f\n", &bj, &row, &col, &imgID, &mrow, &mcol, &mscore) == 7) {
		if (bj != 1) continue;
		if (target_right_ccd >= 0 && imgID != target_right_ccd) continue;
		Seed s;
		s.r = row; s.c = col; s.mr = mrow; s.mc = mcol; s.score = mscore;
		seeds.push_back(s);
	}
	fclose(fp);
	if (seeds.size() < 3) {
		std::cout << "densify: too few seeds (" << seeds.size() << ") for " << seed_match_txt << std::endl;
		return 0;
	}

	FILE* fp_out = fopen(out_append_txt, "a");
	if (!fp_out) return -1;

	const int rows1 = img_1.rows, cols1 = img_1.cols;
	const int rows2 = img_2.rows, cols2 = img_2.cols;
	(void)rows2; (void)cols2;
	const int knn_use = std::max(3, knn);
	const int n_i = rows1 / batch_r;
	const int n_j = cols1 / batch_c;
	if (n_i <= 0 || n_j <= 0) {
		fclose(fp_out);
		return 0;
	}
	const int n_cells = n_i * n_j;

	struct DenseHit {
		int r, c, ccd;
		float mr, mc, score;
	};
	std::vector<DenseHit> hits;
	hits.reserve(static_cast<size_t>(n_cells / 8));

	printf("[densify] grid=%dx%d cells=%d seeds=%d OpenMP...\n", n_i, n_j, n_cells, (int)seeds.size());
	fflush(stdout);

	#pragma omp parallel
	{
		std::vector<DenseHit> local_hits;
		local_hits.reserve(256);
		#pragma omp for schedule(dynamic, 16)
		for (int idx = 0; idx < n_cells; ++idx) {
			const int i = idx / n_j;
			const int j = idx % n_j;
			const int r = i * batch_r + batch_r / 2;
			const int c = j * batch_c + batch_c / 2;
			if (!patchInside(img_1, r, c, half)) continue;

			struct Neigh { float dist2; int sidx; };
			std::vector<Neigh> neigh;
			neigh.reserve(seeds.size());
			for (int s = 0; s < (int)seeds.size(); ++s) {
				const float dr = seeds[s].r - static_cast<float>(r);
				const float dc = seeds[s].c - static_cast<float>(c);
				Neigh nb; nb.dist2 = dr * dr + dc * dc; nb.sidx = s;
				neigh.push_back(nb);
			}
			const int k = std::min(knn_use, (int)neigh.size());
			std::partial_sort(neigh.begin(), neigh.begin() + k, neigh.end(),
				[](const Neigh& a, const Neigh& b) { return a.dist2 < b.dist2; });

			float dx = 0.f, dy = 0.f, wsum = 0.f;
			for (int t = 0; t < k; ++t) {
				const float w = 1.f / (neigh[t].dist2 + 1.f);
				const Seed& s = seeds[neigh[t].sidx];
				dx += w * (s.mr - s.r);
				dy += w * (s.mc - s.c);
				wsum += w;
			}
			if (wsum <= 0.f) continue;
			dx /= wsum; dy /= wsum;

			const int pred_r = static_cast<int>(r + dx + 0.5f);
			const int pred_c = static_cast<int>(c + dy + 0.5f);

			float maxScore = -1e9f;
			int best_r = pred_r, best_c = pred_c;
			for (int ii = -ser_range; ii <= ser_range; ++ii) {
				for (int jj = -ser_range; jj <= ser_range; ++jj) {
					const int rr = pred_r + ii;
					const int cc = pred_c + jj;
					if (!patchInside(img_2, rr, cc, half)) continue;
					const float score = computeScoreB(img_1, img_2, grad_1, grad_2, r, c, rr, cc, half);
					if (score > maxScore) {
						maxScore = score;
						best_r = rr;
						best_c = cc;
					}
				}
			}
			if (maxScore < threshold) continue;

			float s_weight = 0.f;
			float sub_r = 0.f, sub_c = 0.f;
			float best_sub = maxScore;
			const int w_s = 2;
			for (int ii = -w_s; ii <= w_s; ++ii) {
				for (int jj = -w_s; jj <= w_s; ++jj) {
					const int rr = best_r + ii;
					const int cc = best_c + jj;
					if (!patchInside(img_2, rr, cc, half)) continue;
					const float score = computeScoreB(img_1, img_2, grad_1, grad_2, r, c, rr, cc, half);
					if (score > best_sub) best_sub = score;
					double sim01 = 0.5 * (static_cast<double>(score) + 1.0);
					sim01 = std::max(0.0, std::min(1.0, sim01));
					double Es = std::sqrt(static_cast<double>(ii * ii + jj * jj)) / static_cast<double>(w_s);
					double E = (1.0 - sim01) + 1e-8 * Es;
					float weight = static_cast<float>(std::exp(-30.0 * E));
					s_weight += weight;
					sub_r += weight * static_cast<float>(rr);
					sub_c += weight * static_cast<float>(cc);
				}
			}
			if (s_weight > 0.f) { sub_r /= s_weight; sub_c /= s_weight; }
			else { sub_r = static_cast<float>(best_r); sub_c = static_cast<float>(best_c); }

			DenseHit hit;
			hit.r = r; hit.c = c; hit.ccd = target_right_ccd;
			hit.mr = sub_r; hit.mc = sub_c; hit.score = best_sub;
			local_hits.push_back(hit);
		}
		#pragma omp critical
		{
			hits.insert(hits.end(), local_hits.begin(), local_hits.end());
		}
	}

	for (size_t t = 0; t < hits.size(); ++t) {
		const DenseHit& h = hits[t];
		fprintf(fp_out, "%d %d %d %d %.3f %.3f %f\n",
			1, h.r, h.c, h.ccd, h.mr, h.mc, h.score);
	}
	fclose(fp_out);
	const int n_added = static_cast<int>(hits.size());
	std::cout << "densify: added " << n_added << " grid matches -> " << out_append_txt
	          << " (seeds=" << seeds.size() << ")" << std::endl;
	return n_added;
}

int ImageMatch::limit_fea(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* matchpointfile, char* featurepointxt_1, char* featurepointxt_2, char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	Mat grad_1, grad_2;
	if(g_use_affine_patch_score){
		computeGradientMag(img_1, grad_1);
		computeGradientMag(img_2, grad_2);
	}

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	//读取网格点数据
	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}

	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%d %d %d %d %d %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
			fprintf(fp11,"%d %d\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	FILE *fp2=fopen(featurepointxt_2,"r");
	if(fp2==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_2<<std::endl;
		return -1;
	}

	while(!feof(fp2)){
		fscanf(fp2,"%d ",&bj);
		if(bj==1){
			fscanf(fp2,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp2,"%d %d\n",&row,&col);
		}
		KeyPoint_x2.push_back(row);
		KeyPoint_y2.push_back(col);
	}
	fclose(fp2);


	if((fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<0 && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*0+fs_c[5]<0) || 
	   (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]>cols2 &&fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2 && fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]>cols2)){
		return 0;
	}

	//读取控制点位置
	std::vector<int> matchpoint;
    fp1=fopen(matchpointfile,"r");
	if(fp1==NULL){
		std::cout<<"Error opening match point file: "<<matchpointfile<<std::endl;
		return -1;
	}
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			matchpoint.push_back(row);matchpoint.push_back(col);
			matchpoint.push_back(imgID);matchpoint.push_back(mrow);matchpoint.push_back(mcol);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
		}
	}
	fclose(fp1);


	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;
	int count=0;

	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;
		int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];

		//用已有控制数据计算较精确预测位置
		double dx=0;double dy=0;
		int xx=int(fs_c[0]*float(KeyPoint_x1[i])+fs_c[1]*float(KeyPoint_y1[i])+fs_c[2]+0.5);
		int yy=int(fs_c[3]*float(KeyPoint_x1[i])+fs_c[4]*float(KeyPoint_y1[i])+fs_c[5]+0.5);
		const int aff_xx = xx;
		const int aff_yy = yy;

		//搜索邻域内特征点
		std::vector<int> Localmatch;	
		std::vector<double> weighttemp;
		double sum_weight=0;
		int biaoji;
		for(int ii=0;ii<matchpoint.size()/5;ii++){
			biaoji=0;
			double dist=sqrt(double((matchpoint[5*ii]-KeyPoint_x1[i])*(matchpoint[5*ii]-KeyPoint_x1[i])+(matchpoint[5*ii+1]-KeyPoint_y1[i])*(matchpoint[5*ii+1]-KeyPoint_y1[i])));
			if(dist<500 && matchpoint[5*ii+2]==RCCD_id){
				Localmatch.push_back(matchpoint[5*ii]);Localmatch.push_back(matchpoint[5*ii+1]);
				Localmatch.push_back(matchpoint[5*ii+2]);Localmatch.push_back(matchpoint[5*ii+3]);Localmatch.push_back(matchpoint[5*ii+4]);
				if(dist==0){dist=0.0000001;}
				const double sigma_dist = 250.0;
				double wt=exp(-(dist*dist)/(2.0*sigma_dist*sigma_dist));
				weighttemp.push_back(wt);
				sum_weight += wt;
			}
		}

		//计算dx,dy
		if(Localmatch.size()/5>=3){
			for(int ii=0;ii<Localmatch.size()/5;ii++){
				dx += double(Localmatch[5*ii+3]-Localmatch[5*ii+0])*weighttemp[ii];
				dy += double(Localmatch[5*ii+4]-Localmatch[5*ii+1])*weighttemp[ii];
			}
			dx = dx/sum_weight;
			dy = dy/sum_weight;
			xx=KeyPoint_x1[i]+int(dx+0.5);
			yy=KeyPoint_y1[i]+int(dy+0.5);
			LocalGridPrediction pred = selectReliableFeaturePrediction(
				KeyPoint_x1[i], KeyPoint_y1[i],
				aff_xx, aff_yy,
				xx, yy,
				rows2, cols2,
				64,
				128,
				512,
				Localmatch,
				weighttemp);
			xx = pred.row;
			yy = pred.col;
			ser_range = (pred.mode == -1) ? 64 : 128;
		}
		else{
			ser_range=128;
		}


		for(int j=0;j<KeyPoint_x2.size();j++){
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size&&
				abs(KeyPoint_x2[j]-xx)<ser_range && 
				abs(KeyPoint_y2[j]-yy)<ser_range)
			{
				s12=0;s11=0;s22=0;s1=0;s2=0;
				int grey2=img_2.data[KeyPoint_x2[j]*cols2+KeyPoint_y2[j]];
				int sum1=0;int sum2=0;
				for(int k=-w_size/2;k<=w_size/2;k++){
					for(int m=-w_size/2;m<=w_size/2;m++){
						r1=KeyPoint_x1[i]+k;r2=KeyPoint_x2[j]+k;
						c1=KeyPoint_y1[i]+m;c2=KeyPoint_y2[j]+m;
						sum1 += img_1.data[r1*cols1+c1]-grey1;
						sum2 += img_2.data[r2*cols2+c2]-grey2;
						s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
						s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
						s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
						s1  += (img_1.data[r1*cols1+c1]-grey1);
						s2  += (img_2.data[r2*cols2+c2]-grey2);
					}
				}
				s12 = s12/sum1/sum2;
				s22 = s22/sum2/sum2;
				s11 = s11/sum1/sum1;
				s1 = s1/sum1;
				s2 = s2/sum2;
				CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
				if(CC>maxCC && CC>=threshold){
					maxCC=CC;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
		vector<int>().swap(Localmatch);
		vector<double>().swap(weighttemp);
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
			//matched[i]=-1;
			count--;  
		}
	}
	fclose(fp3);
	fclose(fp4);

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);


	// int sf = 2;
	// drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	// //drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);

	delete [] matched;
	delete [] C_match;
	vector<int>().swap(KeyPoint_x1);
	vector<int>().swap(KeyPoint_y1);
	vector<int>().swap(KeyPoint_x2);
	vector<int>().swap(KeyPoint_y2);
	vector<int>().swap(KeyPoint_x11);
	vector<int>().swap(KeyPoint_y11);
	vector<int>().swap(KeyPoint_x22);
	vector<int>().swap(KeyPoint_y22);

	return 0;
}
int ImageMatch::limit_grid(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	Mat grad_1, grad_2;
	if(g_use_affine_patch_score){
		computeGradientMag(img_1, grad_1);
		computeGradientMag(img_2, grad_2);
	}

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%d %d %d %d %d %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
			fprintf(fp11,"%d %d\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matchedx = new int[KeyPoint_x1.size()];
	memset(matchedx,-1,sizeof(int)*KeyPoint_x1.size());
	int * matchedy = new int[KeyPoint_x1.size()];
	memset(matchedy,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-10;
	float loc_th = 100;
	if((fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]<-loc_th && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<-loc_th && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]<-loc_th && fs_c[3]*0+fs_c[4]*0+fs_c[5]<-loc_th) || 
	   (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2+loc_th && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]>cols2+loc_th &&fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2+loc_th && fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]>cols2+loc_th)){
		return 0;
	}/*//*/

	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;

		int xx=fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2];
		int yy=fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5];
		if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size       && //xx-ser_range>=w_size       && yy-ser_range>=w_size && 
			KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size)   //&& xx+ser_range<=rows2-w_size && yy+ser_range<=cols2-w_size)
		{
			int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
			//最小二乘匹配
			for(int ii=-ser_range;ii<ser_range;ii++){
				for(int jj=-ser_range;jj<ser_range;jj++){

					int rr=xx+ii;
					int cc=yy+jj;
					if(rr>=w_size && rr<=rows2-w_size && cc>=w_size && cc<=cols2-w_size){
						s12=0;s11=0;s22=0;s1=0;s2=0;
						int grey2=img_2.data[rr*cols2+cc];
						int sum1=0;int sum2=0;
						for(int k=-w_size/2;k<=w_size/2;k++){
							for(int m=-w_size/2;m<=w_size/2;m++){
								r1=KeyPoint_x1[i]+k;r2=rr+k;
								c1=KeyPoint_y1[i]+m;c2=cc+m;
								sum1 += img_1.data[r1*cols1+c1]-grey1;
								sum2 += img_2.data[r2*cols2+c2]-grey2;
								s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
								s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
								s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
								s1  += (img_1.data[r1*cols1+c1]-grey1);
								s2  += (img_2.data[r2*cols2+c2]-grey2);
							}
						}
						s12 = s12/float(sum1)/float(sum2);
						s22 = s22/float(sum2)/float(sum2);
						s11 = s11/float(sum1)/float(sum1);
						s1 = s1/sum1;
						s2 = s2/sum2;
						CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
						if(CC>maxCC){
							maxCC=CC;
							matchedx[i]=rr;
							matchedy[i]=cc;
							C_match[i]=maxCC;
						}
					}
				}
			}
		}

	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(matchedx[i]);
			KeyPoint_y22.push_back(matchedy[i]);
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
		}
	}
	fclose(fp3);
	fclose(fp4);


	delete [] matchedx;
	delete [] matchedy;
	delete [] C_match;
	vector<int>().swap(KeyPoint_x1);
	vector<int>().swap(KeyPoint_y1);
	vector<int>().swap(KeyPoint_x2);
	vector<int>().swap(KeyPoint_y2);
	vector<int>().swap(KeyPoint_x11);
	vector<int>().swap(KeyPoint_y11);
	vector<int>().swap(KeyPoint_x22);
	vector<int>().swap(KeyPoint_y22);

	return 0;
}
int ImageMatch::limit_grid1(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1, char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	Mat grad_1, grad_2;
	if(g_use_affine_patch_score){
		computeGradientMag(img_1, grad_1);
		computeGradientMag(img_2, grad_2);
	}

	std::vector<float> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,imgID;
	float row,col,mrow,mcol;
	float mscore;

	//读取网格点数据
	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%f %f %d %f %f %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%f %f\n",&row,&col);
			fprintf(fp11,"%f %f\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	float loc_th = 20;
	if((fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]<-loc_th && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<-loc_th && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]<-loc_th && fs_c[3]*0+fs_c[4]*0+fs_c[5]<-loc_th) || 
	   (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2+loc_th && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]>cols2+loc_th &&fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2+loc_th && fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]>cols2+loc_th)){
		return 0;
	}/*//*/


	int stride=2;
	w_size=int(double(w_size)*sqrt(double(stride)));
	int XHcount=0;

	float N=w_size*w_size;
	float CC,E;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matchedx = new int[KeyPoint_x1.size()];
	memset(matchedx,-1,sizeof(int)*KeyPoint_x1.size());
	int * matchedy = new int[KeyPoint_x1.size()];
	memset(matchedy,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float * C_match_sec = new float[KeyPoint_x1.size()];
	memset(C_match_sec,0,sizeof(float)*KeyPoint_x1.size());
	int * C_bj = new int[KeyPoint_x1.size()];
	memset(C_bj,0,sizeof(int)*KeyPoint_x1.size());
	float maxCC=-1000;
	float secCC=-1001;
	float Emin=1000;

	double lamda=0.01;
	int ser_range0=ser_range;
	int plan=0;
	//穷举搜索
	if(plan==0){
		for(int i=0;i<KeyPoint_x1.size();i++){
			int xx,yy;
			maxCC=-1000;
			secCC=-1001;
			float minE=1000;
			double dx,dy,sum_weight;
			dx=dy=sum_weight=0;

			//匹配
			if(KeyPoint_x1[i]>=w_size/2+1 && KeyPoint_y1[i]>=w_size/2+1 && KeyPoint_x1[i]<=rows1-w_size/2 && KeyPoint_y1[i]<=cols1-w_size/2)
			{
				xx=fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2];
				yy=fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5];

				//特征点约束匹配
				int grey1=0;//img_1.data[int(KeyPoint_x1[i]*cols1+KeyPoint_y1[i])];
				//最小二乘匹配
				for(int ii=-ser_range;ii<ser_range;ii+=stride){
					for(int jj=-ser_range;jj<ser_range;jj+=stride){

						int rr=xx+ii;
						int cc=yy+jj;
						if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1){
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int grey2=0;//img_2.data[rr*cols2+cc];
							int sum1=0;int sum2=0;
							for(int k=-w_size/2;k<=w_size/2;k+=1){
								for(int m=-w_size/2;m<=w_size/2;m+=1){
									r1=KeyPoint_x1[i]+k;r2=rr+k;
									c1=KeyPoint_y1[i]+m;c2=cc+m;
									sum1 += img_1.data[r1*cols1+c1]-grey1;
									sum2 += img_2.data[r2*cols2+c2]-grey2;
									s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
									s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
									s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
									s1  += (img_1.data[r1*cols1+c1]-grey1);
									s2  += (img_2.data[r2*cols2+c2]-grey2);
								}
							}
							s12 = s12/float(sum1)/float(sum2);
							s22 = s22/float(sum2)/float(sum2);
							s11 = s11/float(sum1)/float(sum1);
							s1 = s1/sum1;
							s2 = s2/sum2;
							CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
							float E = 1-CC + 0.001*int(sqrt(static_cast<double>(ii*ii+jj*jj)))/10;
							if(E<minE && abs(fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2]-rr)<256 && abs(fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5]-cc)<256){
								minE=E;
								secCC=maxCC;
								maxCC=CC;
								matchedx[i]=rr;
								matchedy[i]=cc;
								C_match[i]=maxCC;
								C_match_sec[i]=secCC;
							}
						}
					}
				}

				//细化
				xx=matchedx[i];
				yy=matchedy[i];
				maxCC=C_match[i];
				int XHbj=0;
				int xh_wsize=19;
				int N1=xh_wsize*xh_wsize;
				minE=1000;
				if(stride>1){
					for(int ii=-4*stride;ii<4*stride;ii++){
						for(int jj=-4*stride;jj<4*stride;jj++){
							int rr=xx+ii;
							int cc=yy+jj;
							if(rr>=xh_wsize && rr<=rows2-xh_wsize && cc>=xh_wsize && cc<=cols2-xh_wsize){
								s12=0;s11=0;s22=0;s1=0;s2=0;
								int grey2=0;//img_2.data[rr*cols2+cc];
								int sum1=0;int sum2=0;
								for(int k=-xh_wsize/2;k<=xh_wsize/2;k++){
									for(int m=-xh_wsize/2;m<=xh_wsize/2;m++){
										r1=KeyPoint_x1[i]+k;r2=rr+k;
										c1=KeyPoint_y1[i]+m;c2=cc+m;
										sum1 += img_1.data[r1*cols1+c1]-grey1;
										sum2 += img_2.data[r2*cols2+c2]-grey2;
										s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
										s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
										s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
										s1  += (img_1.data[r1*cols1+c1]-grey1);
										s2  += (img_2.data[r2*cols2+c2]-grey2);
									}
								}
								s12 = s12/float(sum1)/float(sum2);
								s22 = s22/float(sum2)/float(sum2);
								s11 = s11/float(sum1)/float(sum1);
								s1 = s1/sum1;
								s2 = s2/sum2;
								CC = (s12-s1*s2/N1)/sqrt((s11-s1*s1/N1)*(s22-s2*s2/N1));
								float E = 1-CC;
								if(E<minE && abs(fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2]-rr)<256 && abs(fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5]-cc)<256){
									if(XHbj==0){
										#pragma omp atomic
										XHcount++;
										XHbj=1;
									}
									minE=E;
									secCC=maxCC;
									maxCC=CC;
									matchedx[i]=rr;
									matchedy[i]=cc;
									C_match[i]=maxCC;
									C_match_sec[i]=secCC;
								}
							}
						}
					}
				}
				//细化结束
			}
		}
		//printf("细化次数为：%d\n",XHcount);
	}

	std::vector<float> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		std::cout << " --(!) Error reading feature point file " << std::endl;
		return -1;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold ){//&& C_match[i]/C_match_sec[i]>1.00
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(matchedx[i]);
			KeyPoint_y22.push_back(matchedy[i]);
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %f %f %d %f %f %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %f %f %d %f %f %f\n",bj,row,col,RCCD_id,float(matchedx[i]),float(matchedy[i]),C_match[i]);
					//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",bj,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i],C_bj[i]);
				}
				else{
					fprintf(fp4,"%d %f %f %d %f %f %f\n",bj,row,col,imgID,mrow,mcol,mscore);
					//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",bj,row,col,imgID,mrow,mcol,mscore,C_bj[i]);
				}
			}
			else{
				fscanf(fp3,"%d %f %f\n",&bj,&row,&col);
				fprintf(fp4,"%d %f %f %d %f %f %f\n",1,row,col,RCCD_id,float(matchedx[i]),float(matchedy[i]),C_match[i]);
				//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",1,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i],C_bj[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %f %f %d %f %f %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %f %f %d %f %f %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",bj,row,col,imgID,mrow,mcol,mscore,C_bj[i]);
			}
			else{
				fscanf(fp3,"%d %f %f\n",&bj,&row,&col);
				fprintf(fp4,"%d %f %f\n",bj,row,col);
			}
		}
	}
	fclose(fp3);
	fclose(fp4);


	delete [] matchedx;
	delete [] matchedy;
	delete [] C_match;
	delete [] C_match_sec;
	vector<float>().swap(KeyPoint_x1);
	vector<float>().swap(KeyPoint_y1);
	vector<float>().swap(KeyPoint_x2);
	vector<float>().swap(KeyPoint_y2);
	vector<float>().swap(KeyPoint_x11);
	vector<float>().swap(KeyPoint_y11);
	vector<float>().swap(KeyPoint_x22);
	vector<float>().swap(KeyPoint_y22);
	return 0;
}

int ImageMatch::limit_grid_global(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1, char* outpointxt_1,bool use_grid_controls){
	(void)RCCD_id;

	Mat img_1 = imread(imagepath1, 0);
	if(!img_1.data)
	{
		std::cout<< " --(!) Error reading source image " << std::endl;
		return -1;
	}

	ParsedImagePath src_info,dst_info;
	if(!parse_image_path_info(imagepath1, src_info) || !parse_image_path_info(imagepath2, dst_info)){
		std::cout<<"Error parsing image path for global limit grid."<<std::endl;
		return -1;
	}

	std::vector<int> src_mosaic,dst_mosaic;
	if(!load_mosaic_coefficients_global(src_info.root_path, src_info.seq_id, src_info.level, src_mosaic) ||
	   !load_mosaic_coefficients_global(dst_info.root_path, dst_info.seq_id, dst_info.level, dst_mosaic)){
		std::cout<<"Error loading mosaic coefficients for global limit grid."<<std::endl;
		return -1;
	}
	if(src_info.img_id<0 || src_info.img_id>=(int)src_mosaic.size()/4){
		std::cout<<"Invalid source CCD index for global limit grid."<<std::endl;
		return -1;
	}

	std::vector<GlobalControlMatch> controls;
	load_neighbor_pair_controls(src_info, dst_info, src_mosaic, dst_mosaic, use_grid_controls, controls);

	std::vector<GridPointRecord> records;
	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout<<"Error opening file: "<<featurepointxt_1<<std::endl;
		return -1;
	}
	while(true){
		GridPointRecord rec;
		rec.imgID = -1;
		rec.mrow = 0.0f;
		rec.mcol = 0.0f;
		rec.score = 0.0f;
		if(fscanf(fp1,"%d ",&rec.bj)!=1){
			break;
		}
		if(rec.bj==1){
			if(fscanf(fp1,"%f %f %d %f %f %f\n",&rec.row,&rec.col,&rec.imgID,&rec.mrow,&rec.mcol,&rec.score)!=6){
				break;
			}
		}
		else{
			if(fscanf(fp1,"%f %f\n",&rec.row,&rec.col)!=2){
				break;
			}
		}
		records.push_back(rec);
	}
	fclose(fp1);

	FILE *fp_out=fopen(outpointxt_1,"w");
	if(fp_out==NULL){
		std::cout<<"Error opening output file: "<<outpointxt_1<<std::endl;
		return -1;
	}

	const int src_row_offset = src_mosaic[src_info.img_id*4+0];
	const int src_col_offset = src_mosaic[src_info.img_id*4+2];
	const int dst_ccd_num = (int)dst_mosaic.size()/4;
	std::vector<Mat> dst_image_cache(dst_ccd_num);
	std::vector<Mat> dst_grad_cache(dst_ccd_num);
	Mat grad_1;
	computeGradientMag(img_1, grad_1);

	// 预加载对侧有效 CCD 影像/梯度，避免并行循环内反复 imread
	int preloaded = 0;
	for(int dst_ccd_id=0; dst_ccd_id<dst_ccd_num; ++dst_ccd_id){
		if(!mosaic_ccd_valid(dst_mosaic, dst_ccd_id)) continue;
		if(ensure_cached_image_and_grad(dst_info, dst_ccd_id, dst_image_cache, dst_grad_cache)){
			++preloaded;
		}
	}

	struct GridGlobalOut {
		int bj;
		float row, col;
		int imgID;
		float mrow, mcol, score;
		bool used_fallback;
	};
	const int n_rec = (int)records.size();
	std::vector<GridGlobalOut> outs(n_rec);
	int matched_count = 0;
	int fallback_used_count = 0;
	const int use_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
	const double t0 = omp_get_wtime();
	printf("[limit_grid_global] ScoreB thr=%.3f w=%d ser=%d control=%s points=%d preload=%d threads=%d\n",
		threshold, w_size, ser_range, use_grid_controls ? "grid" : "feature", n_rec, preloaded, use_threads);
	fflush(stdout);

	#pragma omp parallel for schedule(dynamic, 8) if(!omp_in_parallel()) \
		reduction(+:matched_count, fallback_used_count)
	for(int ii=0; ii<n_rec; ++ii){
		const GridPointRecord& rec = records[ii];
		const float src_mosaic_row = rec.row + src_row_offset;
		const float src_mosaic_col = rec.col + src_col_offset;

		DisparityStats stats;
		const bool has_local_stats = estimate_local_disparity(src_mosaic_row, src_mosaic_col, controls, stats);

		float pred_dst_row = 0.0f;
		float pred_dst_col = 0.0f;
		int search_row = ser_range;
		int search_col = ser_range;
		if(has_local_stats){
			pred_dst_row = stats.pred_dst_row;
			pred_dst_col = stats.pred_dst_col;
			search_row = std::max(ser_range, std::min(30, int(3.0f*stats.sigma_row + 0.5f)));
			search_col = std::max(ser_range, std::min(30, int(3.0f*stats.sigma_col + 0.5f)));
		}
		else{
			pred_dst_row = src_mosaic_row*fs_c[0] + src_mosaic_col*fs_c[1] + fs_c[2];
			pred_dst_col = src_mosaic_row*fs_c[3] + src_mosaic_col*fs_c[4] + fs_c[5];
			search_row = std::max(ser_range, 96);
			search_col = std::max(ser_range, 96);
		}

		std::vector<int> candidate_ccds;
		collect_candidate_ccds(pred_dst_row, pred_dst_col, search_row, search_col, dst_mosaic, candidate_ccds);

		CandidateSearchResult best_result;
		best_result.valid = false;
		best_result.imgID = -1;
		best_result.row = 0.0f;
		best_result.col = 0.0f;
		best_result.score = -2.0f;

		for(int jj=0; jj<(int)candidate_ccds.size(); ++jj){
			const int dst_ccd_id = candidate_ccds[jj];
			if(dst_ccd_id < 0 || dst_ccd_id >= dst_ccd_num ||
			   dst_image_cache[dst_ccd_id].data == NULL ||
			   dst_grad_cache[dst_ccd_id].data == NULL){
				continue;
			}

			CandidateSearchResult candidate_result;
			candidate_result.valid = false;
			candidate_result.imgID = dst_ccd_id;
			candidate_result.row = 0.0f;
			candidate_result.col = 0.0f;
			candidate_result.score = -2.0f;

			const int center_row = int(pred_dst_row - dst_mosaic[dst_ccd_id*4+0] + 0.5f);
			const int center_col = int(pred_dst_col - dst_mosaic[dst_ccd_id*4+2] + 0.5f);
			if(!search_best_match_in_candidate(img_1, dst_image_cache[dst_ccd_id],
				grad_1, dst_grad_cache[dst_ccd_id],
				rec.row, rec.col,
				center_row, center_col, search_row, search_col, w_size, candidate_result)){
				continue;
			}
			candidate_result.imgID = dst_ccd_id;
			if(!best_result.valid || candidate_result.score > best_result.score){
				best_result = candidate_result;
			}
		}

		bool used_fallback = false;
		if(!best_result.valid || best_result.score < threshold){
			used_fallback = true;
			fallback_used_count++;
			const float affine_pred_dst_row = src_mosaic_row*fs_c[0] + src_mosaic_col*fs_c[1] + fs_c[2];
			const float affine_pred_dst_col = src_mosaic_row*fs_c[3] + src_mosaic_col*fs_c[4] + fs_c[5];
			const int fallback_search_row = std::max(search_row, 160);
			const int fallback_search_col = std::max(search_col, 160);
			for(int dst_ccd_id=0; dst_ccd_id<dst_ccd_num; ++dst_ccd_id){
				if(!mosaic_ccd_valid(dst_mosaic, dst_ccd_id)){
					continue;
				}
				if(dst_image_cache[dst_ccd_id].data == NULL ||
				   dst_grad_cache[dst_ccd_id].data == NULL){
					continue;
				}

				CandidateSearchResult candidate_result;
				candidate_result.valid = false;
				candidate_result.imgID = dst_ccd_id;
				candidate_result.row = 0.0f;
				candidate_result.col = 0.0f;
				candidate_result.score = -2.0f;

				const int center_row = int(affine_pred_dst_row - dst_mosaic[dst_ccd_id*4+0] + 0.5f);
				const int center_col = int(affine_pred_dst_col - dst_mosaic[dst_ccd_id*4+2] + 0.5f);
				if(!search_best_match_in_candidate(img_1, dst_image_cache[dst_ccd_id],
					grad_1, dst_grad_cache[dst_ccd_id],
					rec.row, rec.col,
					center_row, center_col, fallback_search_row, fallback_search_col, w_size, candidate_result)){
					continue;
				}
				candidate_result.imgID = dst_ccd_id;
				if(!best_result.valid || candidate_result.score > best_result.score){
					best_result = candidate_result;
				}
			}
		}

		GridGlobalOut& out = outs[ii];
		out.used_fallback = used_fallback;
		if(best_result.valid && best_result.score >= threshold){
			matched_count++;
			if(rec.bj == 1 && rec.score >= best_result.score){
				out.bj = rec.bj;
				out.row = rec.row; out.col = rec.col;
				out.imgID = rec.imgID;
				out.mrow = rec.mrow; out.mcol = rec.mcol;
				out.score = rec.score;
			}
			else{
				out.bj = 1;
				out.row = rec.row; out.col = rec.col;
				out.imgID = best_result.imgID;
				out.mrow = best_result.row; out.mcol = best_result.col;
				out.score = best_result.score;
			}
		}
		else if(rec.bj == 1){
			out.bj = rec.bj;
			out.row = rec.row; out.col = rec.col;
			out.imgID = rec.imgID;
			out.mrow = rec.mrow; out.mcol = rec.mcol;
			out.score = rec.score;
		}
		else{
			out.bj = rec.bj;
			out.row = rec.row; out.col = rec.col;
			out.imgID = -1;
			out.mrow = 0.f; out.mcol = 0.f;
			out.score = 0.f;
		}
	}

	for(int ii=0; ii<n_rec; ++ii){
		const GridGlobalOut& out = outs[ii];
		if(out.bj == 1){
			fprintf(fp_out,"%d %f %f %d %f %f %f\n",
				out.bj, out.row, out.col, out.imgID, out.mrow, out.mcol, out.score);
		}
		else{
			fprintf(fp_out,"%d %f %f\n", out.bj, out.row, out.col);
		}
	}

	fclose(fp_out);
	printf("limit_grid_global %s_RED%d control=%s controls=%d points=%d matched=%d fallback=%d elapsed=%.1fs\n",
		src_info.seq_id.c_str(), src_info.img_id, use_grid_controls ? "grid" : "feature", (int)controls.size(), n_rec,
		matched_count, fallback_used_count, omp_get_wtime() - t0);
	return 0;
}

namespace {

// limit_grid4 失败原因统计，写入 /tmp/
// reason: 0=pass 1=left_border 2=window_oob 3=affine_gate 4=no_accept 5=cc_below
void write_limit_grid4_fail_log(
	char* out_path, size_t out_path_len,
	const char* imagepath1, const char* imagepath2,
	const char* matchpointfile, const char* featurepointxt,
	int rccd_id, int w_size, int ser_range, float threshold, float lambda2, int affine_max_dev,
	int n_pts, int n_pass, int n_fail,
	int fail_left_border, int fail_no_candidate, int fail_cc_below,
	int pair_no_overlap,
	int seed0_3_ok, int seed4_5_ok, int seed6p_ok,
	int seed0_3_fail, int seed4_5_fail, int seed6p_fail,
	int n_match_seeds_file, double elapsed_s,
	int fail_window_oob, int fail_affine_gate, int fail_no_accept,
	int mode_affine_fail, int mode_mean_fail, int mode_surf_fail,
	const int* reason, const int* n_seeds_arr, const int* C_bj_arr,
	const int* diag_ser, const int* diag_pred_r, const int* diag_pred_c,
	const int* diag_aff_r, const int* diag_aff_c,
	const int* diag_n_in, const int* diag_n_aok, const int* diag_n_arej,
	const float* diag_best_any, const float* diag_best_aff,
	const int* KeyPoint_x1, const int* KeyPoint_y1,
	const int* matchedx, const int* matchedy, const float* C_match)
{
	time_t now = time(nullptr);
	struct tm tm_buf;
#if defined(_WIN32)
	localtime_s(&tm_buf, &now);
#else
	localtime_r(&now, &tm_buf);
#endif
	char stamp[64];
	strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_buf);
	snprintf(out_path, out_path_len,
		"/tmp/mars_limit_grid4_%s_rccd%d_pid%d.log",
		stamp, rccd_id, (int)getpid());

	FILE* fp = fopen(out_path, "w");
	if (!fp) {
		snprintf(out_path, out_path_len, "/tmp/mars_limit_grid4_rccd%d_pid%d.log",
			rccd_id, (int)getpid());
		fp = fopen(out_path, "w");
	}
	if (!fp) return;

	fprintf(fp, "limit_grid4 fail-stats\n");
	fprintf(fp, "timestamp=%s\n", stamp);
	fprintf(fp, "pid=%d\n", (int)getpid());
	fprintf(fp, "left_image=%s\n", imagepath1 ? imagepath1 : "");
	fprintf(fp, "right_image=%s\n", imagepath2 ? imagepath2 : "");
	fprintf(fp, "match_seeds=%s\n", matchpointfile ? matchpointfile : "");
	fprintf(fp, "grid_points=%s\n", featurepointxt ? featurepointxt : "");
	fprintf(fp, "right_ccd=%d\n", rccd_id);
	fprintf(fp, "w_size=%d ser_range=%d threshold=%.6f lambda2=%.6f affine_max_dev=%d\n",
		w_size, ser_range, threshold, lambda2, affine_max_dev);
	fprintf(fp, "match_seed_records=%d\n", n_match_seeds_file);
	fprintf(fp, "elapsed_s=%.3f\n", elapsed_s);
	fprintf(fp, "\n");
	fprintf(fp, "points_total=%d\n", n_pts);
	fprintf(fp, "points_pass=%d\n", n_pass);
	fprintf(fp, "points_fail=%d\n", n_fail);
	fprintf(fp, "\n");
	fprintf(fp, "# failure reasons (exclusive, searched points)\n");
	fprintf(fp, "fail_pair_affine_no_overlap=%d\n", pair_no_overlap ? n_pts : 0);
	fprintf(fp, "fail_left_border=%d\n", fail_left_border);
	fprintf(fp, "fail_window_oob=%d\n", fail_window_oob);
	fprintf(fp, "fail_affine_gate=%d\n", fail_affine_gate);
	fprintf(fp, "fail_no_accept=%d\n", fail_no_accept);
	fprintf(fp, "fail_cc_below_threshold=%d\n", fail_cc_below);
	fprintf(fp, "fail_no_candidate=%d\n", fail_no_candidate);
	fprintf(fp, "\n");
	fprintf(fp, "# fail by predict mode (C_bj): 0=affine  -1=mean4-5  1=surface>=6\n");
	fprintf(fp, "fail_mode_affine_Cbj0=%d\n", mode_affine_fail);
	fprintf(fp, "fail_mode_mean_Cbj-1=%d\n", mode_mean_fail);
	fprintf(fp, "fail_mode_surface_Cbj1=%d\n", mode_surf_fail);
	fprintf(fp, "\n");
	fprintf(fp, "# seed-count buckets among searched points (excl. left_border / pair-skip)\n");
	fprintf(fp, "seed_0_3_pass=%d fail=%d\n", seed0_3_ok, seed0_3_fail);
	fprintf(fp, "seed_4_5_pass=%d fail=%d\n", seed4_5_ok, seed4_5_fail);
	fprintf(fp, "seed_ge6_pass=%d fail=%d\n", seed6p_ok, seed6p_fail);
	fprintf(fp, "\n");
	fprintf(fp, "notes=\n");
	fprintf(fp, "  left_border: left point too close to image edge for window\n");
	fprintf(fp, "  window_oob: search window had 0 in-bounds right pixels (pred+/-ser outside image)\n");
	fprintf(fp, "  affine_gate: had in-bounds pixels but ALL rejected by affine_max_dev\n");
	fprintf(fp, "  no_accept: some pixels passed affine gate but score/E never updated best\n");
	fprintf(fp, "  cc_below_threshold: best candidate found but CC < threshold  [param: cc_threshold]\n");
	fprintf(fp, "  no_candidate: matchedx<0 (= window_oob + affine_gate + no_accept)\n");
	fprintf(fp, "  pair_affine_no_overlap: whole L-R CCD pair skipped by affine test\n");
	fprintf(fp, "  seeds: feature matches within dist<500 and same right CCD\n");
	fprintf(fp, "  ser_range actual: C_bj==1 uses cfg ser_range; C_bj==-1/0 hardcode 128\n");

	int xt[6][3] = {{0}};
	if (reason && n_seeds_arr) {
		for (int i = 0; i < n_pts; ++i) {
			int r = reason[i];
			if (r < 0 || r > 5) continue;
			int ns = n_seeds_arr[i];
			int b = (ns < 0) ? 0 : (ns <= 3 ? 0 : (ns <= 5 ? 1 : 2));
			xt[r][b]++;
		}
	}
	fprintf(fp, "\n# crosstab reason x seed_bucket (0=0-3,1=4-5,2=ge6)\n");
	const char* rname[6] = {"pass","left_border","window_oob","affine_gate","no_accept","cc_below"};
	for (int r = 0; r < 6; ++r) {
		fprintf(fp, "xt_%s=%d %d %d\n", rname[r], xt[r][0], xt[r][1], xt[r][2]);
	}
	fclose(fp);

	char csv_path[360];
	snprintf(csv_path, sizeof(csv_path),
		"/tmp/mars_limit_grid4_%s_rccd%d_pid%d_detail.csv",
		stamp, rccd_id, (int)getpid());
	int n_csv = 0;
	if (reason && !pair_no_overlap && diag_pred_r && matchedy) {
		FILE* csv = fopen(csv_path, "w");
		if (csv) {
			fprintf(csv,
				"reason,seed_n,C_bj,ser,left_r,left_c,pred_r,pred_c,aff_r,aff_c,"
				"pred_off_aff_r,pred_off_aff_c,n_in,n_aff_ok,n_aff_rej,"
				"best_cc_any,best_cc_aff,matched_r,matched_c,CC\n");
			const int max_per_reason = 800;
			int taken[6] = {0};
			for (int i = 0; i < n_pts; ++i) {
				int r = reason[i];
				if (r <= 0 || r > 5) continue;
				if (taken[r] >= max_per_reason) continue;
				++taken[r];
				++n_csv;
				const int pr = diag_pred_r[i];
				const int pc = diag_pred_c[i];
				const int ar = diag_aff_r[i];
				const int ac = diag_aff_c[i];
				fprintf(csv,
					"%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%d,%d,%.6f\n",
					rname[r], n_seeds_arr[i], C_bj_arr[i], diag_ser[i],
					KeyPoint_x1[i], KeyPoint_y1[i],
					pr, pc, ar, ac, pr - ar, pc - ac,
					diag_n_in[i], diag_n_aok[i], diag_n_arej[i],
					diag_best_any[i], diag_best_aff[i],
					matchedx[i], matchedy[i], C_match[i]);
			}
			fclose(csv);
		}
	} else {
		csv_path[0] = '\0';
	}

	FILE* sum = fopen("/tmp/mars_limit_grid4_summary.log", "a");
	if (sum) {
		fprintf(sum,
			"%s pid=%d rccd=%d total=%d pass=%d fail=%d border=%d "
			"oob=%d aff_gate=%d no_acc=%d cc_low=%d no_cand=%d pair_skip=%d "
			"thr=%.3f adev=%d ser=%d left=%s\n",
			stamp, (int)getpid(), rccd_id, n_pts, n_pass, n_fail,
			fail_left_border, fail_window_oob, fail_affine_gate, fail_no_accept,
			fail_cc_below, fail_no_candidate,
			pair_no_overlap ? 1 : 0, threshold, affine_max_dev, ser_range,
			imagepath1 ? imagepath1 : "");
		fclose(sum);
	}

	FILE* rsum = fopen("/tmp/mars_limit_grid4_reason_summary.log", "a");
	if (rsum) {
		fprintf(rsum,
			"%s pid=%d rccd=%d pass=%d border=%d window_oob=%d affine_gate=%d no_accept=%d cc_below=%d "
			"mode_aff=%d mode_mean=%d mode_surf=%d csv_samples=%d detail=%s\n",
			stamp, (int)getpid(), rccd_id, n_pass,
			fail_left_border, fail_window_oob, fail_affine_gate, fail_no_accept, fail_cc_below,
			mode_affine_fail, mode_mean_fail, mode_surf_fail, n_csv,
			csv_path[0] ? csv_path : "none");
		fclose(rsum);
	}
}

}  // namespace

int ImageMatch::limit_grid4(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* matchpointfile, char* featurepointxt_1, char* outpointxt_1, float lambda2, int affine_max_dev, bool affine_pred_as_match){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	Mat grad_1, grad_2;
	if(g_use_affine_patch_score){
		computeGradientMag(img_1, grad_1);
		computeGradientMag(img_2, grad_2);
	}

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	//读取网格点数据
	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		std::cout << " --(!) Error reading feature point file " << std::endl;
		return -1;
	}
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%d %d %d %d %d %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
			fprintf(fp11,"%d %d\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	if((fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<0 && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*0+fs_c[5]<0) || 
	   (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]>cols2 &&fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2 && fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]>cols2)){
		if (!affine_pred_as_match) {
		char log_path[320];
		write_limit_grid4_fail_log(
			log_path, sizeof(log_path),
			imagepath1, imagepath2, matchpointfile, featurepointxt_1,
			RCCD_id, w_size, ser_range, threshold, lambda2, affine_max_dev,
			(int)KeyPoint_x1.size(), 0, (int)KeyPoint_x1.size(),
			0, 0, 0, 1,
			0, 0, 0, 0, 0, 0, 0, 0.0,
			0, 0, 0, 0, 0, 0,
			nullptr, nullptr, nullptr,
			nullptr, nullptr, nullptr,
			nullptr, nullptr,
			nullptr, nullptr, nullptr,
			nullptr, nullptr,
			nullptr, nullptr,
			nullptr, nullptr, nullptr);
		printf("[limit_grid4] skip pair (affine no-overlap), fail-stats log: %s\n", log_path);
		fflush(stdout);
		return 0;
		}
		printf("[limit_grid4] affine_pred_as_match: ignore pair-level no-overlap skip, check per-point\n");
		fflush(stdout);
	}

	//读取特征点位置
	std::vector<int> matchpoint;
    fp1=fopen(matchpointfile,"r");
	if(fp1==NULL){
		std::cout << " --(!) Error reading matchpoint file " << std::endl;
		return -1;
	}
	while(!feof(fp1)){
		if(fscanf(fp1,"%d ",&bj)!=1) break;
		if(bj==1){
			float frow,fcol,fmrow,fmcol;
			if(fscanf(fp1,"%f %f %d %f %f %f\n",&frow,&fcol,&imgID,&fmrow,&fmcol,&mscore)!=6) break;
			row=(int)frow; col=(int)fcol; mrow=(int)fmrow; mcol=(int)fmcol;
			matchpoint.push_back(row);matchpoint.push_back(col);
			matchpoint.push_back(imgID);matchpoint.push_back(mrow);matchpoint.push_back(mcol);
		}
		else{
			float frow,fcol;
			if(fscanf(fp1,"%f %f\n",&frow,&fcol)!=2) break;
			row=(int)frow; col=(int)fcol;
		}
	}
	fclose(fp1);

	// MI 项在能量中系数为 0，跳过互信息表构建
	const bool use_mi = false;
	(void)use_mi;

	int stride=2;
	w_size=int(double(w_size)*sqrt(double(stride)));
	int XHcount=0;

	float N=w_size*w_size;
	const int half_w = w_size / 2;
	int * matchedx = new int[KeyPoint_x1.size()];
	memset(matchedx,-1,sizeof(int)*KeyPoint_x1.size());
	int * matchedy = new int[KeyPoint_x1.size()];
	memset(matchedy,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	int * C_bj = new int[KeyPoint_x1.size()];
	memset(C_bj,0,sizeof(int)*KeyPoint_x1.size());
	char * left_border = new char[KeyPoint_x1.size()];
	memset(left_border, 0, sizeof(char) * KeyPoint_x1.size());
	int * n_seeds = new int[KeyPoint_x1.size()];
	for (size_t si = 0; si < KeyPoint_x1.size(); ++si) n_seeds[si] = -1;
	int * diag_reason = new int[KeyPoint_x1.size()];
	int * diag_ser = new int[KeyPoint_x1.size()];
	int * diag_pred_r = new int[KeyPoint_x1.size()];
	int * diag_pred_c = new int[KeyPoint_x1.size()];
	int * diag_aff_r = new int[KeyPoint_x1.size()];
	int * diag_aff_c = new int[KeyPoint_x1.size()];
	int * diag_n_in = new int[KeyPoint_x1.size()];
	int * diag_n_aok = new int[KeyPoint_x1.size()];
	int * diag_n_arej = new int[KeyPoint_x1.size()];
	float * diag_best_any = new float[KeyPoint_x1.size()];
	float * diag_best_aff = new float[KeyPoint_x1.size()];
	memset(diag_reason, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_ser, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_pred_r, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_pred_c, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_aff_r, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_aff_c, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_n_in, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_n_aok, 0, sizeof(int) * KeyPoint_x1.size());
	memset(diag_n_arej, 0, sizeof(int) * KeyPoint_x1.size());
	for (size_t si = 0; si < KeyPoint_x1.size(); ++si) {
		diag_best_any[si] = -1e30f;
		diag_best_aff[si] = -1e30f;
	}

	double lamda1 = 1;
	double lamda2 = static_cast<double>(lambda2);
	int ser_range0=ser_range;
	int plan=0;
	const int n_pts = (int)KeyPoint_x1.size();
	int done_pts = 0;
	const int progress_step = std::max(1, n_pts / 20); // ~5%
	const double t0_match = omp_get_wtime();
	printf("[limit_grid4] start RCCD=%d points=%d w=%d ser=%d thr=%.3f lambda2=%.3f affine_max_dev=%d affine_pred_as_match=%d threads=%d\n",
		RCCD_id, n_pts, w_size, ser_range0, threshold, lambda2, affine_max_dev, (int)affine_pred_as_match, omp_get_max_threads());
	fflush(stdout);

	// 诊断/直通模式：仿射预测落在当前右 CCD 影像内 → 直接记为匹配成功，预测坐标即匹配坐标
	if (affine_pred_as_match) {
		const double t0_aff = omp_get_wtime();
		int n_hit = 0, n_miss = 0, n_left_border = 0;
		char rec_path[360];
		snprintf(rec_path, sizeof(rec_path),
			"/tmp/mars_affine_pred_as_match_rccd%d_pid%d.csv", RCCD_id, (int)getpid());
		FILE* rec = fopen(rec_path, "w");
		if (rec) {
			fprintf(rec, "left_r,left_c,pred_r,pred_c,in_right,rccd\n");
		}
		const float score_ok = (threshold <= 1.0f) ? 1.0f : threshold;
		for (int i = 0; i < n_pts; ++i) {
			if (!(KeyPoint_x1[i] >= w_size/2+1 && KeyPoint_y1[i] >= w_size/2+1
				&& KeyPoint_x1[i] <= rows1-w_size/2 && KeyPoint_y1[i] <= cols1-w_size/2)) {
				left_border[i] = 1;
				++n_left_border;
				if (rec) {
					fprintf(rec, "%d,%d,nan,nan,0,%d\n", KeyPoint_x1[i], KeyPoint_y1[i], RCCD_id);
				}
				continue;
			}
			const float pr = fs_c[0]*KeyPoint_x1[i] + fs_c[1]*KeyPoint_y1[i] + fs_c[2];
			const float pc = fs_c[3]*KeyPoint_x1[i] + fs_c[4]*KeyPoint_y1[i] + fs_c[5];
			const int rr = (int)(pr + (pr >= 0.f ? 0.5f : -0.5f));
			const int cc = (int)(pc + (pc >= 0.f ? 0.5f : -0.5f));
			const bool in_right = (rr >= 0 && rr < rows2 && cc >= 0 && cc < cols2);
			if (rec) {
				fprintf(rec, "%d,%d,%.3f,%.3f,%d,%d\n",
					KeyPoint_x1[i], KeyPoint_y1[i], pr, pc, in_right ? 1 : 0, RCCD_id);
			}
			if (in_right) {
				matchedx[i] = rr;
				matchedy[i] = cc;
				C_match[i] = score_ok;
				++n_hit;
			} else {
				++n_miss;
			}
		}
		if (rec) fclose(rec);
		printf("[limit_grid4] affine_pred_as_match RCCD=%d hit=%d miss=%d left_border=%d elapsed=%.2fs record=%s\n",
			RCCD_id, n_hit, n_miss, n_left_border, omp_get_wtime() - t0_aff, rec_path);
		fflush(stdout);

		FILE* sum = fopen("/tmp/mars_affine_pred_as_match_summary.log", "a");
		if (sum) {
			fprintf(sum,
				"pid=%d rccd=%d total=%d hit=%d miss=%d left_border=%d left=%s right=%s record=%s\n",
				(int)getpid(), RCCD_id, n_pts, n_hit, n_miss, n_left_border,
				imagepath1 ? imagepath1 : "", imagepath2 ? imagepath2 : "", rec_path);
			fclose(sum);
		}
	}
	//穷举搜索
	else if(plan==0){
		#pragma omp parallel for schedule(dynamic, 10)
		for(int i=0;i<n_pts;i++){
			int xx,yy;
			float maxCC=-1000;
			float Emin=1000;
			float CC,E;
			float s12,s11,s22,s1,s2;
			int r1,r2,c1,c2;
			int ser_range=ser_range0;
			double dx,dy,sum_weight;
			dx=dy=sum_weight=0;

			//匹配
			if(KeyPoint_x1[i]>=w_size/2+1 && KeyPoint_y1[i]>=w_size/2+1 && KeyPoint_x1[i]<=rows1-w_size/2 && KeyPoint_y1[i]<=cols1-w_size/2
			  //&&KeyPoint_x1[i]>=11000 && KeyPoint_x1[i]<=12000
			  //&& KeyPoint_y1[i]>=1300
			   )
			{
				const float aff_r_f = fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2];
				const float aff_c_f = fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5];
				xx=aff_r_f;
				yy=aff_c_f;
				const int aff_r0 = xx;
				const int aff_c0 = yy;
				int n_in = 0, n_aok = 0, n_arej = 0;
				float best_any = -1e30f, best_aff = -1e30f;
				int pred_r_save = xx, pred_c_save = yy;
				int ser_used = ser_range0;
				bool current_pair_candidate = true;

				//搜索邻域内特征点
				std::vector<int> Localmatch;	
				std::vector<double> weighttemp;
				std::vector<double> grey_L;
				std::vector<double> grey_R;

				//遍历搜索
				int biaoji;
				double m_dI=0;
				for(int ii=0;ii<matchpoint.size()/5;ii++){
					biaoji=0;
					double dist=sqrt(double((matchpoint[5*ii]-KeyPoint_x1[i])*(matchpoint[5*ii]-KeyPoint_x1[i])+(matchpoint[5*ii+1]-KeyPoint_y1[i])*(matchpoint[5*ii+1]-KeyPoint_y1[i])));
					if(dist<500 && matchpoint[5*ii+2]==RCCD_id){
						Localmatch.push_back(matchpoint[5*ii]);Localmatch.push_back(matchpoint[5*ii+1]);
						Localmatch.push_back(matchpoint[5*ii+2]);Localmatch.push_back(matchpoint[5*ii+3]);Localmatch.push_back(matchpoint[5*ii+4]);
						if(dist==0){dist=0.0000001;}
						const double sigma_dist = 250.0;
						double wt=exp(-(dist*dist)/(2.0*sigma_dist*sigma_dist));
						weighttemp.push_back(wt);
						sum_weight += wt;
						m_dI += wt*(img_1.data[matchpoint[5*ii]*cols1+matchpoint[5*ii+1]] - img_2.data[matchpoint[5*ii+3]*cols2+matchpoint[5*ii+4]]);
						grey_L.push_back(img_1.data[matchpoint[5*ii]*cols1+matchpoint[5*ii+1]]);
						grey_R.push_back(img_2.data[matchpoint[5*ii+3]*cols2+matchpoint[5*ii+4]]);
					}
				}
				n_seeds[i] = (int)(Localmatch.size() / 5);
				m_dI=m_dI/sum_weight;

				//计算dx,dy
				if(Localmatch.size()>3*5 && Localmatch.size()<6*5){
					for(int ii=0;ii<Localmatch.size()/5;ii++){
						dx += double(Localmatch[5*ii+3]-Localmatch[5*ii+0])*weighttemp[ii];
						dy += double(Localmatch[5*ii+4]-Localmatch[5*ii+1])*weighttemp[ii];
						//sum_weight += weighttemp[ii];
					}
					dx = dx/sum_weight;
					dy = dy/sum_weight;
					xx=KeyPoint_x1[i]+int(dx+0.5);
					yy=KeyPoint_y1[i]+int(dy+0.5);
					C_bj[i]=-1;
					current_pair_candidate = shouldSearchCurrentRccd(
						aff_r0, aff_c0, xx, yy, rows2, cols2, 128);
				}
				else if(Localmatch.size()>=6*5){
					int n = int(Localmatch.size()/5);
					//加权均值
					dx=0;dy=0;
					/*for(int ii=0;ii<Localmatch.size()/5;ii++){
						dx += double(Localmatch[5*ii+3]-Localmatch[5*ii+0])*weighttemp[ii];
						dy += double(Localmatch[5*ii+4]-Localmatch[5*ii+1])*weighttemp[ii];
						//sum_weight += weighttemp[ii];
					}
					dx = dx/sum_weight;
					dy = dy/sum_weight;//*/

					//移动曲面
					float *A = new float[n*6];
					float *Lx = new float[n];
					float *Ly = new float[n];
					float *P = new float[n*n];
					memset(A,0,sizeof(float)*n*6);
					memset(Lx,0,sizeof(float)*n);
					memset(Ly,0,sizeof(float)*n);
					memset(P,0,sizeof(float)*n*n);
					for(int j=0;j<n;j++){
						A[j*6+0]=float((Localmatch[5*j+0]-KeyPoint_x1[i])*(Localmatch[5*j+0]-KeyPoint_x1[i]));
						A[j*6+1]=float((Localmatch[5*j+1]-KeyPoint_y1[i])*(Localmatch[5*j+0]-KeyPoint_x1[i]));
						A[j*6+2]=float((Localmatch[5*j+1]-KeyPoint_y1[i])*(Localmatch[5*j+1]-KeyPoint_y1[i]));
						A[j*6+3]=float(Localmatch[5*j+0]-KeyPoint_x1[i]);
						A[j*6+4]=float(Localmatch[5*j+1]-KeyPoint_y1[i]);
						A[j*6+5]=1.0;

						Lx[j]=float(Localmatch[5*j+3]-Localmatch[5*j+0]);
						Ly[j]=float(Localmatch[5*j+4]-Localmatch[5*j+1]);
						P[j*n+j]=float(weighttemp[j]);
					}
					MatrixXf A_ = (Map<MatrixXf>(A,6,n)).transpose(); //列优先
					MatrixXf Lx_ = Map<MatrixXf>(Lx,n,1);
					MatrixXf Ly_ = Map<MatrixXf>(Ly,n,1);
					MatrixXf P_ = Map<MatrixXf>(P,n,n);
					VectorXf x = (A_.transpose()*P_*A_).inverse()*A_.transpose()*P_*Lx_;
					VectorXf y = (A_.transpose()*P_*A_).inverse()*A_.transpose()*P_*Ly_;
					delete[] A;delete[] Lx;delete[] Ly;delete[] P;

					dx=x(5);
					dy=y(5);//*/

					//局部仿射
					//拟合关系式：x2=a0x+a1y+a2;y2=b0x+b1y+b2; V=Ax-L
					/*float *A = new float[2*n*6];
					float *L = new float[2*n*1];
					float *P = new float[2*n*2*n];
					memset(A,0,sizeof(float)*2*n*6);
					memset(L,0,sizeof(float)*2*n*1);
					memset(P,0,sizeof(float)*2*n*2*n);
					for(int j=0;j<n;j++){
						A[(2*j)*6+0]=Localmatch[5*j+0];
						A[(2*j)*6+1]=Localmatch[5*j+1];
						A[(2*j)*6+2]=1;
						A[(2*j+1)*6+3]=Localmatch[5*j+0];
						A[(2*j+1)*6+4]=Localmatch[5*j+1];
						A[(2*j+1)*6+5]=1;

						L[2*j]=Localmatch[5*j+3];
						L[2*j+1]=Localmatch[5*j+4];
						P[(2*j)*2*n+2*j]=float(weighttemp[j]);
						P[(2*j+1)*2*n+2*j+1]=float(weighttemp[j]);
					}

					MatrixXf A_ = (Map<MatrixXf>(A,6,2*n)).transpose();
					MatrixXf L_ = Map<MatrixXf>(L,2*n,1);
					MatrixXf P_ = Map<MatrixXf>(P,2*n,2*n);
					VectorXf x = (A_.transpose()*P_*A_).inverse()*A_.transpose()*P_*L_;

					//权迭代一次剔除大粗差
					VectorXf v = A_*x-L_;
					for(int j=0;j<n;j++){
						P[(2*j)*2*n+2*j]=float(1/(abs(v(2*j))+0.00001));
						P[(2*j+1)*2*n+2*j+1]=float(1/(abs(v(2*j+1))+0.00001));
					}
					P_ = Map<MatrixXf>(P,2*n,2*n);
					x = (A_.transpose()*P_*A_).inverse()*A_.transpose()*P_*L_;
					
					delete [] A;delete [] L;delete [] P;

					xx=x(0)*KeyPoint_x1[i]+x(1)*KeyPoint_y1[i]+x(2);
					yy=x(3)*KeyPoint_x1[i]+x(4)*KeyPoint_y1[i]+x(5);
					dx=xx-KeyPoint_x1[i];
					dy=yy-KeyPoint_y1[i];//*/

					//计算匹配点
					xx=KeyPoint_x1[i]+int(dx+0.5);
					yy=KeyPoint_y1[i]+int(dy+0.5);
					C_bj[i]=1;
					current_pair_candidate = shouldSearchCurrentRccd(
						aff_r0, aff_c0, xx, yy, rows2, cols2, ser_range0);
					if (current_pair_candidate) {
						LocalGridPrediction pred = selectReliableGridPrediction(
							KeyPoint_x1[i], KeyPoint_y1[i],
							aff_r0, aff_c0,
							xx, yy,
							rows2, cols2,
							ser_range0,
							affine_max_dev,
							Localmatch,
							weighttemp);
						xx = pred.row;
						yy = pred.col;
						C_bj[i] = pred.mode;
					}
				}
				else {
					current_pair_candidate = searchWindowTouchesImage(
						xx, yy, rows2, cols2, 128);
				}

				if(current_pair_candidate && C_bj[i]==1){
					ser_range=ser_range0;
					//特征点约束匹配
					float sum1_L=0.f, s11_L=0.f, s1_L=0.f;
					accumLeftPatchSums(img_1, KeyPoint_x1[i], KeyPoint_y1[i], half_w, sum1_L, s11_L, s1_L);
					//最小二乘匹配
					for(int ii=-ser_range;ii<ser_range;ii+=stride){
						for(int jj=-ser_range;jj<ser_range;jj+=stride){

							int rr=xx+ii;
							int cc=yy+jj;
							if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1){
								CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
									KeyPoint_x1[i], KeyPoint_y1[i], rr, cc, half_w,
									sum1_L, s11_L, s1_L, N, fs_c);

								//计算平滑损失
								const double Es = sqrt(double(ii*ii+jj*jj))/double(ser_range);

								//Es = abs(jj)+0.5*abs(ii);
								E = lamda1*(1-CC) + lamda2*Es;

								//E = 1-CC + lamda*(m_dI-abs(grey1-grey2));
								{
									++n_in;
									const bool aok = (std::fabs(aff_r_f - rr)<affine_max_dev
										&& std::fabs(aff_c_f - cc)<affine_max_dev);
									if (!aok) ++n_arej; else ++n_aok;
									if (CC > best_any) best_any = CC;
									if (aok && CC > best_aff) best_aff = CC;
								}
								if(E<Emin && std::fabs(aff_r_f - rr)<affine_max_dev && std::fabs(aff_c_f - cc)<affine_max_dev){
									maxCC=CC;
									Emin = E;
									matchedx[i]=rr;
									matchedy[i]=cc;
									C_match[i]=maxCC;
								}
							}
						}
					}
					//细化
					pred_r_save = xx;
					pred_c_save = yy;
					ser_used = ser_range;
					xx=matchedx[i];
					yy=matchedy[i];
					maxCC=C_match[i];
					int XHbj=0;
					int xh_wsize=19;
					int N1=xh_wsize*xh_wsize;
					const int half_xh = xh_wsize / 2;
					float sum1_Lh=0.f, s11_Lh=0.f, s1_Lh=0.f;
					accumLeftPatchSums(img_1, KeyPoint_x1[i], KeyPoint_y1[i], half_xh, sum1_Lh, s11_Lh, s1_Lh);
					int ser_range1=4*stride;
					if(stride>1){
						for(int ii=-ser_range1;ii<ser_range1;ii++){
							for(int jj=-ser_range1;jj<ser_range1;jj++){
								int rr=xx+ii;
								int cc=yy+jj;
								if(rr>=xh_wsize && rr<=rows2-xh_wsize && cc>=xh_wsize && cc<=cols2-xh_wsize){
									CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
										KeyPoint_x1[i], KeyPoint_y1[i], rr, cc, half_xh,
										sum1_Lh, s11_Lh, s1_Lh, static_cast<float>(N1), fs_c);

									//计算平滑损失
									const double Es = sqrt(double(ii*ii+jj*jj))/double(ser_range1);
									//Es = abs(jj)+0.5*abs(ii);
									E = lamda1*(1-CC) + lamda2*Es;

									//E = 1-CC + lamda*(m_dI-abs(grey1-grey2));
									if(E<Emin && std::fabs(aff_r_f - rr)<affine_max_dev && std::fabs(aff_c_f - cc)<affine_max_dev){
										if(XHbj==0){
											#pragma omp atomic
											XHcount++;
											XHbj=1;
										}
										maxCC=CC;
										Emin = E;
										matchedx[i]=rr;
										matchedy[i]=cc;
										C_match[i]=maxCC;
									}
								}
							}
						}
					}
					//细化结束
					//matchedx[i]=KeyPoint_x1[i]+int(dx+0.5);
					//matchedy[i]=KeyPoint_y1[i]+int(dy+0.5);
					//C_match[i]=0.8;
				}
				else if(current_pair_candidate && C_bj[i]==-1){
					ser_range=128;
					//特征点约束匹配
					float sum1_L=0.f, s11_L=0.f, s1_L=0.f;
					accumLeftPatchSums(img_1, KeyPoint_x1[i], KeyPoint_y1[i], half_w, sum1_L, s11_L, s1_L);
					//最小二乘匹配
					for(int ii=-ser_range;ii<ser_range;ii+=stride){
						for(int jj=-ser_range;jj<ser_range;jj+=stride){

							int rr=xx+ii;
							int cc=yy+jj;
							if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1){
								CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
									KeyPoint_x1[i], KeyPoint_y1[i], rr, cc, half_w,
									sum1_L, s11_L, s1_L, N, fs_c);

								//计算平滑损失
								const double Es = sqrt(double(ii*ii+jj*jj))/double(ser_range);
								//Es = abs(jj)+0.5*abs(ii);
								E = lamda1*(1-CC) + lamda2*Es;
								//E = 1-CC + lamda*(m_dI-abs(grey1-grey2));
								{
									++n_in;
									const bool aok = (std::fabs(aff_r_f - rr)<affine_max_dev
										&& std::fabs(aff_c_f - cc)<affine_max_dev);
									if (!aok) ++n_arej; else ++n_aok;
									if (CC > best_any) best_any = CC;
									if (aok && CC > best_aff) best_aff = CC;
								}
								if(E<Emin && std::fabs(aff_r_f - rr)<affine_max_dev && std::fabs(aff_c_f - cc)<affine_max_dev){
									maxCC=CC;
									Emin = E;
									matchedx[i]=rr;
									matchedy[i]=cc;
									C_match[i]=maxCC;
								}
							}
						}
					}
					//细化
					pred_r_save = xx;
					pred_c_save = yy;
					ser_used = ser_range;
					xx=matchedx[i];
					yy=matchedy[i];
					maxCC=C_match[i];
					int XHbj=0;
					int xh_wsize=19;
					int N1=xh_wsize*xh_wsize;
					const int half_xh = xh_wsize / 2;
					float sum1_Lh=0.f, s11_Lh=0.f, s1_Lh=0.f;
					accumLeftPatchSums(img_1, KeyPoint_x1[i], KeyPoint_y1[i], half_xh, sum1_Lh, s11_Lh, s1_Lh);
					int ser_range1=4*stride;
					if(stride>1){
						for(int ii=-4*stride;ii<4*stride;ii++){
							for(int jj=-4*stride;jj<4*stride;jj++){
								int rr=xx+ii;
								int cc=yy+jj;
								if(rr>=xh_wsize && rr<=rows2-xh_wsize && cc>=xh_wsize && cc<=cols2-xh_wsize){
									CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
										KeyPoint_x1[i], KeyPoint_y1[i], rr, cc, half_xh,
										sum1_Lh, s11_Lh, s1_Lh, static_cast<float>(N1), fs_c);

									//计算平滑损失
									const double Es = sqrt(double(ii*ii+jj*jj))/double(ser_range1);
									//Es = abs(jj)+0.5*abs(ii);
									E = lamda1*(1-CC) + lamda2*Es;

									//E = 1-CC + lamda*(m_dI-abs(grey1-grey2));
									if(E<Emin && std::fabs(aff_r_f - rr)<affine_max_dev && std::fabs(aff_c_f - cc)<affine_max_dev){
										if(XHbj==0){
											#pragma omp atomic
											XHcount++;
											XHbj=1;
										}
										maxCC=CC;
										Emin = E;
										matchedx[i]=rr;
										matchedy[i]=cc;
										C_match[i]=maxCC;
									}
								}
							}
						}
					}
					//细化结束（保留 4–5 邻域种子约束下的搜索结果，不再强制清零）
				}
				else if(current_pair_candidate && C_bj[i]==0){
					ser_range=128;
					//特征点约束匹配
					float sum1_L=0.f, s11_L=0.f, s1_L=0.f;
					accumLeftPatchSums(img_1, KeyPoint_x1[i], KeyPoint_y1[i], half_w, sum1_L, s11_L, s1_L);
					//最小二乘匹配
					for(int ii=-ser_range;ii<ser_range;ii+=stride){
						for(int jj=-ser_range;jj<ser_range;jj+=stride){

							int rr=xx+ii;
							int cc=yy+jj;
							if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1){
								CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
									KeyPoint_x1[i], KeyPoint_y1[i], rr, cc, half_w,
									sum1_L, s11_L, s1_L, N, fs_c);

									//E = 1-CC + lamda*(m_dI-abs(grey1-grey2));

								{
									++n_in;
									const bool aok = (std::fabs(aff_r_f - rr)<affine_max_dev
										&& std::fabs(aff_c_f - cc)<affine_max_dev);
									if (!aok) ++n_arej; else ++n_aok;
									if (CC > best_any) best_any = CC;
									if (aok && CC > best_aff) best_aff = CC;
								}
								if(CC>maxCC && std::fabs(aff_r_f - rr)<affine_max_dev && std::fabs(aff_c_f - cc)<affine_max_dev){
									maxCC=CC;
									//Emin = E;
									matchedx[i]=rr;
									matchedy[i]=cc;
									C_match[i]=maxCC;
								}
							}
						}
					}
					//细化
					pred_r_save = xx;
					pred_c_save = yy;
					ser_used = ser_range;
					xx=matchedx[i];
					yy=matchedy[i];
					maxCC=C_match[i];
					int XHbj=0;
					int xh_wsize=19;
					int N1=xh_wsize*xh_wsize;
					const int half_xh = xh_wsize / 2;
					float sum1_Lh=0.f, s11_Lh=0.f, s1_Lh=0.f;
					accumLeftPatchSums(img_1, KeyPoint_x1[i], KeyPoint_y1[i], half_xh, sum1_Lh, s11_Lh, s1_Lh);
					if(stride>1){
						for(int ii=-4*stride;ii<4*stride;ii++){
							for(int jj=-4*stride;jj<4*stride;jj++){
								int rr=xx+ii;
								int cc=yy+jj;
								if(rr>=xh_wsize && rr<=rows2-xh_wsize && cc>=xh_wsize && cc<=cols2-xh_wsize){
									CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
										KeyPoint_x1[i], KeyPoint_y1[i], rr, cc, half_xh,
										sum1_Lh, s11_Lh, s1_Lh, static_cast<float>(N1), fs_c);

									//E = 1-CC + lamda*(m_dI-abs(grey1-grey2));
									if(CC>maxCC && std::fabs(aff_r_f - rr)<affine_max_dev && std::fabs(aff_c_f - cc)<affine_max_dev){
										if(XHbj==0){
											#pragma omp atomic
											XHcount++;
											XHbj=1;
										}
										maxCC=CC;
										//Emin = E;
										matchedx[i]=rr;
										matchedy[i]=cc;
										C_match[i]=maxCC;
									}
								}
							}
						}
					}
					//细化结束
				}
				diag_aff_r[i] = aff_r0;
				diag_aff_c[i] = aff_c0;
				diag_pred_r[i] = pred_r_save;
				diag_pred_c[i] = pred_c_save;
				diag_ser[i] = ser_used;
				diag_n_in[i] = n_in;
				diag_n_aok[i] = n_aok;
				diag_n_arej[i] = n_arej;
				diag_best_any[i] = best_any;
				diag_best_aff[i] = best_aff;
				if (C_match[i] >= threshold) {
					diag_reason[i] = 0; // pass
				} else if (matchedx[i] >= 0) {
					diag_reason[i] = 5; // cc_below
				} else if (n_in == 0) {
					diag_reason[i] = 2; // window_oob
				} else if (n_aok == 0) {
					diag_reason[i] = 3; // affine_gate
				} else {
					diag_reason[i] = 4; // no_accept
				}
				vector<int>().swap(Localmatch);
				vector<double>().swap(weighttemp);
			}
			else {
				left_border[i] = 1;
				diag_reason[i] = 1;
			}
			int cur_done;
			#pragma omp atomic capture
			cur_done = ++done_pts;
			if (cur_done % progress_step == 0 || cur_done == n_pts) {
				#pragma omp critical(limit_grid4_progress)
				{
					printf("[limit_grid4] %d/%d (%.0f%%) elapsed=%.1fs refine=%d\n",
						cur_done, n_pts, 100.0 * cur_done / std::max(1, n_pts),
						omp_get_wtime() - t0_match, XHcount);
					fflush(stdout);
				}
			}
		}
		int n_pass = 0;
		int fail_left_border = 0;
		int fail_no_candidate = 0;
		int fail_cc_below = 0;
		int fail_window_oob = 0;
		int fail_affine_gate = 0;
		int fail_no_accept = 0;
		int mode_affine_fail = 0, mode_mean_fail = 0, mode_surf_fail = 0;
		int seed0_3_ok = 0, seed4_5_ok = 0, seed6p_ok = 0;
		int seed0_3_fail = 0, seed4_5_fail = 0, seed6p_fail = 0;
		for (int ii = 0; ii < n_pts; ++ii) {
			const bool ok = (C_match[ii] >= threshold);
			if (ok) ++n_pass;
			if (left_border[ii]) {
				if (!ok) ++fail_left_border;
			} else if (!ok) {
				if (matchedx[ii] < 0) ++fail_no_candidate;
				else ++fail_cc_below;
				if (diag_reason[ii] == 2) ++fail_window_oob;
				else if (diag_reason[ii] == 3) ++fail_affine_gate;
				else if (diag_reason[ii] == 4) ++fail_no_accept;
				if (C_bj[ii] == 0) ++mode_affine_fail;
				else if (C_bj[ii] == -1) ++mode_mean_fail;
				else if (C_bj[ii] == 1) ++mode_surf_fail;
			}
			const int ns = n_seeds[ii];
			if (ns >= 0) {
				if (ok) {
					if (ns <= 3) ++seed0_3_ok;
					else if (ns <= 5) ++seed4_5_ok;
					else ++seed6p_ok;
				} else if (!left_border[ii]) {
					if (ns <= 3) ++seed0_3_fail;
					else if (ns <= 5) ++seed4_5_fail;
					else ++seed6p_fail;
				}
			}
		}
		const int n_fail = n_pts - n_pass;
		const double elapsed = omp_get_wtime() - t0_match;
		printf("[limit_grid4] done RCCD=%d pass=%d/%d fail=%d (border=%d oob=%d aff_gate=%d no_acc=%d cc_low=%d) refine=%d elapsed=%.1fs\n",
			RCCD_id, n_pass, n_pts, n_fail,
			fail_left_border, fail_window_oob, fail_affine_gate, fail_no_accept, fail_cc_below,
			XHcount, elapsed);
		fflush(stdout);

		char log_path[320];
		write_limit_grid4_fail_log(
			log_path, sizeof(log_path),
			imagepath1, imagepath2, matchpointfile, featurepointxt_1,
			RCCD_id, w_size, ser_range0, threshold, lambda2, affine_max_dev,
			n_pts, n_pass, n_fail,
			fail_left_border, fail_no_candidate, fail_cc_below, 0,
			seed0_3_ok, seed4_5_ok, seed6p_ok,
			seed0_3_fail, seed4_5_fail, seed6p_fail,
			(int)(matchpoint.size() / 5), elapsed,
			fail_window_oob, fail_affine_gate, fail_no_accept,
			mode_affine_fail, mode_mean_fail, mode_surf_fail,
			diag_reason, n_seeds, C_bj,
			diag_ser, diag_pred_r, diag_pred_c,
			diag_aff_r, diag_aff_c,
			diag_n_in, diag_n_aok, diag_n_arej,
			diag_best_any, diag_best_aff,
			KeyPoint_x1.data(), KeyPoint_y1.data(),
			matchedx, matchedy, C_match);
		printf("[limit_grid4] fail-stats log: %s  (summary: /tmp/mars_limit_grid4_reason_summary.log)\n", log_path);
		fflush(stdout);
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		printf("Error opening file: %s\n", featurepointxt_1);
		delete [] matchedx;
		delete [] matchedy;
		delete [] C_match;
		delete [] C_bj;
		delete [] left_border;
		delete [] n_seeds;
		delete [] diag_reason;
		delete [] diag_ser;
		delete [] diag_pred_r;
		delete [] diag_pred_c;
		delete [] diag_aff_r;
		delete [] diag_aff_c;
		delete [] diag_n_in;
		delete [] diag_n_aok;
		delete [] diag_n_arej;
		delete [] diag_best_any;
		delete [] diag_best_aff;
		return -1;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(matchedx[i]);
			KeyPoint_y22.push_back(matchedy[i]);
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
					//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",bj,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i],C_bj[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
					//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",bj,row,col,imgID,mrow,mcol,mscore,C_bj[i]);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
				//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",1,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i],C_bj[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				//fprintf(fp4,"%d %d %d %d %d %d %f %d\n",bj,row,col,imgID,mrow,mcol,mscore,C_bj[i]);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
		}
	}
	fclose(fp3);
	fclose(fp4);


	delete [] matchedx;
	delete [] matchedy;
	delete [] C_match;
	delete [] C_bj;
	delete [] left_border;
	delete [] n_seeds;
	delete [] diag_reason;
	delete [] diag_ser;
	delete [] diag_pred_r;
	delete [] diag_pred_c;
	delete [] diag_aff_r;
	delete [] diag_aff_c;
	delete [] diag_n_in;
	delete [] diag_n_aok;
	delete [] diag_n_arej;
	delete [] diag_best_any;
	delete [] diag_best_aff;
	vector<int>().swap(KeyPoint_x1);
	vector<int>().swap(KeyPoint_y1);
	vector<int>().swap(KeyPoint_x2);
	vector<int>().swap(KeyPoint_y2);
	vector<int>().swap(KeyPoint_x11);
	vector<int>().swap(KeyPoint_y11);
	vector<int>().swap(KeyPoint_x22);
	vector<int>().swap(KeyPoint_y22);
	vector<int>().swap(matchpoint);
	return 0;
}
int ImageMatch::limit_grid5(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1, char* outpointxt_1, float lambda2, int affine_max_dev){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;
	Mat grad_1, grad_2;
	if(g_use_affine_patch_score){
		computeGradientMag(img_1, grad_1);
		computeGradientMag(img_2, grad_2);
	}

	std::vector<float> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;
	std::vector<int> match_image;

	int bj,imgID;
	float row,col,mrow,mcol;
	float mscore;

	//printf("dddd\n");

	//读取网格点数据
	FILE *fp1=fopen(featurepointxt_1,"r");
	if(fp1==NULL){
		printf("Error opening file: %s\n", featurepointxt_1);
		return -1;
	}
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			//printf("aaaa\n");
			fscanf(fp1,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%f %f %d %f %f %f\n",row,col,imgID,mrow,mcol,mscore);
			Biaoji.push_back(bj);
			match_image.push_back(imgID);
			KeyPoint_x1.push_back(row);
			KeyPoint_y1.push_back(col);
			KeyPoint_x2.push_back(mrow);
			KeyPoint_y2.push_back(mcol);
			//printf("%f %f\n",mrow,mcol);
		}
		else{
			fscanf(fp1,"%f %f\n",&row,&col);
			fprintf(fp11,"%f %f\n",row,col);
			Biaoji.push_back(bj);
			match_image.push_back(-1);
			KeyPoint_x1.push_back(row);
			KeyPoint_y1.push_back(col);
			KeyPoint_x2.push_back(-1);
			KeyPoint_y2.push_back(-1);
		}
	}
	fclose(fp1);
	fclose(fp11);

	if((fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<0 && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*0+fs_c[5]<0) || 
	   (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*rows1+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2 && fs_c[3]*rows1+fs_c[4]*cols1+fs_c[5]>cols2)){
		   //printf("bbbb\n");
		return 0;
	}

	//printf("cccc\n");
	
	int stride=1;
	w_size=int(double(w_size)*sqrt(double(stride)));
	int XHcount=0;
	(void)XHcount;

	float N=w_size*w_size;
	const int half_w = w_size / 2;
	float * matchedx = new float[KeyPoint_x1.size()];
	memset(matchedx,0,sizeof(float)*KeyPoint_x1.size());
	float * matchedy = new float[KeyPoint_x1.size()];
	memset(matchedy,0,sizeof(float)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	int * C_bj = new int[KeyPoint_x1.size()];
	memset(C_bj,0,sizeof(int)*KeyPoint_x1.size());

	double lamda1 = 1;
	double lamda2 = static_cast<double>(lambda2);
	int ser_range0=ser_range;
	const int n_pts = (int)KeyPoint_x1.size();
	int done_pts = 0;
	const int progress_step = std::max(1, n_pts / 20);
	const double t0_match = omp_get_wtime();
	(void)affine_max_dev; // follow-up 路径不做仿射偏差门控；参数保留以与 limit_grid4 对齐
	printf("[limit_grid5] start RCCD=%d points=%d w=%d ser=%d thr=%.3f lambda2=%.3f threads=%d\n",
		RCCD_id, n_pts, w_size, ser_range0, threshold, lambda2, omp_get_max_threads());
	fflush(stdout);
	//穷举搜索（点级并行）
	#pragma omp parallel for schedule(dynamic, 16)
	for(int i=0;i<n_pts;i++){
		//匹配
		if(KeyPoint_x1[i]>=w_size/2+1 && KeyPoint_y1[i]>=w_size/2+1 && KeyPoint_x1[i]<=rows1-w_size/2 && KeyPoint_y1[i]<=cols1-w_size/2 && match_image[i]==RCCD_id && KeyPoint_x2[i]>=0)
		{
			int xx=KeyPoint_x2[i];
			int yy=KeyPoint_y2[i];
			int ser_range=ser_range0;
			float sum1_L=0.f, s11_L=0.f, s1_L=0.f;
			accumLeftPatchSums(img_1, int(KeyPoint_x1[i]), int(KeyPoint_y1[i]), half_w, sum1_L, s11_L, s1_L);
			float s_weight=0;
			for(int ii=-ser_range;ii<=ser_range;ii+=stride){
				for(int jj=-ser_range;jj<=ser_range;jj+=stride){
					int rr=xx+ii;
					int cc=yy+jj;
					if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1){
						float CC = gridPatchScore(img_1, img_2, grad_1, grad_2,
							int(KeyPoint_x1[i]), int(KeyPoint_y1[i]), rr, cc, half_w,
							sum1_L, s11_L, s1_L, N, fs_c);
						if(CC>C_match[i]){
							C_match[i]=CC;
						}
						double Es = sqrt(double(ii*ii+jj*jj));
						float E = lamda1*(1-CC) + lamda2*Es;
						float weight = exp(-3*E);
						s_weight += weight;
						matchedx[i]+=weight*float(rr);
						matchedy[i]+=weight*float(cc);
					}
				}
			}
			if(s_weight>0){
				matchedx[i]/=s_weight;
				matchedy[i]/=s_weight;
			}
			else{
				matchedx[i]=KeyPoint_x2[i];
				matchedy[i]=KeyPoint_y2[i];
			}
		}
		int cur_done;
		#pragma omp atomic capture
		cur_done = ++done_pts;
		if (cur_done % progress_step == 0 || cur_done == n_pts) {
			#pragma omp critical(limit_grid5_progress)
			{
				printf("[limit_grid5] %d/%d (%.0f%%) elapsed=%.1fs\n",
					cur_done, n_pts, 100.0 * cur_done / std::max(1, n_pts),
					omp_get_wtime() - t0_match);
				fflush(stdout);
			}
		}
	}
	{
		int n_active = 0;
		for (int ii = 0; ii < n_pts; ++ii) {
			if (match_image[ii] == RCCD_id && KeyPoint_x2[ii] >= 0) ++n_active;
		}
		printf("[limit_grid5] done RCCD=%d active=%d/%d elapsed=%.1fs\n",
			RCCD_id, n_active, n_pts, omp_get_wtime() - t0_match);
		fflush(stdout);
	}

	FILE *fp3=fopen(featurepointxt_1,"r");
	if(fp3==NULL){
		printf("特征点文件打开失败！\n");
		return 0;
	}
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(match_image[i]==RCCD_id){
			fscanf(fp3,"%d %f %f %d %f %f %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp4,"%d %f %f %d %f %f %f\n",1,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
		}
		else if(match_image[i]!=-1){
			fscanf(fp3,"%d %f %f %d %f %f %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp4,"%d %f %f %d %f %f %f\n",1,row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp3,"%d %f %f\n",&bj,&row,&col);
			fprintf(fp4,"%d %f %f\n",0,row,col);
		}
	}
	fclose(fp3);
	fclose(fp4);


	delete [] matchedx;
	delete [] matchedy;
	delete [] C_match;
	vector<float>().swap(KeyPoint_x1);
	vector<float>().swap(KeyPoint_y1);
	vector<float>().swap(KeyPoint_x2);
	vector<float>().swap(KeyPoint_y2);
	vector<int>().swap(Biaoji);
	vector<int>().swap(match_image);
	return 0;
}
int ImageMatch::limit_dense2(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* matchpointfile, char* featurepointxt_1, char* outpointxt_1,int offY,int bYsize,float* MI_table){
	GDALAllRegister();

	//判定重叠范围
	int cols1=2048;
	int cols2=2048;
	if((fs_c[3]*0+fs_c[4]*0+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<0) || (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2)){
		printf("无重叠范围！\n");
		return 0;
	}

	//////////////////////用GDAL读取匹配影像(读取影像对应需要的部分即可)/////////////////////////////////
	//定义变量
	GDALDataType Type0 = GDT_Float32;
	GDALDataset *poDataset;
	GDALRasterBand *poBand;
	int nXSize1,nYSize1,nXSize2,nYSize2,nBands;
	float *paf1,*paf2;
	uchar* data1,*data2;

	//左影像读取
	poDataset = (GDALDataset*) GDALOpen( imagepath1,GA_ReadOnly);
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepath1);
		return 0;
	}
	if(poDataset->GetRasterCount()<1){
		printf("视差图波段小于1！\n");
		return 0;
	}
	nXSize1 = poDataset->GetRasterBand(1)->GetXSize();
	nYSize1 = poDataset->GetRasterBand(1)->GetYSize();
	nBands = poDataset->GetRasterCount();
	int rows1=nYSize1;
	cols1=nXSize1;
	Type0 = poDataset->GetRasterBand(1)->GetRasterDataType();

	int offY_=offY;
	int bYsize_=bYsize;
	if(offY_-ser_range-w_size/2>=0){
		bYsize_ += ser_range+w_size/2; 
		offY_ -= ser_range+w_size/2;
	}
	else{
		bYsize_ += offY_; 
		offY_=0;
	}

	if(offY_+bYsize_+ser_range+w_size/2<rows1){
		bYsize_ += ser_range+w_size/2;
	}
	else{
		bYsize_ = rows1-offY_;
	}

	data1 = new uchar[nXSize1*bYsize_];
	poDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, offY_, nXSize1, bYsize_, 
		data1, nXSize1, bYsize_, Type0, 0, 0 );
	GDALClose(poDataset);

	//右影像读取
	poDataset = (GDALDataset*) GDALOpen( imagepath2,GA_ReadOnly);
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepath2);
		return 0;
	}
	if(poDataset->GetRasterCount()<1){
		printf("视差图波段小于1！\n");
		return 0;
	}
	nXSize1 = poDataset->GetRasterBand(1)->GetXSize();
	nYSize1 = poDataset->GetRasterBand(1)->GetYSize();
	int rows2=nYSize1;
	cols2=nXSize1;
	nBands = poDataset->GetRasterCount();
	Type0 = poDataset->GetRasterBand(1)->GetRasterDataType();

	//计算左影像分块对应右影像的分块（最小最大行）
	int minY=rows2;
	int maxY=-1;
	int CornerRy[4];
	CornerRy[0]=fs_c[0]*offY_+fs_c[1]*0+fs_c[2];CornerRy[1]=fs_c[0]*offY_+fs_c[1]*cols1+fs_c[2];
	CornerRy[2]=fs_c[0]*(offY_+bYsize_)+fs_c[1]*0+fs_c[2];CornerRy[3]=fs_c[0]*(offY_+bYsize_)+fs_c[1]*cols1+fs_c[2];

	for(int i=0;i<4;i++){
		if(CornerRy[i]<minY){
			if(CornerRy[i]-w_size/2>=0){  //放宽w_size
				minY=CornerRy[i]-w_size/2;
			}
			else{
				minY=0;
			}
		}
		if(CornerRy[i]>maxY){
			if(CornerRy[i]+w_size/2<rows2){
				maxY=CornerRy[i]+w_size/2;
			}
			else{
				maxY=rows2;
			}
		}
	}

	data2 = new uchar[nXSize1*(maxY-minY)];
	poDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, minY, nXSize1, (maxY-minY), 
		data2, nXSize1, (maxY-minY), Type0, 0, 0 );
	GDALClose(poDataset);

	////////////////////////////////////////////拷贝视差图///////////////////////////////////////////////////////////
	//原视差图读取
	poDataset = (GDALDataset*) GDALOpen( featurepointxt_1,GA_ReadOnly);
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",featurepointxt_1);
		return 0;
	}
	if(poDataset->GetRasterCount()<1){
		printf("视差图波段小于1！\n");
		return 0;
	}

	nXSize1 = poDataset->GetRasterBand(1)->GetXSize();
	nYSize1 = poDataset->GetRasterBand(1)->GetYSize();
	nBands = poDataset->GetRasterCount();

	//创建备份视差图
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(outpointxt_1, nXSize1, bYsize, nBands, GDT_Float32, ppszOptions);

	//数据写入
	for(int i=0;i<nBands;i++){
		poBand = poDataset->GetRasterBand(i+1);
		Type0 = poBand->GetRasterDataType();
		
		//读取
		paf1 = new float[nXSize1*bYsize];
		poBand->RasterIO(GF_Read, 0, 0, nXSize1, bYsize, 
			paf1, nXSize1, bYsize, Type0, 0, 0 );

		//写入
		dst->GetRasterBand(i+1)->RasterIO(GF_Write, 0, 0, nXSize1, bYsize, 
			paf1, nXSize1, bYsize, Type0, 0, 0 );

		if (dst == nullptr)
		{
			GDALClose(dst);
			printf("写入失败!");
			return 0;
		}
		delete []paf1;
	}

	GDALClose(poBand);
	GDALClose(poDataset);


	//读取控制点位置
	int bj,row,col,imgID,mrow,mcol;
	float mscore;
	std::vector<int> matchpoint;
	FILE *fp1=fopen(matchpointfile,"r");
	if(fp1==NULL){
		printf("读取匹配点文件失败！\n");
		return 0;
	}
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			matchpoint.push_back(row);matchpoint.push_back(col);
			matchpoint.push_back(imgID);matchpoint.push_back(mrow);matchpoint.push_back(mcol);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
		}
	}
	fclose(fp1);
	
	int stride=1;
	w_size=int(double(w_size)*sqrt(double(stride)));
	int XHcount=0;

	float N=w_size*w_size;
	float CC,E;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;

	int C_bj=0;

	//mingID,mrow,mcol,mscore
	float *match_info = new float[4];
	memset(match_info,0,sizeof(float)*4);

	float maxCC=-1000;
	float Emin=1000;

	double lamda=0.01;
	int plan=0;
	int countTH=0;
	//穷举搜索
	if(plan==0){
		for(int mp_i=offY;mp_i<offY+bYsize;mp_i++){
			if((mp_i-offY)%10==0){
				printf("线程%d——影像%d已完成：%d%%\n",offY/bYsize,RCCD_id,(mp_i-offY)*100/bYsize);
			}
			for(int mp_j=0;mp_j<cols1;mp_j++){
				int xx,yy;
				maxCC=-1000;
				Emin=1000;
				double dx,dy,sum_weight;
				dx=dy=sum_weight=0;
				C_bj=0;

				if(mp_i>=w_size/2+1 && mp_j>=w_size/2+1 && mp_i<=rows1-w_size/2 && mp_j<=cols1-w_size/2
					//&&i>=0 && i<=1000
					//&& j>=1300
					)
				{
					xx=fs_c[0]*mp_i+fs_c[1]*mp_j+fs_c[2];
					yy=fs_c[3]*mp_i+fs_c[4]*mp_j+fs_c[5];

					//搜索邻域内特征点
					std::vector<int> Localmatch;	
					std::vector<double> weighttemp;
					std::vector<double> grey_L;
					std::vector<double> grey_R;

					//遍历搜索
					int biaoji;
					double m_dI=0;
					for(int ii=0;ii<matchpoint.size()/5;ii++){
						biaoji=0;
						double dist=sqrt(double((matchpoint[5*ii]-mp_i)*(matchpoint[5*ii]-mp_i)+(matchpoint[5*ii+1]-mp_j)*(matchpoint[5*ii+1]-mp_j)));
						if(dist<500 && matchpoint[5*ii+2]==RCCD_id && matchpoint[5*ii]-offY>=0 && matchpoint[5*ii]-offY<bYsize){
							Localmatch.push_back(matchpoint[5*ii]);Localmatch.push_back(matchpoint[5*ii+1]);
							Localmatch.push_back(matchpoint[5*ii+2]);Localmatch.push_back(matchpoint[5*ii+3]);Localmatch.push_back(matchpoint[5*ii+4]);
							if(dist==0){dist=0.0000001;}
							const double sigma_dist = 250.0;
							double wt=exp(-(dist*dist)/(2.0*sigma_dist*sigma_dist));
							weighttemp.push_back(wt);
							sum_weight += wt;
							//m_dI += wt*(data1[(matchpoint[5*ii]-offY)*cols1+matchpoint[5*ii+1]] - data2[matchpoint[5*ii+3]*cols2+matchpoint[5*ii+4]]);
							grey_L.push_back(data1[(matchpoint[5*ii]-offY_)*cols1+matchpoint[5*ii+1]]);
							grey_R.push_back(data2[(matchpoint[5*ii+3]-minY)*cols2+matchpoint[5*ii+4]]);
						}
					}
					//m_dI=m_dI/sum_weight;

					//计算dx,dy
					if(Localmatch.size()>3*5 && Localmatch.size()<6*5){
						for(int ii=0;ii<Localmatch.size()/5;ii++){
							dx += double(Localmatch[5*ii+3]-Localmatch[5*ii+0])*weighttemp[ii];
							dy += double(Localmatch[5*ii+4]-Localmatch[5*ii+1])*weighttemp[ii];
						}
						dx = dx/sum_weight;
						dy = dy/sum_weight;
						xx=mp_i+int(dx+0.5);
						yy=mp_j+int(dy+0.5);
						C_bj=-1;
					}
					else if(Localmatch.size()>=6*5){
						int n = int(Localmatch.size()/5);
						//加权均值
						dx=0;dy=0;
						for(int ii=0;ii<Localmatch.size()/5;ii++){
							dx += double(Localmatch[5*ii+3]-Localmatch[5*ii+0])*weighttemp[ii];
							dy += double(Localmatch[5*ii+4]-Localmatch[5*ii+1])*weighttemp[ii];
						}
						dx = dx/sum_weight;
						dy = dy/sum_weight;

						//计算匹配点
						xx=mp_i+int(dx+0.5);
						yy=mp_j+int(dy+0.5);
						C_bj=1;
					}

					if(C_bj==1 || C_bj==-1 ||C_bj==0){
						ser_range=16;
						//特征点约束匹配
						int grey1=data1[(mp_i-offY_)*cols1+mp_j];
						//最小二乘匹配
						for(int ii=-ser_range;ii<ser_range;ii+=stride){
							for(int jj=-ser_range;jj<ser_range;jj+=stride){

								int rr=xx+ii;
								int cc=yy+jj;
								if(rr-minY>=w_size/2+1 && rr-minY<=(maxY-minY)-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1){
									int grey2=data2[(rr-minY)*cols2+cc];
									//计算相关系数
									s12=0;s11=0;s22=0;s1=0;s2=0;
									int sum1=0;int sum2=0;
									for(int k=-w_size/2;k<=w_size/2;k+=1){
										for(int m=-w_size/2;m<=w_size/2;m+=1){
											r1=mp_i-offY_+k;r2=rr-minY+k;
											c1=mp_j+m;c2=cc+m;
											sum1 += data1[r1*cols1+c1]-grey1;
											sum2 += data2[r2*cols2+c2]-grey2;
											s12 += (data1[r1*cols1+c1]-grey1)*(data2[r2*cols2+c2]-grey2);
											s22 += (data2[r2*cols2+c2]-grey2)*(data2[r2*cols2+c2]-grey2);
											s11 += (data1[r1*cols1+c1]-grey1)*(data1[r1*cols1+c1]-grey1);
											s1  += (data1[r1*cols1+c1]-grey1);
											s2  += (data2[r2*cols2+c2]-grey2);
										}
									}
									s12 = s12/float(sum1)/float(sum2);
									s22 = s22/float(sum2)/float(sum2);
									s11 = s11/float(sum1)/float(sum1);
									s1 = s1/sum1;
									s2 = s2/sum2;
									CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));

									//计算平滑损失
									double Es=0;
									for(int k=0;k<grey_L.size();k++){
										Es += weighttemp[k]*double(abs((grey_L[k]-grey1)-(grey_R[k]-grey2)))/sum_weight;
									}
									E = 1-CC + lamda*Es +lamda*(1-MI_table[grey1*255+grey2]);

									if(E<Emin && abs(fs_c[0]*mp_i+fs_c[1]*mp_j+fs_c[2]-rr)<256 && abs(fs_c[3]*mp_i+fs_c[4]*mp_j+fs_c[5]-cc)<256){
										maxCC=CC;
										Emin = E;

										match_info[0]=RCCD_id;
										match_info[1]=rr-mp_i;
										match_info[2]=cc-mp_j;
										match_info[3]=maxCC;
									}
								}
							}
						}
					}

					vector<int>().swap(Localmatch);
					std::vector<double>().swap(weighttemp);
					std::vector<double>().swap(grey_L);
					std::vector<double>().swap(grey_R);

					float maxCC0 = 0;
					dst->GetRasterBand(4)->RasterIO(GF_Read, mp_j, mp_i-offY, 1, 1, 
						&maxCC0, 1, 1, Type0, 0, 0 );

					if(maxCC>=threshold && maxCC>=maxCC0){
						countTH++;
						for(int ii=0;ii<nBands;ii++){
							float temp = match_info[ii];
							//写入:mingID,mrow,mcol,mscore
							dst->GetRasterBand(ii+1)->RasterIO(GF_Write, mp_j, mp_i-offY, 1, 1, 
								&temp, 1, 1, Type0, 0, 0 );

							if (dst == nullptr)
							{
								printf("写入失败!");
								return 0;
							}
						}
					}
				}
				//一个点结束
			}
		}
	}
	//printf("应该输出：%d\n",countTH);

	GDALClose(dst);  //关闭写入影像

	//更新密集匹配文件
	int rem=remove(featurepointxt_1);
	int ren=rename(outpointxt_1,featurepointxt_1);

	//delete[] MI_table;
	delete[] data1;delete[] data2;
	matchpoint.clear();
	std::vector<int>().swap(matchpoint);
	return 0;
}
int ImageMatch::intra_CCD_match(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size,int CCD_id,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<float> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;

	std::vector<float> matchedx,matchedy;
	std::vector<float> C_match;
	
	int count=-1;
	//int batch_size = 48;
	for(int i=0;(i+1)*batch_size<=rows1;i++){
		for(int j=0;(j+1)*batch_size<=48;j++){
			//寻找梯度响应最大的点
			int maxG=-1000;
			int r,c;
			for(int ii=i*batch_size+1;ii<(i+1)*batch_size-1;ii++){
				for(int jj=j*batch_size+1;jj<(j+1)*batch_size-1;jj++){
					int Grad=abs(img_1.data[(ii-1)*cols1+jj]-img_1.data[(ii+1)*cols1+jj])+abs(img_1.data[(ii)*cols1+jj-1]-img_1.data[(ii)*cols1+jj]+1);
					if(Grad>maxG){
						maxG=Grad;
						r=ii;
						c=jj+cols1-48;
					}
				}
			}
			//int r=i*batch_size+batch_size/2;
			//int c=j*batch_size+batch_size/2+cols1-48;

			if(r>w_size/2 && c>w_size/2 && r<rows1-w_size && c<cols1-w_size){
				KeyPoint_x1.push_back(r);KeyPoint_y1.push_back(c);
				int grey1=0;//img_1.data[r*cols1+c];
				count++;
				float maxCC=-1000;

				matchedx.push_back(float(r));matchedy.push_back(float(c));
				C_match.push_back(-1);

				//最小二乘匹配
				for(int ii=-ser_range;ii<ser_range;ii++){
					for(int jj=-ser_range;jj<ser_range;jj++){
						int rr=r+ii;
						int cc=c-cols1+48+jj;
						if(rr>w_size && rr<rows2-w_size && cc>w_size && cc<cols2-w_size){
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int grey2=0;//img_2.data[rr*cols2+cc];
							int sum1=0;int sum2=0;
							for(int k=-w_size/2;k<=w_size/2;k++){
								for(int m=-w_size/2;m<=w_size/2;m++){
									r1=r+k;r2=rr+k;
									c1=c+m;c2=cc+m;
									sum1 += img_1.data[r1*cols1+c1]-grey1;
									sum2 += img_2.data[r2*cols2+c2]-grey2;
									s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
									s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
									s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
									s1  += (img_1.data[r1*cols1+c1]-grey1);
									s2  += (img_2.data[r2*cols2+c2]-grey2);
								}
							}
							s12 = s12/float(sum1)/float(sum2);
							s22 = s22/float(sum2)/float(sum2);
							s11 = s11/float(sum1)/float(sum1);
							s1 = s1/sum1;
							s2 = s2/sum2;
							CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
							if(CC>maxCC){
								maxCC=CC;
								matchedx[count]=rr;
								matchedy[count]=cc;
								C_match[count]=maxCC;
							}
						}

					}
				}

				grey1=img_1.data[int(KeyPoint_x1[count]*cols1+KeyPoint_y1[count])];

				//子像素
				float s_weight=0;
				int s_r=matchedx[count];
				int s_c=matchedy[count];
				matchedx[count]=0;
				matchedy[count]=0;
				float sigma=0;
				int w_s=2;
				for(int ii=-w_s;ii<=w_s;ii+=1){
					for(int jj=-w_s;jj<=w_s;jj+=1){
						int rr=s_r+ii;
						int cc=s_c+jj;
						if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1 ){
							int grey2=img_2.data[rr*cols2+cc];
							//计算相关系数
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int sum1=0;int sum2=0;
							for(int k=-w_size/2;k<=w_size/2;k+=1){
								for(int m=-w_size/2;m<=w_size/2;m+=1){
									//grey1=0;
									//grey2=0;
									r1=KeyPoint_x1[count]+k;r2=rr+k;
									c1=KeyPoint_y1[count]+m;c2=cc+m;
									//printf("%d %d %d %d\n",r1,c1,r2,c2);
									sum1 += img_1.data[r1*cols1+c1]-grey1;
									sum2 += img_2.data[r2*cols2+c2]-grey2;
									s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
									s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
									s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
									s1  += (img_1.data[r1*cols1+c1]-grey1);
									s2  += (img_2.data[r2*cols2+c2]-grey2);
								}
							}
							s12 = s12/float(sum1)/float(sum2);
							s22 = s22/float(sum2)/float(sum2);
							s11 = s11/float(sum1)/float(sum1);
							s1 = s1/sum1;
							s2 = s2/sum2;
							CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));

							if(CC>C_match[count]){
								C_match[count]=CC;
							}

							grey1=img_1.data[int(KeyPoint_x1[count]*cols1+KeyPoint_y1[count])];
							grey2=img_2.data[rr*cols2+cc];

							//计算平滑损失
							double Es = sqrt(double(ii*ii+jj*jj))/w_s;

							double E = 1*(1-CC) + 0.0001*double(Es);

							float weight = exp(-30*E);
							s_weight += weight;
							matchedx[count]+=weight*float(rr);
							matchedy[count]+=weight*float(cc);
						}
					}
				}
				if(s_weight>0){
					matchedx[count]/=s_weight;
					matchedy[count]/=s_weight;
					//cout<<s_weight<<" "<<matchedx[count]<<" "<<matchedy[count]<<endl;
				}
				else{
					matchedx[count]=float(s_r);
					matchedy[count]=float(s_c);
				}//*/

			}

		}
	}

	FILE *fp4;
	if(CCD_id==0){
		fp4=fopen(outpointxt_1,"w");
	}
	else{
		fp4=fopen(outpointxt_1,"w");
	}
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			fprintf(fp4,"%d %f %f %d %f %f %f\n",CCD_id,KeyPoint_x1[i],KeyPoint_y1[i],CCD_id+1,matchedx[i],matchedy[i],C_match[i]);
		}
	}
	fclose(fp4);


	vector<float>().swap(matchedx);
	vector<float>().swap(matchedy);
	vector<float>().swap(C_match);

	vector<float>().swap(KeyPoint_x1);
	vector<float>().swap(KeyPoint_y1);

	return 0;
}
int ImageMatch::intra_CCD_match(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size_r,int batch_size_c,int CCD_id,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<float> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;

	std::vector<float> matchedx,matchedy;
	std::vector<float> C_match;
	
	int count=-1;
	//int batch_size = 48;
	for(int i=0;(i+1)*batch_size_r<=rows1;i++){
		for(int j=0;(j+1)*batch_size_c<=48;j++){
			//寻找梯度响应最大的点
			int maxG=-1000;
			int r,c;
			//for(int ii=i*batch_size_r+1;ii<(i+1)*batch_size_r-1;ii++){
			//	for(int jj=j*batch_size_c+1;jj<(j+1)*batch_size_c-1;jj++){
			//		int jj_=cols1-48+jj;
			//		int Grad=abs(img_1.data[(ii-1)*cols1+jj_]-img_1.data[(ii+1)*cols1+jj_])+abs(img_1.data[(ii)*cols1+jj_-1]-img_1.data[(ii)*cols1+jj_]+1);
			//		if(Grad>maxG){
			//			maxG=Grad;
			//			r=ii;
			//			c=jj_;
			//		}
			//	}
			//}
			r=i*batch_size_r+batch_size_r/2;
			c=j*batch_size_c+batch_size_c/2+cols1-48;

			if(r>w_size/2 && c>w_size/2 && r<rows1-w_size && c<cols1-w_size){
				KeyPoint_x1.push_back(r);KeyPoint_y1.push_back(c);
				int grey1=0;//img_1.data[r*cols1+c];
				count++;
				float maxCC=-1000;

				matchedx.push_back(float(r));matchedy.push_back(float(c));
				C_match.push_back(-1);

				//最小二乘匹配
				for(int ii=-ser_range;ii<ser_range;ii++){
					for(int jj=-ser_range;jj<ser_range;jj++){
						int rr=r+ii;
						int cc=c-cols1+48+jj;
						if(rr>w_size && rr<rows2-w_size && cc>w_size && cc<cols2-w_size){
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int grey2=0;//img_2.data[rr*cols2+cc];
							int sum1=0;int sum2=0;
							for(int k=-w_size/2;k<=w_size/2;k++){
								for(int m=-w_size/2;m<=w_size/2;m++){
									r1=r+k;r2=rr+k;
									c1=c+m;c2=cc+m;
									sum1 += img_1.data[r1*cols1+c1]-grey1;
									sum2 += img_2.data[r2*cols2+c2]-grey2;
									s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
									s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
									s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
									s1  += (img_1.data[r1*cols1+c1]-grey1);
									s2  += (img_2.data[r2*cols2+c2]-grey2);
								}
							}
							s12 = s12/float(sum1)/float(sum2);
							s22 = s22/float(sum2)/float(sum2);
							s11 = s11/float(sum1)/float(sum1);
							s1 = s1/sum1;
							s2 = s2/sum2;
							CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
							if(CC>maxCC){
								maxCC=CC;
								matchedx[count]=rr;
								matchedy[count]=cc;
								C_match[count]=maxCC;
							}
						}

					}
				}

				grey1=img_1.data[int(KeyPoint_x1[count]*cols1+KeyPoint_y1[count])];

				//子像素
				float s_weight=0;
				int s_r=matchedx[count];
				int s_c=matchedy[count];
				matchedx[count]=0;
				matchedy[count]=0;
				float sigma=0;
				int w_s=2;
				for(int ii=-w_s;ii<=w_s;ii+=1){
					for(int jj=-w_s;jj<=w_s;jj+=1){
						int rr=s_r+ii;
						int cc=s_c+jj;
						if(rr>=w_size/2+1 && rr<=rows2-w_size/2-1 && cc>=w_size/2+1 && cc<=cols2-w_size/2-1 ){
							int grey2=img_2.data[rr*cols2+cc];
							//计算相关系数
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int sum1=0;int sum2=0;
							for(int k=-w_size/2;k<=w_size/2;k+=1){
								for(int m=-w_size/2;m<=w_size/2;m+=1){
									//grey1=0;
									//grey2=0;
									r1=KeyPoint_x1[count]+k;r2=rr+k;
									c1=KeyPoint_y1[count]+m;c2=cc+m;
									//printf("%d %d %d %d\n",r1,c1,r2,c2);
									sum1 += img_1.data[r1*cols1+c1]-grey1;
									sum2 += img_2.data[r2*cols2+c2]-grey2;
									s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
									s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
									s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
									s1  += (img_1.data[r1*cols1+c1]-grey1);
									s2  += (img_2.data[r2*cols2+c2]-grey2);
								}
							}
							s12 = s12/float(sum1)/float(sum2);
							s22 = s22/float(sum2)/float(sum2);
							s11 = s11/float(sum1)/float(sum1);
							s1 = s1/sum1;
							s2 = s2/sum2;
							CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));

							if(CC>C_match[count]){
								C_match[count]=CC;
							}

							grey1=img_1.data[int(KeyPoint_x1[count]*cols1+KeyPoint_y1[count])];
							grey2=img_2.data[rr*cols2+cc];

							//计算平滑损失
							double Es = sqrt(double(ii*ii+jj*jj))/w_s;

							double E = 1*(1-CC) + 0.00000001*double(Es);

							float weight = exp(-30*E);
							s_weight += weight;
							matchedx[count]+=weight*float(rr);
							matchedy[count]+=weight*float(cc);
						}
					}
				}
				if(s_weight>0){
					matchedx[count]/=s_weight;
					matchedy[count]/=s_weight;
					//cout<<s_weight<<" "<<matchedx[count]<<" "<<matchedy[count]<<endl;
				}
				else{
					matchedx[count]=float(s_r);
					matchedy[count]=float(s_c);
				}//*/

			}

		}
	}

	FILE *fp4;
	if(CCD_id==0){
		fp4=fopen(outpointxt_1,"w");
	}
	else{
		fp4=fopen(outpointxt_1,"w");
	}
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			fprintf(fp4,"%d %f %f %d %f %f %f\n",CCD_id,KeyPoint_x1[i],KeyPoint_y1[i],CCD_id+1,matchedx[i],matchedy[i],C_match[i]);
		}
	}
	fclose(fp4);

	vector<float>().swap(matchedx);
	vector<float>().swap(matchedy);
	vector<float>().swap(C_match);

	vector<float>().swap(KeyPoint_x1);
	vector<float>().swap(KeyPoint_y1);

	return 0;
}


// =========================
// 方案A：Census + Gradient
// =========================
int ImageMatch::intra_CCD_match_A(char* imagepath1, char* imagepath2,
                                  int w_size, int ser_range, float threshold,
                                  int batch_size_r, int batch_size_c,
                                  int CCD_id, char* outpointxt_1)
{
    return intra_CCD_match_core(imagepath1, imagepath2,
                                w_size, ser_range, threshold,
                                batch_size_r, batch_size_c,
                                CCD_id, outpointxt_1,
                                MatchMode::CensusGrad);
}


// ===================================
// 方案B：Gray ZNCC + Gradient ZNCC
// ===================================
int ImageMatch::intra_CCD_match_B(char* imagepath1, char* imagepath2,
                                  int w_size, int ser_range, float threshold,
                                  int batch_size_r, int batch_size_c,
                                  int CCD_id, char* outpointxt_1)
{
    return intra_CCD_match_core(imagepath1, imagepath2,
                                w_size, ser_range, threshold,
                                batch_size_r, batch_size_c,
                                CCD_id, outpointxt_1,
                                MatchMode::GrayGrad);
}


int ImageMatch::Hijitreg1(char* imagepathL, char* imagepathR, int OverlapSamples, char* outpath){
	//读入影像
	Mat img_1 = imread(imagepathL, 0);
	Mat img_2 = imread(imagepathR, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	uchar* imgData1=img_1.data;
	uchar* imgData2=img_2.data;

	//生成格网点
	int SampleRateLine = 10;
	int SampleRateSample = 4;

	//格网点灰度匹配
	int dl,ds;
	int w_size=5;
	int N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int dl_max=0;int ds_max=0;
	float maxCC=-1000;
	float CC;
	for(dl=-100;dl<100;dl++){
		for(ds=-5;ds<5;ds++){
			CC=0;
			int count=0;
			for(int i=0+w_size/2;i<rows1-w_size/2;i++){
				for(int j=0+w_size/2;j<OverlapSamples-w_size/2;j++){
					int ii=i-dl;int jj=j-ds;
					if(i%SampleRateLine==0 && j%SampleRateSample==0 && ii>=w_size/2 && ii<=rows2-w_size/2 && jj>=w_size/2 && jj<=cols2-w_size/2){
						s12=0;s11=0;s22=0;s1=0;s2=0;
						int sum1=0;int sum2=0;
						int grey1=imgData1[i*cols1+cols1-OverlapSamples+j];
						int grey2=imgData2[(ii)*cols2+jj];
						for(int k=-w_size/2;k<w_size/2;k++){
							for(int m=-w_size/2;m<w_size/2;m++){
								r1=i+k;r2=ii+k;
								c1=cols1-OverlapSamples+j+m;c2=jj+m;
								sum1 += imgData1[r1*cols1+c1]-grey1;
								sum2 += imgData2[r2*cols2+c2]-grey2;
								s12 += (imgData1[r1*cols1+c1]-grey1)*(imgData2[r2*cols2+c2]-grey2);
								s22 += (imgData2[r2*cols2+c2]-grey2)*(imgData2[r2*cols2+c2]-grey2);
								s11 += (imgData1[r1*cols1+c1]-grey1)*(imgData1[r1*cols1+c1]-grey1);
								s1  += (imgData1[r1*cols1+c1]-grey1);
								s2  += (imgData2[r2*cols2+c2]-grey2);
							}
						}
						if(sum1==0) sum1=1;
						if(sum2==0) sum2=1;
						s12 = s12/sum1/sum2;
						s22 = s22/sum2/sum2;
						s11 = s11/sum1/sum1;
						s1 = s1/sum1;
						s2 = s2/sum2;
						float temp=(s11-s1*s1/N)*(s22-s2*s2/N);
						if(temp<=0) temp=0.0001;
						CC += (s12-s1*s2/N)/sqrt(temp);
						count++;
					}
				}
			}
			CC=CC/count;
			if(CC>maxCC){
				maxCC=CC;
				dl_max=dl;
				ds_max=ds;
			}
		}
	}
	//最小二乘思想（dx，dy）精确对齐

	//精确的offsetX+dx和offsetY+dy

	//拼接
	Mat img;
	int beginR = 0<=dl_max ? 0:dl_max;
	int endR = rows1>rows2+dl_max ? rows1:rows2+dl_max;
	img.create(endR-beginR,cols1+cols2-OverlapSamples+ds_max,CV_8UC1);
	int rows=img.rows;
	int cols=img.cols;
	uchar* pImg=img.data;

	if(beginR==0){
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				pImg[i*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				pImg[(i+dl_max)*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	else{
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				pImg[(i-dl_max)*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				pImg[i*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	//imshow("mosaic",img);
	imwrite(outpath,img);
	//waitKey(0);
}

//外方位元素的三次多项式拟合EO(t)=EO(t0)+c1(t-t0)+c2(t-t0)^2+c3(t-t0)^3
void ImageMatch::Polynomial3_EO(char* EO_txt,double* Poly_C, char* polyCC_txt){
	FILE *fp=fopen(EO_txt,"r");
	if(fp==nullptr){
		printf("EO_txt file not found!\n");
		return;
	}
	double beginT,LR;
	fscanf(fp,"%lf %lf\n",&beginT,&LR);
	std::vector<double> EO;
	double et,Xs,Ys,Zs,phi,w,ka;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",&et,&Xs,&Ys,&Zs,&phi,&w,&ka);
		EO.push_back(et);
		EO.push_back(Xs);EO.push_back(Ys);EO.push_back(Zs);
		EO.push_back(phi);EO.push_back(w);EO.push_back(ka);
	}
	fclose(fp);

	double t,t0;
	t0=EO[0];
	int number=EO.size()/7;
	MatrixXf x;
	FILE *fp1=fopen(polyCC_txt,"w");
	//fprintf(fp1,"%lf %lf\n",beginT,LR);
	for(int j=1;j<7;j++){
		MatrixXf A=MatrixXf::Zero(number,4);
		MatrixXf L=MatrixXf::Zero(number,1);
		for(int i=0;i<number;i++){
			VectorXf temp1(4);
			t=EO[i*7];
			temp1 << 1.0,(t-t0),pow((t-t0),2),pow((t-t0),3);
			A.row(i)=temp1;
			L(i)=EO[i*7+j];
		}
		x = (A.transpose()*A).inverse()*A.transpose()*L;
		Poly_C[5*(j-1)+0]=t0;
		Poly_C[5*(j-1)+1]=x(0);
		Poly_C[5*(j-1)+2]=x(1);
		Poly_C[5*(j-1)+3]=x(2);
		Poly_C[5*(j-1)+4]=x(3);
		fprintf(fp1,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",t0,x(0),x(1),x(2),x(3));
	}
	fclose(fp1);
}
//根据拟合公式求出对应行的外方位元素
void ImageMatch::Get_PolyEO(double et,double* Poly_C,float* EO){
	for(int i=0;i<6;i++){
		double t=et;
		double t0=Poly_C[5*i];
		EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*pow((t-t0),2)+Poly_C[5*i+4]*pow((t-t0),3);
	}
}
//内方位纠正;IO:x0,dx/ds,dx/dl,y0,dy/ds,dy/dl,k0,k1,k2,f
void ImageMatch::IO_correct(int sample,int BIN,int TDI,float* IO,float* xp,float* yp){
	int cols = 2048;
	float s = (float(sample)-0.5)*float(BIN)-cols/2;
	float l = float(TDI)/2-64-(float(BIN)/2-0.5);

	float x = IO[0]+IO[1]*s+IO[2]*l;
	float y = IO[3]+IO[4]*s+IO[5]*l;

	float r = sqrt(x*x+y*y);
	float dr_r = IO[6]+IO[7]*r*r+IO[8]*r*r*r*r;

	*xp = x-dr_r*x;
	*yp = y-dr_r*y;
}
void ImageMatch::IO_correct_arc(float xp,float yp,float* IO,int BIN,int TDI,int* RC){
	int cols=2048;

	float r = sqrt(xp*xp+yp*yp);
	float dr_r = IO[6]+IO[7]*r*r+IO[8]*r*r*r*r;

	float x = xp/(1-dr_r);
	float y = yp/(1-dr_r);

	float l = (IO[4]*(x-IO[0])-IO[1]*(y-IO[3]))/(IO[2]*IO[4]-IO[1]*IO[5]);
	float s = (IO[5]*(x-IO[0])-IO[2]*(y-IO[3]))/(IO[1]*IO[5]-IO[2]*IO[4]);

	int sample = int(float(s+cols/2)/float(BIN)+0.5);
	int line = l>=0 ? int(l+0.5) : int(l-0.5);

	RC[0] = line;
	RC[1] = sample;
}
//欧拉角计算旋转矩阵
void ImageMatch::Eul2R(float phi,float w,float k,float* R){
	R[0]=cos(phi)*cos(k)-sin(phi)*sin(w)*sin(k);
	R[1]=-cos(phi)*sin(k)-sin(phi)*sin(w)*cos(k);
	R[1]=-R[1];
	R[2]=-sin(phi)*cos(w);
	R[3]=cos(w)*sin(k);
	R[3]=-R[3];
	R[4]=cos(w)*cos(k);
	R[5]=-sin(w);
	R[5]=-R[5];
	R[6]=sin(phi)*cos(k)+cos(phi)*sin(w)*sin(k);
	R[7]=-sin(phi)*sin(k)+cos(phi)*sin(w)*cos(k);
	R[7]=-R[7];
	R[8]=cos(phi)*cos(w);
}


//地面匹配
//根据DEM和单张影像获取地面点坐标
//EO：Xs,Ys,Zs,phi,w,k
void ImageMatch::Get_groundtruth(char* DEMdoc, int sample, int line,int BIN, int TDI, float meanZ, float* EO, float* IO, float* GC){
	double rMars=3396190;
	//读取坐标系信息
	GDALAllRegister();
	GDALDataset *poDataset = (GDALDataset*) GDALOpen( DEMdoc,GA_ReadOnly );

	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",DEMdoc);
		return;
	}
	const char *projRef =poDataset->GetProjectionRef();

	//内定向
	float xp,yp;
	IO_correct(sample,BIN,TDI,IO,&xp,&yp);
	float * R = new float[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float f = IO[9];
	float * xi = new float[3];
	xi[0] = xp;
	xi[1] = yp;
	xi[2] = f;

	//R转换到eigen格式，通过eigen矩阵运算计算X，Y
	MatrixXf R_ = (Map<MatrixXf>(R,3,3)).transpose();
	MatrixXf xi_ = Map<MatrixXf>(xi,3,1);
	VectorXf Rx = R_*xi_;

	float Xs = EO[0];
	float Ys = EO[1];
	float Zs = EO[2];

	double rate=rMars/sqrt(Xs*Xs+Ys*Ys+Zs*Zs);
	double meanX=Xs*rate;
	double meanY=Ys*rate;
	meanZ=Zs*rate;

	float X,Y,Z;
	float dX,dY,dZ;
	dX=1000000;dY=1000000;dZ=1000000;
	Z = meanZ;

    
	//投影到球面
	double A=Rx(0)*Rx(0)+Rx(1)*Rx(1)+Rx(2)*Rx(2);
	double B=2*(Rx(0)*Xs+Rx(1)*Ys+Rx(2)*Zs);
	double C=Xs*Xs+Ys*Ys+Zs*Zs-rMars*rMars;
	double nlamda1=(-1*B+sqrt(B*B-4*A*C))/2/A;
	double nlamda2=(-1*B-sqrt(B*B-4*A*C))/2/A;
	double nlamda=1;
	
	if(nlamda1*nlamda1>nlamda2*nlamda2){
		nlamda=nlamda2;
	}
	else{
		nlamda=nlamda1;
	}

	X=nlamda*Rx(0)+Xs;
	Y=nlamda*Rx(1)+Ys;
	Z=nlamda*Rx(2)+Zs;
	
	/*
	float lamda = (Z-Zs)/Rx(2);
	X = lamda*Rx(0)+Xs;
	Y = lamda*Rx(1)+Ys;

	float dX0,dY0,dZ0;
	double N,E,H;
	float temp;
	int count = 0;
	while(count<100){
		dZ0=dZ;
		temp=Z;
		
		
		My_rec2NEH(projRef,double(X),double(Y),double(Z),&N,&E,&H);
		LoadDEM(DEMdoc, float(E), float(N), &Z);
		H=double(Z);
		double XX,YY,ZZ;
		My_NEH2rec(projRef,N,E,H,&XX,&YY,&ZZ);
		X=float(XX);Y=float(YY);Z=float(ZZ);

		dZ = abs(temp-Z);
		lamda = (Z-Zs)/Rx(2);

		dX0 = dX;dY0 = dY;
		temp = X;
		X = lamda*Rx(0)+Xs;
		dX = abs(temp-X);
		temp = Y;
		Y = lamda*Rx(1)+Ys;
		dY = abs(temp-Y);
		if(dX>=dX0 && dY>=dY0 && dZ>=dZ0){
			break;
		}
		count++;
		if(count==100){
			printf("%s\n","000000!");
		}
	}*/
	GC[0]=X;
	GC[1]=Y;
	GC[2]=Z;
}
void ImageMatch::Get_groundtruth(float* IC, float* EO, float f, float Z, float* GC){
	//计算旋转矩阵
	float * R = new float[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float * xi = new float[3];
	xi[0] = IC[0];
	xi[1] = IC[1];
	xi[2] = f;   //f取正值

	//R转换到eigen格式，通过eigen矩阵运算计算X，Y
	MatrixXf R_ = (Map<MatrixXf>(R,3,3)).transpose();
	MatrixXf xi_ = Map<MatrixXf>(xi,3,1);
	VectorXf Rx = R_*xi_;

	float Xs = EO[0];
	float Ys = EO[1];
	float Zs = EO[2];

	float X,Y;

	X = (Z-Zs)*Rx(0)/Rx(2)+Xs;
	Y = (Z-Zs)*Rx(1)/Rx(2)+Ys;
	
	GC[0]=X;
	GC[1]=Y;
	GC[2]=Z;
}
void ImageMatch::Get_groundtruth1(float xp ,float yp, float* EO, float f, float rMars, float* GC){
	//double rMars=3396190;
	//读取坐标系信息
	
	float * R = new float[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float * xi = new float[3];
	xi[0] = xp;
	xi[1] = yp;
	xi[2] = f;

	//R转换到eigen格式，通过eigen矩阵运算计算X，Y
	MatrixXf R_ = (Map<MatrixXf>(R,3,3)).transpose();
	MatrixXf xi_ = Map<MatrixXf>(xi,3,1);
	VectorXf Rx = R_*xi_;

	float Xs = EO[0];
	float Ys = EO[1];
	float Zs = EO[2];

	float X,Y,Z;

	//投影到球面
	double A=Rx(0)*Rx(0)+Rx(1)*Rx(1)+Rx(2)*Rx(2);
	double B=2*(Rx(0)*Xs+Rx(1)*Ys+Rx(2)*Zs);
	double C=Xs*Xs+Ys*Ys+Zs*Zs-rMars*rMars;
	double nlamda1=(-1*B+sqrt(B*B-4*A*C))/2/A;
	double nlamda2=(-1*B-sqrt(B*B-4*A*C))/2/A;
	double nlamda=1;
	
	if(nlamda1*nlamda1>nlamda2*nlamda2){
		nlamda=nlamda2;
	}
	else{
		nlamda=nlamda1;
	}

	X=nlamda*Rx(0)+Xs;
	Y=nlamda*Rx(1)+Ys;
	Z=nlamda*Rx(2)+Zs;
	
	GC[0]=X;
	GC[1]=Y;
	GC[2]=Z;
}

//根据地面点及外方位元素获取影像坐标
void ImageMatch::Get_imageCoor(float* EO, float* IO, float* GC, float* IC){
	float * R = new float[9];
	Eul2R(EO[3],EO[4],EO[5],R);
	MatrixXf R_inv = (Map<MatrixXf>(R,3,3)).inverse();

	float* XYZ = new float[3];
	XYZ[0] = GC[0]-EO[0];
	XYZ[1] = GC[1]-EO[1];
	XYZ[2] = GC[2]-EO[2];
	MatrixXf XYZ_ = Map<MatrixXf>(XYZ,3,1);

	VectorXf Rx = R_inv*XYZ_;

	float x0 = IO[0];
	float y0 = IO[3];
	float f = IO[9];
	float lamda = -f/Rx(2);

	IC[0] = lamda*Rx(0)+x0;
	IC[1] = lamda*Rx(1)+y0;
}
void ImageMatch::Get_imageCoor(float* GC, float* EO, float f, float* IC){
	float * R = new float[9];
	Eul2R(EO[3],EO[4],EO[5],R);
	MatrixXf R_inv = (Map<MatrixXf>(R,3,3)); //求逆用转置替代(列优先，所以不再需要转置)

	float* XYZ = new float[3];
	XYZ[0] = GC[0]-EO[0];
	XYZ[1] = GC[1]-EO[1];
	XYZ[2] = GC[2]-EO[2];
	MatrixXf XYZ_ = Map<MatrixXf>(XYZ,3,1);

	VectorXf Rx = R_inv*XYZ_;

	float lamda = f/Rx(2);   //f取正值

	IC[0] = lamda*Rx(0);
	IC[1] = lamda*Rx(1);

	delete []R;
	delete []XYZ;
}
void ImageMatch::G2I(float* GC, float* RIO, double* RPoly_C, int* Rrc, double RbeginT, double LR, int Rrows, int BIN, int TDI){
	float* IC =new float[2];
	//地面点反投至右影像（需要拟合方程）
	int diedai_count = 0;
	float* REO = new float[6];
	double et = RbeginT + int(Rrows/2)*LR*BIN;
	Get_PolyEO(et,RPoly_C,REO);
	Get_imageCoor(GC, REO, RIO[9], IC);
	IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
	while(abs(Rrc[0]-0)>=1 && diedai_count<100){
		//printf("%d\n",BIN);
		et = et+Rrc[0]*LR*BIN;
		Get_PolyEO(et,RPoly_C,REO);
		Get_imageCoor(GC, REO, RIO[9], IC);
		//printf("%f %f\n",IC[0],IC[1]);
		IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
		//printf("%d %d\n",Rrc[0],Rrc[1]);
		diedai_count++;
	}

	if(Rrc[0]==0){
		Rrc[0]=int((et-RbeginT)/LR+0.5);
	}
	if(diedai_count==100){
		printf("diedai:%d\n",diedai_count);
	}
}

//合起来，根据影像坐标获取地面坐标，再回去右影像找到大概区域进行约束匹配。
void ImageMatch::L2G2R(int* Lrc, float* LIO, float* LEO, float Z, float* RIO, double* RPoly_C, int* Rrc, double RbeginT, double LR, int Rrows, int BIN, int TDI){
	//内定向
	float xp,yp;
	IO_correct(Lrc[1],BIN,TDI,LIO,&xp,&yp);
	float *IC = new float[2];
	IC[0]=xp;
	IC[1]=yp;

	//获取地面坐标
	float* GC = new float[3];
	Get_groundtruth(IC, LEO, LIO[9], Z, GC);

	//地面点反投至右影像（需要拟合方程）
	int diedai_count = 0;
	float* REO = new float[6];
	double et = RbeginT + int(Rrows/2)*LR*BIN;
	Get_PolyEO(et,RPoly_C,REO);
	Get_imageCoor(GC, REO, RIO[9], IC);
	IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
	while(abs(Rrc[0]-0)>=1 && diedai_count<100){
		et = et+Rrc[0]*LR*BIN;
		Get_PolyEO(et,RPoly_C,REO);
		Get_imageCoor(GC, REO, RIO[9], IC);
		IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
		diedai_count++;
	}

	if(Rrc[0]==0){
		Rrc[0]=int((et-RbeginT)/LR+0.5);
	}
	else{
		Rrc[0]=-1;
	}
}
void ImageMatch::L2G2R(int* Lrc, float* LIO, double* LPoly_C, float Z, int* Rrc, float* RIO, double* RPoly_C, double LbeginT, double LLR, double RbeginT, double RLR, int Rrows, int BIN, int TDI){
	//内定向
	float xp,yp;
	IO_correct(Lrc[1],BIN,TDI,LIO,&xp,&yp);
	float *IC = new float[2];
	IC[0]=xp;
	IC[1]=yp;

	//获取地面坐标
	float* GC = new float[3];
	float* LEO = new float[6];
    double et = LbeginT + Lrc[0]*LLR*BIN;
	Get_PolyEO(et,LPoly_C,LEO);
	Get_groundtruth(IC, LEO, LIO[9], Z, GC);
	

	//地面点反投至右影像（需要拟合方程）
	int diedai_count = 0;
	float* REO = new float[6];
	et = RbeginT + int(Rrows/2)*RLR*BIN;
	Get_PolyEO(et,RPoly_C,REO);
	//printf("%f\n",RIO[9]);
	Get_imageCoor(GC, REO, RIO[9], IC);
	IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
	while(abs(Rrc[0]-0)>=1 && diedai_count<100){
		et = et+Rrc[0]*RLR*BIN;
		Get_PolyEO(et,RPoly_C,REO);
		Get_imageCoor(GC, REO, RIO[9], IC);
		IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
		diedai_count++;
	}

	if(Rrc[0]==0){
		Rrc[0]=int((et-RbeginT)/RLR+0.5);
	}
	else{
		Rrc[0]=-1;
	}
}
void ImageMatch::G_M(char* DEMdoc,int sample, int line, int BIN, int TDI, float meanZ, float* EO, float* IO, int RLines, double DLINE, double RBIN, double RTDI, double et0, double* Poly_C, float* RIO, float* IC){
	float* GC = new float[3];
	Get_groundtruth(DEMdoc, sample, line, BIN, TDI, meanZ, EO, IO, GC);

	double LR = ( 74.0 + DLINE/16.0 )/1000000;
	double et1 = et0 + LR*(BIN-TDI)/2;
	double et;
	//float* IC = new float[2];
	float* REO = new float[6];
	for(int i=0;i<RLines;i++){
		et = et1 + i*LR*BIN;
		Get_PolyEO(et,Poly_C,REO);
		Get_imageCoor(REO, RIO, GC, IC);
		if(abs(IC[1]-0)<2){
			break;
		}
	}
}

void ImageMatch::g2itest(char* filepath,char* xulie_ID1,char* xulie_ID2){
	char* FItxt = new char[80]; 
	sprintf( FItxt, "%s%s%s%s%s", "../data/result/", xulie_ID1,"_", xulie_ID2, "_FI.txt");
	char* EOfile = "../data/EO";
	char* IOtxt = "../data/IO/IO.txt";
	double* Poly_C = new double[30];

	int mark=5;

	//读取时间信息
	double et,beginET,LR;
	char* EOtxt = new char[80]; 
	sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", mark, "_0.txt" );
	FILE* fp_eo=fopen(EOtxt,"r");
	if(fp_eo==NULL){
		printf("File %s not found!\n",EOtxt);
		return;
	}
	fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
	fclose(fp_eo);

	//读取polyC
	sprintf( EOtxt, "%s", "../data/result/PSP_001777_1650_EOre.txt" );
	fp_eo=fopen(EOtxt,"r");
	if(fp_eo==NULL){
		printf("File %s not found!\n",EOtxt);
		return;
	}
	double a1,a2,a3,a4,a5;
	for(int i=0;i<6;i++){
		int ii;
		if(i<3){
			ii=i+3;
		}
		else{
			ii=i-3;
		}
		fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
		Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
		Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
	}
	fclose(fp_eo);

	//内方位元素
	FILE* fp_io = fopen(IOtxt, "r");
	if(fp_io==NULL){
		printf("File %s not found!\n",IOtxt);
		return;
	}

	float** IO = new float*[10];
	for(int i=0;i<10;i++){
		IO[i]=new float[10];
	}
	for(int j=0;j<10;j++){
		int temp;
		float temp1;
		for(int k=0;k<11;k++){
			if(k==0){
				fscanf(fp_io,"%d ",&temp);
			}
			else if(k==8||k==9){
				fscanf(fp_io,"%e ",&IO[j][k-1]);
			}
			else{
				fscanf(fp_io,"%f ",&IO[j][k-1]);
			}
		}
		fscanf(fp_io,"\n");
	}
	fclose(fp_io);

	//地面点
	float* GC = new float[3];

	//影像点
	int* RC = new int[2];

	int ground_num=0;
	double X,Y,Z,res;
	FILE *fp_fi=fopen(FItxt,"r");
	if(fp_fi==NULL){
		printf("File %s not found!\n",FItxt);
		return;
	}
	fscanf(fp_fi,"%d\n",&ground_num);
	for(int i=0;i<ground_num;i++){
		fscanf(fp_fi,"%lf %lf %lf %lf\n",&X,&Y,&Z,&res);
		GC[0]=X;GC[1]=Y;GC[2]=Z;
		G2I(GC, IO[mark], Poly_C, RC, beginET, LR, 40000, 1, 128);
		printf("%d %d\n",RC[0],RC[1]);
	}
	fclose(fp_fi);
}

void ImageMatch::gmtest(char* filepath,char* xulie_ID1,char* xulie_ID2){
	char* FItxt = new char[80]; 
	sprintf( FItxt, "%s%s%s%s%s", "../data/result/", xulie_ID1,"_", xulie_ID2, "_FI.txt");
	char* EOfile = "../data/EO";
	char* IOtxt = "../data/IO/IO.txt";
	double* Poly_C = new double[30];

	int mark=5;

	//读取时间信息
	double et,beginET,LR;
	char* EOtxt = new char[80]; 
	sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", mark, "_0.txt" );
	FILE* fp_eo=fopen(EOtxt,"r");
	if(fp_eo==NULL){
		printf("File %s not found!\n",EOtxt);
		return;
	}
	fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
	fclose(fp_eo);

	//读取polyC
	sprintf( EOtxt, "%s", "../data/result/PSP_001777_1650_EOre.txt" );
	fp_eo=fopen(EOtxt,"r");
	if(fp_eo==NULL){
		printf("File %s not found!\n",EOtxt);
		return;
	}
	double a1,a2,a3,a4,a5;
	for(int i=0;i<6;i++){
		int ii;
		if(i<3){
			ii=i+3;
		}
		else{
			ii=i-3;
		}
		fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
		Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
		Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
	}
	fclose(fp_eo);

	//内方位元素
	FILE* fp_io = fopen(IOtxt, "r");
	if(fp_io==NULL){
		printf("File %s not found!\n",IOtxt);
		return;
	}

	float** IO = new float*[10];
	for(int i=0;i<10;i++){
		IO[i]=new float[10];
	}
	for(int j=0;j<10;j++){
		int temp;
		float temp1;
		for(int k=0;k<11;k++){
			if(k==0){
				fscanf(fp_io,"%d ",&temp);
			}
			else if(k==8||k==9){
				fscanf(fp_io,"%e ",&IO[j][k-1]);
			}
			else{
				fscanf(fp_io,"%f ",&IO[j][k-1]);
			}
		}
		fscanf(fp_io,"\n");
	}
	fclose(fp_io);

	//地面点
	float* GC = new float[3];

	//影像点
	int* RC = new int[2];

	int ground_num=0;
	double X,Y,Z,res;
	FILE *fp_fi=fopen(FItxt,"r");
	if(fp_fi==NULL){
		printf("File %s not found!\n",FItxt);
		return;
	}
	fscanf(fp_fi,"%d\n",&ground_num);
	for(int i=0;i<ground_num;i++){
		fscanf(fp_fi,"%lf %lf %lf %lf\n",&X,&Y,&Z,&res);
		GC[0]=X;GC[1]=Y;GC[2]=Z;
		G2I(GC, IO[mark], Poly_C, RC, beginET, LR, 40000, 1, 128);
		printf("%d %d\n",RC[0],RC[1]);
	}
	fclose(fp_fi);
}
