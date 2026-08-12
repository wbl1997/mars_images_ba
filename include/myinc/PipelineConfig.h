#ifndef _PIPELINECONFIG_H_
#define _PIPELINECONFIG_H_

struct PipelineConfig {
    struct StepSwitches {
        bool preprocess = false;
        bool intra_ccd_mosaic = false;
        bool downsample = false;
        bool feature_mosaic = false;
        bool feature_extract = false;
        bool feature_match = false;
        bool grid_match = false;
        bool inter_ccd_match = true;
        bool ba = true;
    };

    struct MosaicParams {
        int overlap_samples = 48;
    };

    struct FeatureExtractParams {
        int channel = 6;
        int localmax_win[5] = {64, 51, 51, 31, 15};
    };

    // Step 6 特征匹配：三块 —— fenfu_match / densify_grid / iterative_refinement
    struct FeatureMatchParams {
        // ---------- 1) fenfu_match（金字塔/L0 特征匹配）----------
        // true: 运行金字塔 fenfu_match1(fs,0)；false: 跳过
        bool pyramid_open = true;
        int coarse_window_size = 13;
        int guided_window_factor = 6;
        int final_layer_window_size = 27;
        int search_range_factor = 70;
        float coarse_cc_threshold = 0.8f;
        float guided_cc_threshold = 0.7f;
        float final_layer_cc_threshold = 0.65f;
        float ransac_sigma_factor = 50.0f;
        int ransac_iterations = 200000;
        // false: 原始 0.9*ZNCC(gray)+0.1*ZNCC(grad)；true: 中心加权鲁棒 ScoreB（较慢）
        bool use_robust_scoreb = false;
        // 仅在 use_local_affine 且局部仿射场可用时，对右图 patch 做局部仿射重采样后评分
        bool use_affine_patch_score = false;
        // 分块局部仿射引导（写入 *_match.txt）
        bool use_local_affine = true;
        int local_tiles_r = 12;
        int local_tiles_c = 10;
        // 局部仿射唯一残差尺度：建场 RANSAC、局部 inlier 标记、最终 keep 共用
        float local_affine_sigma = 220.0f;
        int local_min_points = 8;
        float local_keep_score_min = 0.42f;
        // use_local_affine=false 时：相对全局 fs_c 的 keep 门限（像素）
        // keep = (√(vx²+vy²) < global_keep_radius) && (|vx| < global_keep_abs) && (|vy| < global_keep_abs)
        float global_keep_radius = 500.0f;
        float global_keep_abs = 250.0f;

        // ---------- 2) densify_grid（种子密格网成网）----------
        bool densify_grid = true;
        int densify_batch_r = 64;
        int densify_batch_c = 64;
        int densify_window_size = 15;
        int densify_search_range = 28;
        float densify_cc_threshold = 0.50f;
        int densify_knn = 8;

        // ---------- 3) iterative_refinement（Fea IR / limit_fea）----------
        int iterative_refinement_iterations = 5;
        int iterative_window_size = 19;
        // 相对种子预测位置的搜索半径（像素，行/列同）；配置值原样生效
        int iterative_search_range = 128;
        float iterative_cc_threshold = 0.85f;
        // E = λ1*(1-ScoreB) + λ2*Es；λ1 固定为 1
        float lambda2 = 0.1f;
        // 邻域种子：欧氏距离 < seed_radius 且同右 CCD；数量 < seed_min 则跳过该点
        int iterative_seed_radius = 500;
        int iterative_seed_min = 3;
    };

    struct GridMatchParams {
        int refinement_iterations = 3;
        int batch_size = 64;
        int window_size = 19;
        int search_range = 64;
        float cc_threshold = 0.7f;
        // false: 原始 ScoreB；true: 中心加权鲁棒 ScoreB（较慢）
        bool use_robust_scoreb = false;
        // true: 先按当前仿射线性项重采样右图 patch，再计算相似度（较慢）
        bool use_affine_patch_score = false;
        // false: Step7 使用全局仿射；true: 每个左 CCD 用当前+相邻 CCD 匹配点重估仿射
        bool use_neighbor_affine = false;
        // E = λ1*(1-CC) + λ2*Es；λ1 固定为 1
        float lambda2 = 0.1f;
        // 候选点相对全局仿射预测的最大像素偏差（原硬编码 256）
        int affine_max_dev = 512;
        // true: 不做相关搜索；仿射预测落在当前右 CCD 影像内则直接记为匹配成功
        bool affine_pred_as_match = false;
        int followup_window_size = 15;
        int followup_search_range = 2;
        float followup_cc_threshold = 0.7f;
        // 仅 Grid 侧 Iterative_refinement(mark!=0) 备用；Fea IR 用 feature_match.iterative_*
        int iterative_window_size = 19;
        int iterative_search_range = 128;
        float iterative_cc_threshold = 0.85f;
    };

    struct InterCcdMatchParams {
        int intra_window_size = 13;
        int intra_search_range = 60;
        int intra_batch_size_r = 10;
        int intra_batch_size_c = 48;
        float intra_cc_threshold = 0.5f;
        int cross_window_size = 15;
        int cross_search_range = 100;
        float cross_cc_threshold = 0.7f;
        // false: 原始 ScoreB；true: 中心加权鲁棒 ScoreB（较慢）
        bool use_robust_scoreb = false;
        // true: 先按当前仿射线性项重采样右图 patch，再计算相似度（较慢）
        bool use_affine_patch_score = false;
        // 跨航带局部预测控制来源: 0=feature(_match.txt), 1=grid(_grid.txt)
        int control_source = 1;
        bool run_global_refine = true;

        bool control_use_grid() const { return control_source != 0; }
        bool control_use_feature() const { return control_source == 0; }
    };

    // Step 9 光束法平差 / 前交
    struct BaParams {
        // FI 观测来源: 0=feature(_match.txt), 1=grid(_grid.txt)
        // yaml 可写 feature / grid / 0 / 1
        int fi_source = 1;
        // 是否先用特征点做 BA 更新 EO（写 *_EOre.txt），再做 FI
        bool run_feature_ba = true;
        // true: 优先 Jitter_adjustment，失败回退 Block；false: 直接 Block
        bool use_jitter = true;
        // Ceres 迭代上限
        int block_max_iterations = 100;
        int jitter_max_iterations = 50;
        int fi_max_iterations = 30;
        // CauchyLoss 尺度（像方残差）
        float cauchy_loss_ba = 1.0f;
        float cauchy_loss_fi = 0.5f;
        float function_tolerance = 1e-10f;

        // Tichu_CX(mark=0/1，及 mark=4 共用) 粗差剔除
        // 相对全局仿射残差门：sqrt(vx^2+vy^2)、|vx|、|vy|
        float tichu_affine_max_residual = 500.0f;
        float tichu_affine_max_vx = 150.0f;
        float tichu_affine_max_vy = 250.0f;
        // 局部视差邻域半径（像素）与 σ 倍数
        float tichu_local_radius = 500.0f;
        float tichu_sigma_factor = 3.0f;

        bool fi_use_grid() const { return fi_source != 0; }
        bool fi_use_feature() const { return fi_source == 0; }
    };

    char srcfilepath[512] = "";  // 原始 PDS 数据路径
    char filepath[512] = "";     // 处理后数据路径
    char xulie_ID1[128] = "";
    char xulie_ID2[128] = "";
    int rows1 = 0;
    int rows2 = 0;
    int cols = 0;
    int CCD_num = 0;
    // 处理 CCD 半开区间 [CCD_begin, CCD_end)；CCD_end<0 表示 CCD_num
    int CCD_begin = 0;
    int CCD_end = -1;

    StepSwitches steps;
    MosaicParams mosaic;
    FeatureExtractParams feature_extract;
    FeatureMatchParams feature_match;
    GridMatchParams grid_match;
    InterCcdMatchParams inter_ccd_match;
    BaParams ba;

    // 有效 CCD 起始（含）
    int ccd_begin() const {
        int b = CCD_begin;
        if (b < 0) b = 0;
        if (CCD_num > 0 && b > CCD_num) b = CCD_num;
        return b;
    }
    // 有效 CCD 结束（不含）；默认 CCD_num
    int ccd_end() const {
        int e = (CCD_end < 0) ? CCD_num : CCD_end;
        if (e < 0) e = 0;
        if (e > CCD_num) e = CCD_num;
        const int b = ccd_begin();
        if (e < b) e = b;
        return e;
    }
};

#endif
