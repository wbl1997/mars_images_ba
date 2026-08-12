#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdarg>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#include "BA.h"
#include "ImageProcess.h"
#include "Observation.h"
#include "PipelineConfig.h"

using namespace std;

// 旧接口使用 char*，封装 const_cast
#define CC(s) const_cast<char*>(s)

// ============================================================
// 工具函数
// ============================================================
static bool file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    fclose(f);
    return true;
}

static bool file_nonempty(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz > 0;
}

static bool file_exists(const string& path) {
    return file_exists(path.c_str());
}

static bool file_nonempty(const string& path) {
    return file_nonempty(path.c_str());
}

static string format_string(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap_count;
    va_copy(ap_count, ap);
    const int needed = vsnprintf(NULL, 0, fmt, ap_count);
    va_end(ap_count);
    if (needed < 0) {
        va_end(ap);
        return string();
    }

    vector<char> buf(static_cast<size_t>(needed) + 1);
    vsnprintf(buf.data(), buf.size(), fmt, ap);
    va_end(ap);
    return string(buf.data());
}

static bool ensure_dir(const char* path) {
    if (mkdir(path, 0755) == 0) return true;
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static string timestamp_string() {
    time_t now = time(NULL);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return string(buf);
}

static string elapsed_hms(double seconds) {
    if (seconds < 0) seconds = 0;
    const long total = static_cast<long>(seconds + 0.5);
    const long h = total / 3600;
    const long m = (total % 3600) / 60;
    const long s = total % 60;
    char buf[64];
    snprintf(buf, sizeof(buf), "%02ld:%02ld:%02ld", h, m, s);
    return string(buf);
}

static double wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
}

class PipelineLogger {
public:
    PipelineLogger(const PipelineConfig& cfg, const char* cfg_path)
        : start_time_(time(NULL)) {
        const bool launched_from_bin = file_exists("../data") || file_exists("../cfg.yaml");
        const string root = launched_from_bin ? "../logs" : "logs";
        ensure_dir(root.c_str());
        ensure_dir((root + "/p").c_str());
        run_dir_ = root + "/p/" + timestamp_string();
        ensure_dir(run_dir_.c_str());
        main_log_path_ = run_dir_ + "/main.log";
        main_ = fopen(main_log_path_.c_str(), "w");
        info("RUN_START cfg=%s dataset=%s<->%s ccd=[%d,%d)",
             cfg_path, cfg.xulie_ID1, cfg.xulie_ID2, cfg.ccd_begin(), cfg.ccd_end());
    }

    ~PipelineLogger() {
        info("RUN_END elapsed=%s", elapsed_hms(difftime(time(NULL), start_time_)).c_str());
        if (main_) fclose(main_);
    }

    const string& run_dir() const { return run_dir_; }
    const string& main_log_path() const { return main_log_path_; }

    void info(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        write("INFO", fmt, ap);
        va_end(ap);
    }

    void warn(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        write("WARN", fmt, ap);
        va_end(ap);
    }

private:
    void write(const char* level, const char* fmt, va_list ap) {
        char msg[2048];
        vsnprintf(msg, sizeof(msg), fmt, ap);
        const string ts = timestamp_string();
        printf("[PIPELINE][%s] %s\n", level, msg);
        fflush(stdout);
        if (main_) {
            fprintf(main_, "%s [%s] %s\n", ts.c_str(), level, msg);
            fflush(main_);
        }
    }

    FILE* main_ = NULL;
    string run_dir_;
    string main_log_path_;
    time_t start_time_;
};

class ScopedOutputRedirect {
public:
    explicit ScopedOutputRedirect(const string& path) {
        fflush(stdout);
        fflush(stderr);
        std::cout.flush();
        std::cerr.flush();
        out_fd_ = dup(fileno(stdout));
        err_fd_ = dup(fileno(stderr));
        file_ = fopen(path.c_str(), "w");
        if (file_) {
            dup2(fileno(file_), fileno(stdout));
            dup2(fileno(file_), fileno(stderr));
        }
    }

    ~ScopedOutputRedirect() {
        fflush(stdout);
        fflush(stderr);
        std::cout.flush();
        std::cerr.flush();
        if (out_fd_ >= 0) {
            dup2(out_fd_, fileno(stdout));
            close(out_fd_);
        }
        if (err_fd_ >= 0) {
            dup2(err_fd_, fileno(stderr));
            close(err_fd_);
        }
        if (file_) fclose(file_);
    }

    bool active() const { return file_ != NULL; }

private:
    int out_fd_ = -1;
    int err_fd_ = -1;
    FILE* file_ = NULL;
};

typedef bool (*StepFn)(const PipelineConfig&);

static bool run_logged_step(PipelineLogger& logger,
                            int step_no,
                            const char* key,
                            const char* title,
                            bool enabled,
                            StepFn fn,
    const PipelineConfig& cfg) {
    if (!enabled) {
        logger.info("STEP_SKIP step=%d key=%s title=\"%s\" status=skipped elapsed_sec=0.000 elapsed=00:00:00",
                    step_no, key, title);
        return true;
    }

    char detail_name[256];
    snprintf(detail_name, sizeof(detail_name), "/step%02d_%s.log", step_no, key);
    const string detail_path = logger.run_dir() + detail_name;
    logger.info("STEP_BEGIN step=%d key=%s title=\"%s\" detail=%s",
                step_no, key, title, detail_path.c_str());

    const double t0 = wall_seconds();
    bool ok = false;
    {
        ScopedOutputRedirect redirect(detail_path);
        if (!redirect.active()) {
            printf("[PIPELINE][WARN] detail log redirect failed: %s\n", detail_path.c_str());
        }
        ok = fn(cfg);
    }
    const double elapsed = wall_seconds() - t0;
    logger.info("STEP_END step=%d key=%s status=%s elapsed_sec=%.3f elapsed=%s detail=%s",
                step_no, key, ok ? "ok" : "failed", elapsed, elapsed_hms(elapsed).c_str(),
                detail_path.c_str());
    return ok;
}

static string trim_copy(const string& value) {
    size_t begin = 0;
    while (begin < value.size() && isspace(static_cast<unsigned char>(value[begin]))) {
        begin++;
    }

    size_t end = value.size();
    while (end > begin && isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(begin, end - begin);
}

static string strip_inline_comment(const string& line) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (size_t i = 0; i < line.size(); i++) {
        const char ch = line[i];
        if (ch == '\'' && !in_double_quote) {
            in_single_quote = !in_single_quote;
        } else if (ch == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
        } else if (ch == '#' && !in_single_quote && !in_double_quote) {
            return line.substr(0, i);
        }
    }
    return line;
}

static string unquote_copy(const string& value) {
    string result = trim_copy(value);
    if (result.size() >= 2) {
        const char first = result.front();
        const char last = result.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            result = result.substr(1, result.size() - 2);
        }
    }
    return result;
}

static void copy_config_string(char* dst, size_t dst_size, const string& value) {
    snprintf(dst, dst_size, "%s", unquote_copy(value).c_str());
}

static string lower_copy(string value) {
    for (size_t i = 0; i < value.size(); i++) {
        value[i] = static_cast<char>(tolower(static_cast<unsigned char>(value[i])));
    }
    return value;
}

static bool parse_bool_value(const string& value, bool* out) {
    const string v = lower_copy(unquote_copy(value));
    if (v == "true" || v == "1" || v == "yes" || v == "on") {
        *out = true;
        return true;
    }
    if (v == "false" || v == "0" || v == "no" || v == "off") {
        *out = false;
        return true;
    }
    return false;
}

static bool parse_int_value(const string& value, int* out) {
    const string v = unquote_copy(value);
    char* end = NULL;
    const long parsed = strtol(v.c_str(), &end, 10);
    if (end == v.c_str() || !trim_copy(end ? string(end) : string()).empty()) {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

static bool parse_float_value(const string& value, float* out) {
    const string v = unquote_copy(value);
    char* end = NULL;
    const float parsed = strtof(v.c_str(), &end);
    if (end == v.c_str() || !trim_copy(end ? string(end) : string()).empty()) {
        return false;
    }
    *out = parsed;
    return true;
}

static bool parse_int_array5(const string& value, int out[5]) {
    string v = trim_copy(value);
    if (v.size() >= 2 && v.front() == '[' && v.back() == ']') {
        v = v.substr(1, v.size() - 2);
    }

    size_t start = 0;
    int count = 0;
    while (start <= v.size() && count < 5) {
        const size_t comma = v.find(',', start);
        const size_t end = (comma == string::npos) ? v.size() : comma;
        int parsed = 0;
        if (!parse_int_value(v.substr(start, end - start), &parsed)) {
            return false;
        }
        out[count++] = parsed;
        if (comma == string::npos) {
            break;
        }
        start = comma + 1;
    }
    return count == 5;
}

static bool assign_config_value(PipelineConfig& cfg, const string& key,
                                const string& value, int line_no) {
    bool ok = true;

    if (key == "dataset.srcfilepath") copy_config_string(cfg.srcfilepath, sizeof(cfg.srcfilepath), value);
    else if (key == "dataset.filepath") copy_config_string(cfg.filepath, sizeof(cfg.filepath), value);
    else if (key == "dataset.xulie_ID1") copy_config_string(cfg.xulie_ID1, sizeof(cfg.xulie_ID1), value);
    else if (key == "dataset.xulie_ID2") copy_config_string(cfg.xulie_ID2, sizeof(cfg.xulie_ID2), value);
    else if (key == "dataset.rows1") ok = parse_int_value(value, &cfg.rows1);
    else if (key == "dataset.rows2") ok = parse_int_value(value, &cfg.rows2);
    else if (key == "dataset.cols") ok = parse_int_value(value, &cfg.cols);
    else if (key == "dataset.CCD_num") ok = parse_int_value(value, &cfg.CCD_num);
    else if (key == "dataset.CCD_begin") ok = parse_int_value(value, &cfg.CCD_begin);
    else if (key == "dataset.CCD_end") ok = parse_int_value(value, &cfg.CCD_end);

    else if (key == "steps.preprocess") ok = parse_bool_value(value, &cfg.steps.preprocess);
    else if (key == "steps.intra_ccd_mosaic") ok = parse_bool_value(value, &cfg.steps.intra_ccd_mosaic);
    else if (key == "steps.downsample") ok = parse_bool_value(value, &cfg.steps.downsample);
    else if (key == "steps.feature_mosaic") ok = parse_bool_value(value, &cfg.steps.feature_mosaic);
    else if (key == "steps.feature_extract") ok = parse_bool_value(value, &cfg.steps.feature_extract);
    else if (key == "steps.feature_match") ok = parse_bool_value(value, &cfg.steps.feature_match);
    else if (key == "steps.grid_match") ok = parse_bool_value(value, &cfg.steps.grid_match);
    else if (key == "steps.inter_ccd_match") ok = parse_bool_value(value, &cfg.steps.inter_ccd_match);
    else if (key == "steps.ba") ok = parse_bool_value(value, &cfg.steps.ba);

    else if (key == "mosaic.overlap_samples") ok = parse_int_value(value, &cfg.mosaic.overlap_samples);

    else if (key == "feature_extract.channel") ok = parse_int_value(value, &cfg.feature_extract.channel);
    else if (key == "feature_extract.localmax_win") ok = parse_int_array5(value, cfg.feature_extract.localmax_win);

    else if (key == "feature_match.coarse_window_size") ok = parse_int_value(value, &cfg.feature_match.coarse_window_size);
    else if (key == "feature_match.guided_window_factor") ok = parse_int_value(value, &cfg.feature_match.guided_window_factor);
    else if (key == "feature_match.final_layer_window_size") ok = parse_int_value(value, &cfg.feature_match.final_layer_window_size);
    else if (key == "feature_match.search_range_factor") ok = parse_int_value(value, &cfg.feature_match.search_range_factor);
    else if (key == "feature_match.coarse_cc_threshold") ok = parse_float_value(value, &cfg.feature_match.coarse_cc_threshold);
    else if (key == "feature_match.guided_cc_threshold") ok = parse_float_value(value, &cfg.feature_match.guided_cc_threshold);
    else if (key == "feature_match.final_layer_cc_threshold") ok = parse_float_value(value, &cfg.feature_match.final_layer_cc_threshold);
    else if (key == "feature_match.ransac_sigma_factor") ok = parse_float_value(value, &cfg.feature_match.ransac_sigma_factor);
    else if (key == "feature_match.ransac_iterations") ok = parse_int_value(value, &cfg.feature_match.ransac_iterations);
    else if (key == "feature_match.use_robust_scoreb") ok = parse_bool_value(value, &cfg.feature_match.use_robust_scoreb);
    else if (key == "feature_match.use_affine_patch_score") ok = parse_bool_value(value, &cfg.feature_match.use_affine_patch_score);
    else if (key == "feature_match.pyramid_open") ok = parse_bool_value(value, &cfg.feature_match.pyramid_open);
    else if (key == "feature_match.use_local_affine") ok = parse_bool_value(value, &cfg.feature_match.use_local_affine);
    else if (key == "feature_match.local_tiles_r") ok = parse_int_value(value, &cfg.feature_match.local_tiles_r);
    else if (key == "feature_match.local_tiles_c") ok = parse_int_value(value, &cfg.feature_match.local_tiles_c);
    else if (key == "feature_match.local_affine_sigma") ok = parse_float_value(value, &cfg.feature_match.local_affine_sigma);
    else if (key == "feature_match.local_ransac_sigma") ok = parse_float_value(value, &cfg.feature_match.local_affine_sigma);
    else if (key == "feature_match.local_min_points") ok = parse_int_value(value, &cfg.feature_match.local_min_points);
    else if (key == "feature_match.local_inlier_sigma") ok = parse_float_value(value, &cfg.feature_match.local_affine_sigma);
    else if (key == "feature_match.local_keep_sigma") ok = parse_float_value(value, &cfg.feature_match.local_affine_sigma);
    else if (key == "feature_match.local_keep_sigma_factor") ok = true;
    else if (key == "feature_match.local_keep_score_min") ok = parse_float_value(value, &cfg.feature_match.local_keep_score_min);
    else if (key == "feature_match.global_keep_radius") ok = parse_float_value(value, &cfg.feature_match.global_keep_radius);
    else if (key == "feature_match.global_keep_abs") ok = parse_float_value(value, &cfg.feature_match.global_keep_abs);
    else if (key == "feature_match.densify_grid") ok = parse_bool_value(value, &cfg.feature_match.densify_grid);
    else if (key == "feature_match.densify_batch_r") ok = parse_int_value(value, &cfg.feature_match.densify_batch_r);
    else if (key == "feature_match.densify_batch_c") ok = parse_int_value(value, &cfg.feature_match.densify_batch_c);
    else if (key == "feature_match.densify_window_size") ok = parse_int_value(value, &cfg.feature_match.densify_window_size);
    else if (key == "feature_match.densify_search_range") ok = parse_int_value(value, &cfg.feature_match.densify_search_range);
    else if (key == "feature_match.densify_cc_threshold") ok = parse_float_value(value, &cfg.feature_match.densify_cc_threshold);
    else if (key == "feature_match.densify_knn") ok = parse_int_value(value, &cfg.feature_match.densify_knn);
    else if (key == "feature_match.iterative_refinement_iterations") ok = parse_int_value(value, &cfg.feature_match.iterative_refinement_iterations);
    else if (key == "feature_match.iterative_window_size") ok = parse_int_value(value, &cfg.feature_match.iterative_window_size);
    else if (key == "feature_match.iterative_search_range") ok = parse_int_value(value, &cfg.feature_match.iterative_search_range);
    else if (key == "feature_match.iterative_cc_threshold") ok = parse_float_value(value, &cfg.feature_match.iterative_cc_threshold);
    else if (key == "feature_match.lambda2") ok = parse_float_value(value, &cfg.feature_match.lambda2);
    else if (key == "feature_match.iterative_seed_radius") ok = parse_int_value(value, &cfg.feature_match.iterative_seed_radius);
    else if (key == "feature_match.iterative_seed_min") ok = parse_int_value(value, &cfg.feature_match.iterative_seed_min);

    else if (key == "grid_match.refinement_iterations") ok = parse_int_value(value, &cfg.grid_match.refinement_iterations);
    else if (key == "grid_match.batch_size") ok = parse_int_value(value, &cfg.grid_match.batch_size);
    else if (key == "grid_match.window_size") ok = parse_int_value(value, &cfg.grid_match.window_size);
    else if (key == "grid_match.search_range") ok = parse_int_value(value, &cfg.grid_match.search_range);
    else if (key == "grid_match.cc_threshold") ok = parse_float_value(value, &cfg.grid_match.cc_threshold);
    else if (key == "grid_match.use_robust_scoreb") ok = parse_bool_value(value, &cfg.grid_match.use_robust_scoreb);
    else if (key == "grid_match.use_affine_patch_score") ok = parse_bool_value(value, &cfg.grid_match.use_affine_patch_score);
    else if (key == "grid_match.use_neighbor_affine") ok = parse_bool_value(value, &cfg.grid_match.use_neighbor_affine);
    else if (key == "grid_match.lambda2") ok = parse_float_value(value, &cfg.grid_match.lambda2);
    else if (key == "grid_match.affine_max_dev") ok = parse_int_value(value, &cfg.grid_match.affine_max_dev);
    else if (key == "grid_match.affine_pred_as_match") ok = parse_bool_value(value, &cfg.grid_match.affine_pred_as_match);
    else if (key == "grid_match.followup_window_size") ok = parse_int_value(value, &cfg.grid_match.followup_window_size);
    else if (key == "grid_match.followup_search_range") ok = parse_int_value(value, &cfg.grid_match.followup_search_range);
    else if (key == "grid_match.followup_cc_threshold") ok = parse_float_value(value, &cfg.grid_match.followup_cc_threshold);
    else if (key == "grid_match.iterative_window_size") ok = parse_int_value(value, &cfg.grid_match.iterative_window_size);
    else if (key == "grid_match.iterative_search_range") ok = parse_int_value(value, &cfg.grid_match.iterative_search_range);
    else if (key == "grid_match.iterative_cc_threshold") ok = parse_float_value(value, &cfg.grid_match.iterative_cc_threshold);

    else if (key == "inter_ccd_match.intra_window_size") ok = parse_int_value(value, &cfg.inter_ccd_match.intra_window_size);
    else if (key == "inter_ccd_match.intra_search_range") ok = parse_int_value(value, &cfg.inter_ccd_match.intra_search_range);
    else if (key == "inter_ccd_match.intra_batch_size_r") ok = parse_int_value(value, &cfg.inter_ccd_match.intra_batch_size_r);
    else if (key == "inter_ccd_match.intra_batch_size_c") ok = parse_int_value(value, &cfg.inter_ccd_match.intra_batch_size_c);
    else if (key == "inter_ccd_match.intra_cc_threshold") ok = parse_float_value(value, &cfg.inter_ccd_match.intra_cc_threshold);
    else if (key == "inter_ccd_match.cross_window_size") ok = parse_int_value(value, &cfg.inter_ccd_match.cross_window_size);
    else if (key == "inter_ccd_match.cross_search_range") ok = parse_int_value(value, &cfg.inter_ccd_match.cross_search_range);
    else if (key == "inter_ccd_match.cross_cc_threshold") ok = parse_float_value(value, &cfg.inter_ccd_match.cross_cc_threshold);
    else if (key == "inter_ccd_match.use_robust_scoreb") ok = parse_bool_value(value, &cfg.inter_ccd_match.use_robust_scoreb);
    else if (key == "inter_ccd_match.use_affine_patch_score") ok = parse_bool_value(value, &cfg.inter_ccd_match.use_affine_patch_score);
    else if (key == "inter_ccd_match.control_source") {
        const string v = lower_copy(unquote_copy(value));
        if (v == "feature" || v == "fea" || v == "match" || v == "0") {
            cfg.inter_ccd_match.control_source = 0;
        } else if (v == "grid" || v == "1") {
            cfg.inter_ccd_match.control_source = 1;
        } else {
            ok = false;
        }
    }
    else if (key == "inter_ccd_match.run_global_refine") ok = parse_bool_value(value, &cfg.inter_ccd_match.run_global_refine);

    else if (key == "ba.fi_source") {
        const string v = lower_copy(unquote_copy(value));
        if (v == "feature" || v == "fea" || v == "0") {
            cfg.ba.fi_source = 0;
        } else if (v == "grid" || v == "1") {
            cfg.ba.fi_source = 1;
        } else {
            ok = false;
        }
    }
    else if (key == "ba.run_feature_ba") ok = parse_bool_value(value, &cfg.ba.run_feature_ba);
    else if (key == "ba.use_jitter") ok = parse_bool_value(value, &cfg.ba.use_jitter);
    else if (key == "ba.block_max_iterations") ok = parse_int_value(value, &cfg.ba.block_max_iterations);
    else if (key == "ba.jitter_max_iterations") ok = parse_int_value(value, &cfg.ba.jitter_max_iterations);
    else if (key == "ba.fi_max_iterations") ok = parse_int_value(value, &cfg.ba.fi_max_iterations);
    else if (key == "ba.cauchy_loss_ba") ok = parse_float_value(value, &cfg.ba.cauchy_loss_ba);
    else if (key == "ba.cauchy_loss_fi") ok = parse_float_value(value, &cfg.ba.cauchy_loss_fi);
    else if (key == "ba.function_tolerance") ok = parse_float_value(value, &cfg.ba.function_tolerance);
    else if (key == "ba.tichu_affine_max_residual") ok = parse_float_value(value, &cfg.ba.tichu_affine_max_residual);
    else if (key == "ba.tichu_affine_max_vx") ok = parse_float_value(value, &cfg.ba.tichu_affine_max_vx);
    else if (key == "ba.tichu_affine_max_vy") ok = parse_float_value(value, &cfg.ba.tichu_affine_max_vy);
    else if (key == "ba.tichu_local_radius") ok = parse_float_value(value, &cfg.ba.tichu_local_radius);
    else if (key == "ba.tichu_sigma_factor") ok = parse_float_value(value, &cfg.ba.tichu_sigma_factor);
    else {
        printf("[WARN][配置] 未识别的配置项 %s (line %d)，已忽略\n", key.c_str(), line_no);
        return true;
    }

    if (!ok) {
        printf("[ERROR][配置] 配置项 %s 的值无效: %s (line %d)\n",
               key.c_str(), value.c_str(), line_no);
    }
    return ok;
}

static bool validate_pipeline_config(const PipelineConfig& cfg) {
    bool ok = true;
    if (cfg.srcfilepath[0] == '\0') {
        printf("[ERROR][配置] dataset.srcfilepath 不能为空\n");
        ok = false;
    }
    if (cfg.filepath[0] == '\0') {
        printf("[ERROR][配置] dataset.filepath 不能为空\n");
        ok = false;
    }
    if (cfg.xulie_ID1[0] == '\0' || cfg.xulie_ID2[0] == '\0') {
        printf("[ERROR][配置] dataset.xulie_ID1/xulie_ID2 不能为空\n");
        ok = false;
    }
    if (cfg.rows1 <= 0 || cfg.rows2 <= 0 || cfg.cols <= 0 || cfg.CCD_num <= 0) {
        printf("[ERROR][配置] dataset.rows1/rows2/cols/CCD_num 必须为正数\n");
        ok = false;
    }
    if (cfg.CCD_begin < 0) {
        printf("[ERROR][配置] dataset.CCD_begin 不能为负\n");
        ok = false;
    }
    if (cfg.CCD_end >= 0 && cfg.CCD_end < cfg.CCD_begin) {
        printf("[ERROR][配置] dataset.CCD_end(%d) 必须 >= CCD_begin(%d)（或设为 -1 表示 CCD_num）\n",
               cfg.CCD_end, cfg.CCD_begin);
        ok = false;
    }
    if (cfg.CCD_begin >= cfg.CCD_num) {
        printf("[ERROR][配置] dataset.CCD_begin(%d) 必须 < CCD_num(%d)\n",
               cfg.CCD_begin, cfg.CCD_num);
        ok = false;
    }
    if (cfg.mosaic.overlap_samples <= 0 || cfg.grid_match.batch_size <= 0) {
        printf("[ERROR][配置] mosaic.overlap_samples 和 grid_match.batch_size 必须为正数\n");
        ok = false;
    }
    if (cfg.feature_match.local_affine_sigma <= 0.f ||
        cfg.feature_match.local_min_points <= 0) {
        printf("[ERROR][配置] feature_match.local_affine_sigma 和 local_min_points 必须为正数\n");
        ok = false;
    }
    if (cfg.ba.fi_source != 0 && cfg.ba.fi_source != 1) {
        printf("[ERROR][配置] ba.fi_source 必须是 feature(0) 或 grid(1)\n");
        ok = false;
    }
    if (cfg.inter_ccd_match.control_source != 0 && cfg.inter_ccd_match.control_source != 1) {
        printf("[ERROR][配置] inter_ccd_match.control_source 必须是 feature(0) 或 grid(1)\n");
        ok = false;
    }
    if (cfg.ba.block_max_iterations <= 0 || cfg.ba.jitter_max_iterations <= 0 ||
        cfg.ba.fi_max_iterations <= 0) {
        printf("[ERROR][配置] ba.*_max_iterations 必须为正数\n");
        ok = false;
    }
    if (cfg.ba.cauchy_loss_ba <= 0.f || cfg.ba.cauchy_loss_fi <= 0.f) {
        printf("[ERROR][配置] ba.cauchy_loss_* 必须为正数\n");
        ok = false;
    }
    if (cfg.ba.tichu_affine_max_residual <= 0.f ||
        cfg.ba.tichu_affine_max_vx <= 0.f ||
        cfg.ba.tichu_affine_max_vy <= 0.f ||
        cfg.ba.tichu_local_radius <= 0.f ||
        cfg.ba.tichu_sigma_factor <= 0.f) {
        printf("[ERROR][配置] ba.tichu_* 阈值必须为正数\n");
        ok = false;
    }
    return ok;
}

static bool load_pipeline_config(const char* cfg_path, PipelineConfig& cfg) {
    ifstream in(cfg_path);
    if (!in.is_open()) {
        printf("[ERROR][配置] 无法打开配置文件: %s\n", cfg_path);
        return false;
    }

    string section;
    string raw_line;
    int line_no = 0;
    while (getline(in, raw_line)) {
        line_no++;
        string line = trim_copy(strip_inline_comment(raw_line));
        if (line.empty()) {
            continue;
        }

        const size_t first_non_space = raw_line.find_first_not_of(" \t");
        const bool nested = first_non_space != string::npos && first_non_space > 0;
        const size_t colon = line.find(':');
        if (colon == string::npos) {
            printf("[ERROR][配置] 缺少冒号: %s (line %d)\n", raw_line.c_str(), line_no);
            return false;
        }

        const string key = trim_copy(line.substr(0, colon));
        const string value = trim_copy(line.substr(colon + 1));
        if (key.empty()) {
            printf("[ERROR][配置] 配置项为空 (line %d)\n", line_no);
            return false;
        }

        if (!nested && value.empty()) {
            section = key;
            continue;
        }

        const string full_key = nested && !section.empty() ? section + "." + key : key;
        if (!assign_config_value(cfg, full_key, value, line_no)) {
            return false;
        }
    }

    return validate_pipeline_config(cfg);
}

static void print_config_summary(const PipelineConfig& cfg, const char* cfg_path) {
    printf("[INFO][配置] 已加载: %s\n", cfg_path);
    printf("[INFO][配置] 数据集: %s <-> %s, rows=(%d,%d), cols=%d, CCD=%d, CCD_range=[%d,%d)\n",
           cfg.xulie_ID1, cfg.xulie_ID2, cfg.rows1, cfg.rows2, cfg.cols, cfg.CCD_num,
           cfg.ccd_begin(), cfg.ccd_end());
    printf("[INFO][配置] 关键参数: overlap=%d, feature_iter=%d, feature_lambda2=%.3f, pyramid_open=%d, grid_iter=%d, grid_window=%d, grid_threshold=%.3f, grid_lambda2=%.3f, grid_affine_max_dev=%d, grid_neighbor_affine=%d, grid_affine_patch=%d\n",
           cfg.mosaic.overlap_samples,
           cfg.feature_match.iterative_refinement_iterations,
           cfg.feature_match.lambda2,
           (int)cfg.feature_match.pyramid_open,
           cfg.grid_match.refinement_iterations,
           cfg.grid_match.window_size,
           cfg.grid_match.cc_threshold,
           cfg.grid_match.lambda2,
           cfg.grid_match.affine_max_dev,
           (int)cfg.grid_match.use_neighbor_affine,
           (int)cfg.grid_match.use_affine_patch_score);
    printf("[INFO][配置] ScoreB: feature_robust=%d, grid_robust=%d, inter_robust=%d, feature_affine_patch=%d, grid_affine_patch=%d, inter_affine_patch=%d\n",
           (int)cfg.feature_match.use_robust_scoreb,
           (int)cfg.grid_match.use_robust_scoreb,
           (int)cfg.inter_ccd_match.use_robust_scoreb,
           (int)cfg.feature_match.use_affine_patch_score,
           (int)cfg.grid_match.use_affine_patch_score,
           (int)cfg.inter_ccd_match.use_affine_patch_score);
    printf("[INFO][配置] CCD间: control_source=%s\n",
           cfg.inter_ccd_match.control_use_grid() ? "grid" : "feature");
    printf("[INFO][配置] BA: fi_source=%s, run_feature_ba=%d, use_jitter=%d, "
           "iters(block/jitter/fi)=%d/%d/%d, cauchy(ba/fi)=%.3f/%.3f\n",
           cfg.ba.fi_use_grid() ? "grid" : "feature",
           (int)cfg.ba.run_feature_ba,
           (int)cfg.ba.use_jitter,
           cfg.ba.block_max_iterations,
           cfg.ba.jitter_max_iterations,
           cfg.ba.fi_max_iterations,
           cfg.ba.cauchy_loss_ba,
           cfg.ba.cauchy_loss_fi);
    printf("[INFO][配置] BA Tichu_CX: residual<%.0f |vx|<%.0f |vy|<%.0f local_r=%.0f sigma×%.1f\n",
           cfg.ba.tichu_affine_max_residual,
           cfg.ba.tichu_affine_max_vx,
           cfg.ba.tichu_affine_max_vy,
           cfg.ba.tichu_local_radius,
           cfg.ba.tichu_sigma_factor);
}

// 统计指定后缀的非空文件数量（level-0，仅 CCD 处理区间）
static int count_valid_files(const PipelineConfig& c, const char* xulie_ID,
                              const char* suffix) {
    int count = 0;
    for (int j = c.ccd_begin(); j < c.ccd_end(); j++) {
        const string path = format_string("%s/%s/downsample/0/%s_RED%d%s",
                                          c.filepath, xulie_ID, xulie_ID, j, suffix);
        if (file_nonempty(path)) count++;
    }
    return count;
}

// 统计 _grid.txt 中含有 bj=1 的文件数量（仅 CCD 处理区间）
static int count_grid_matched(const PipelineConfig& c, const char* xulie_ID) {
    int count = 0;
    for (int j = c.ccd_begin(); j < c.ccd_end(); j++) {
        const string path = format_string("%s/%s/downsample/0/%s_RED%d_grid.txt",
                                          c.filepath, xulie_ID, xulie_ID, j);
        FILE* f = fopen(path.c_str(), "r");
        if (!f) continue;
        char line[256]; int bj;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "%d", &bj) == 1 && bj == 1) { count++; break; }
        }
        fclose(f);
    }
    return count;
}

static float* alloc_fsc() {
    float* fs_c = new float[6];
    memset(fs_c, 0, sizeof(float) * 6);
    fs_c[0] = fs_c[4] = 1.0f;
    return fs_c;
}

// ============================================================
// Step 1: 图像预处理 (PDS .img/.IMG -> 半幅 TIF)
// ============================================================
static bool resolve_pds_img_path(string& path,
                                 const char* srcfilepath, const char* xulie_ID,
                                 int ccd, int half) {
    path = format_string("%s/%s/%s_RED%d_%d.img",
                         srcfilepath, xulie_ID, xulie_ID, ccd, half);
    if (file_exists(path)) {
        return true;
    }
    path = format_string("%s/%s/%s_RED%d_%d.IMG",
                         srcfilepath, xulie_ID, xulie_ID, ccd, half);
    return file_exists(path);
}

static bool check_step_preprocess(const PipelineConfig& c) {
    string path;
    const int ccd0 = c.ccd_begin();
    if (!resolve_pds_img_path(path, c.srcfilepath, c.xulie_ID1, ccd0, 0)) {
        printf("[WARN][预处理] 原始 .img/.IMG 文件不存在: %s/%s/%s_RED%d_0.img|.IMG\n",
               c.srcfilepath, c.xulie_ID1, c.xulie_ID1, ccd0);
        return false;
    }
    return true;
}

static bool run_step_preprocess(const PipelineConfig& c) {
    if (!check_step_preprocess(c)) return false;
    printf("=== Step 1: 图像预处理 ===\n");
    ImageProcess::xulie_process(c);
    printf("=== Step 1 完成 ===\n");
    return true;
}

// ============================================================
// Step 2: CCD 内拼接（_0/_1.tif -> RED{j}.tif）
// ============================================================
static bool check_step_intra_ccd_mosaic(const PipelineConfig& c) {
    const int ccd0 = c.ccd_begin();
    string path = format_string("%s/%s/downsample/0/%s_RED%d_0.tif",
                                c.filepath, c.xulie_ID1, c.xulie_ID1, ccd0);
    if (!file_exists(path)) {
        printf("[WARN][CCD内拼接] 半幅 TIF 不存在: %s\n"
               "  → 请先运行 Step 1 (图像预处理)\n", path.c_str());
        return false;
    }
    path = format_string("%s/%s/downsample/0/%s_RED%d_1.tif",
                         c.filepath, c.xulie_ID1, c.xulie_ID1, ccd0);
    if (!file_exists(path)) {
        printf("[WARN][CCD内拼接] 半幅 TIF 不存在: %s\n"
               "  → 请先运行 Step 1 (图像预处理)\n", path.c_str());
        return false;
    }
    return true;
}

static bool run_step_intra_ccd_mosaic(const PipelineConfig& c) {
    if (!check_step_intra_ccd_mosaic(c)) return false;
    printf("=== Step 2: CCD 内拼接 ===\n");
    ImageProcess::intra_ccd_mosaic(c);
    printf("=== Step 2 完成 ===\n");
    return true;
}

// ============================================================
// Step 3: 影像降采样（生成金字塔 level 1~4）
// ============================================================
static bool check_step_downsample(const PipelineConfig& c) {
    const int ccd0 = c.ccd_begin();
    const string path = format_string("%s/%s/downsample/0/%s_RED%d.tif",
                                      c.filepath, c.xulie_ID1, c.xulie_ID1, ccd0);
    if (!file_exists(path)) {
        printf("[WARN][降采样] level-0 TIF 不存在: %s\n"
               "  → 请先运行 Step 2 (CCD 内拼接)\n", path.c_str());
        return false;
    }
    return true;
}

static bool run_step_downsample(const PipelineConfig& c) {
    if (!check_step_downsample(c)) return false;
    printf("=== Step 3: 影像降采样 ===\n");
    Observation OB;
    OB.cfg_ = c;
    OB.xulie_downsample();
    printf("=== Step 3 完成 ===\n");
    return true;
}

// ============================================================
// Step 4: 序列拼接参数准备（xulie_mosaic1 + 多层 mosaic.txt）
// ============================================================
static bool check_step_feature_mosaic(const PipelineConfig& c) {
    const int ccd0 = c.ccd_begin();
    string path = format_string("%s/%s/downsample/0/%s_RED%d.tif",
                                c.filepath, c.xulie_ID1, c.xulie_ID1, ccd0);
    if (!file_exists(path)) {
        printf("[WARN][序列拼接] level-0 TIF 不存在: %s\n"
               "  → 请先运行 Step 2 (CCD 内拼接)\n", path.c_str());
        return false;
    }
    path = format_string("%s/%s/downsample/4/%s_RED%d.tif",
                         c.filepath, c.xulie_ID1, c.xulie_ID1, ccd0);
    if (!file_exists(path)) {
        printf("[WARN][序列拼接] 第4层 TIF 不存在: %s\n"
               "  → 请先运行 Step 3 (降采样)\n", path.c_str());
        return false;
    }
    return true;
}

static bool run_step_feature_mosaic(const PipelineConfig& c) {
    if (!check_step_feature_mosaic(c)) return false;
    printf("=== Step 4: 序列拼接参数准备 ===\n");
    Observation OB;
    OB.cfg_ = c;
    OB.prepare_feature_match_mosaic(0);
    printf("=== Step 4 完成 ===\n");
    return true;
}

// ============================================================
// Step 5: 特征提取（金字塔各层特征点检测）
// 输出: {xulie_ID1/2}_RED{j}.txt (level 0~4)
// ============================================================
static bool check_step_feature_extract(const PipelineConfig& c) {
    const int ccd0 = c.ccd_begin();
    for (int level = 0; level <= 4; level++) {
        const string path = format_string("%s/%s/downsample/%d/%s_RED%d.tif",
                                          c.filepath, c.xulie_ID1, level, c.xulie_ID1, ccd0);
        if (!file_exists(path)) {
            printf("[WARN][特征提取] 第%d层 TIF 不存在: %s\n"
                   "  → 请先运行 Step 3 (降采样)\n", level, path.c_str());
            return false;
        }
    }
    return true;
}

static bool run_step_feature_extract(const PipelineConfig& c) {
    if (!check_step_feature_extract(c)) return false;
    printf("=== Step 5: 特征提取 ===\n");
    Observation OB;
    OB.cfg_ = c;
    OB.fenfu_extract(0);
    printf("=== Step 5 完成 ===\n");
    return true;
}

// ============================================================
// Step 6: 特征匹配（金字塔 + 迭代优化）
// 输出: {xulie_ID1}_RED{j}_match.txt (level 0)
// ============================================================
static bool check_step_feature_match(const PipelineConfig& c) {
    const int ccd0 = c.ccd_begin();
    for (int level = 0; level <= 4; level++) {
        string path = format_string("%s/%s/downsample/%d/%s_RED%d.txt",
                                    c.filepath, c.xulie_ID1, level, c.xulie_ID1, ccd0);
        if (!file_exists(path)) {
            printf("[WARN][特征匹配] 第%d层特征点文件不存在: %s\n"
                   "  → 请先运行 Step 5 (特征提取)\n", level, path.c_str());
            return false;
        }
        path = format_string("%s/%s/downsample/%d/mosaic.txt",
                             c.filepath, c.xulie_ID1, level);
        if (!file_exists(path)) {
            printf("[WARN][特征匹配] 第%d层 mosaic.txt 不存在: %s\n"
                   "  → 请先运行 Step 4 (序列拼接参数准备)\n", level, path.c_str());
            return false;
        }
    }
    return true;
}

static bool run_step_feature_match(const PipelineConfig& c) {
    if (!check_step_feature_match(c)) return false;
    printf("=== Step 6: 特征匹配 ===\n");
    ImageMatch::SetScoreBOptions(c.feature_match.use_robust_scoreb,
                                 c.feature_match.use_affine_patch_score);
    printf("[INFO][特征匹配] ScoreB: robust=%d affine_patch=%d\n",
           (int)c.feature_match.use_robust_scoreb,
           (int)c.feature_match.use_affine_patch_score);
    Observation OB;
    OB.cfg_ = c;
    float* fs_c = alloc_fsc();

    if (c.feature_match.pyramid_open) {
        OB.fenfu_match1(fs_c, 0);
    } else {
        printf("[INFO][特征匹配] pyramid_open=false，跳过 fenfu_match1(fs,0)\n");
    }

    // OB.Compute_fsc(fs_c, 0);
    // // OB.fenfu_extract(1);
    // OB.fenfu_match1(fs_c, 1);

    // OB.Compute_fsc(fs_c, 0);
    // // OB.fenfu_extract(1);
    // OB.fenfu_match1(fs_c, 1);

    for (int i = 0; i < c.feature_match.iterative_refinement_iterations; i++) {
        printf("Fea_IR_%d\n", i);
        float* fs_c2 = alloc_fsc();
        OB.Compute_fsc(fs_c2, 0);
        OB.MatchGet(0, fs_c2);
        OB.Iterative_refinement(fs_c2, 0);
        OB.Compute_fsc(fs_c2, 0);
        OB.MatchGet(0, fs_c2);
        delete[] fs_c2;
    }
    delete[] fs_c;

    int n = count_valid_files(c, c.xulie_ID1, "_match.txt");
    printf("=== Step 6 完成: %d/%d 个 CCD 有效匹配文件 ===\n", n, c.ccd_end() - c.ccd_begin());
    return true;
}

// ============================================================
// Step 7: 格网匹配（SemiDenseGrid，两轮迭代优化）
// 输出: {xulie_ID1}_RED{j}_grid.txt (level 0)
// ============================================================
static bool check_step_grid_match(const PipelineConfig& c) {
    int n = count_valid_files(c, c.xulie_ID1, "_match.txt");
    if (n == 0) {
        printf("[WARN][格网匹配] 未找到有效的 _match.txt 文件\n"
               "  → 请先运行 Step 6 (特征匹配)\n");
        return false;
    }
    const string path = format_string("%s/%s/downsample/0/mosaic.txt",
                                      c.filepath, c.xulie_ID1);
    if (!file_exists(path)) {
        printf("[WARN][格网匹配] mosaic.txt 不存在: %s\n"
               "  → 请先运行 Step 4 (序列拼接参数准备)\n", path.c_str());
        return false;
    }
    printf("[INFO][格网匹配] 前提检查通过，找到 %d/%d 个有效 _match.txt\n",
           n, c.ccd_end() - c.ccd_begin());
    return true;
}

static bool run_step_grid_match(const PipelineConfig& c) {
    if (!check_step_grid_match(c)) return false;
    printf("=== Step 7: 格网匹配 ===\n");
    ImageMatch::SetScoreBOptions(c.grid_match.use_robust_scoreb,
                                 c.grid_match.use_affine_patch_score);
    printf("[INFO][格网匹配] ScoreB: robust=%d affine_patch=%d neighbor_affine=%d\n",
           (int)c.grid_match.use_robust_scoreb,
           (int)c.grid_match.use_affine_patch_score,
           (int)c.grid_match.use_neighbor_affine);
    Observation OB;
    OB.cfg_ = c;
    float* fs_c = alloc_fsc();

    for (int i = 0; i < c.grid_match.refinement_iterations; i++) {
        printf("Grid_IR_%d\n", i);
        OB.Compute_fsc(fs_c, 0);
        if (i > 0) {
            OB.MatchGet(1, fs_c);
        }
        OB.SemiDenseGrid_match1(fs_c);
    }
    delete[] fs_c;

    int ng = count_grid_matched(c, c.xulie_ID1);
    printf("=== Step 7 完成: %d/%d 个 CCD 格网文件含有效匹配点 ===\n",
           ng, c.ccd_end() - c.ccd_begin());
    return true;
}

// ============================================================
// Step 8: 生成CCD间连接点
// 输出: {xulie_ID1}_RED{j}_{j+1}_intra_match.txt (level 0)
// ============================================================
static bool check_step_inter_ccd_match(const PipelineConfig& c) {
    int ng = count_grid_matched(c, c.xulie_ID1);
    if (ng == 0) {
        printf("[WARN][CCD间连接点] _grid.txt 中没有已匹配的格网点 (bj=1)\n"
               "  → 请先运行 Step 7 (格网匹配)\n");
        return false;
    }
    printf("[INFO][CCD间连接点] 前提检查通过: %d 个含格网匹配的 CCD (Step 7)\n", ng);
    return true;
}

static bool run_step_inter_ccd_match(const PipelineConfig& c) {
    if (!check_step_inter_ccd_match(c)) return false;
    printf("=== Step 8: 生成CCD间连接点 ===\n");
    ImageMatch::SetScoreBOptions(c.inter_ccd_match.use_robust_scoreb,
                                 c.inter_ccd_match.use_affine_patch_score);
    printf("[INFO][CCD间连接点] ScoreB: robust=%d affine_patch=%d\n",
           (int)c.inter_ccd_match.use_robust_scoreb,
           (int)c.inter_ccd_match.use_affine_patch_score);
    printf("[INFO][CCD间连接点] params: intra(w=%d ser=%d thr=%.2f batch=%dx%d) "
           "cross(w=%d ser=%d thr=%.2f control=%s) run_global_refine=%d CCD=[%d,%d)\n",
           c.inter_ccd_match.intra_window_size,
           c.inter_ccd_match.intra_search_range,
           c.inter_ccd_match.intra_cc_threshold,
           c.inter_ccd_match.intra_batch_size_r,
           c.inter_ccd_match.intra_batch_size_c,
           c.inter_ccd_match.cross_window_size,
           c.inter_ccd_match.cross_search_range,
           c.inter_ccd_match.cross_cc_threshold,
           c.inter_ccd_match.control_use_grid() ? "grid" : "feature",
           (int)c.inter_ccd_match.run_global_refine,
           c.ccd_begin(), c.ccd_end());
    Observation OB;
    OB.cfg_ = c;
    float* fs_c = alloc_fsc();
    OB.Compute_fsc(fs_c, 0);
    OB.Generate_matchPoint_Between_CCD(fs_c);
    printf("=== Step 8: 汇总绘制 Intra__ 到 mosaic DS4 ===\n");
    OB.Draw_InterCCDMatch_OnMosaic(0);
    // OB.Tichu_CX_ByGlobalControl(fs_c, 0);
    // OB.Tichu_CX_ByGlobalControl(fs_c, 1);
    if (c.inter_ccd_match.run_global_refine) {
        printf("[INFO][CCD间连接点] run_global_refine=true → SemiDenseGrid_match2\n");
        OB.SemiDenseGrid_match2(fs_c, 1);
    } else {
        printf("[INFO][CCD间连接点] run_global_refine=false，跳过 SemiDenseGrid_match2\n");
    }
    delete[] fs_c;

    // 统计输出文件数量（仅范围内相邻对）
    int count = 0;
    const int n_pairs = c.ccd_end() - c.ccd_begin() - 1;
    for (int j = c.ccd_begin(); j < c.ccd_end() - 1; j++) {
        const string path = format_string("%s/%s/downsample/0/%s_RED%d_%d_intra_match.txt",
                                          c.filepath, c.xulie_ID1, c.xulie_ID1, j, j + 1);
        if (file_nonempty(path)) count++;
    }
    printf("=== Step 8 完成: %d/%d 个CCD间连接点文件生成 ===\n",
           count, n_pairs > 0 ? n_pairs : 0);
    return true;
}

// ============================================================
// Step 9: 光束法平差（BA）
// 输出: data/result/{xulie_ID1}_{xulie_ID2}_FI.txt
// ============================================================
static bool check_step_ba(const PipelineConfig& c) {
    const bool need_feature = c.ba.run_feature_ba || c.ba.fi_use_feature();
    const bool need_grid = c.ba.fi_use_grid();
    int nm = count_valid_files(c, c.xulie_ID1, "_match.txt");
    int ng = count_grid_matched(c, c.xulie_ID1);

    if (need_feature && nm == 0) {
        printf("[WARN][BA] 未找到有效的 _match.txt 文件\n"
               "  → 请先运行 Step 6 (特征匹配)；或改 ba.fi_source / ba.run_feature_ba\n");
        return false;
    }
    if (need_grid && ng == 0) {
        printf("[WARN][BA] _grid.txt 中没有已匹配的格网点 (bj=1)\n"
               "  → 请先运行 Step 7 (格网匹配)；或把 ba.fi_source 设为 feature\n");
        return false;
    }
    printf("[INFO][BA] 前提检查通过: fi_source=%s, match=%d, grid=%d, run_feature_ba=%d\n",
           c.ba.fi_use_grid() ? "grid" : "feature", nm, ng, (int)c.ba.run_feature_ba);
    return true;
}

static bool run_step_ba(const PipelineConfig& c) {
    if (!check_step_ba(c)) return false;
    printf("=== Step 9: BA 平差 ===\n");
    BA_main(c);

    const string result_path = format_string("../data/result/%s_%s_FI.txt",
                                             c.xulie_ID1, c.xulie_ID2);
    if (file_nonempty(result_path)) {
        FILE* f = fopen(result_path.c_str(), "r");
        int lines = 0; char line[512];
        while (fgets(line, sizeof(line), f)) lines++;
        fclose(f);
        printf("=== Step 9 完成: DEM 共 %d 个点 → %s ===\n", lines, result_path.c_str());
    } else {
        printf("=== Step 9 完成 ===\n");
    }
    return true;
}

// ============================================================
// main
// ============================================================
int main(int argc, const char* argv[])
{
    cout << "Hello Mars!" << endl;

    PipelineConfig cfg;
    const char* cfg_path = argc > 1 ? argv[1] : "cfg.yaml";
    if (!load_pipeline_config(cfg_path, cfg)) {
        return 1;
    }
    PipelineLogger logger(cfg, cfg_path);
    print_config_summary(cfg, cfg_path);
    logger.info("CONFIG rows=(%d,%d) cols=%d CCD=%d feature_iter=%d grid_iter=%d ba=%d",
                cfg.rows1, cfg.rows2, cfg.cols, cfg.CCD_num,
                cfg.feature_match.iterative_refinement_iterations,
                cfg.grid_match.refinement_iterations,
                (int)cfg.steps.ba);

    run_logged_step(logger, 1, "preprocess", "图像预处理",
                    cfg.steps.preprocess, run_step_preprocess, cfg);
    run_logged_step(logger, 2, "intra_ccd_mosaic", "CCD内拼接",
                    cfg.steps.intra_ccd_mosaic, run_step_intra_ccd_mosaic, cfg);
    run_logged_step(logger, 3, "downsample", "影像降采样",
                    cfg.steps.downsample, run_step_downsample, cfg);
    run_logged_step(logger, 4, "feature_mosaic", "序列拼接参数准备",
                    cfg.steps.feature_mosaic, run_step_feature_mosaic, cfg);
    run_logged_step(logger, 5, "feature_extract", "特征提取",
                    cfg.steps.feature_extract, run_step_feature_extract, cfg);
    run_logged_step(logger, 6, "feature_match", "特征匹配",
                    cfg.steps.feature_match, run_step_feature_match, cfg);
    run_logged_step(logger, 7, "grid_match", "格网匹配",
                    cfg.steps.grid_match, run_step_grid_match, cfg);
    run_logged_step(logger, 8, "inter_ccd_match", "CCD间连接点",
                    cfg.steps.inter_ccd_match, run_step_inter_ccd_match, cfg);
    run_logged_step(logger, 9, "ba", "BA平差",
                    cfg.steps.ba, run_step_ba, cfg);

    logger.info("MAIN_LOG %s", logger.main_log_path().c_str());

    return 0;
}
