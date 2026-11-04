#include "camera.hpp"
#include "npz.hpp"
#include <cviruntime.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <csignal>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

using json = nlohmann::ordered_json;
using Clock = std::chrono::high_resolution_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

namespace model_const
{
    constexpr int INPUT_W = 640;
    constexpr int INPUT_H = 640;
    constexpr int OUTPUT_CHANNELS = 36;
    constexpr int OUTPUT_NUM_ANCHOR = 8400;
    constexpr int INPUT_PLANE = INPUT_W * INPUT_H;
}

const char* const CLASSES[] = {
    "G_Blue","G_Red","G_N","1_Blue","1_Red","1_N","2_Blue","2_Red","2_N",
    "3_Blue","3_Red","3_N","4_Blue","4_Red","4_N","5_Blue","5_Red","5_N",
    "O_Blue","O_Red","Bs_Blue","Bs_Red","Bb_Blue","Bb_Red"
};
constexpr size_t CLASS_COUNT = sizeof(CLASSES)/sizeof(CLASSES[0]);

struct Options
{
    std::string model_path = "/root/first_model_sg2002_balanced_mix.cvimodel";
    std::string source = "rtsp://127.0.0.1:8554/live";
    std::string image_path;
    std::string camera_log = "camera.log";

    std::string color_filter = "all";
    std::string center_mode = "bbox";

    std::string save_json;
    std::string save_image;
    std::string save_raw;

    double conf_thresh = 0.25;
    double iou_thresh = 0.45;
    int max_frames = 0;

    bool use_csi = false;
    bool enable_profile = false;
};

static volatile sig_atomic_t running = 1;

static void sig_handler(int)
{
    running = 0;
}

static void print_help()
{
    fprintf(stderr,
R"(armor_detector 装甲板检测器
Usage:
armor_detector [OPTIONS]
Options:
    --model <path>          cvimodel模型路径
    --source <rtsp>         视频流地址
    --image <path>          单张图片测试，与--start-csi互斥
    --start-csi             使用CSI摄像头
    --camera-log <file>     相机日志
    --color all|Blue|Red|N 过滤颜色
    --center-method bbox|keypoints 目标中心点计算方式
    --conf <float>          置信度阈值 0~1
    --iou <float>           NMS iou阈值 0~1
    --max-frames N          最大处理帧数，0代表无限
    --save-json <file>      输出jsonl结果文件
    --save-image <file>     保存标注后的图片
    --save-raw <file>       保存模型输出npz
    --profile               输出性能统计
    -h, --help              打印帮助
)");
}

bool parse_args(int argc, char** argv, Options& out_opt)
{
    for (int i = 1; i < argc;)
    {
        std::string key = argv[i];
        if(key == "-h" || key == "--help")
        {
            print_help();
            return false;
        }
        if(key == "--start-csi")
        {
            out_opt.use_csi = true;
            i++;
            continue;
        }
        if(key == "--profile")
        {
            out_opt.enable_profile = true;
            i++;
            continue;
        }
        if(i+1 >= argc)
        {
            fprintf(stderr, "Error: option %s needs argument\n", key.c_str());
            return false;
        }
        std::string val = argv[i+1];
        if(key == "--model")          out_opt.model_path = val;
        else if(key == "--source")    out_opt.source = val;
        else if(key == "--image")     out_opt.image_path = val;
        else if(key == "--camera-log")out_opt.camera_log = val;
        else if(key == "--color")     out_opt.color_filter = val;
        else if(key == "--center-method") out_opt.center_mode = val;
        else if(key == "--save-json") out_opt.save_json = val;
        else if(key == "--save-image")out_opt.save_image = val;
        else if(key == "--save-raw")  out_opt.save_raw = val;
        else if(key == "--conf")
        {
            size_t used;
            double v = std::stod(val, &used);
            if (used != val.size() || !std::isfinite(v)) throw std::runtime_error("invalid --conf");
            out_opt.conf_thresh = v;
        }
        else if(key == "--iou")
        {
            size_t used;
            double v = std::stod(val, &used);
            if (used != val.size() || !std::isfinite(v)) throw std::runtime_error("invalid --iou");
            out_opt.iou_thresh = v;
        }
        else if(key == "--max-frames")
        {
            size_t used;
            out_opt.max_frames = std::stoi(val, &used);
            if (used != val.size() || out_opt.max_frames < 0) throw std::runtime_error("invalid --max-frames");
        }
        else
        {
            fprintf(stderr, "Error: unknown argument %s\n", key.c_str());
            return false;
        }
        i += 2;
    }

    if(out_opt.conf_thresh < 0.0 || out_opt.conf_thresh > 1.0 ||
       out_opt.iou_thresh < 0.0 || out_opt.iou_thresh > 1.0)
    {
        fprintf(stderr,"conf/iou must between 0~1\n");
        return false;
    }
    if(out_opt.color_filter != "all" && out_opt.color_filter != "Blue" &&
       out_opt.color_filter != "Red" && out_opt.color_filter != "N")
    {
        fprintf(stderr,"--color only support all|Blue|Red|N\n");
        return false;
    }
    if(out_opt.center_mode != "bbox" && out_opt.center_mode != "keypoints")
    {
        fprintf(stderr,"--center-method only support bbox|keypoints\n");
        return false;
    }
    if(!out_opt.image_path.empty() && out_opt.use_csi)
    {
        fprintf(stderr,"--image and --start-csi cannot use together\n");
        return false;
    }
    return true;
}

// SG2002 的 CVI Runtime 接口
class CviInfer
{
public:
    explicit CviInfer(const std::string& model_file);
    ~CviInfer();
    CviInfer(const CviInfer&) = delete;
    CviInfer& operator=(const CviInfer&) = delete;
// 禁止拷贝对象
    double forward();

    float* get_input_buffer()  { return m_input; }
    float* get_output_buffer() { return m_output; }

private:
    CVI_MODEL_HANDLE m_handle = nullptr;
    CVI_TENSOR* m_input_tensors = nullptr;
    CVI_TENSOR* m_output_tensors = nullptr;
    float* m_input = nullptr;
    float* m_output = nullptr;
};

CviInfer::CviInfer(const std::string& model_file)
{
    int ret = CVI_NN_RegisterModel(model_file.c_str(), &m_handle);
    if(ret != 0)
        throw std::runtime_error("CVI_NN_RegisterModel failed ret="+std::to_string(ret));

    try {
        int32_t ni{0}, no{0};
        ret = CVI_NN_GetInputOutputTensors(m_handle, &m_input_tensors, &ni, &m_output_tensors, &no);
        if(ret != 0)
            throw std::runtime_error("CVI_NN_GetInputOutputTensors failed ret="+std::to_string(ret));

        auto shape_eq = [](CVI_TENSOR* t, std::initializer_list<int> expect) -> bool {
            auto s = CVI_NN_TensorShape(t);
            if(static_cast<size_t>(s.dim_size) != expect.size()) return false;
            return std::equal(expect.begin(), expect.end(), s.dim);
        };
        if(ni != 1 || no != 1)
            throw std::runtime_error("expect 1 input 1 output tensor");
        if(!shape_eq(m_input_tensors, {1,3,model_const::INPUT_H,model_const::INPUT_W}))
            throw std::runtime_error("input tensor shape expect [1,3,640,640]");
        if(!shape_eq(m_output_tensors, {1,model_const::OUTPUT_CHANNELS,model_const::OUTPUT_NUM_ANCHOR,1}))
            throw std::runtime_error("output tensor shape expect [1,36,8400,1]");
        if(m_input_tensors->fmt != CVI_FMT_FP32 || m_output_tensors->fmt != CVI_FMT_FP32)
            throw std::runtime_error("tensor must be FP32 format");
        if(CVI_NN_TensorSize(m_input_tensors) != 3*model_const::INPUT_PLANE*sizeof(float) ||
           CVI_NN_TensorSize(m_output_tensors) != model_const::OUTPUT_CHANNELS*model_const::OUTPUT_NUM_ANCHOR*sizeof(float))
            throw std::runtime_error("unexpected tensor buffer size");

        m_input  = static_cast<float*>(CVI_NN_TensorPtr(m_input_tensors));
        m_output = static_cast<float*>(CVI_NN_TensorPtr(m_output_tensors));
        if(!m_input || !m_output)
            throw std::runtime_error("tensor buffer ptr is null");
    } catch (...) {
        CVI_NN_CleanupModel(m_handle);
        m_handle = nullptr;
        throw;
    }
}

CviInfer::~CviInfer()
{
    if(m_handle)
        CVI_NN_CleanupModel(m_handle);
}

double CviInfer::forward()
{
    auto t0 = Clock::now();
    int rc = CVI_NN_Forward(m_handle, m_input_tensors, 1, m_output_tensors, 1);
    auto t1 = Clock::now();
    if(rc != 0)
        throw std::runtime_error("CVI_NN_Forward error rc="+std::to_string(rc));
    return DurationMs(t1 - t0).count();
}

// 保存缩放和补边参数，后处理用来还原坐标。
struct PreprocMeta
{
    int src_w{0};
    int src_h{0};
    int pad_left{0};
    int pad_top{0};
    double scale{1.0};
};

// 缩放缓冲区重复使用，处理结果直接写入模型输入。
class ArmorPreproc
{
public:
    ArmorPreproc()
    {
        for(int i = 0; i < 256; i++)
            m_norm_table[i] = static_cast<float>(i) / 255.f;
    }

    PreprocMeta process(const cv::Mat& src_bgr, float* tensor_chw);

private:
    std::array<float, 256> m_norm_table{};
    cv::Mat m_resized_buf;
};

PreprocMeta ArmorPreproc::process(const cv::Mat& src_bgr, float* tensor_chw)
{
    if(src_bgr.empty() || src_bgr.type() != CV_8UC3 || !tensor_chw)
        throw std::runtime_error("preprocess needs a BGR image and input buffer");

    PreprocMeta meta;
    meta.src_w = src_bgr.cols;
    meta.src_h = src_bgr.rows;

    const int target_w = model_const::INPUT_W;
    const int target_h = model_const::INPUT_H;

    meta.scale = std::min(double(target_w) / meta.src_w, double(target_h) / meta.src_h);
    int nw = std::max(1, static_cast<int>(meta.src_w * meta.scale));
    int nh = std::max(1, static_cast<int>(meta.src_h * meta.scale));

    meta.pad_left = (target_w - nw) / 2;
    meta.pad_top  = (target_h - nh) / 2;

    cv::resize(src_bgr, m_resized_buf, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    const float pad_val = m_norm_table[114];
    for(int c = 0; c < 3; c++)
    {
        float* plane = tensor_chw + c * model_const::INPUT_PLANE;
        std::fill_n(plane, meta.pad_top * target_w, pad_val);
        auto bottom_start = plane + (meta.pad_top + nh) * target_w;
        std::fill(bottom_start, plane + model_const::INPUT_PLANE, pad_val);
        if(nw != target_w)
        {
            for(int y = meta.pad_top; y < meta.pad_top + nh; y++)
            {
                float* row = plane + y * target_w;
                std::fill_n(row, meta.pad_left, pad_val);
                std::fill(row + meta.pad_left + nw, row + target_w, pad_val);
            }
        }
    }

    for(int y = 0; y < nh; y++)
    {
        auto src_row = m_resized_buf.ptr<cv::Vec3b>(y);
        int pos_in_plane = (y + meta.pad_top) * target_w + meta.pad_left;
        float* r_ptr = tensor_chw + pos_in_plane;
        float* g_ptr = r_ptr + model_const::INPUT_PLANE;
        float* b_ptr = g_ptr + model_const::INPUT_PLANE;

        for(int x = 0; x < nw; x++)
        {
            auto& px = src_row[x];
            r_ptr[x] = m_norm_table[px[2]];
            g_ptr[x] = m_norm_table[px[1]];
            b_ptr[x] = m_norm_table[px[0]];
        }
    }
    return meta;
}

static double round_double(double v, int digits)
{
    if(digits == 2)
        return std::nearbyint(v * 100.0) / 100.0;
    else
        return std::nearbyint(v * 1000000.0) / 1000000.0;
}

struct DetectResult
{
    int cls_id{0};
    std::string cls_name;
    float score{0.f};
    cv::Rect2d bbox;
    std::array<std::array<double, 2>, 4> kpt;
    double center_x{0};
    double center_y{0};
};

// 后处理
// 先在模型坐标下按类别做 NMS，再还原到原图
std::vector<DetectResult> post_process(const float* raw, const PreprocMeta& meta, const Options& opt)
{
    constexpr int n = model_const::OUTPUT_NUM_ANCHOR;
    for(int i = 0; i < n * model_const::OUTPUT_CHANNELS; ++i)
        if(!std::isfinite(raw[i])) throw std::runtime_error("non-finite model output");

    struct Box {
        int index, cls;
        float score;
        std::array<float, 4> xyxy;
        float area;
    };
    std::array<std::vector<Box>, CLASS_COUNT> groups;
    std::array<bool, CLASS_COUNT> allowed;
    for(size_t c = 0; c < CLASS_COUNT; ++c) {
        const char* suffix = std::strrchr(CLASSES[c], '_') + 1;
        allowed[c] = opt.color_filter == "all" || opt.color_filter == suffix;
    }
    for(int i = 0; i < n; ++i) {
        int cls = 0;
        float score = raw[4*n + i];
        for(size_t c = 1; c < CLASS_COUNT; ++c) {
            float s = raw[(4+c)*n + i];
            if(s > score) { score = s; cls = int(c); }
        }
        float w = raw[2*n + i], h = raw[3*n + i];
        if(score < opt.conf_thresh || !allowed[cls] || w <= 0 || h <= 0) continue;
        float x = raw[i], y = raw[n + i];
        std::array<float, 4> box{x - w/2, y - h/2, x + w/2, y + h/2};
        float area = std::max(0.f, box[2] - box[0]) * std::max(0.f, box[3] - box[1]);
        groups[cls].push_back({i, cls, score, box, area});
    }

    std::vector<Box> kept;
    for(auto& group : groups) {
        std::stable_sort(group.begin(), group.end(),
            [](const Box& a, const Box& b) { return a.score > b.score; });
        std::vector<unsigned char> removed(group.size(), 0);
        for(size_t i = 0; i < group.size(); ++i) {
            if(removed[i]) continue;
            const Box& a = group[i];
            kept.push_back(a);
            for(size_t j = i + 1; j < group.size(); ++j) {
                if(removed[j]) continue;
                const Box& b = group[j];
                float intersection = std::max(0.f, std::min(a.xyxy[2], b.xyxy[2]) - std::max(a.xyxy[0], b.xyxy[0])) *
                    std::max(0.f, std::min(a.xyxy[3], b.xyxy[3]) - std::max(a.xyxy[1], b.xyxy[1]));
                float iou = intersection / (a.area + b.area - intersection + 1e-9f);
                if(iou > opt.iou_thresh) removed[j] = 1;
            }
        }
    }
    std::stable_sort(kept.begin(), kept.end(),
        [](const Box& a, const Box& b) { return a.score > b.score; });

    std::vector<DetectResult> result;
    for(const Box& box : kept) {
        double x1 = (double(box.xyxy[0]) - meta.pad_left) / meta.scale;
        double y1 = (double(box.xyxy[1]) - meta.pad_top) / meta.scale;
        double x2 = (double(box.xyxy[2]) - meta.pad_left) / meta.scale;
        double y2 = (double(box.xyxy[3]) - meta.pad_top) / meta.scale;
        if(x2 < 0 || y2 < 0 || x1 >= meta.src_w || y1 >= meta.src_h) continue;

        DetectResult d;
        d.cls_id = box.cls;
        d.cls_name = CLASSES[box.cls];
        d.score = box.score;
        for(int j = 0; j < 4; ++j) {
            d.kpt[j][0] = (double(raw[(28+2*j)*n + box.index]) - meta.pad_left) / meta.scale;
            d.kpt[j][1] = (double(raw[(29+2*j)*n + box.index]) - meta.pad_top) / meta.scale;
        }
        // 中心沿用未裁剪的框，避免边缘目标因裁剪而偏移
        d.center_x = (x1 + x2) / 2;
        d.center_y = (y1 + y2) / 2;
        if(opt.center_mode == "keypoints") {
            d.center_x = d.center_y = 0;
            for(const auto& p : d.kpt) { d.center_x += p[0]; d.center_y += p[1]; }
            d.center_x /= 4;
            d.center_y /= 4;
        }
        if(d.center_x < 0 || d.center_x >= meta.src_w || d.center_y < 0 || d.center_y >= meta.src_h) continue;
        x1 = std::clamp(x1, 0.0, double(meta.src_w - 1));
        y1 = std::clamp(y1, 0.0, double(meta.src_h - 1));
        x2 = std::clamp(x2, 0.0, double(meta.src_w - 1));
        y2 = std::clamp(y2, 0.0, double(meta.src_h - 1));
        d.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
        result.push_back(std::move(d));
    }
    return result;
}

cv::Mat draw_detection(const cv::Mat& src, const std::vector<DetectResult>& dets)
{
    cv::Mat out = src.clone();
    for(auto& d : dets)
    {
        cv::Point pt1(cvRound(d.bbox.x), cvRound(d.bbox.y));
        cv::Point pt2(cvRound(d.bbox.x + d.bbox.width), cvRound(d.bbox.y + d.bbox.height));
        cv::rectangle(out, pt1, pt2, cv::Scalar(0, 255, 0), 1);
        cv::drawMarker(out, cv::Point(cvRound(d.center_x), cvRound(d.center_y)),
                       cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 1);

        char buf[128];
        snprintf(buf, sizeof(buf), "%s %.2f (%d,%d)", d.cls_name.c_str(), d.score,
                 int(std::nearbyint(d.center_x)), int(std::nearbyint(d.center_y)));
        cv::putText(out, buf, cv::Point(std::max(0, pt1.x), std::max(12, pt1.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 0), 1);
    }
    return out;
}

json detection_to_json(const std::vector<DetectResult>& dets, const PreprocMeta& meta)
{
    json arr = json::array();
    for(auto& d : dets)
    {
        json jkpt = json::array();
        for(auto& p : d.kpt)
            jkpt.push_back({round_double(p[0], 2), round_double(p[1], 2)});

        arr.push_back({
            {"class_id", d.cls_id},
            {"class_name", d.cls_name},
            {"confidence", round_double(d.score, 6)},
            {"center", {round_double(d.center_x, 2), round_double(d.center_y, 2)}},
            {"center_normalized", {round_double(d.center_x / meta.src_w, 6), round_double(d.center_y / meta.src_h, 6)}},
            {"offset_from_image_center", {round_double(d.center_x - (meta.src_w - 1) / 2.0, 2), round_double(d.center_y - (meta.src_h - 1) / 2.0, 2)}},
            {"bbox_xyxy", {
                round_double(d.bbox.x, 2),
                round_double(d.bbox.y, 2),
                round_double(d.bbox.x + d.bbox.width, 2),
                round_double(d.bbox.y + d.bbox.height, 2)
            }},
            {"keypoints", jkpt}
        });
    }
    return arr;
}

struct PerfStat
{
    bool enable{false};
    size_t skip_warmup{3};
    size_t count{0};
    double wait_ms{0}, pre_ms{0}, infer_ms{0}, post_ms{0}, output_ms{0};

    void add(double w, double pre, double inf, double post, double out)
    {
        if(!enable) return;
        if(count < skip_warmup) { count++; return; }
        wait_ms   += w;
        pre_ms    += pre;
        infer_ms  += inf;
        post_ms   += post;
        output_ms += out;
        count++;
        if((count - skip_warmup) % 30 == 0)
            print();
    }

    void print()
    {
        if(count <= skip_warmup) return;
        size_t samples = count - skip_warmup;
        double avg_wait  = wait_ms   / samples;
        double avg_pre   = pre_ms    / samples;
        double avg_infer = infer_ms  / samples;
        double avg_post  = post_ms   / samples;
        double avg_out   = output_ms / samples;
        double total = avg_wait + avg_pre + avg_infer + avg_post + avg_out;
        double fps = 1000.0 / total;
        fprintf(stderr, "[PROFILE] n=%zu wait=%.2f pre=%.2f infer=%.2f post=%.2f out=%.2f loop=%.2f fps=%.2f\n",
                samples, avg_wait, avg_pre, avg_infer, avg_post, avg_out, total, fps);
    }

    ~PerfStat()
    {
        if(enable && count > skip_warmup && (count - skip_warmup) % 30 != 0) print();
    }
};

int main(int argc, char** argv)
{
    try {
        signal(SIGINT, sig_handler);
        signal(SIGTERM, sig_handler);

        Options opt;
        if(!parse_args(argc, argv, opt))
        {
            for(int i = 1; i < argc; ++i)
                if(std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") return 0;
            return 1;
        }

        // 保存原始stdout用于JSON输出，之后把stdout重定向到stderr
        int fd = dup(STDOUT_FILENO);
        if(fd < 0) throw std::runtime_error("dup stdout failed");
        FILE* stream = fdopen(fd, "w");
        if(!stream) { close(fd); throw std::runtime_error("fdopen failed"); }
        std::unique_ptr<FILE, decltype(&fclose)> json_stdout(stream, fclose);
        fflush(stdout);
        if(dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("redirect stdout failed");

        cv::setNumThreads(1);
        setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 0);

        CviInfer infer(opt.model_path);

        CameraProcess camera;
        std::unique_ptr<LatestFrames> frame_source;
        cv::Mat single_image;

        if(opt.use_csi) {
            camera.start(opt.camera_log);
            frame_source = camera.waitForFrames();
        } else if(!opt.image_path.empty()) {
            single_image = cv::imread(opt.image_path);
            if(single_image.empty()) {
                fprintf(stderr, "cannot read image %s\n", opt.image_path.c_str());
                return 1;
            }
        } else {
            frame_source = std::make_unique<LatestFrames>(opt.source);
        }

        std::ofstream json_file;
        if(!opt.save_json.empty()) {
            json_file.open(opt.save_json);
            if(!json_file.is_open()) {
                fprintf(stderr, "cannot open save json file %s\n", opt.save_json.c_str());
                return 1;
            }
        }

        ArmorPreproc preproc;
        PerfStat perf;
        perf.enable = opt.enable_profile;

        uint64_t frame_seq = 0;
        uint64_t processed_cnt = 0;

        while(running)
        {
            auto t_loop_start = Clock::now();
            auto capture_ts = t_loop_start;
            cv::Mat frame;
            if(!single_image.empty()) {
                frame = single_image;
            } else {
                frame = frame_source->next(frame_seq, capture_ts);
                if(!running) break;
            }
            auto t_frame_acquired = Clock::now();

            PreprocMeta meta = preproc.process(frame, infer.get_input_buffer());
            auto t_pre_done = Clock::now();

            double infer_cost_ms = infer.forward();
            auto t_infer_done = Clock::now();

            auto det_list = post_process(infer.get_output_buffer(), meta, opt);
            auto t_post_done = Clock::now();

            json det_json = detection_to_json(det_list, meta);
            json result{
                {"frame_id", processed_cnt},
                {"timestamp", std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()},
                {"image_size", {meta.src_w, meta.src_h}},
                {"coordinate_system", "original_image_pixels_top_left_xy"},
                {"center_method", opt.center_mode},
                {"target", det_json.empty() ? json(nullptr) : det_json[0]},
                {"inference_ms", round_double(infer_cost_ms, 2)},
                {"frame_age_ms", round_double(DurationMs(Clock::now() - capture_ts).count(), 2)},
                {"found", !det_json.empty()},
                {"detections", det_json}
            };
            std::string out_line = result.dump();
            if(fprintf(json_stdout.get(), "%s\n", out_line.c_str()) < 0 || fflush(json_stdout.get()))
                throw std::runtime_error("JSON stdout write failed");
            if(json_file.is_open()) {
                json_file << out_line << '\n';
                json_file.flush();
                if(!json_file) throw std::runtime_error("JSON file write failed");
            }

            if(!opt.save_image.empty()) {
                cv::Mat vis = draw_detection(frame, det_list);
                if(!cv::imwrite(opt.save_image, vis)) throw std::runtime_error("image write failed");
            }
            if(!opt.save_raw.empty()) {
                saveRaw(opt.save_raw, infer.get_output_buffer());
            }
            auto t_output_done = Clock::now();

            double wait_ms  = DurationMs(t_frame_acquired - t_loop_start).count();
            double pre_ms   = DurationMs(t_pre_done - t_frame_acquired).count();
            double post_ms  = DurationMs(t_post_done - t_infer_done).count();
            double out_ms   = DurationMs(t_output_done - t_post_done).count();
            perf.add(wait_ms, pre_ms, infer_cost_ms, post_ms, out_ms);

            processed_cnt++;

            if(!single_image.empty()) break;
            if(opt.max_frames > 0 && processed_cnt >= static_cast<uint64_t>(opt.max_frames)) break;
        }

        if(json_file.is_open()) json_file.close();
        fprintf(stderr, "exit, processed %llu frames\n", static_cast<unsigned long long>(processed_cnt));
        return 0;
    } catch(const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
