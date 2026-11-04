#include <limits>
#define main detector_main
#include "armor_detector.cpp"
#undef main

void check(bool ok, const char* message)
{
    if(!ok) throw std::runtime_error(message);
}

int main()
{
    Options opt;
    ArmorPreproc pre;
    std::vector<float> input(3*model_const::INPUT_PLANE);
    auto meta = pre.process(cv::Mat::zeros(100,200,CV_8UC3), input.data());
    check(meta.pad_top==160 && meta.pad_left==0 && meta.scale==3.2, "letterbox");
    check(input[0]==114.f/255 && input[160*640]==0, "padding");
    constexpr int n = model_const::OUTPUT_NUM_ANCHOR;
    std::vector<float> raw(36*n,0);
    check(post_process(raw.data(),meta,opt).empty(), "empty output");
    for(int i=0;i<2;++i) {
        raw[i]=320; raw[n+i]=320; raw[2*n+i]=64; raw[3*n+i]=32;
        raw[7*n+i]=i==0 ? .9f : .8f;
        for(int j=0;j<4;++j) {
            raw[(28+2*j)*n+i]=288;
            raw[(29+2*j)*n+i]=320;
        }
    }
    auto det = post_process(raw.data(),meta,opt);
    check(det.size()==1 && det[0].center_x==100 && det[0].center_y==50, "same class NMS and coordinates");
    auto output = detection_to_json(det,meta);
    check(output[0].contains("center_normalized") && output[0].contains("offset_from_image_center"), "JSON fields");
    raw[7*n+1]=0; raw[8*n+1]=.8f;
    check(post_process(raw.data(),meta,opt).size()==2, "different classes must survive");
    opt.center_mode="keypoints";
    check(post_process(raw.data(),meta,opt)[0].center_x==90, "keypoint center");
    opt.center_mode="bbox";
    // Unclipped center is 5; clipped box center would be 7.5.
    raw.assign(36*n,0);
    PreprocMeta edge;
    edge.src_w=200; edge.src_h=100;
    raw[0]=5; raw[n]=50; raw[2*n]=20; raw[3*n]=10; raw[7*n]=.9f;
    check(post_process(raw.data(),edge,opt)[0].center_x==5, "center before clipping");
    raw[35*n]=std::numeric_limits<float>::quiet_NaN();
    bool rejected=false;
    try { post_process(raw.data(),edge,opt); } catch(const std::runtime_error&) { rejected=true; }
    check(rejected,"non-finite output");
    PerfStat perf;
    perf.enable=true;
    perf.print(); // No samples: must not underflow size_t.
    fprintf(stderr,"checks passed\n");
}
