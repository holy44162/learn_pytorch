#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

#include <iostream> // tested by Holy 2109010810

static int detect_squeezenet(const cv::Mat &bgr, std::vector<float> &cls_scores)
{
    ncnn::Net squeezenet;

    // squeezenet.opt.use_vulkan_compute = true; // hided by Holy 2109010810

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    squeezenet.load_param("shufflenet_v2_x1_0.param");
    squeezenet.load_model("shufflenet_v2_x1_0.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    // const float mean_vals[3] = {104.f, 117.f, 123.f};
    // in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    // added by Holy 2109010810
    ex.set_light_mode(true);
    ex.set_num_threads(4); // should be commented in qnx
    // end of addition 2109010810

    ex.input("x", in);

    ncnn::Mat out;
    ex.extract("y", out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float> &cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char *imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    print_topk(cls_scores, 2);

    return 0;
}