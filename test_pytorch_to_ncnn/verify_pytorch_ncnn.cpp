#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "net.h"

using namespace std;

vector<float> get_output(const ncnn::Mat &m)
{
    vector<float> res;
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                res.push_back(ptr[x]);
            }
            ptr += m.w;
        }
    }
    return res;
}

int main()
{
    cv::Mat img = cv::imread("d:/data_seq/gongqiWinding/Z75_DF-4105H-BD/210820/shrinkVideo/smallDatasets/test/imgs/img00001.jpg");
    int w = img.cols;
    int h = img.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 224, 224);

    ncnn::Net net;
    net.load_param("resnet18.param");
    net.load_model("resnet18.bin");
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("x", in);
    ncnn::Mat feat;
    ex.extract("y", feat);
    vector<float> res = get_output(feat);
    vector<float>::iterator max_id = max_element(res.begin(), res.end());
    printf("predicted class: %d, predicted value: %f", max_id - res.begin(), res[max_id - res.begin()]);
    // net.clear();
    return 0;
}
