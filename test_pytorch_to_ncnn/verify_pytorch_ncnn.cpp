#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "net.h"

#include <iostream> // tested by Holy 2109010810

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
    std::cout << "res[534]: " << res[534] << std::endl; // tested by Holy 2109010810
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
    // tested by Holy 2109010810
    cout << "y.c: " << feat.c << endl;
    cout << "y.w: " << feat.w << endl;
    cout << "y.h: " << feat.h << endl;
    // end of test 2109010810
    vector<float> res = get_output(feat);
    vector<float>::iterator max_id = max_element(res.begin(), res.end());
    printf("predicted class: %d, predicted value: %f", max_id - res.begin(), res[max_id - res.begin()]);
    // tested by Holy 2109010810
    printf("\r\n max_id: %d, res.begin: %d\r\n", max_id, res.begin());
    printf("\r\n max_id - res.begin: %d\r\n", max_id-res.begin());
    cout << "size of res: " << res.size() << endl;
    // end of test 2109010810
    // net.clear();
    return 0;
}
