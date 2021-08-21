// super 2021.07.30


#include "net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

#include <io.h>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int classify_wuSNet(const cv::Mat& bgr, std::vector<float>& cls_scores , string paramPath,string binPath)
{

    ncnn::Net wuSNet;
    
    wuSNet.opt.use_vulkan_compute = false;
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    wuSNet.load_param(paramPath.data());
    wuSNet.load_model(binPath.data());
	
	cv::Mat roiImg=bgr(Range(276,276+201),Range(25,25+681)).clone();//提取Z75数据集卷扬区域的图像

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roiImg.data, ncnn::Mat::PIXEL_BGR2RGB, roiImg.cols, roiImg.rows, 224, 224);//recommend

    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};//recommend 均值
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};//recommend 方差

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = wuSNet.create_extractor();
    ex.input("input.1", in);
    ncnn::Mat out;
    ex.extract("191", out);

    cls_scores.resize(out.w);

    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }
    int top1_index=vec[0].second;
    return top1_index;
}


int main(int argc, char** argv)
{
    using namespace std;
    using namespace cv;

    string path = argv[1];//输入图片文件夹地址
    string inputPath=argv[2];//输入模型地址文件名

    // added by Holy 2108061500
    string strDatasetPrefix = path;
    path += "imgs";
    // end of addition 2108061500

    string paramPath,binPath; 
    int result;
    paramPath= inputPath+".param";
    binPath = inputPath+".bin";

    printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
    printf("\nparamPath: %s\nbinPath: %s\n",paramPath.data(),binPath.data());//加载模型地址
    printf("\nimageDir: %s\n",path.data());//图片文件夹地址

    vector<String> fn;
    glob(path, fn, false);
    size_t count = fn.size();
    printf("\nimage Number: %d\n",int(count));

    printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
    int count0=0,count1=0;

    // added by Holy 2108061500
    bool bMess;
	double dF1;	
	vector<string> vecStrImgPathName;
	vector<bool> vecBMess, vecBMessYTest, vecBResult;
    char cAYTestLine[16];	
    string strYTest = strDatasetPrefix + "y_Test.txt";

    std::ifstream fInYTest(strYTest);

	if (!fInYTest.is_open())
	{
		cout << "cannot open file: " << strYTest << endl;
	}
	else
	{
		while (!fInYTest.eof())
		{
			fInYTest.getline(cAYTestLine, 16);
			if (cAYTestLine[0] != '\0')
			{
				istringstream(cAYTestLine) >> bMess;
				vecBMessYTest.push_back(bMess);
			}
		}
	}
	vecBResult.resize(vecBMessYTest.size());
    // end of addition 2108061500

    for(int i=0;i<count;i++){
        printf("number:%d\n",i);
        cout << fn[i] << endl;
        cv::Mat m = cv::imread(fn[i], 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", fn[i].c_str());//fn[i].c_str()
            return -1;
        }
        printf("load image ok!!\n");

        std::vector<float> cls_scores;
        classify_wuSNet(m, cls_scores,paramPath,binPath);

        result=print_topk(cls_scores, 2);

        // added by Holy 2108061500
        vecBMess.push_back(bool(1-result));
        cout << "vecBMess: " << vecBMess[i] << endl;
        cout << "vecBMessYTest: " << vecBMessYTest[i] << endl;
        cout << "i Count: " << i + 1 << endl;
        // end of addition 2108061500

        if(result==0){
            count0++;
        }
        else{count1++;};
        printf("*****************************\n");

    }
    printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
    printf("\nparamPath: %s\nbinPath: %s",paramPath.data(),binPath.data());//加载模型地址
    printf("\nimageDir: %s",path.data());//图片文件夹地址
    printf("\nimage Number: %d\n",int(count));//测试图片数量
    printf("0: %d,  1: %d\n",count0,count1);//结果为1及0的图片分别的数量

    // added by Holy 2108061500
    transform(vecBMess.begin(), vecBMess.end(),
              vecBMessYTest.begin(), vecBResult.begin(), logical_and<bool>());
    uint_fast32_t tp, fp, fn1;
    tp = static_cast<uint_fast32_t>(std::count(vecBResult.begin(), vecBResult.end(), true));

    vecBMessYTest.flip();
    transform(vecBMess.begin(), vecBMess.end(),
              vecBMessYTest.begin(), vecBResult.begin(), logical_and<bool>());
    fp = static_cast<uint_fast32_t>(std::count(vecBResult.begin(), vecBResult.end(), true));

    vector<uint_fast32_t> indFn, indFp;
    for (uint_fast32_t ind = 0; ind < static_cast<uint_fast32_t>(vecBResult.size()); ind++)
    {
        if (vecBResult[ind])
        {
            indFp.push_back(ind);
        }
    }

    vecBMessYTest.flip();
    vecBMess.flip();
    transform(vecBMess.begin(), vecBMess.end(),
              vecBMessYTest.begin(), vecBResult.begin(), logical_and<bool>());
    fn1 = static_cast<uint_fast32_t>(std::count(vecBResult.begin(), vecBResult.end(), true));

    for (uint_fast32_t ind1 = 0; ind1 < static_cast<uint_fast32_t>(vecBResult.size()); ind1++)
    {
        if (vecBResult[ind1])
        {
            indFn.push_back(ind1);
        }
    }

    double prec = static_cast<double>(tp) / static_cast<double>(tp + fp);
    double rec = static_cast<double>(tp) / static_cast<double>(tp + fn1);
    dF1 = 2 * prec * rec / (prec + rec);

    printf("data: F1 is %f, prec is %f, rec is %f, tp is %lu, fp is %lu, fn is %lu\r\n",
           dF1, prec, rec, tp, fp, fn1);

    cout << "indFn is: ";
    copy(indFn.begin(), indFn.end(), ostream_iterator<uint_fast32_t>(cout, " "));
    cout << endl;
    cout << "indFp is: ";
    copy(indFp.begin(), indFp.end(), ostream_iterator<uint_fast32_t>(cout, " "));
    cout << endl;
    // end of addition 2108061500

    return 0;
}
