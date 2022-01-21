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

// added by Holy 2201201549
#include <ctime>
#include "mman.h"
#include <sys/stat.h>
#include <fcntl.h>
// end of addition 2201201549

using namespace std;
using namespace cv;

// added by Holy 2201201549
void handle_error(const char* msg) {
    perror(msg); 
    exit(255);
}

const char* map_file(const char* fname, size_t& length)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1)
        handle_error("open");

    // obtain file size
    struct stat sb;
    if (fstat(fd, &sb) == -1)
        handle_error("fstat");

    length = sb.st_size;

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));
    if (addr == MAP_FAILED)
        handle_error("mmap");

    // TODO close fd at some point in time, call munmap(...)
    return addr;
}
// end of addition 2201201549

int classify_wuSNet(const cv::Mat& bgr, std::vector<float>& cls_scores , string paramPath,string binPath)
{

    ncnn::Net wuSNet;
    
    wuSNet.opt.use_vulkan_compute = false;
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    wuSNet.load_param(paramPath.data());
    wuSNet.load_model(binPath.data());
	
	// cv::Mat roiImg=bgr(Range(245,474),Range(5,708)).clone();//提取Z75数据集卷扬区域的图像

    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(roiImg.data, ncnn::Mat::PIXEL_BGR2RGB, roiImg.cols, roiImg.rows, 174, 174);//recommend

    // cv::Mat roiImg = bgr(Range(276,276+201),Range(25,25+681)).clone(); // added by Holy 2109090810
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(roiImg.data, ncnn::Mat::PIXEL_BGR, roiImg.cols, roiImg.rows, 224, 224); // added by Holy 2109090810
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224); // added by Holy 2109131500

    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};//recommend 均值
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};//recommend 方差

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = wuSNet.create_extractor();
    // ex.input("input.1", in);
    ex.input("x", in); // added by Holy 2109090810
    ncnn::Mat out;
    // ex.extract("78", out);
    ex.extract("y", out); // added by Holy 2109090810

    cls_scores.resize(out.w);

    // added by Holy 2111181500
    cout << "length of out: " << out.w << endl;
    // best_threshold = array(40.58083, dtype=float32)
    // density.mean = tensor([0.0028, 0.0086, 0.0213,  ..., 0.0125, 0.0163, 0.0005])
    // end of addition 2111181500

    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, vector<float> data_vector, vector<float> data_vector_mean, float data_threshold)
{
    // partial sort topk with index
    int size = cls_scores.size();

    // added by Holy 2111201500
    vector<float> normalizedData_l2;
    normalize(cls_scores, normalizedData_l2, 1.0, 0.0, NORM_L2);

    Mat mat_data_vector_sample(1, 1024, CV_32FC1, (float *)normalizedData_l2.data());
    
    Mat mat_data_vector_mean(1, 1024, CV_32FC1, (float *)data_vector_mean.data());
    Mat mat_data_vector(1024, 1024, CV_32FC1, (float *)data_vector.data());
    
    double Maha = Mahalanobis(mat_data_vector_sample, mat_data_vector_mean, mat_data_vector);
	cout << "Maha distance:\t" << Maha << endl;

    int cls = 0;
    if (Maha < data_threshold)
    {
        cls = 1;
    }
    return cls;
    // end of addition 2111201500

    // hided by Holy 2111201500
    /*
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
    return top1_index;*/
    // end of hide 2111201500
}


int main(int argc, char** argv)
{
    using namespace std;
    using namespace cv;

    string path = argv[1];//输入图片文件夹地址
    string inputPath=argv[2];//输入模型地址文件名
    
    clock_t clock_t_clock_time = clock(); // added by Holy 2201201549
    
    // added by Holy 2111201500
    // read inv_cov
    string txtPath = "d:/backup/project/learn_pytorch/test_cutpaste/data_inv_cov.txt";

    ifstream in_file;
    in_file.open(txtPath);

    vector<float> data_vector;

    if (in_file)
    {
        string line;
        float number;

        for (int i = 0; i < 1; ++i)
        {
            getline(in_file, line);
            istringstream iss(line);
            while (iss >> number)
            {
                data_vector.push_back(number);
            }
        }
        in_file.close();
        in_file.clear();
    }
    else
    {
        throw runtime_error("document error");
    }

    cout << "data_vector.size: " << data_vector.size() << endl;

    // added by Holy 2201201549
    clock_t_clock_time = clock() - clock_t_clock_time;
    printf("read inv_cov: %f seconds\n", (static_cast<float>(clock_t_clock_time)) / CLOCKS_PER_SEC);

    size_t length;
    auto f = map_file(txtPath.c_str(), length);
    // auto f = map_file("d:/temp/install_v1.3_211027/bin/param_w6013.ini", length);
    auto l = f + length;

    std::cout << "f = " << f << "\n";

    uintmax_t m_numLines = 0;
    while (f && f != l)
    {
        if ((f = static_cast<const char *>(memchr(f, '\n', l - f))))
        {
            m_numLines++, f++;            
        }
    }

    std::cout << "m_numLines = " << m_numLines << "\n";
    std::cout << "file length = " << length << "\n";

    return 0;
    // end of addition 2201201549
    
    // read mean
    txtPath = "d:/backup/project/learn_pytorch/test_cutpaste/data_mean.txt";

    ifstream in_file_mean;
    in_file_mean.open(txtPath);

    vector<float> data_vector_mean;

    if (in_file_mean)
    {
        string line_mean;
        float number_mean;

        for (int i = 0; i < 1; ++i)
        {
            getline(in_file_mean, line_mean);
            istringstream iss(line_mean);
            while (iss >> number_mean)
            {
                data_vector_mean.push_back(number_mean);
            }
        }
        in_file_mean.close();
        in_file_mean.clear();
    }
    else
    {
        throw runtime_error("document error");
    }

    cout << "data_vector_mean.size: " << data_vector_mean.size() << endl;
    cout << "data_vector_mean[0]: " << data_vector_mean[0] << endl;
    cout << "data_vector_mean[100]: " << data_vector_mean[100] << endl;

    // read best threshold
    txtPath = "d:/backup/project/learn_pytorch/test_cutpaste/data_threshold.txt";

    ifstream in_file_threshold;
    in_file_threshold.open(txtPath);

    float data_threshold;

    if (in_file_threshold)
    {
        string line_threshold;        

        for (int i = 0; i < 1; ++i)
        {
            getline(in_file_threshold, line_threshold);
            istringstream iss(line_threshold);
            iss >> data_threshold;            
        }
        in_file_threshold.close();
        in_file_threshold.clear();
    }
    else
    {
        throw runtime_error("document error");
    }

    cout << "best threshold: " << data_threshold << endl;
    // end of addition 2111201500

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

        result=print_topk(cls_scores, data_vector, data_vector_mean, data_threshold);

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
