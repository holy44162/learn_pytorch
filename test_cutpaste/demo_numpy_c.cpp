#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

std::vector< std::vector<float> > to_2d( const std::vector<float>& flat_vec, std::size_t ncols )
{
    // ncols is the number of cols in the matrix
    // the size of the flat vector must be an integral multiple of some non-zero ncols
    // if the check fails, report an error (by throwing an exception)
    if( ncols == 0 || flat_vec.size()%ncols != 0 ) throw std::domain_error( "bad #cols" ) ;

    // compute the number of rows in the matrix ie. flat_vec.size() == nrows * ncols
    const auto nrows = flat_vec.size() / ncols ;

    // declare an empty matrix. eventually, we build this up to an nrows x ncols matrix
    // the final matrix would be a collection of nrows rows
    // exch row in the matrix would a collection (a vector) containing ncols elements
    std::vector< std::vector<float> > mtx ;

    // get an iterator to the beginning of the flat vector
    // http://en.cppreference.com/w/cpp/iterator/begin
    // if you are unfamiliar with iterators, see: https://cal-linux.com/tutorials/STL.html
    const auto begin = std::begin(flat_vec) ;

    // add rows one by one to the matrix
    for( std::size_t row = 0 ; row < nrows ; ++row ) // for each row [0,nrows-1] in the matrix
    {
        // add the row (a vector of ncols elements)
        // for example, if ncols = 12,
        // row 0 would contain elements in positions 0, 1, 2, ...10, 11 ie. [0,12)
        // row 1 would contain elements in positions 12, 13, ...23 ie. [12,24)
        // row 2 would contain elements in positions 24, 25, ...35 ie. [24,36)
        // in general, row r would contain elements at positions [ r*12, (r+1)*12 ]
        mtx.push_back( { begin + row*ncols, begin + (row+1)*ncols } ) ;
        // the above as akin to:
        // construct the row containing elements at positions as described earlier
        // const std::vector<T> this_row( begin + row*ncols, begin + (row+1)*ncols ) ;
        // mtx.push_back(this_row) ; // and add this row to the back of the vector
    }

    return mtx ; // return the fully populated matrix
}

int main(int argc, char *argv[])
{
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

    std::vector< std::vector<float> > data_vector_2d;

    size_t n_cols = 1024;
    data_vector_2d = to_2d(data_vector, n_cols);

    cout << "data_vector_2d.size: " << data_vector_2d.size() << endl;
    cout << "data_vector_2d[0].size: " << data_vector_2d[0].size() << endl;
    cout << "data_vector_2d[0][0]: " << data_vector_2d[0][0] << endl;
    cout << "data_vector_2d[1][1]: " << data_vector_2d[1][1] << endl;

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

    // calculate mahalanobis distance
    Mat Pt(2, 5, CV_64FC1);
	Pt.at<double>(0, 0) = 2;
	Pt.at<double>(1, 0) = 4;
 
	Pt.at<double>(0, 1) = 2;
	Pt.at<double>(1, 1) = 2;
 
	Pt.at<double>(0, 2) = 3;
	Pt.at<double>(1, 2) = 3;
 
	Pt.at<double>(0, 3) = 4;
	Pt.at<double>(1, 3) = 4;
 
	Pt.at<double>(0, 4) = 4;
	Pt.at<double>(1, 4) = 2;
	cout << Pt << endl;
 
  	// calculate covariance matrix
	Mat coVar, meanVar;
	calcCovarMatrix(Pt, coVar, meanVar, COVAR_NORMAL|COVAR_COLS);
	cout << "Covar is:\n" << coVar << endl;
	cout << "Mean is:\n" << meanVar << endl;
	// calculate inverse of covariance matrix
	Mat iCovar;
	invert(coVar, iCovar, DECOMP_SVD);
 
	Mat pt1(2, 1, CV_64FC1);
	Mat pt2(2, 1, CV_64FC1);
	pt1.at<double>(0, 0) = 1;
	pt1.at<double>(1, 0) = 1;
	pt2.at<double>(0, 0) = 5;
	pt2.at<double>(1, 0) = 5;
 
	double Maha1 = Mahalanobis(pt1, meanVar, iCovar);
	double Maha2 = Mahalanobis(pt2, meanVar, iCovar);
	cout << "Maha distance 1:\t" << Maha1 << endl;
	cout << "Maha distance 2:\t" << Maha2 << endl;

    vector<float> data_vector_sample(1024, 10.);
    data_vector_sample[6] = 7.;

    Mat mat_data_vector_sample(1, 1024, CV_32FC1, (float *)data_vector_sample.data());
    cout << "mat_data_vector_sample(0,6): " << mat_data_vector_sample.at<float>(0,6) << endl;
    cout << "mat_data_vector_sample(0,16): " << mat_data_vector_sample.at<float>(0,16) << endl;

    Mat mat_data_vector_mean(1, 1024, CV_32FC1, (float *)data_vector_mean.data());
    Mat mat_data_vector(1024, 1024, CV_32FC1, (float *)data_vector.data());
    cout << "mat_data_vector_mean(0,6): " << mat_data_vector_mean.at<float>(0,6) << endl;
    cout << "mat_data_vector_mean(0,16): " << mat_data_vector_mean.at<float>(0,16) << endl;
    cout << "mat_data_vector(10,6): " << mat_data_vector.at<float>(10,6) << endl;
    cout << "mat_data_vector(20,16): " << mat_data_vector.at<float>(20,16) << endl;

    double Maha3 = Mahalanobis(mat_data_vector_sample, mat_data_vector_mean, mat_data_vector);
	cout << "Maha distance 3:\t" << Maha3 << endl;
    
    /*
    // to_2d( flat, 12 ) returns a 10 x 12 matrix
    // range based loop: http://www.stroustrup.com/C++11FAQ.html#for
    // auto: http://www.stroustrup.com/C++11FAQ.html#auto
    for( const auto& row : to_2d( data_vector, 1024 ) ) // for each row in the returned matrix
    {
        // print each int in that row
        for( int v : row ) std::cout << v << ' ' ;
        std::cout << '\n' ;
    }
    // note: we could write the above as
    // const auto mtx = to_2d( flat, 12 ) ; // get the matrix returned by the function
    // for( const auto& row : mtx ) // for each row in the matrix
    // { etc...

    std::cout << "---------------------------\n" ;

    for( const auto& row : to_2d( data_vector, 256 ) ) // 4096 x 256
    {
        for( int v : row ) std::cout << v << ' ' ;
        std::cout << '\n' ;
    }

    std::cout << "---------------------------\n" ;

    try // 17 is an invalid value for the number of columns
    {
        // to_2d( flat, 17 ) would report an error by throwing an exception
        // when that happens, control would be transferred to the catch block
        for( const auto& row : to_2d( data_vector, 17 ) ) // bad ncols
        {
            for( int v : row ) std::cout << v << ' ' ;
            std::cout << '\n' ;
        }
    }
    catch( const std::exception& e ) // catch the error (exception)
    {
        // and print out an error messagr
        std::cerr << "*** error: " << e.what() << '\n' ;
    }*/
    
    return 0;
}
