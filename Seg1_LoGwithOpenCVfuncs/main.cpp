#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat myLoG(Mat src, int ddepth, int ksize){
    Mat GBImg, LoG;
    GaussianBlur(src, GBImg, Size(3,3),0,0);
    Laplacian(GBImg,LoG,ddepth,1);
    return LoG;
}

int main(int argc, char** argv)
{
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -1;

    /*
    Mat GBImg;
    Mat Gkernel = getGaussianKernel(4,25);
    filter2D(src, GBImg,ddepth,Gkernel);


    Mat GBImg;
    GaussianBlur(src, GBImg, Size(3,3),0,0);

    Mat LoG;
    Laplacian(GBImg,LoG,CV_8U,1);
    */

    float c=-1;
    Mat LoG = myLoG(src,CV_8U,1);
    Mat output = (src+c*LoG);

    imshow("Original",src);
    imshow("Laplacian of Gaussian", LoG);
    imshow("Output", output);

    waitKey(0);
    return 0;
}







