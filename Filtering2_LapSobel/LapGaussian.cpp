#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -1;

    Mat output;
    GaussianBlur(src,output,Size(3,3),0,0);

    Mat lap_output;
    Mat lap_src;
    Laplacian(src,lap_src,CV_8U);
    Laplacian(output,lap_output,CV_8U);

    float c=-1;
    Mat res_src = (src+c*lap_src);
    Mat res_output = (src+c*lap_output);

    imshow("Original", src);
    imshow("Gaussian Blur", output);
    imshow("Lap_src", lap_src);
    imshow("Lap_output", lap_output);
    imshow("Res_src", res_src);
    imshow("Res_output", res_output);


    waitKey(0);
    return 0;
}
