#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat mean3x3(Mat);
Vec3b submean3x3(Mat,int,int);

Mat mean3x3_gs(Mat);
unsigned char submean3x3_gs(Mat,int,int);

int main(int argc, char** argv)
{
    Mat img = imread(argv[1],IMREAD_COLOR);
    if( img.empty() ) return -1;

    Mat img_gs = imread(argv[1],IMREAD_GRAYSCALE);
    if (img_gs.empty() ) return -1;


    imshow("window_gs",img_gs);
    imshow("window_gs2",mean3x3_gs(img_gs));

    waitKey(0);

    imshow("window",img);
    imshow("window2",mean3x3(img));

    waitKey(0);
    return 0;
}


Mat mean3x3(Mat img){

    //1.
    Mat imgPadded;
    Scalar value(0,0,0);    //black padding
    copyMakeBorder(img,imgPadded,1,1,1,1,BORDER_CONSTANT,value);

    //2.
    Mat result(img.rows,img.cols,img.type()); //CV_8UC3

    //3.
    int i,j;
    for(i=1; i<imgPadded.rows-1; i++)   //2 -> num pixel added left/right
        for(j=1; j<imgPadded.cols-1; j++)   //2 -> num pixel added top/bottom
            result.at<Vec3b>(i-1,j-1) = submean3x3(imgPadded,i,j);

    return result;
}

Vec3b submean3x3(Mat img, int istart, int jstart){
    Vec3i sum(0,0,0);
    for(int i=istart-1; i<istart+2; i++)
        for(int j=jstart-1; j<jstart+2; j++)
            sum += img.at<Vec3b>(i,j);

    return Vec3b(sum/9);
}


//-----------------------------

Mat mean3x3_gs(Mat img){

    //1.
    Mat imgPadded;
    Scalar value(0);    //black padding
    copyMakeBorder(img,imgPadded,1,1,1,1,BORDER_CONSTANT,value);

    //2.
    Mat result(img.rows,img.cols,img.type()); //CV_8UC3

    //3.
    int i,j;
    for(i=1; i<imgPadded.rows-1; i++)   //2 -> num pixel added left/right
        for(j=1; j<imgPadded.cols-1; j++)   //2 -> num pixel added top/bottom
            result.at<unsigned char>(i-1,j-1) = submean3x3_gs(imgPadded,i,j);

    return result;
}

unsigned char submean3x3_gs(Mat img, int istart, int jstart){
    int sum = 0;
    for(int i=istart-1; i<istart+2; i++)
        for(int j=jstart-1; j<jstart+2; j++)
            sum += img.at<unsigned char>(i,j);

    return (unsigned char)(sum/9);
}











