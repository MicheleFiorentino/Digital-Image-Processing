#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

void printMask(Mat mask){
    for(int i=0; i<mask.rows; i++){
        for(int j=0; j<mask.cols; j++)
            cout<<mask.at<float>(i,j)<<" ";
        cout<<"\n";
    }

    cout<<"\n\nrows: "<<mask.rows<<"\tcols: "<<mask.cols<<"\n";
}

uchar calcFilteredValue(Mat neighb, Mat mask, int offx, int offy){
    float value=0;
    for(int i=0; i<mask.rows; i++)
        for(int j=0; j<mask.cols; j++)
            value += mask.at<float>(i,j)*neighb.at<uchar>(i+offx,j+offy);
    return uchar(value);
}

Mat correlation(Mat src, Mat mask){
    Mat dst(src.rows,src.cols,src.type());

    Mat pad;
    int bw = mask.rows/2; //border width, 3/2 =1, 5/2 =2, etc.
    copyMakeBorder(src,pad,bw,bw,bw,bw,BORDER_REFLECT);

    for(int i=0; i<dst.rows; i++)
        for(int j=0; j<dst.cols; j++)
            dst.at<uchar>(i,j) = calcFilteredValue(pad,mask,i,j);

    return dst;
}

Mat convolution(Mat src, Mat mask){
    Mat temp;
    rotate(src,temp,ROTATE_180);
    return correlation(temp, mask);
}




int main(int argc, char** argv){

    Mat img = imread(argv[1],IMREAD_GRAYSCALE);
    if(img.empty()) return -1;

    int dim=3;
    Mat mask = Mat::ones(dim,dim,CV_32F)/(float)(dim*dim);
    printMask(mask);

    /* weighted average mask
    float data[]={
        1.0,2.0,1.0,
        2.0,4.0,2.0,
        1.0,2.0,1.0
    };
    for(int i=0; i<9; i++) data[i] /= 16.0;
    Mat wmask = Mat(3,3,CV_32F,data);
    */

    Mat filter2Dimg;
    filter2D(img,filter2Dimg,img.type(),mask);

    Mat corrImg = correlation(img, mask);

    Mat convImg = convolution(img, mask);

    Mat corrImg2 = correlation2(img,mask);

    imshow("original",img);
    imshow("filter2D",filter2Dimg);
    imshow("correlation",corrImg);
    imshow("correlation2",corrImg2);
    imshow("convolution",convImg);

    waitKey(0);
    return 0;
}



