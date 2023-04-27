#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void hysteresisThresholding(Mat &img, Mat &out, int lth, int hth){

    for(int i=1; i<img.rows-1; i++){
        for(int j=1; j<img.cols-1; j++){
            if(img.at<uchar>(i,j) > hth){
                out.at<uchar>(i,j) = 255;
                for(int k=-1; k<=1; k++){
                    for(int l=-1; l<=1; l++){
                        if(img.at<uchar>(i+k,j+l) > lth && img.at<uchar>(i+k,j+l) < hth)
                            out.at<uchar>(i+k,j+l) = 255;
                    }
                }
            }
            else if(img.at<uchar>(i,j < lth))
                out.at<uchar>(i,j) = 0;
        }
    }

}

void nonMaximaSuppression(Mat &mag, Mat &orientations, Mat &nms){

    float angle;
    for(int i=1; i<mag.rows-1; i++){
        for(int j=1; j<mag.cols-1; j++){
            angle = orientations.at<float>(i,j);
            angle = angle > 180 ? angle - 360 : angle;

            if( (-22.5 < angle && angle <= 22.5) || (-157.5 < angle && angle <= 157.5) ){
                if(mag.at<uchar>(i,j) >= mag.at<uchar>(i,j-1) && mag.at<uchar>(i,j) >= mag.at<uchar>(i,j+1))
                    nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
            }
            else if( (-67.5 < angle && angle <= -22.5) || (112.5 < angle && angle <= 157.5) ){
                if(mag.at<uchar>(i,j) >= mag.at<uchar>(i-1,j-1) && mag.at<uchar>(i,j) >= mag.at<uchar>(i+1,j+1))
                    nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
            }
            else if( (-112.5 < angle && angle <= -67.5) || (67.5 < angle && angle <= 112.5) ){
                if(mag.at<uchar>(i,j) >= mag.at<uchar>(i-1,j) && mag.at<uchar>(i,j) >= mag.at<uchar>(i+1,j))
                    nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
            }
            else if( (-157.5 < angle && angle <= -112.5) || (22.5 < angle && angle <= 67.5) ){
                if(mag.at<uchar>(i,j) >= mag.at<uchar>(i-1,j+1) && mag.at<uchar>(i,j) >= mag.at<uchar>(i+1,j-1))
                    nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
            }
        }
    }

}

void myCanny(Mat &src, Mat &output, int lth, int hth, int ksize){

    // 1.
    Mat Gauss;
    GaussianBlur(src,Gauss,Size(3,3),0,0);

    //2.
    Mat Dx, Dy;
    Sobel(Gauss,Dx,CV_32FC1,1,0,ksize);
    Sobel(Gauss,Dy,CV_32FC1,0,1,ksize);

    Mat mag,Dx2,Dy2;
    pow(Dx,2,Dx2);
    pow(Dy,2,Dy2);
    sqrt(Dx2+Dy2,mag);
    normalize(mag,mag,0,255,NORM_MINMAX,CV_8U);
    imshow("mag",mag);

    Mat orientations;
    phase(Dx,Dy,orientations,true);

    //3.
    Mat nms=Mat::zeros(mag.rows,mag.cols,CV_8U);
    nonMaximaSuppression(mag,orientations,nms);
    imshow("nms",nms);

    //4.
    Mat out=Mat::zeros(mag.rows,mag.cols,CV_8U);
    hysteresisThresholding(nms,out,lth,hth);


    output=out;
}

void circleCorners(Mat &src, Mat &dst, int th){
    for(int i=0; i<src.rows; i++)
        for(int j=0; j<src.cols; j++)
            if( (int)src.at<float>(i,j) > th)
                circle(dst,Point(j,i),5,Scalar(0),2,8,0);
}

void myHarris(Mat &src, Mat &dst, int ksize, float k, int th){

    //1.
    Mat Dx,Dy;
    Sobel(src,Dx,CV_32FC1,1,0,ksize);
    Sobel(src,Dy,CV_32FC1,0,1,ksize);

    //2.
    Mat Dx2,Dy2,Dxy;
    pow(Dx,2,Dx2);
    pow(Dy,2,Dy2);
    multiply(Dx,Dy,Dxy);

    //3-4.
    Mat C00,C01,C10,C11;
    GaussianBlur(Dx2,C00,Size(7,7),2,0);
    GaussianBlur(Dy2,C11,Size(7,7),0,2);
    GaussianBlur(Dxy,C01,Size(7,7),2,2);
    C10=C01;

    //5.
    Mat det, PPD, PSD, trace, trace2, R;

    multiply(C00,C11,PPD);  //Product of the Principal Diagonal
    multiply(C01,C10,PSD);  //Product of the Secondary Diagonal
    det = PPD-PSD;

    trace = C00+C11;
    pow(trace,2,trace2);

    R = det-k*trace2;

    //6.
    normalize(R,R,0,255,NORM_MINMAX,CV_32FC1);
    convertScaleAbs(R, dst);

    //7.
    circleCorners(R,dst,th);
}

int main(int argc, char** argv)
{
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -1;

    /*//CANNY
    // di OpenCV
    Mat blurSrc, CEdges;
    blur(src,blurSrc,Size(3,3));
    Canny(blurSrc,CEdges,30,90);

    // mia
    Mat CannyEdges;
    int lth=30, hth=80, cksize=3;
    myCanny(src,CannyEdges,lth,hth,cksize);


    imshow("Canny original", CEdges);
    imshow("myCanny", CannyEdges);*/




    //HARRIS
    // di OpenCV
    Mat CHarris, CH_norm, CH_norm_scaled;
    int blockSize=2, hksize=3;
    float k=0.04;
    cornerHarris(src,CHarris, blockSize, hksize,k);
    normalize(CHarris,CH_norm,0,255,NORM_MINMAX, CV_32FC1,Mat());
    convertScaleAbs(CH_norm, CH_norm_scaled);

    circleCorners(CH_norm,CH_norm_scaled,200);

    // mia
    Mat HarrisCorners;
    int th=200, ksize=3;
    float k_2=0.1;
    myHarris(src,HarrisCorners,ksize,k_2,th);


    imshow("Harris original", CH_norm_scaled);
    imshow("myHarris", HarrisCorners);


    waitKey(0);

    return 0;
}







