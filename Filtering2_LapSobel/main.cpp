#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

Mat myLaplacian(Mat src, int ddepth, int ksize){

    //Kernel assignment
    float data[9];
    switch(ksize){
    case 1:
        {
        float data90[]={
            0.0,1.0,0.0,
            1.0,-4.0,1.0,
            0.0,1.0,0.0
        };
        copy(data90, data90+sizeof(data90)/sizeof(data90[0]), data);
        break;
        }

    case 3:
        {
        float data45[]={
            1.0,1.0,1.0,
            1.0,-8.0,1.0,
            1.0,1.0,1.0
        };
        copy(data45, data45+sizeof(data45)/sizeof(data45[0]), data);
        break;
        }
    }

    //Mask application
    Mat mask = Mat(3,3,CV_32F,data);
    Mat dst;
    filter2D(src,dst,ddepth,mask);

    return dst;
}

Mat mySobel(Mat src, int ddepth, int xorder, int yorder){

    float sobelX[]={
        -1.0,-2.0,-1.0,
        0.0,0.0,0.0,
        1.0,2.0,1.0
    };

    float sobelY[]={
        -1.0,0.0,1.0,
        -2.0,0.0,2.0,
        -1.0,0.0,1.0
    };

    //Get Filters
    Mat maskSX = Mat(3,3,CV_32F,sobelX);
    Mat maskSY = Mat(3,3,CV_32F,sobelY);

    //Mask application
    Mat dst;
    if(xorder == 1 && yorder == 0){ //SobelX
        filter2D(src,dst,ddepth,maskSX);
    }
    else if(xorder == 0 && yorder == 1){ //SobelY
        filter2D(src,dst,ddepth,maskSY);
    }
    else if(xorder == 1 && yorder == 1){

        Mat imgSobX = mySobel(src,ddepth,1,0);
        Mat imgSobY = mySobel(src,ddepth,0,1);

        dst = abs(imgSobX)+abs(imgSobY);

        /*Mat isx2, isy2;
        pow(imgSobX,2,isx2);
        pow(imgSobY,2,isy2);
        sqrt(isx2+isy2,dst);*/
        //Versione più precisa, ma dobbiamo convertire da CV_8U a CV_32F
    }

    return dst;
}



int main(int argc, char** argv)
{
    Mat img = imread(argv[1],IMREAD_GRAYSCALE);
    if(img.empty()) return -1;


    //LAPLACIAN

    Mat myLapImg45 = myLaplacian(img,CV_8U,3);
    Mat myLapImg90 = myLaplacian(img,CV_8U,1);

    Mat lapImg45;
    Laplacian(img,lapImg45,CV_8U,3);
    Mat lapImg90;
    Laplacian(img,lapImg90,CV_8U,1);

    Mat test;
    float c=-1; // -1 <= c <= 0
    test = (img+c*lapImg45);

    imshow("original",img);

    /*imshow("laplacian_f2D_45", myLapImg45);
    imshow("laplacian_f2D_90", myLapImg90);
    imshow("laplacian_fun_90", lapImg90);
    imshow("laplacian_fun_45", lapImg45);
    imshow("test",test);*/
    waitKey(0);
    destroyAllWindows();



    //SOBEL
    Mat sobXImg = mySobel(img,CV_8U,1,0);
    Mat sobYImg = mySobel(img,CV_8U,0,1);
    Mat Mag = mySobel(img,CV_8U,1,1);

    Mat sobel;
    Sobel(img, sobel, CV_8U, 1,1);

    imshow("original",img);
    imshow("mySobelX", sobXImg);
    imshow("mySobelY", sobYImg);
    imshow("Mag", Mag);
    imshow("Sobel", sobel);
    waitKey(0);


    return 0;
}









