#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void drawHoughLines(Mat& src, Mat& dst, vector<Vec2f>& lines){

    cvtColor(src,dst,COLOR_GRAY2BGR);
    float rho, theta;
    double cos_t, sin_t, x0, y0;
    int alpha = 1000;
    Point pt1, pt2;

    for(size_t i=0; i<lines.size(); i++){
        rho = lines[i][0]; theta = lines[i][1];
        cos_t = cos(theta); sin_t = sin(theta);
        x0 = rho*cos_t; y0 = rho*sin_t;
        pt1.x = cvRound(x0 + alpha*(-sin_t));
        pt1.y = cvRound(y0 + alpha*(cos_t));
        pt2.x = cvRound(x0 - alpha*(-sin_t));
        pt2.y = cvRound(y0 - alpha*(cos_t));
        line(dst, pt1, pt2, Scalar(0,0,255),3,16); //CV_AA = 16
    }

}

void drawHoughCircles(Mat &src, Mat& dst, vector<Vec3f>& circles){

    cvtColor(src,dst,COLOR_GRAY2BGR);
    Point center;
    int radius;

    for(size_t i=0; i<circles.size(); i++){
        center.x = cvRound(circles[i][0]);
        center.y = cvRound(circles[i][1]);
        radius = cvRound(circles[i][2]);

        // circle center and outline
        circle(dst, center, 3, Scalar(0,255,0), -1, 8, 0);
        circle(dst, center, radius, Scalar(0,0,255), 3, 8, 0);
    }

}

void myHoughLines(Mat& src, vector<Vec2f>& lines, int cannyLTH, int Hth){

    Mat gsrc;
    GaussianBlur(src,gsrc,Size(5,5),0,0);

    //1.
    int dist = hypot(src.rows, src.cols);
    Mat votes = Mat::zeros(dist*2,180,CV_8U);

    //2.
    Mat edgeCanny;
    Canny(gsrc, edgeCanny, cannyLTH, cannyLTH*2,3);

    //3.
    double rho, theta;
    for(int x=0; x<edgeCanny.rows; x++)
        for(int y=0; y<edgeCanny.cols; y++)
            if(edgeCanny.at<uchar>(x,y) == 255)
                for(theta=0; theta<180; theta++){
                    rho = dist + y*cos((theta-90)*CV_PI/180) + x*sin((theta-90)*CV_PI/180);
                    votes.at<uchar>(rho,theta)++;
                }

    //4.
    cout<<votes.size();
    for(int i=0; i<votes.rows; i++)
        for(int j=0; j<votes.cols; j++)
            if(votes.at<uchar>(i,j) >= Hth){
                lines.push_back(Vec2f(i,j));
            }
}

void profHoughLines(Mat& src, int cannyLTH, int Hth){

    Mat gsrc;
    GaussianBlur(src,gsrc,Size(5,5),0,0);

    //1.
    int dist = hypot(src.rows, src.cols);
    Mat votes = Mat::zeros(dist*2,180,CV_8U);

    //2.
    Mat edgeCanny;
    Canny(gsrc, edgeCanny, cannyLTH, cannyLTH*2,3);

    //3.
    double rho, theta;
    for(int x=0; x<edgeCanny.rows; x++)
        for(int y=0; y<edgeCanny.cols; y++)
            if(edgeCanny.at<uchar>(x,y) == 255)
                for(theta=0; theta<180; theta++){
                    rho = dist + y*cos((theta-90)*CV_PI/180) + x*sin((theta-90)*CV_PI/180);
                    votes.at<uchar>(rho,theta)++;
                }

    //4.
    for(int r=0; r<votes.rows; r++)
        for(int t=0; t<votes.cols; t++)
            if(votes.at<uchar>(r,t) >= Hth){
                theta = (t-90)*CV_PI/180;
                int x = (r-dist)*cos(theta);
                int y = (r-dist)*sin(theta);
                double sin_t = sin(theta);
                double cos_t = cos(theta);
                Point pt1(cvRound(x+dist*(-sin_t)), cvRound(y+dist*cos_t));
                Point pt2(cvRound(x-dist*(-sin_t)), cvRound(y-dist*cos_t));
                line(src, pt1, pt2, Scalar(255), 2, 0);
            }

    imshow("lines",src);
}

void profHoughCircles(Mat& src, int cannyLTH, int cannyHTH, int Hth, int Rmin, int Rmax){

    Mat gsrc;
    GaussianBlur(src,gsrc,Size(9,9),0,0);

    //1.
    Mat edgeCanny;
    Canny(gsrc, edgeCanny, cannyLTH, cannyHTH,3);
    imshow("edges", edgeCanny);

    //2.
    int sizes[]={edgeCanny.rows, edgeCanny.cols, Rmax-Rmin+1};
    Mat votes = Mat(3,sizes,CV_8U,Scalar(0));

    //3.
    for(int x=0; x<edgeCanny.rows; x++)
        for(int y=0; y<edgeCanny.cols; y++)
            if(edgeCanny.at<uchar>(x,y) == 255)
                for(int r=Rmin; r<=Rmax; r++)
                    for(int theta=0; theta<360; theta++){
                        int a = y - r*cos(theta*CV_PI/180);
                        int b = x - r*sin(theta*CV_PI/180);
                        if(a>=0 && a<edgeCanny.cols && b>=0 && b<edgeCanny.rows)
                            votes.at<uchar>(b,a,r-Rmin)++;
                    }

    //4.
    for(int r=Rmin; r<Rmax; r++)
        for(int b=0; b<edgeCanny.rows; b++)
            for(int a=0; a<edgeCanny.cols; a++)
                if(votes.at<uchar>(b,a,r-Rmin)>Hth){
                    circle(src, Point(a,b), 3, Scalar(0), 2, 8, 0);
                    circle(src, Point(a,b), r, Scalar(0), 2, 8, 0);
                }

    imshow("HoughCircles", src);
}

int main(int argc, char** argv)
{
    if(argc!=2) return -1;
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -2;

    imshow("src",src);

    //GaussianBlur(src,src,Size(5,5),0,0);


    //OPENCV functions
    //lines
    Mat edgeCanny;
    //Canny(src, edgeCanny,80,160,3);

    /*
    vector<Vec2f> HLines;
    int rho = 1, th = 150;
    double theta = CV_PI/180;
    HoughLines(edgeCanny,HLines,rho,theta,th,0,0);

    Mat dst;
    drawHoughLines(src,dst,HLines);

    imshow("Canny",edgeCanny);
    imshow("HoughLines", dst);

    //circles
    vector<Vec3f> HCircles;
    int dimH = 10, minDist = src.rows/8;
    int CannyTh = 100, HTh = 100;
    HoughCircles(src,HCircles,HOUGH_GRADIENT,dimH, minDist, CannyTh, HTh, 0, 20);

    Mat cdst;
    drawHoughCircles(src,cdst,HCircles);

    imshow("HoughCircles", cdst);

    */

    //Implementations
    //lines
    vector<Vec2f> myHLines;
    int cannyLTH = 150, cannyHTH = 230, Hth = 130;
    /*//myHoughLines(src,myHLines,cannyLTH,Hth);
    Mat dst;
    drawHoughLines(src,dst,myHLines);

    imshow("myHoughLines",dst);
    imshow("edgeCanny", edgeCanny);

    profHoughLines(src,cannyLTH,Hth);*/


    //circles
    profHoughCircles(src,cannyLTH, cannyHTH, Hth,40,130);



    waitKey(0);
    return 0;
}
