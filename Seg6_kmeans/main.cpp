#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>
#include <time.h>

using namespace std;
using namespace cv;

void kmeans(Mat& src, Mat& dst, int k){

    srand(time(NULL));

    //1.
    vector<uchar> c(k,0);
    for(int i=0; i<k; i++){
        int randRow = rand()%src.rows;
        int randCol = rand()%src.cols;
        c[i] = src.at<uchar>(randRow,randCol);
    }

    //2 and 3
    double epsilon = 0.01f;
    bool is_c_varied = true;
    int maxIterations = 50;
    int it;
    vector<double> oldmean(k,0.0f);
    vector<double> newmean(k,0.0f);
    vector<vector<Point>> cluster(k);

    vector<uchar> dist(k,0);
    int minDistIdx;

    while(is_c_varied && it++ < maxIterations){

        is_c_varied = false;
        for(int i=0; i<k; i++) cluster[i].clear();
        for(int i=0; i<k; i++) oldmean[i] = newmean[i];

        for(int x=0; x<src.rows; x++){
            for(int y=0; y<src.cols; y++){
                for(int i=0; i<k; i++){
                    dist[i] = abs(c[i] - src.at<uchar>(x,y));
                }
                minDistIdx = min_element(dist.begin(),dist.end())-dist.begin();
                cluster[minDistIdx].push_back(Point(x,y));
            }
        }

        for(int i=0; i<k; i++){
            int csize = static_cast<int>(cluster[i].size());
            for(int j=0; j<csize; j++){
                int cx = cluster[i][j].x;
                int cy = cluster[i][j].y;
                newmean[i] += src.at<uchar>(cx,cy);
            }
            newmean[i] /= csize;
            c[i] = uchar(newmean[i]);
        }

        for(int i=0; i<k; i++)
            if( !(abs(newmean[i]-oldmean[i]) <= epsilon) )
                is_c_varied = true;

        cout<<it<<endl;
    }

    //4.
    dst = src.clone();
    for(int i=0; i<k; i++){
        int csize = static_cast<int>(cluster[i].size());
        for(int j=0; j<csize; j++){
            int cx = cluster[i][j].x;
            int cy = cluster[i][j].y;
            dst.at<uchar>(cx,cy) = c[i];
        }
    }

}

void kmeansRGB(Mat& src, Mat& dst, int k){

    srand(time(NULL));

    //1.
    vector<Vec3b> c(k,0);
    for(int i=0; i<k; i++){
        int randRow = rand()%src.rows;
        int randCol = rand()%src.cols;
        c[i] = src.at<Vec3b>(randRow,randCol);
    }

    //2 and 3
    double epsilon = 0.01f;
    bool is_c_varied = true;
    int maxIterations = 50;
    int it;
    vector<Vec3d> oldmean(k,0.0f);
    vector<Vec3d> newmean(k,0.0f);
    vector<vector<Point>> cluster(k);

    double diffBlue, diffGreen, diffRed;
    vector<uchar> dist(k,0);
    int minDistIdx;

    vector<Mat> srcChannels(3);
    split(src, srcChannels);

    while(is_c_varied && it++ < maxIterations){

        is_c_varied = false;
        for(int i=0; i<k; i++) cluster[i].clear();
        for(int i=0; i<k; i++) oldmean[i] = newmean[i];

        for(int x=0; x<src.rows; x++){
            for(int y=0; y<src.cols; y++){
                for(int i=0; i<k; i++){
                    diffBlue  = c[i].val[0] - srcChannels[0].at<uchar>(x,y);
                    diffGreen = c[i].val[1] - srcChannels[1].at<uchar>(x,y);
                    diffRed   = c[i].val[2] - srcChannels[2].at<uchar>(x,y);
                    dist[i] = sqrt(pow(diffBlue, 2) + pow(diffGreen,2) + pow(diffRed,2));
                }
                minDistIdx = min_element(dist.begin(),dist.end())-dist.begin();
                cluster[minDistIdx].push_back(Point(x,y));
            }
        }

        for(int i=0; i<k; i++){
            int csize = static_cast<int>(cluster[i].size());
            for(int j=0; j<csize; j++){
                int cx = cluster[i][j].x;
                int cy = cluster[i][j].y;
                newmean[i].val[0] += srcChannels[0].at<uchar>(cx,cy);
                newmean[i].val[1] += srcChannels[1].at<uchar>(cx,cy);
                newmean[i].val[2] += srcChannels[2].at<uchar>(cx,cy);
            }
            newmean[i] /= csize;
            c[i] = newmean[i];
        }

        double val = 0.0f;
        for(int i=0; i<k; i++){
            for(int ch=0; ch<3; ch++)
                val += newmean[i].val[ch]-oldmean[i].val[ch];
            val /= 3;
            if( abs(val) <= epsilon)
                is_c_varied = true;
        }

        cout<<it<<endl;
    }

    //4.
    dst = src.clone();
    for(int i=0; i<k; i++){
        int csize = static_cast<int>(cluster[i].size());
        for(int j=0; j<csize; j++){
            int cx = cluster[i][j].x;
            int cy = cluster[i][j].y;
            dst.at<Vec3b>(cx,cy) = c[i];
        }
    }

}

int main(int argc, char** argv)
{
    Mat src = imread( argv[1]);
    if(src.empty()) return -1;

    Mat greySrc;
    cvtColor(src, greySrc, COLOR_BGR2GRAY);

    Mat dst;
    kmeans(greySrc,dst,2);

    Mat dstColor;
    kmeansRGB(src,dstColor,3);

    imshow("src",src);
    imshow("kmeans",dst);
    imshow("kmeansRGB",dstColor);

    waitKey(0);
    return 0;
}








