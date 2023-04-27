#include <iostream>
#include <stack>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void grow(Mat& src, Mat &dst, Mat& mask, Point seed, int th){

    const Point pointShift2D[8] = //8intorno
    {
        Point(-1,-1), Point(-1,0), Point(-1,1),
        Point(0,-1), Point(0,1),
        Point(1,-1), Point(1,0), Point(1,1)
    };

    stack<Point> point_stack;
    point_stack.push(seed);

    Point ePoint; //estimated Point
    Point center;
    while(!point_stack.empty()){
        center = point_stack.top();
        mask.at<uchar>(center) = 1;
        point_stack.pop();

        for(int i=0; i<8; i++){
            ePoint = center+pointShift2D[i];
            if( ePoint.x < 0 || ePoint.x > src.cols-1 ||
                ePoint.y < 0 || ePoint.y > src.rows-1){
                    continue;
            } else {
                int delta = int(pow(src.at<cv::Vec3b>(center)[0] - src.at<cv::Vec3b>(ePoint)[0], 2)
                + pow(src.at<cv::Vec3b>(center)[1] - src.at<cv::Vec3b>(ePoint)[1], 2)
                + pow(src.at<cv::Vec3b>(center)[2] - src.at<cv::Vec3b>(ePoint)[2], 2));
                if(dst.at<uchar>(ePoint) == 0
                   && mask.at<uchar>(ePoint) == 0
                   && delta < th){
                    mask.at<uchar>(ePoint) = 1;
                    point_stack.push(ePoint);
                }
            }
        }
    }

}

vector<Mat> regionGrowing(Mat& src, float minRegionAreaFactor, int maxRegionNum, int th){

    int minRegionArea = int(minRegionAreaFactor*src.rows*src.cols);
    uchar label = 1;
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    int maskArea;
    vector<Mat> masks;

    for(int x=0; x<src.cols; x++){
        for(int y=0; y<src.rows; y++){
            if(dst.at<uchar>(Point(x,y)) == 0){
                grow(src,dst,mask,Point(x,y),th);
                maskArea = (int)sum(mask).val[0];
                if(maskArea > minRegionArea){
                    dst += mask*label;
                    label++;
                    masks.push_back(mask.clone()*255);
                    if(label > maxRegionNum) exit(-2); //oversegmentation
                } else {
                    dst += mask*255; //don't care area
                }
                mask -= mask;
            }
        }
    }
    return masks;
}

//Split and Merge

/*
class TNode{
private:
    Rect region;
    TNode *UL, *UR, *LL, *LR;
    vector<TNode*> merged;
    vector<bool> mergedB = vector<bool>(4,false);
    double stddev, mean;

public:
    TNode(Rect R){ region=R; UL=UR=LL=LR=nullptr; };

    TNode *getUL(){ return UL; }
    TNode *getUR(){ return UR; }
    TNode *getLL(){ return LL; }
    TNode *getLR(){ return LR; }

    void setUL(TNode *N){ UL=N; }
    void setUR(TNode *N){ UR=N; }
    void setLL(TNode *N){ LL=N; }
    void setLR(TNode *N){ LR=N; }

    double getStdDev(){ return stddev; }
    double detMean(){ return mean; }

    void setStdDev(double stddev){ this->stddev=stddev; }
    void setMean(double mean){ this->mean=mean; }

    void addRegion(TNode *R){ merged.push_back(R); }
    vector<TNode*> &getMerged(){ return merged; }

    void setMergedB(int i){ mergedB[i] = true; }
    bool getMergedB(int i){ return mergedB[i]; }
};
*/

int tsize;
double th;

class TNode{
public:
    Rect region;
    TNode *UL, *UR, *LL, *LR;
    vector<TNode*> merged;
    vector<bool> mergedB = vector<bool>(4,false);
    double stddev, mean;

    TNode(Rect R){ region=R; UL=UR=LL=LR=nullptr; }
    void addRegion(TNode *R){ merged.push_back(R); }
    void setMergedB(int i){ mergedB[i] = true; }
};

TNode* split(Mat& src, Rect R){

    TNode* root = new TNode(R);

    Scalar mean, stddev;
    meanStdDev(src(R),mean,stddev);
    root->mean = mean[0];
    root->stddev = stddev[0];

    if(R.width > tsize && root->stddev > th){

        Rect ul(R.x,R.y,R.height/2,R.width/2);
        root->UL=split(src,ul);

        Rect ur(R.x,R.y+R.width/2,R.height/2,R.width/2);
        root->UR=split(src,ur);

        Rect ll(R.x+R.height/2,R.y,R.height/2,R.width/2);
        root->LL=split(src,ll);

        Rect lr(R.x+R.height/2,R.y+R.width/2,R.height/2,R.width/2);
        root->LR=split(src,lr);
    }

    rectangle(src,R,Scalar(0));
    return root;
}

void merge(TNode *root){

    if(root->region.width > tsize && root->stddev > th){
        if(root->UL->stddev <= th && root->UR->stddev <= th){ //UL-UR
            root->addRegion(root->UL); root->setMergedB(0);
            root->addRegion(root->UR); root->setMergedB(1);
            if(root->LL->stddev <= th && root->LR->stddev <= th){
                root->addRegion(root->LL); root->setMergedB(3);
                root->addRegion(root->LR); root->setMergedB(2);
            } else {
                merge(root->LL);
                merge(root->LR);
            }
        } else if(root->UR->stddev <= th && root->LR->stddev <= th){ //UR-LR
            root->addRegion(root->UR); root->setMergedB(1);
            root->addRegion(root->LR); root->setMergedB(2);
            if(root->UL->stddev <= th && root->LL->stddev <= th){
                root->addRegion(root->UL); root->setMergedB(0);
                root->addRegion(root->LL); root->setMergedB(3);
            } else {
                merge(root->UL);
                merge(root->LL);
            }
        } else if(root->LL->stddev <= th && root->LR->stddev <= th){ //LL-LR
            root->addRegion(root->LL); root->setMergedB(3);
            root->addRegion(root->LR); root->setMergedB(2);
            if(root->UL->stddev <= th && root->UR->stddev <= th){
                root->addRegion(root->UL); root->setMergedB(0);
                root->addRegion(root->UR); root->setMergedB(1);
            } else {
                merge(root->UL);
                merge(root->UR);
            }
        } else if(root->UL->stddev <= th && root->LL->stddev <= th){ //UL-LL
            root->addRegion(root->UL); root->setMergedB(0);
            root->addRegion(root->LL); root->setMergedB(3);
            if(root->UR->stddev <= th && root->LR->stddev <= th){
                root->addRegion(root->UR); root->setMergedB(1);
                root->addRegion(root->LR); root->setMergedB(2);
            } else {
                merge(root->UR);
                merge(root->LR);
            }
        } else {
            merge(root->UL);
            merge(root->UR);
            merge(root->LL);
            merge(root->LR);
        }
    } else {
        root->addRegion(root); for(int i=0; i<4; i++) root->setMergedB(i);
    }

}

void segment(TNode *root, Mat& src){

    vector<TNode*> tmp = root->merged;

    if(tmp.size() == 0){
        segment(root->UL, src);
        segment(root->UR, src);
        segment(root->LR, src);
        segment(root->LL, src);
    } else {
        double val=0;       //calc means
        for(auto x:tmp)
            val+=(int)x->mean;
        val/=tmp.size();

        for(auto x:tmp)     //assign mean value to regions
            src(x->region) = (int)val;

        if(tmp.size() > 1){
            if(!root->mergedB[0])
                segment(root->UL,src);
            if(!root->mergedB[1])
                segment(root->UR,src);
            if(!root->mergedB[2])
                segment(root->LR,src);
            if(!root->mergedB[3])
                segment(root->LL,src);
        }

    }
}

void splitAndMerge(Mat src, Mat& segmented, float t_size, int threshold){

    tsize = t_size;
    th = threshold;

    int exponent = log(min(src.rows,src.cols))/log(2);
    int s = pow(2.0, double(exponent));
    Rect square = Rect(0,0,s,s);
    src = src(square).clone();
    GaussianBlur(src,src,Size(3,3),0,0);

    TNode* root = split(src,Rect(0,0,src.rows,src.cols));
    merge(root);

    segmented = src.clone();
    segment(root,segmented);
}


int main(int argc, char** argv)
{
    Mat src = imread(argv[1],IMREAD_COLOR);
    if(src.empty()) return -1;


    vector<Mat> masks = regionGrowing(src,0.01,100,40);
    for(auto x:masks){
        imshow("win",x);
        waitKey(0);
    }

    /*
    Mat segmented;
    splitAndMerge(src,segmented,6,4);
    imshow("src",src);
    imshow("seg",segmented);
    */



    waitKey(0);

    return 0;
}
