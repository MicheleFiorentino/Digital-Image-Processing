#include <iostream>
#include <opencv2/opencv.hpp>

#define maxSmallerThan1 0.999999940395355224609
//1 is 0x3f800000, so 0x3f800000-1 would be its predecessor

using namespace std;
using namespace cv;

vector<double> normalizedHistogram(Mat img){
    vector<double> his(256, 0.0f);
    for(int y=0; y<img.rows; y++)
        for(int x=0; x<img.cols; x++)
            his[img.at<uchar>(y,x)]++;
    for(int i=0; i<256; i++)
        his[i]/=img.rows*img.cols;
    return his;
}

int Otsu(Mat src){

    //1.
    vector<double> his = normalizedHistogram(src);

    //2.
    vector<double> prob1(256,0.0f);
    prob1[0]=his[0];
    for(int i=1; i<256; i++)
        prob1[i] = prob1[i-1]+his[i];

    //3.
    vector<double> cumMean(256,0.0f);
    for(int i=1; i<256; i++)
        cumMean[i] = cumMean[i-1]+i*his[i];

    //4.
    double gCumMean = 0.0f;
    for(int i=0; i<256; i++)
        gCumMean+=i*his[i];

    //5. nb: if 1 is 0x3f800000, then mST1 is 0x3f7fffff, so 0.999999940395355224609
    vector<double> intVariance(256,0.0f);
    for(int i=0; i<256; i++){
        if(prob1[i] == 0.0f){
            intVariance[0]=0.0f;
        } else {
            if(prob1[i] == 1.0f)
                prob1[i] = maxSmallerThan1;
            intVariance[i]=pow(gCumMean*prob1[i]-cumMean[i],2)/(prob1[i]*(1-prob1[i]));
        }
    }

    //6.
    auto maxVariance = max_element(intVariance.begin(), intVariance.end());
    int thresh = distance(intVariance.begin(), maxVariance); //k*

    /*DEBUG
    for(int i=0; i<256; i++){
        cout<<"hisNorm["<<i<<"]\t: "<<his[i]<<endl;
        cout<<"prob1["<<i<<"]\t: "<<prob1[i]<<endl;
        cout<<"cumMean["<<i<<"]\t: "<<cumMean[i]<<endl;
        cout<<"intVar["<<i<<"]\t: "<<intVariance[i]<<endl<<endl;
    }
    cout<<*max_element(intVariance.begin(), intVariance.end())<<endl;
    cout<<thresh<<endl;
    cout<<gCumMean<<endl<<endl;
    */

    return thresh;
}

int OtsuImp(Mat src){

    vector<double> his = normalizedHistogram(src);

    double gCumMean=0.0f;
    for(int i=0; i<256; i++)
        gCumMean+=i*his[i];

    double currProb1=0.0f;
    double currCumMean=0.0f;
    double currIntVariance=0.0f;
    double maxVariance=0.0f;
    int thresh=0;
    for(int i=0; i<256; i++){
        currProb1+=his[i];
        currCumMean+=i*his[i];
        currIntVariance=pow(gCumMean*currProb1-currCumMean,2)/(currProb1*(1-currProb1));
        if(currIntVariance > maxVariance){
            maxVariance = currIntVariance;
            thresh = i;
        }

        /*DEBUG
        cout<<"hisNorm["<<i<<"]\t: "<<his[i]<<endl;
        cout<<"prob1["<<i<<"]\t: "<<currProb1<<endl;
        cout<<"cumMean["<<i<<"]\t: "<<currCumMean<<endl;
        cout<<"intVar["<<i<<"]\t: "<<currIntVariance<<endl<<endl;
        */
    }

    return thresh;
}

vector<int> Otsu2Imp(Mat src){

    vector<double> his = normalizedHistogram(src);

    double gCumMean=0.0f;
    for(int i=0; i<256; i++)
        gCumMean+=i*his[i];

    vector<double> currProb(3,0.0f);
    vector<double> currCumMean(3,0.0f);
    double currIntVariance=0.0f;
    double maxVariance=0.0f;
    vector<int> thresh(2,0);
    for(int i=0; i<256-2; i++){
        currProb[0]+=his[i];
        currCumMean[0]+=i*his[i];
        for(int j=i+1; j<256-1; j++){
            currProb[1]+=his[j];
            currCumMean[1]+=j*his[1];
            for(int k=j+1; k<256; k++){
                currProb[2]+=his[k];
                currCumMean[2]+=k*his[k];
                currIntVariance=0.0f;
                for(int w=0; w<3; w++)
                    currIntVariance+=currProb[w]*pow(currCumMean[w]/currProb[w]-gCumMean,2);
                if(currIntVariance>maxVariance){
                    maxVariance=currIntVariance;
                    thresh[0]=i;
                    thresh[1]=j;
                }
            }
        currProb[2]=currCumMean[2]=0.0f;
        }
    currProb[1]=currCumMean[1]=0.0f;
    }

    return thresh;
}

void multipleThreshold(Mat src, Mat& dst, vector<int> thresh){
    dst = Mat::zeros(src.size(), src.type());
    for(int y=0; y<src.rows; y++)
        for(int x=0; x<src.cols; x++)
            if(src.at<uchar>(y,x) >= thresh[1])
                dst.at<uchar>(y,x) = 255;
            else if(src.at<uchar>(y,x) >= thresh[0])
                dst.at<uchar>(y,x) = 127;
}


int main(int argc, char** argv)
{
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -1;

    GaussianBlur(src, src, Size(3,3), 0);

    cout<<Otsu(src)<<" "<<OtsuImp(src)<<endl;
    vector<int> res = Otsu2Imp(src); cout<<"("<<res[0]<<","<<res[1]<<")"<<endl;

    Mat otsuImg, otsuImpImg, otsu2ImpImg;
    threshold(src,otsuImg,Otsu(src),255,THRESH_BINARY);
    threshold(src,otsuImpImg,OtsuImp(src),255,THRESH_BINARY);
    multipleThreshold(src,otsu2ImpImg,Otsu2Imp(src));


    imshow("src",src);
    //imshow("Otsu 1th",otsuImg);
    imshow("Otsu imp 1th",otsuImpImg);
    imshow("Otsu imp 2th",otsu2ImpImg);
    waitKey(0);
    return 0;
}
