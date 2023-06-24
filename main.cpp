#include <opencv2\opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat src;

    src = imread("fruit.jpg"); //opencv 이미지 불러오기 
    cv::namedWindow("original",WINDOW_AUTOSIZE);
    cv::imshow("original", src); //opencv 이미지 출력
    cv::waitKey(0);
  
    blur(src, src, Size(15, 15));//이미지 blur 처리 
    imshow("blurred", src); 

    Mat p = Mat::zeros(src.cols * src.rows, 5, CV_32F); // CV_32F: float, Mat 클래스는 행렬 표현 클래스
    Mat bestLabels, centers, clustered;
    vector<Mat> bgr;
    cv::split(src, bgr);
    // bgr 색상 구별하기 
    for (int i = 0; i < src.cols * src.rows; i++) {
        p.at<float>(i, 0) = (i / src.cols) / src.rows; //opencv에서 픽셀값 접근하려면 at을 사용한다. 행과 열을 각각 구해 입력받음.
        p.at<float>(i, 1) = (i % src.cols) / src.cols;
        p.at<float>(i, 2) = bgr[0].data[i] / 255.0;
        p.at<float>(i, 3) = bgr[1].data[i] / 255.0;
        p.at<float>(i, 4) = bgr[2].data[i] / 255.0;
    }

    int K = 8; // cluster 할 군집의 총 수 (구별할 색상의 수) 설정 k-means의 k를 의미
    cv::kmeans(p, K, bestLabels,
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers); 
    //cv::kmeans(InputArray data, int K, InputOutputArray best labels, TermCriteria,int attempts, int flags, OutputArray centers= noArray())

    int* colors = new int[K];
    for (int i = 0; i < K; i++) {
        colors[i] = 255 / (i + 1);
    }

    clustered = Mat(src.rows, src.cols, CV_32F); 
    for (int i = 0; i < src.cols * src.rows; i++) {
        clustered.at<float>(i / src.cols, i % src.cols) = (float)(colors[bestLabels.at<int>(i)]);
    }

    clustered.convertTo(clustered, CV_8U); //CV_8U: unsigned char
    imshow("clustered", clustered);

    waitKey(0);
    return 0;
}

