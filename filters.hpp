#ifndef FILTERS_H
#define FILTERS_H

// Grey Scale transformation
int greyscale(cv::Mat &src, cv::Mat &dst);

// Sepia toner
int sepiatoner(cv::Mat &src,cv::Mat &dst);

// 5x5 blur filter with at
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

// 5X5 blur filter with pointer
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// 3x3 Sobel X 
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

//3x3 Sobel Y
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Magnitude image from X and Y Sobel images
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// Blur and quantize a color image 
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// Grayscale background of face
int greyscaleBackground(cv::Mat &src, cv::Mat &dst,std::vector<cv::Rect> &faces);

// Negative Image 
int negativeImage(cv::Mat &src, cv::Mat &dst);

// Rotate Image
int rotateImage(cv::Mat &src, cv::Mat &dst, double rotationAngle);

// Contour Image
int contourImage(cv::Mat &src, cv::Mat &dst);

#endif
