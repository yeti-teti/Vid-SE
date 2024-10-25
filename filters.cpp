/*
 
  Saugat Malla
  January 26, 2024
*/


#include<iostream>
#include<string>
#include "opencv2/opencv.hpp"
#include<math.h>
#include<algorithm>


// The function used is the averge grayscale transformation unlike the default luminosity transformation
int greyscale(cv::Mat &src, cv::Mat &dst){
  
  src.copyTo(dst);

  // Luma Method
  for(int i=0;i<src.rows;i++){
    for(int j=0;j<src.cols;j++){

        double Y = 0.299 * src.at<cv::Vec3b>(i, j)[2] + \
          0.587 * src.at<cv::Vec3b>(i, j)[1] +\
          0.114 * src.at<cv::Vec3b>(i, j)[0];
        
        
        //dst.at<unsigned char>(i, j) = static_cast<unsigned char>(Y);
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<unsigned char>(Y),
                                          static_cast<unsigned char>(Y),
                                          static_cast<unsigned char>(Y));
    }
  }

  return 0;

}


// Function that is used for sepia toner filter 
int sepiatoner(cv::Mat &src,cv::Mat &dst){
  
  // To ensure modified value is not copied
  //cv::Mat newSrc = src.clone();

  dst = src.clone();

  for(int i=0;i<src.rows;i++){
    for(int j=0;j<src.cols;j++){
        
      cv::Vec3b image = src.at<cv::Vec3b>(i,j);
      
      // Apply sepia tone filter
      float blue = image[0];
      float green = image[1];
      float red = image[2];

      float newBlue = 0.272 * red + 0.534 * green + 0.131 * blue;
      float newGreen = 0.349 * red + 0.686 * green + 0.168 * blue;
      float newRed = 0.393 * red + 0.769 * green + 0.189 * blue;

      //dst.at<cv::Vec3b>(i,j)[0] = sobelB;
      //dst.at<cv::Vec3b>(i,j)[1] = sobelG;
      //dst.at<cv::Vec3b>(i,j)[2] = sobelR;

      // Saturate pixel intensifies
      newBlue = cv::saturate_cast<uchar>(newBlue);
      newGreen = cv::saturate_cast<uchar>(newGreen);
      newRed = cv::saturate_cast<uchar>(newRed);

      // Update pixel values in the new Mat 
      dst.at<cv::Vec3b>(i,j) = cv::Vec3b(newBlue, newGreen, newRed);
      
    }
  }

  return 0;
}

// Function that applies blur filter using at method 
int blur5x5_1(cv::Mat &src, cv::Mat &dst){
 
  src.copyTo(dst);

  for(int i=2;i<src.rows-2;i++){
    for(int j=2;j<src.cols-2;j++){
      for(int k=0;k<src.channels();k++){

      int sum = src.at<cv::Vec3b>(i-2, j-2)[k] + 2*src.at<cv::Vec3b>(i-2,j-1)[k] + 4*src.at<cv::Vec3b>(i-2, j)[k] + 2*src.at<cv::Vec3b>(i-2, j+1)[k] + src.at<cv::Vec3b>(i-2, j+2)[k] + 2*src.at<cv::Vec3b>(i-1, j-2)[k] + 4*src.at<cv::Vec3b>(i-1, j-1)[k] + 8*src.at<cv::Vec3b>(i-1, j)[k] +  4*src.at<cv::Vec3b>(i-1, j+1)[k] + 2*src.at<cv::Vec3b>(i-1, j+2)[k] + 4*src.at<cv::Vec3b>(i, j-2)[k] + 8*src.at<cv::Vec3b>(i, j-1)[k] + 16*src.at<cv::Vec3b>(i,j)[k] + 8*src.at<cv::Vec3b>(i, j+1)[k] + 4*src.at<cv::Vec3b>(i, j+2)[k] + 2*src.at<cv::Vec3b>(i+1, j-2)[k] + 4*src.at<cv::Vec3b>(i+1, j-1)[k] + 8*src.at<cv::Vec3b>(i+1, j)[k] + 4*src.at<cv::Vec3b>(i+1, j+1)[k] + 2*src.at<cv::Vec3b>(i+1, j+2)[k] +  src.at<cv::Vec3b>(i+2, j-2)[k] + 2*src.at<cv::Vec3b>(i+2, j-1)[k] + 4*src.at<cv::Vec3b>(i+2, j)[k] + 2*src.at<cv::Vec3b>(i+2, j+1)[k] +  src.at<cv::Vec3b>(i+2, j+2)[k] ; 


      dst.at<cv::Vec3b>(i,j)[k] = sum/100; 

      }
    } 
  }
      return 0; 
}


// Function that applies blur filter using ptr method
int blur5x5_2(cv::Mat &src, cv::Mat &dst){

  src.copyTo(dst);
  
  for(int i=2;i<src.rows-2;i++){
    
    cv::Vec3b *ptrup_2 = src.ptr<cv::Vec3b>(i-2);
    cv::Vec3b *ptrup_1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *ptrmd = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *ptrdn_1 = src.ptr<cv::Vec3b>(i+1);
    cv::Vec3b *ptrdn_2 = src.ptr<cv::Vec3b>(i+2);
    cv::Vec3b *dptr = src.ptr<cv::Vec3b>(i);

    for(int j=2;j<src.cols-2;j++){
      for(int k=0;k<src.channels();k++){
      

      int sum = ptrup_2[j-2][k] + 2*ptrup_2[j-1][k] + 4*ptrup_2[j][k] + \
                  2*ptrup_2[j+1][k] + ptrup_2[j+2][k] + \
                  2*ptrup_1[j-2][k] + 4*ptrup_1[j-1][k] + \
                  8*ptrmd[j][k] + 4*ptrmd[j+1][k] + \
                  2*ptrmd[j+2][k] + 4*ptrdn_1[j-2][k] + \
                  8*ptrdn_1[j-1][k] + 16*ptrdn_1[j][k] + \
                  8*ptrdn_1[j+1][k] + 4*ptrdn_1[j+2][k] + \
                  2*ptrdn_2[j-2][k] + 4*ptrdn_2[j-1][k] + \
                  8*ptrdn_2[j][k] + 4*ptrdn_2[j+1][k] + \
                  2*ptrdn_2[j+2][k] + dptr[j-2][k] + \
                  2*dptr[j-1][k] + 4*dptr[j][k] + \
                  2*dptr[j+1][k] + dptr[j+2][k]; 
     
        /*
          for (int m = -2; m <= 2; m++) {
                    for (int n = -2; n <= 2; n++) {
                        sum += filter[m + 2][n + 2] * ptrmd1[j + n][k];
                    }
                } 
        */

      sum /= 100;

      dptr[j][k] = sum; 

      }
    } 
  }
  
  return 0;
}


// Function that applied sobelX filter
int sobelX3x3(cv::Mat &src, cv::Mat &dst){
  
  //src.copyTo(dst);
  
  dst.create(src.size(), CV_16SC3);

  for(int i=1;i<src.rows-1;i++){
    for(int j=1;j<src.cols-1;j++){
      for(int k=0;k<src.channels();k++){ 

        int sum = (-1)*src.at<cv::Vec3b>(i-1, j-1)[k] + \
          0*src.at<cv::Vec3b>(i-1, j)[k] \
          + src.at<cv::Vec3b>(i-1,j+1)[k] + (-2)*src.at<cv::Vec3b>(i, j-1)[k] + \
          0*src.at<cv::Vec3b>(i,j)[k] + 2*src.at<cv::Vec3b>(i, j+1)[k] + \
          (-1)*src.at<cv::Vec3b>(i+1, j-1)[k] + (0)* src.at<cv::Vec3b>(i+1, j)[k] + \
          src.at<cv::Vec3b>(i+1, j+1)[k];
        
        //sum /= 16;
        
        dst.at<cv::Vec3s>(i,j)[k] = sum;
        dst.convertTo(dst, CV_16SC3);

      }
    }
  }
   

  return 0;
}

// Function that applied sobelY filter 
int sobelY3x3(cv::Mat &src, cv::Mat &dst){
  
  //src.copyTo(dst);
  
  dst.create(src.size(), CV_16SC3);


  for (int i = 1; i < src.rows - 1; i++) {
    for (int j = 1; j < src.cols - 1; j++) {
      for (int k = 0; k < src.channels(); k++) {

        int sum = (-1) * src.at<cv::Vec3b>(i - 1, j - 1)[k] + \
                  (-2) * src.at<cv::Vec3b>(i - 1, j)[k] + \
                  (-1) * src.at<cv::Vec3b>(i - 1, j + 1)[k] + (0) * src.at<cv::Vec3b>(i, j - 1)[k] + \
                  0 * src.at<cv::Vec3b>(i, j)[k] + 0 * src.at<cv::Vec3b>(i, j + 1)[k] + \
                  (1) * src.at<cv::Vec3b>(i + 1, j - 1)[k] + (2) * src.at<cv::Vec3b>(i + 1, j)[k] + \
                  src.at<cv::Vec3b>(i + 1, j + 1)[k];
        
        //sum /= 16;
        
        dst.at<cv::Vec3s>(i, j)[k] = sum;
        dst.convertTo(dst,CV_16SC3);

      }
    }
  }

  return 0;
}


// Function that generates a gradient magnitude image from sobel X and Y filters
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
  
  // Ensure sx, sy are of type CV_32SC3
  sx.convertTo(sx, CV_16SC3);
  sy.convertTo(sy, CV_16SC3);

  // Default CV_8UC3
  dst.create(sx.size(), CV_8UC3);
  
  for(int i=0;i<sx.rows;i++){
    for(int j=0;j<sx.cols;j++){
      for(int k=0;k<sx.channels();k++){
      
        float I = sqrt(static_cast<float>(sx.at<cv::Vec3s>(i,j)[k] * sx.at<cv::Vec3s>(i,j)[k] + \
                                          sy.at<cv::Vec3s>(i,j)[k] * sy.at<cv::Vec3s>(i,j)[k]));

        // To ensure magnitude is within th uchar range [0,255]
        I = std::min(255.0f, std::max(0.0f, I));

        dst.at<cv::Vec3b>(i,j)[k] = static_cast<uchar>(I);

      }
    }
  }

  return 0; 

}

// Function that blurs and quantizes a color image 
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels){
 
  cv::Mat blurred;
  blur5x5_1(src, blurred);
  
  src.copyTo(dst);
  float b = 255.0/(levels-1), xt, xf; 

  for(int i=1;i<src.rows-1;i++){
    for(int j=1;j<src.cols-1;j++){
      for(int k=0;k<src.channels();k++){

       xt = blurred.at<cv::Vec3b>(i,j)[k]/b;
       xf = round(xt)*b;
      // Not rounding gave undesired result

      dst.at<cv::Vec3b>(i,j)[k] = static_cast<uchar>(xf);
      }
    }
  }
  

  return 0;
}


// Function that makes only the background in the video greyscale while the face is colored
int greyscaleBackground(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces) {
    src.copyTo(dst);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            bool insideFace = false;

            for (const auto& face : faces) {
                // Check if the pixel is inside any of the detected faces
                if (j >= face.x && j < (face.x + face.width) && i >= face.y && i < (face.y + face.height)) {
                    insideFace = true;
                    break;
                }
            }

            if (!insideFace) {
                
               double Y = 0.299 * src.at<cv::Vec3b>(i, j)[2] + \
          0.587 * src.at<cv::Vec3b>(i, j)[1] +\
          0.114 * src.at<cv::Vec3b>(i, j)[0];
        
        
        //dst.at<unsigned char>(i, j) = static_cast<unsigned char>(Y);
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<unsigned char>(Y),
                                          static_cast<unsigned char>(Y),
                                          static_cast<unsigned char>(Y));

            }
        }
    }

    return 0;
}


// Function that gets the negative of an image itself.
int negativeImage(cv::Mat &src, cv::Mat &dst){
  
  src.copyTo(dst);

  for(int i=0;i<src.rows;i++){
    for(int j=0;j<src.cols;j++){

      dst.at<cv::Vec3b>(i,j) = cv::Vec3b(255-src.at<unsigned char>(i,j), 255-src.at<unsigned char>(i,j), 255-src.at<unsigned char>(i,j));
    }
  }

  return 0;
}


// Function that rotates the image 90 degrees 
int rotateImage(cv::Mat &src, cv::Mat &dst, double rotationAngle){
    src.copyTo(dst);

    // Get the rotation matrix 
    cv::Point2f center(static_cast<float>(src.cols/2), static_cast<float>(src.rows/2));
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotationAngle, 1.0);
    
    // Apply the rotation 
    cv::warpAffine(src, dst, rotationMatrix, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));

    return 0;
}


// Function that gets the contour of an image 
int contourImage(cv::Mat &src, cv::Mat &dst){
  
  src.copyTo(dst);

  // Convert image to greyscale 
  cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);

  // Applying a binary threshold or edge detectiion as needed.
  cv::threshold(dst, dst, 128, 255, cv::THRESH_BINARY);

  // Find contours in the binary image 
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Draw contours on the destination image 
  dst = cv::Mat::zeros(dst.size(), CV_8UC3);
  cv::drawContours(dst, contours, -1, cv::Scalar(0, 255, 0), 2);

  return 0;
}
