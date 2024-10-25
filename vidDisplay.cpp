// Name: Saugat Malla
// Date: 
// .cpp 


// Display video Live
#include<iostream>
#include<string>
#include<sys/time.h>
#include "opencv2/opencv.hpp"
#include "filters.hpp"
#include "faceDetect.h"

using namespace std;

double getTime(){
  struct timeval cur;

  gettimeofday(&cur, NULL);
  return(cur.tv_sec + cur.tv_usec / 1000000.0);
}

// Main function
int main (){
  cv::VideoCapture *capdev;

  double startAt, endAt, startPTR, endPTR;
  char prevKey;

  //Open the video device
  //capdev = new cv::VideoCapture(0)
  //capdev = new cv::VideoCapture(0,cv::CAP_AVFOUNDATION);
  capdev = new cv::VideoCapture(0);
  if( !capdev -> isOpened() ){
    printf("Unable to open video device\n");
    return (-1);
  }
 

  //Get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT)
            );
  printf("Expected size: %d %d\n", refS.width, refS.height);

  cv::namedWindow("Video",1); // identifies a window 
  cv::Mat frame, grayscale, grayscale_self, sepia_self, blurWith_at, sobelx, sobely, gradientMag, blurQuant, grayscaleFace, vidBrightnessHigh100, vidBrightnesslow100, negImage, rotateImg, contourImg;
  std::vector<cv::Rect> detectedFaces;

  for(;;){
    *capdev >> frame; //get a new frame from the camera, rear as a stream
    
    if(frame.empty()){
      printf("Frame is empty\n");
      break;
    }

    imshow("Video", frame);

    //see if there is a waiting keystroke
    char key = cv::waitKey(10);
    if(key=='q'){
      
    printf("Time for at<> method: %.5f\n", (endAt-startAt) / 3);
    printf("Time for ptr<> method: %.5f\n", (endPTR-startPTR) / 3);

      break;
    }

    if(key=='s'){
      cv::Mat imgCaptured; 
      //*capdev >> imgCaptured;
      
      // Switch case to check the last entered key so as to save the image based on the last key press 
      switch(prevKey){
        case 'g':
          imwrite("grayscale.jpg", grayscale);
          break;
        case 'h':
          imwrite("grayscale_self.jpg", grayscale_self);
          break;
        case 't':
          imwrite("sepia_self.jpg", sepia_self);
          break;
        case 'b':
          imwrite("blur.jpg", blurWith_at);
          break;
        case 'x':
          imwrite("sobelx.jpg", sobelx);
          break;
        case 'y':
          imwrite("sobely.jpg", sobely);
          break;
        case 'm':
          imwrite("gradientMag.jpg", gradientMag);
          break;
        case 'l':
          imwrite("blurQuant.jpg", blurQuant);
          break;
        case 'f':
          imwrite("FaceDetection.jpg",frame);
          break;
        case 'i':
          imwrite("GrayscaleFace.jpg",grayscaleFace);
          break;
        case 'p':
          imwrite("vidBright.jpg", vidBrightnessHigh100);
          break;
        case 'd':
          imwrite("vidDim.jpg", vidBrightnesslow100);
          break;
        case 'n':
          imwrite("NegativeImage.jpg",negImage);
          break;
        case 'r':
          imwrite("RotateImage.jpg",rotateImg);
          break;
        case 'c':
          imwrite("ContourImage.jpg",contourImg);
          break;
        default:
          *capdev >> imgCaptured;
          imwrite("normal.jpg", imgCaptured);
      }

      
    }
    
    // To get greyscale using default OpenCV function
    if(key=='g'){
      prevKey = 'g';
      //cv::Mat grayscale;
      cv::cvtColor(frame, grayscale, cv::COLOR_RGB2GRAY); 
      imshow("Grayscale CV2", grayscale);
    }
    
    // To get greyscale using a self implemented funciton 
    if(key=='h'){
      prevKey = 'h';
      //cv::Mat grayscale_self;
      greyscale(frame, grayscale_self);
      imshow("Grayscale Self", grayscale_self);
    }
    
    // To get a sepia tone filter 
    if(key=='t'){
      prevKey = 't';
      //cv::Mat sepia_self;
      sepiatoner(frame,sepia_self);
      imshow("Sepia", sepia_self);
    }
    
    // To get a blur image with both at and ptr method 
    if(key=='b') {
      prevKey = 'b';
      //cv::Mat blurWith_at;
      startAt = getTime();
      blur5x5_1(frame, blurWith_at);
      endAt = getTime();
      imshow("Blur with at", blurWith_at);

      cv::Mat blurWith_ptr;
      startPTR = getTime();
      blur5x5_2(frame, blurWith_ptr);
      endPTR = getTime();
      imshow("Blur with Ptr", blurWith_ptr);
    }
    
    // To get a sobelX filter 
    if(key=='x'){
      prevKey = 'x';
      //cv::Mat sobelx;
      sobelX3x3(frame, sobelx);
      cv::convertScaleAbs(sobelx,sobelx);
      imshow("SobelX", sobelx);

    }
    
    // To get a sobelY filter 
    if(key=='y'){
      prevKey = 'y';
      //cv::Mat sobely;
      sobelY3x3(frame, sobely);
      cv::convertScaleAbs(sobely,sobely);
      imshow("SobelY", sobely);

    }
    
    // To get a gradient magnitude image using sobelx and sobely 
    if(key=='m'){
      prevKey = 'm';
      sobelX3x3(frame, sobelx);
      cv::convertScaleAbs(sobelx, sobelx);
      sobelY3x3(frame, sobely);
      cv::convertScaleAbs(sobely, sobely);
      magnitude(sobelx,sobely, gradientMag);
      imshow("Gradient magnitude", gradientMag);

    }
    
    //  To get a blur and quantized image 
    if(key=='l'){
      prevKey = 'l';
      blurQuantize(frame, blurQuant, 10);
      imshow("Blur and Quantize", blurQuant);
    }
    
    // To detect faces 
    if(key=='f'){
      prevKey = 'f';
      
      //std::vector<cv::Rect> detectedFaces;
      
      // Convert frame to grayscale before detecting faces, or else assert in the function of faceDetect resize will fail  
      cv::Mat grayscaleFrame;
      cv::cvtColor(frame, grayscaleFrame, cv::COLOR_RGB2GRAY);

      detectFaces(grayscaleFrame, detectedFaces);
      //drawBoxes(grayscaleFrame, detectedFaces, 10,1.0);
      drawBoxes(frame, detectedFaces, 20,1.0);
      imshow("Face Detection", frame);

    }
    
    // Face color remains while rest(background) is greyscale
    if(key=='i'){
      prevKey = 'i'; 

      cv::Mat grayscaleFrame;
      cv::cvtColor(frame, grayscaleFrame, cv::COLOR_RGB2GRAY);

      detectFaces(grayscaleFrame, detectedFaces);
      //drawBoxes(grayscaleFrame, detectedFaces, 10,1.0);
      drawBoxes(frame, detectedFaces, 20,1.0);

      greyscaleBackground(frame,grayscaleFace, detectedFaces );
      imshow("Grayscale face Bacground", grayscaleFace);
    }

    //Brightness increase
    if(key=='p'){
      prevKey = 'p';
      
      cv::Mat grayscaleFrame;
      cv::cvtColor(frame, grayscaleFrame, cv::COLOR_RGB2GRAY);

      detectFaces(grayscaleFrame, detectedFaces);
      //drawBoxes(grayscaleFrame, detectedFaces, 10,1.0);
      drawBoxes(frame, detectedFaces, 20,1.0);

      //cv::Mat vidBrightnessHigh100;
      frame.copyTo(vidBrightnessHigh100);
      vidBrightnessHigh100.convertTo(vidBrightnessHigh100, -1,1,100);
      imshow("Bright Video", vidBrightnessHigh100);
    }

    //Brightness decrease
    if(key=='d'){
      prevKey = 'd';
      
      cv::Mat grayscaleFrame;
      cv::cvtColor(frame, grayscaleFrame, cv::COLOR_RGB2GRAY);

      detectFaces(grayscaleFrame, detectedFaces);
      //drawBoxes(grayscaleFrame, detectedFaces, 10,1.0);
      drawBoxes(frame, detectedFaces, 20,1.0);

      //cv::Mat vidBrightnesslow100;
      frame.copyTo(vidBrightnesslow100);
      vidBrightnesslow100.convertTo(vidBrightnesslow100, -1,1,-100);
      imshow("Dim Video", vidBrightnesslow100);
    }

    // Negative of image
    if(key=='n'){
      
      prevKey = 'n';

      cv::Mat grayscaleFrame;
      cv::cvtColor(frame, grayscaleFrame, cv::COLOR_RGB2GRAY);

      detectFaces(grayscaleFrame, detectedFaces);
      //drawBoxes(grayscaleFrame, detectedFaces, 10,1.0);
      drawBoxes(frame, detectedFaces, 20,1.0);

      negativeImage(frame, negImage);
      imshow("Negative Image", negImage);

    }


    // Extensions
    
    // Rotate an image
    if(key=='r'){

      prevKey = 'r';

      rotateImage(frame, rotateImg, 180.0);
      imshow("Rotate Video", rotateImg);

    }
    
    // Contours
    if(key=='c'){
      
      prevKey = 'c';
      contourImage(frame, contourImg);
      imshow("Contour Image", contourImg);

    }

  } 

  delete capdev;

  return 0;
}
