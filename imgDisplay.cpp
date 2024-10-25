#include<cstdio>
#include<cstring>
#include "opencv2/opencv.hpp"

int main(int argc, char *argv[]){

  cv::Mat src;
  cv::Mat dst;
  char filename[256];
  
  // Check if enough command line arguments
  if(argc <2){
    printf("Usage: %s <image filename>\n", argv[0]);
    exit(-1);
  }
  strcpy(filename, argv[1]);

  // Read the image 
  src = cv::imread(filename);
  
  // Cheking if the iamge read was successfull
  if(src.data == NULL){
    // If no image data read from file 
    printf("Error: Unable to read image %s\n", filename);
    exit(-1);
  }

  cv::namedWindow(filename, 1);
  cv::imshow(filename, src);

  // To keep the Window open until an event
  cv::waitKey(0);
  cv::destroyWindow(filename);

  printf("Terminating\n");

}
