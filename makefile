CXX = g++
CXXFLAGS = -std=c++11
LIBS = -L/opt/homebrew/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_videoio -lopencv_videostab -lopencv_objdetect
INCLUDES = -I/opt/homebrew/include/opencv4


imgDisplay: imgDisplay.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

vidDisplay: vidDisplay.cpp filters.cpp faceDetect.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

.PHONY: clean

clean:
	rm -f imgDisplay
