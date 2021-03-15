
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<io.h>
#include <iostream>

using namespace cv;
using namespace std;

#define text "3180103570 LuJiaying"
vector<String> get_image_names(string file_path);

/**
* @function main
* @brief Main function
*/
int main(int argc,char *argv[])
{
	string dir;
	if (argc == 2) {
		dir = argv[1];
		cout << "TARGET:"<<dir<<endl;
	}
	else {
		cout << "illegal input" << endl;
		return 0;
	}
	
	//load video
	vector<String> videoAddrs = get_image_names(dir+"/*.avi");
	String Addr = dir+"/"+videoAddrs[0];
	cv::VideoCapture capture(Addr);
	//int height = capture.get(4);
	double height = capture.get(CAP_PROP_FRAME_HEIGHT);
	double width = capture.get(CAP_PROP_FRAME_WIDTH);
	double fps = capture.get(CAP_PROP_FPS);
	double framecount = capture.get(CAP_PROP_FRAME_COUNT);

	//read images
	cv::Size size = Size(width, height);
	vector<String> imagesAddr = get_image_names(dir + "/*.jpg");
	vector <cv::Mat> images;
	for (int i = 0; i < imagesAddr.size(); i++) {
		cv::Mat image = cv::imread(dir+"/"+imagesAddr[i]);
		cv::resize(image, image, size);
		cv::putText(image, text, Point(width / 2 - 150, height - 50), FONT_HERSHEY_DUPLEX, 2, Scalar(255, 255, 255), 4, 8, false);
		images.push_back(image);
	}

	//write
	VideoWriter Writer(dir + "/output.avi", CV_FOURCC('D', 'I', 'V', 'X'), fps, size, true);
	
	//write images
	for (int i = 0; i < images.size(); i++) {
		for (int j = 0; j < 60; j++) {
			Writer.write(images[i]);
		}
		for (int j = 0; j < 40; j++) {
			if (i == images.size()-1) {
				Writer.write(images[i]);
			}
			else {
				cv::Mat midImg;
				cv::addWeighted(images[i], (40 - j) *0.025, images[i + 1], j *0.025, 3, midImg);
				Writer.write(midImg);
			}
		}
	}

	//write video
	cv::Mat frame;
	capture >> frame;
	for (; !frame.empty(); capture >> frame) {
		cv::putText(frame, text, Point(width / 2 - 150, height - 50), FONT_HERSHEY_DUPLEX, 2, Scalar(255, 255, 255), 4, 8, false);
		Writer.write(frame);
	}
	cout << "OUTPUT:"<<dir<<"/output.avi" << endl;
	capture.release();
	Writer.release();
	system("pause");
	return 0;
}


vector<String> get_image_names(string file_path) {
	vector<String>file_names;
	intptr_t hFile = 0;
	_finddata_t fileInfo;
	hFile = _findfirst(file_path.c_str(), &fileInfo);
	if (hFile != -1) {
		do {
			if ((fileInfo.attrib&_A_SUBDIR)) {
				continue;
			}
			else {
				file_names.push_back(fileInfo.name);
				cout << fileInfo.name << endl;
			}
		} while (_findnext(hFile,&fileInfo)==0);
		_findclose(hFile);
	}
	return file_names;
}

