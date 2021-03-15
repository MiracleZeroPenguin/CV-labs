#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_DEPRECATE

#include <opencv2/opencv.hpp>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<string>
#include<io.h>
#include<windows.h>
#include<direct.h>

using namespace std;
using namespace cv;

/*全局变量声明*/
vector<String> imgList;	//目标图片列表，保存其路径
Size image_size;	//记录输入图像的尺寸
Size board_size;	//标定板上每行每列的角点数steor(9,6),calibration(12,12)
vector<Point2f> image_corners;  // 每幅图像上检测到的角点数组
vector<vector<Point2f>> all_corners; //所有图像角点数组
Size cell_size = Size(100.0, 100.0);  //实际测量得到的每个棋盘格的大小
vector<vector<Point3f>> object_points;// 标定板上角点的三维坐标数组
Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参数矩阵
Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); //摄像机的5个畸变系数：k1,k2,p1,p2,k3
vector<Mat> tvecsMat;  //图像的旋转向量数组
vector<Mat> rvecsMat;//图像的平移向量数组

/*函数声明*/
void getConrners(char* dataDir);
vector<String> get_image_names(string file_path);
void cameraCalibeate();
void writeCalibrate();
void correctImages(char* dataDir);
void birdEyeSee(char* dataDir);

/*主函数*/
int main(int argc,char* argv[]) {
	//argv[1] 数据库路径
	//argv[2] argv[3] boardSize
		//stereoData:  9 6
		//calibration: 12 12
	if (argc < 4) {
		cout << "参数不足" << endl;
		cout << "argv[1] 数据库路径" << endl;
		cout << "argv[2] argv[3] boardSize" << endl;
		cout << "\tstereoData:  9 6" << endl;
		cout << "\tcalibration: 12 12" << endl;
		return 0;
	}
	board_size = Size(atoi(argv[2]), atoi(argv[3]));
	getConrners(argv[1]);
	cameraCalibeate();
	writeCalibrate();
	correctImages(argv[1]);
	birdEyeSee(argv[1]);
	system("pause");
}

/*
void birdEyeSee(char* dataDir)
找到每幅图像中的4对三维和像素顶点
通过计算透视变换矩阵
生成鸟瞰图
*/
void birdEyeSee(char* dataDir) {
	cout << "开始生成鸟瞰图" << endl;
	for (int i = 0; i < imgList.size(); i++) {
		Mat h = Mat(3, 3, CV_32F, Scalar::all(0));//找到单应矩阵
		vector<Point2f> objPts(4);//选定的4对顶点
		vector<Point2f> imgPts(4);
		int indexArray[4] = {								//每对顶点在顶点数组中的index
			0,												//No.1:左上角(0,0)
			board_size.width - 1,							//No.2:右上角(w-1,0)
			(board_size.height - 1)*board_size.width,		//No.3:左下角(0,h-1)
			board_size.height*board_size.width - 1			//No.4:右下角(w-1,h-1)
		};
		//给选定的4对顶点赋值：必须是point2f类型，所以objPts只取x,y坐标
		for (int j = 0; j < 4; j++) {
			objPts[j].x = object_points[i][indexArray[j]].x;
			objPts[j].y = object_points[i][indexArray[j]].y;
			imgPts[j] = all_corners[i][indexArray[j]];
		}

		h = getPerspectiveTransform(objPts, imgPts);

		Mat imageInput = imread((string)dataDir + "/corners/" + imgList[i]);
		Mat imageBird = imageInput.clone();
		//使用单应矩阵来remap view
		warpPerspective(imageInput, imageBird, h, image_size, CV_INTER_LINEAR + CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS);
		
		string folder = (string)dataDir + "/bird";
		if (access(folder.c_str(), 0) == -1) {
			mkdir(folder.c_str());
		}//新建文件夹bird
		imwrite((string)dataDir + "/bird/" + imgList[i], imageBird);
		cout<< "鸟瞰图" << (string)dataDir << "/bird/" << imgList[i] << "已保存" << endl;
	}
}

/*
void correctImages()
根据内参与畸变系数
对图像进行矫正与保存
*/
void correctImages(char* dataDir) {
	cout << "开始矫正图像" << endl;
	Mat map1 = Mat(image_size, CV_32FC1);
	Mat map2 = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	string folder = (string)dataDir + "/correct";
	if (access(folder.c_str(), 0) == -1) {
		cout << "新建文件夹correct";
		mkdir(folder.c_str());
	}//新建文件夹correct
	for (int i = 0; i < imgList.size(); i++) {
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, map1, map2);
		Mat imageInput = imread((string)dataDir + "/" + imgList[i]);
		Mat correctImage = imageInput.clone();
		remap(imageInput, correctImage, map1, map2, INTER_LINEAR);
		imwrite((string)dataDir + "/correct/" + imgList[i], correctImage);
		cout << "图片" << (string)dataDir << "/correct/" << imgList[i] << "矫正完成" << endl;
	}
	cout << "矫正结束";
}

/*
void writeCalibrate()
1. 相机内参数矩阵
2. 畸变系数
3. n幅图片的旋转矩阵与平移向量
储存为.xml
*/
void writeCalibrate() {
	cout << "开始保存intrinsics.xml" << endl;
	FileStorage fs("intrinsics.xml", FileStorage::WRITE);
	fs << "imageWidth" << image_size.width;
	fs << "imageHeight" << image_size.height;
	fs << "cameraMatric" << cameraMatrix;
	fs << "distCoeffs" << distCoeffs;
	fs.release();
	cout << "intrinsicx.xml保存完成" << endl;
}

/*
摄像机标定
根据坐标和图片大小，得出相机内参外参与畸变系数
*/
void cameraCalibeate() {
	cout << "正在进行相机内外参标定" << endl;
	for (int k = 0; k < imgList.size(); k++) {
		vector<Point3f>positionList;
		//标定板上角点的三维坐标
		for (int j = 0; j < board_size.height; j++) {
			for (int i = 0; i < board_size.width; i++) {
				Point3f tmpPoint;
				tmpPoint.x = (float)j*cell_size.width;
				tmpPoint.y = (float)i*cell_size.height;
				tmpPoint.z = 0;
				positionList.push_back(tmpPoint);
			}
		}
		object_points.push_back(positionList);
	}
	calibrateCamera(object_points, all_corners, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);//标定
	cout << "角点标定完成" << endl;
}

/*
void getConrners(char* dataDir)
获得所有图像的角点，并进行亚像素精确化
保存在all_corners数组中
*/
void getConrners(char* dataDir) {
	cout << "正在进行角点提取" << endl;
	string dir = (string)dataDir + "/*.jpg";
	imgList = get_image_names(dir);
	string folder = (string)dataDir + "/corners";
	if (access(folder.c_str(), 0) == -1) {
		mkdir(folder.c_str());
	}//新建文件夹corners

	for (int i = 0; i < imgList.size(); i++) {
		string imgDir = (string)dataDir + +"/"+imgList[i];//将图片名称补充为完整路径
		Mat imageInput = imread(imgDir);
		if (i == 0) {
			//读取第一张图时，记录尺寸
			image_size.height = imageInput.rows;
			image_size.width = imageInput.cols;
		}
		//提取角点
		if (!findChessboardCorners(imageInput, board_size, image_corners)) {
			cout << "在图片" << imgList[i] << "中找不到角点" << endl;
			exit(1);
		}
		else {
			Mat imageGray;
			cvtColor(imageInput, imageGray, CV_RGB2GRAY);
			find4QuadCornerSubpix(imageGray, image_corners, board_size);//亚像素精确化
			all_corners.push_back(image_corners);//保存图像角点
			drawChessboardCorners(imageInput, board_size, image_corners, false);
			imwrite((string)dataDir+"/corners/"+imgList[i], imageInput);//保存标定角点后的图片，保存在原路径下corners文件夹中
			cout << "图片" << imgDir << "角点标注完成" << endl;
		}
	}
	cout << "全部图片角点标注完成" << endl;
	return;
}

/*
vector<String> get_image_names(string file_path)
获得路径下所有符合要求的文件名并返回
*/
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
			}
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}
	return file_names;
}