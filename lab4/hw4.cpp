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

/*ȫ�ֱ�������*/
vector<String> imgList;	//Ŀ��ͼƬ�б�������·��
Size image_size;	//��¼����ͼ��ĳߴ�
Size board_size;	//�궨����ÿ��ÿ�еĽǵ���steor(9,6),calibration(12,12)
vector<Point2f> image_corners;  // ÿ��ͼ���ϼ�⵽�Ľǵ�����
vector<vector<Point2f>> all_corners; //����ͼ��ǵ�����
Size cell_size = Size(100.0, 100.0);  //ʵ�ʲ����õ���ÿ�����̸�Ĵ�С
vector<vector<Point3f>> object_points;// �궨���Ͻǵ����ά��������
Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // ������ڲ�������
Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); //�������5������ϵ����k1,k2,p1,p2,k3
vector<Mat> tvecsMat;  //ͼ�����ת��������
vector<Mat> rvecsMat;//ͼ���ƽ����������

/*��������*/
void getConrners(char* dataDir);
vector<String> get_image_names(string file_path);
void cameraCalibeate();
void writeCalibrate();
void correctImages(char* dataDir);
void birdEyeSee(char* dataDir);

/*������*/
int main(int argc,char* argv[]) {
	//argv[1] ���ݿ�·��
	//argv[2] argv[3] boardSize
		//stereoData:  9 6
		//calibration: 12 12
	if (argc < 4) {
		cout << "��������" << endl;
		cout << "argv[1] ���ݿ�·��" << endl;
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
�ҵ�ÿ��ͼ���е�4����ά�����ض���
ͨ������͸�ӱ任����
�������ͼ
*/
void birdEyeSee(char* dataDir) {
	cout << "��ʼ�������ͼ" << endl;
	for (int i = 0; i < imgList.size(); i++) {
		Mat h = Mat(3, 3, CV_32F, Scalar::all(0));//�ҵ���Ӧ����
		vector<Point2f> objPts(4);//ѡ����4�Զ���
		vector<Point2f> imgPts(4);
		int indexArray[4] = {								//ÿ�Զ����ڶ��������е�index
			0,												//No.1:���Ͻ�(0,0)
			board_size.width - 1,							//No.2:���Ͻ�(w-1,0)
			(board_size.height - 1)*board_size.width,		//No.3:���½�(0,h-1)
			board_size.height*board_size.width - 1			//No.4:���½�(w-1,h-1)
		};
		//��ѡ����4�Զ��㸳ֵ��������point2f���ͣ�����objPtsֻȡx,y����
		for (int j = 0; j < 4; j++) {
			objPts[j].x = object_points[i][indexArray[j]].x;
			objPts[j].y = object_points[i][indexArray[j]].y;
			imgPts[j] = all_corners[i][indexArray[j]];
		}

		h = getPerspectiveTransform(objPts, imgPts);

		Mat imageInput = imread((string)dataDir + "/corners/" + imgList[i]);
		Mat imageBird = imageInput.clone();
		//ʹ�õ�Ӧ������remap view
		warpPerspective(imageInput, imageBird, h, image_size, CV_INTER_LINEAR + CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS);
		
		string folder = (string)dataDir + "/bird";
		if (access(folder.c_str(), 0) == -1) {
			mkdir(folder.c_str());
		}//�½��ļ���bird
		imwrite((string)dataDir + "/bird/" + imgList[i], imageBird);
		cout<< "���ͼ" << (string)dataDir << "/bird/" << imgList[i] << "�ѱ���" << endl;
	}
}

/*
void correctImages()
�����ڲ������ϵ��
��ͼ����н����뱣��
*/
void correctImages(char* dataDir) {
	cout << "��ʼ����ͼ��" << endl;
	Mat map1 = Mat(image_size, CV_32FC1);
	Mat map2 = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	string folder = (string)dataDir + "/correct";
	if (access(folder.c_str(), 0) == -1) {
		cout << "�½��ļ���correct";
		mkdir(folder.c_str());
	}//�½��ļ���correct
	for (int i = 0; i < imgList.size(); i++) {
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, map1, map2);
		Mat imageInput = imread((string)dataDir + "/" + imgList[i]);
		Mat correctImage = imageInput.clone();
		remap(imageInput, correctImage, map1, map2, INTER_LINEAR);
		imwrite((string)dataDir + "/correct/" + imgList[i], correctImage);
		cout << "ͼƬ" << (string)dataDir << "/correct/" << imgList[i] << "�������" << endl;
	}
	cout << "��������";
}

/*
void writeCalibrate()
1. ����ڲ�������
2. ����ϵ��
3. n��ͼƬ����ת������ƽ������
����Ϊ.xml
*/
void writeCalibrate() {
	cout << "��ʼ����intrinsics.xml" << endl;
	FileStorage fs("intrinsics.xml", FileStorage::WRITE);
	fs << "imageWidth" << image_size.width;
	fs << "imageHeight" << image_size.height;
	fs << "cameraMatric" << cameraMatrix;
	fs << "distCoeffs" << distCoeffs;
	fs.release();
	cout << "intrinsicx.xml�������" << endl;
}

/*
������궨
���������ͼƬ��С���ó�����ڲ���������ϵ��
*/
void cameraCalibeate() {
	cout << "���ڽ����������α궨" << endl;
	for (int k = 0; k < imgList.size(); k++) {
		vector<Point3f>positionList;
		//�궨���Ͻǵ����ά����
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
	calibrateCamera(object_points, all_corners, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);//�궨
	cout << "�ǵ�궨���" << endl;
}

/*
void getConrners(char* dataDir)
�������ͼ��Ľǵ㣬�����������ؾ�ȷ��
������all_corners������
*/
void getConrners(char* dataDir) {
	cout << "���ڽ��нǵ���ȡ" << endl;
	string dir = (string)dataDir + "/*.jpg";
	imgList = get_image_names(dir);
	string folder = (string)dataDir + "/corners";
	if (access(folder.c_str(), 0) == -1) {
		mkdir(folder.c_str());
	}//�½��ļ���corners

	for (int i = 0; i < imgList.size(); i++) {
		string imgDir = (string)dataDir + +"/"+imgList[i];//��ͼƬ���Ʋ���Ϊ����·��
		Mat imageInput = imread(imgDir);
		if (i == 0) {
			//��ȡ��һ��ͼʱ����¼�ߴ�
			image_size.height = imageInput.rows;
			image_size.width = imageInput.cols;
		}
		//��ȡ�ǵ�
		if (!findChessboardCorners(imageInput, board_size, image_corners)) {
			cout << "��ͼƬ" << imgList[i] << "���Ҳ����ǵ�" << endl;
			exit(1);
		}
		else {
			Mat imageGray;
			cvtColor(imageInput, imageGray, CV_RGB2GRAY);
			find4QuadCornerSubpix(imageGray, image_corners, board_size);//�����ؾ�ȷ��
			all_corners.push_back(image_corners);//����ͼ��ǵ�
			drawChessboardCorners(imageInput, board_size, image_corners, false);
			imwrite((string)dataDir+"/corners/"+imgList[i], imageInput);//����궨�ǵ���ͼƬ��������ԭ·����corners�ļ�����
			cout << "ͼƬ" << imgDir << "�ǵ��ע���" << endl;
		}
	}
	cout << "ȫ��ͼƬ�ǵ��ע���" << endl;
	return;
}

/*
vector<String> get_image_names(string file_path)
���·�������з���Ҫ����ļ���������
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