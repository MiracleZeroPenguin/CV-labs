#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;

void findEllipse(IplImage*);
int threType=0;//Ĭ��ʹ����Ӧ����ֵ
int main(int argc, char** argv) {
	//const char* filename = "input.png";
	IplImage* imageSrc;
	const char* filename;
	if (argc !=2) {
		cout << "please input your image dir" << endl;
		system("pause");
		return 0;
	}
	else {
		filename = argv[1];
	}
	cout << "chose cvThreshold or cvAdaptiveThreshold?(1 or 0):";
	cin >> threType;
	if ((imageSrc = cvLoadImage(filename, 0)) == 0) {
		cout << "illegal file";
	}
	else {
		findEllipse(imageSrc);
	}
	return 0;
}



//����imageThreshold������ֵ��
void findEllipse(IplImage* imageSrc) {
	CvMemStorage* storage;
	CvSeq* contour;
	//��̬�����Ĵ���
	storage = cvCreateMemStorage(0);
	contour = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	//�ֲ���ֵ��
	IplImage* imageThreshold = cvCloneImage(imageSrc);
	if (threType == 1) {
		cout << "use cvThreshold" << endl;
		cvThreshold(imageSrc, imageThreshold, 125, 255, CV_THRESH_BINARY);
	}
	else {
		cout << "use cvAdaptiveThreshold" << endl;
		cvAdaptiveThreshold(imageSrc, imageThreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 125, 10);
	}
	//cvSaveImage("./Threshold.png", imageThreshold);
	//��ʴ
	IplImage* imageErode = cvCloneImage(imageSrc);
	cvErode(imageThreshold, imageErode);
	//��Ե
	IplImage* imageCanny= cvCloneImage(imageSrc);
	cvCanny(imageErode, imageCanny, 100, 150, 3);
	//cvSaveImage("./Canny.png", imageCanny);
	//��ȡ����
	cvFindContours(imageCanny, storage, &contour, sizeof(CvContour),CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	//ellipse
	IplImage* imageOut = cvCloneImage(imageSrc);
	IplImage* imageCompare = cvCloneImage(imageSrc);
	cvZero(imageOut);
	int numEllipse = 0;
	for (; contour; contour = contour->h_next) {
		int count = contour->total; // �����е������
		CvPoint center;
		CvSize size;
		CvBox2D box;

		// Ϊ������ֹ�С����Բ�����õ���������Сֵ
		if (count < 10)
			continue;
		imageThreshold++;
		CvMat* points_f = cvCreateMat(1, count, CV_32FC2);
		CvMat points_i = cvMat(1, count, CV_32SC2, points_f->data.ptr);
		cvCvtSeqToArray(contour, points_f->data.ptr, CV_WHOLE_SEQ);
		cvConvert(&points_i, points_f);

		//�Ե�ǰ����������Բ���
		box = cvFitEllipse2(points_f);

		// ��Բ�Ļ���
		center = cvPointFrom32f(box.center);
		size.width = cvRound(box.size.width*0.5);
		size.height = cvRound(box.size.height*0.5);

		cvEllipse(imageOut, center, size,box.angle, 0, 360,CV_RGB(0, 0, 255), 1, CV_AA, 0);
		cvEllipse(imageCompare, center, size, box.angle, 0, 360, CV_RGB(0, 0, 255), 1, CV_AA, 0);
		cvReleaseMat(&points_f);
	}

	//����
	cvSaveImage("./output.png", imageOut);
	cvSaveImage("./compare.png", imageCompare);
}

