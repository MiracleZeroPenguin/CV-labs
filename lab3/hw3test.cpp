#include <iostream>

#include <opencv2/opencv.hpp>

const int num_people = 40;
const int img_per_person = 10;
const int width_resize = 30;
const int height_resize = 30;

class tester
{
private:
	int width_resized, height_resized;
	cv::Mat A_T;
	std::string datasetDirent;
	std::vector<std::string> fileNames;
	std::vector<cv::Mat> dataset;
public:
	tester(std::string modelName, std::string datasetDirent);
	void test(std::string fileName);
};

int main32(int argc, char* argv[])
{
	if (argc >= 4) {
		tester t(argv[2], argv[3]);
		t.test(argv[1]);
	}
	else {
		std::cout << "²ÎÊý²»×ã";
	}
	
	return 0;
}

tester::tester(std::string modelName, std::string datasetDirent)
{
	if ('/' != *(datasetDirent.end() - 1))
	{
		datasetDirent.push_back('/');
	}
	this->datasetDirent = datasetDirent;
	std::ifstream in(modelName);
	double energyPercent;
	int trainFileNum;
	in >> width_resized >> height_resized >> energyPercent >> trainFileNum;

	A_T = cv::Mat(cv::Size(width_resized * height_resized * energyPercent * 0.01, width_resized * height_resized), CV_64F);
	for (int i = 0; i < trainFileNum; ++i)
	{
		std::string fileName;
		in >> fileName;
		fileNames.push_back(fileName);
	}
	for (int i = 0; i < A_T.rows; ++i)
	{
		for (int j = 0; j < A_T.cols; ++j)
		{
			in >> A_T.at<double>(i, j);
		}
	}

	for (auto it : fileNames)
	{
		cv::Mat img = cv::imread(datasetDirent + it);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		cv::resize(img, img, cv::Size(width_resize, height_resize));
		img.reshape(0, 1).convertTo(img, CV_64F);
		cv::Mat Y_T = img * A_T;
		dataset.push_back(Y_T);
	}

}

void tester::test(std::string fileName)
{
	cv::Mat img_read, img;
	img_read = cv::imread(fileName);
	cv::cvtColor(img_read, img, cv::COLOR_BGR2GRAY);
	cv::resize(img, img, cv::Size(30, 30));
	img.reshape(0, 1).convertTo(img, CV_64F);
	cv::Mat coordinate = img * A_T;

	double min = -1;
	int index = 0;
	for (int i = 0; i < dataset.size(); ++i)
	{
		cv::Mat diff = coordinate - dataset[i];
		double mo = cv::norm(diff);
		if (mo < min || min < 0)
		{
			min = mo;
			index = i;
		}
	}

	cv::Mat origin = cv::imread(datasetDirent + fileNames[index]);

	cv::Mat blend;
	cv::addWeighted(img_read, 0.6, origin, 0.4, 0, blend);

	std::cout << fileNames[index][1] << fileNames[index][2] << '\n';

	cv::imwrite("test.jpg", img_read);
	cv::imwrite("blend.jpg", blend);
	cv::imwrite("origin.jpg", origin);
}