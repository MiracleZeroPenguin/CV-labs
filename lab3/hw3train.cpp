#include <iostream>

#include <opencv2/opencv.hpp>

const int num_people = 40;
const int img_per_person = 10;
const int width_resize = 30;
const int height_resize = 30;

class trainer
{
private:
	std::vector<cv::Mat> dataset;
	std::vector<std::string> trainFiles;
	cv::Mat A_T;
	double energyPercent;
public:
	trainer(std::string datasetDirent, int trainPercent);
	void train(double energyPercent);
	void write(std::string modelName);
};

int main(int argc, char* argv[])
{
	if (argc >= 4) {
		double energyPer = atof(argv[1]);
		trainer t(argv[3], 90);
		t.train(energyPer);
		t.write(argv[2]);
	}
	else {
		std::cout << "²ÎÊý²»×ã";
	}
	return 0;
}

trainer::trainer(std::string datasetDirent, int trainPercent)
{
	if ('/' != *(datasetDirent.end() - 1))
	{
		datasetDirent.push_back('/');
	}
	std::string fileName;
	for (int i = 1; i <= num_people; ++i)
	{
		for (int j = 1; j <= (int)(img_per_person * trainPercent * 0.01); ++j)
		{
			fileName = 's' + std::to_string(i) + "/" + std::to_string(j) + ".bmp";
			trainFiles.push_back(fileName);
			cv::Mat gray, imgResized;
			cv::cvtColor(cv::imread(datasetDirent + fileName), gray, cv::COLOR_BGR2GRAY);
			cv::resize(gray, imgResized, cv::Size(width_resize, height_resize));
			dataset.push_back(imgResized);
		}
	}
}

void trainer::train(double energyPercent)
{
	this->energyPercent = energyPercent;
	cv::Mat covar, mean;
	cv::Mat eigenValue, eigenVector;
	cv::calcCovarMatrix(dataset, covar, mean, cv::COVAR_NORMAL);
	cv::eigen(covar, eigenValue, eigenVector);

	A_T = eigenVector.colRange(0, eigenVector.cols * energyPercent * 0.01);

	std::vector<cv::Mat> img_samples;
	cv::Mat img_cat;
	for (int i = 0; i < 10; ++i)
	{
		cv::Mat tmp(cv::Size(width_resize, height_resize), CV_64F);
		cv::Mat tmp_int(cv::Size(width_resize, height_resize), CV_8UC1);
		for (int j = 0; j < width_resize * height_resize; ++j)
		{
			tmp.at<double>(j / width_resize, j % width_resize) = eigenVector.at<double>(i, j);
		}
		cv::normalize(tmp, tmp, 255, 0, cv::NORM_MINMAX);
		tmp.convertTo(tmp_int, CV_8UC1);
		img_samples.push_back(tmp_int);
	}
	cv::hconcat(img_samples, img_cat);
	cv::imwrite("cat.jpg", img_cat);
}

void trainer::write(std::string modelName)
{
	std::ofstream out(modelName);
	out << width_resize << ' ' << height_resize << '\n';
	out << energyPercent << '\n';
	out << trainFiles.size() << '\n';
	for (const auto& it : trainFiles)
	{
		out << it << '\n';
	}
	for (int i = 0; i < A_T.rows; ++i)
	{
		for (int j = 0; j < A_T.cols; ++j)
		{
			out << A_T.at<double>(i, j) << ' ';
		}
	}
	out << '\n';
}