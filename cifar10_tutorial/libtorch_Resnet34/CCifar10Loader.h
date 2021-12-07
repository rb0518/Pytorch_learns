#pragma once
#include <string>
#include <vector>


#include <opencv2/opencv.hpp>

class CCifar10Loader
{
public:
	typedef enum enumDATATYPE {
		_TRAIN_ = 0,
		_TEST_
	}DATATYPE;

public:
	explicit CCifar10Loader(const std::string& folder, DATATYPE type);
	~CCifar10Loader() {};


	std::vector<cv::Mat> GetImages() { return images_; };
	std::vector<uchar> GetLabels() { return labels_id_; };
private:
	std::string folder_;
	DATATYPE datatype_;

	std::vector<cv::Mat> images_;
	std::vector<uchar> labels_id_;

private:
	void read_cifar10_bin();
};

