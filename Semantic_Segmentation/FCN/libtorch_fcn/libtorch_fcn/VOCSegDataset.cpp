#include "VOCSegDataset.h"
#include <filesystem>

#include <glog/logging.h>

#include "myutils.h"

std::vector<std::string> class_names = {
	"backgroud", "areoplane", "bicycle", "bird", "boat",
	"bottle", "bus","car", "cat","chair",
	"cow", "table","dog", "horse", "motorbike",
	"person","plant","sheep", "sofa"," train",
	"monitor"
};

std::vector<cv::Scalar> color_list = {
	cv::Scalar(0, 0, 0),
	cv::Scalar(128, 0, 0),
	cv::Scalar(0, 128, 0),
	cv::Scalar(128, 128, 0),
	cv::Scalar(0, 0, 128),
	cv::Scalar(128, 0, 128),
	cv::Scalar(0, 128, 128),
	cv::Scalar(128, 128, 128),
	cv::Scalar(64, 0, 0),
	cv::Scalar(192, 0, 0),
	cv::Scalar(64, 128, 0),
	cv::Scalar(192, 128, 0),
	cv::Scalar(64, 0, 128),
	cv::Scalar(192, 0, 128),
	cv::Scalar(64, 128, 128),
	cv::Scalar(192, 128, 128),
	cv::Scalar(0, 64, 0),
	cv::Scalar(128, 64, 0),
	cv::Scalar(0, 192, 0),
	cv::Scalar(128, 192, 0),
	cv::Scalar(0, 64, 128),
};


template<typename T>
T RandomNum(T _min, T _max)
{
	T temp;
	if (_min > _max)
	{
		temp = _max;
		_max = _min;
		_min = temp;
	}
	return rand() / (double)RAND_MAX * (_max - _min) + _min;
}


cv::Mat centerCrop(cv::Mat srcImage, int width, int height) {
	int srcHeight = srcImage.rows;
	int srcWidth = srcImage.cols;
	int maxHeight = srcHeight > height ? srcHeight : height;
	int maxWidth = srcWidth > width ? srcWidth : width;
	cv::Mat maxImage = cv::Mat::zeros(cv::Size(maxWidth, maxHeight), srcImage.type());
	int h_max_start = int((maxHeight - srcHeight) / 2);
	int w_max_start = int((maxWidth - srcWidth) / 2);
	srcImage.clone().copyTo(maxImage(cv::Rect(w_max_start, h_max_start, srcWidth, srcHeight)));

	int h_start = int((maxHeight - height) / 2);
	int w_start = int((maxWidth - width) / 2);
	cv::Mat dstImage = maxImage(cv::Rect(w_start, h_start, width, height)).clone();
	return dstImage;
}

cv::Mat RotateImage(cv::Mat src, float angle, float scale, int interpolation, int boder_mode)
{
	cv::Mat dst;

	//make output size same with input after scaling
	cv::Size dst_sz(src.cols, src.rows);
	scale = 1 + scale;
	cv::resize(src, src, cv::Size(int(src.cols * scale), int(src.rows * scale)));
	src = centerCrop(src, dst_sz.width, dst_sz.height);

	//center for rotating 
	cv::Point2f center(static_cast<float>(src.cols / 2.), static_cast<float>(src.rows / 2.));

	//rotate matrix     
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

	cv::warpAffine(src, dst, rot_mat, dst_sz, interpolation, boder_mode);
	return dst;
}

const int const_height = 720, const_weight = 960;
//int batch_size = 1;
//�����input image�ĳߴ�����˲ü�����ʺ�����ṹ�м�����������
const int train_h = 480;  
const int train_w = 320;  

const int val_h = 480;
const int val_w = 320;


VOCSegDataset::VOCSegDataset(std::string data_root, std::string run_mode, int64_t num_class)
	:data_root_(data_root), run_mode_(run_mode), num_class_(num_class)
{
	if (run_mode_ == "train")
	{
		list_file_ = std::filesystem::path(data_root).append("ImageSets/Segmentation/train.txt").string();
	}
	else if (run_mode_ == "val")
	{
		list_file_ = std::filesystem::path(data_root).append("ImageSets/Segmentation/val.txt").string();
	}
	else {
		LOG(FATAL) << "must be train or val";
	}

	std::vector<std::string> imageset_files;
	MyUtils::read_lines_from_file(list_file_, imageset_files);
	std::filesystem::path imagespath = std::filesystem::path(data_root_).append("JPEGImages");
	std::filesystem::path labelpath = std::filesystem::path(data_root_).append("SegmentationClass");
	for_each(imageset_files.cbegin(), imageset_files.cend(), [&](const std::string& val) {
		std::string image_name = val + std::string(".jpg");
		std::string label_name = val + std::string(".png");
		image_files_.push_back(imagespath.string() + "\\"+ image_name);
		label_files_.push_back(labelpath.string() + "\\"+ label_name);
		});

}

torch::data::Example<> VOCSegDataset::get(size_t index)
{
	// image ԭʼͼ��
	// label ��ǩͼ��shape {height, weight}, ����ֵΪ��Ӧ��classvalue
	// target one-hot����ShapeΪ{channel = 21<VOC2012������Ϊ20�����ϱ���Ϊ21, height, weight(0..1)}
	std::string image_name = image_files_.at(index);
	std::string label_name = label_files_.at(index);
/*	LOG(INFO) << "index: " << index << " image: " << image_name << "label: " << label_name;*/

	cv::Mat img = cv::imread(image_name);		// ͼ������ΪBGR
	cv::Mat label_img = cv::imread(label_name);


	std::vector<cv::Mat> vectImg;
	vectImg.push_back(img);
	vectImg.push_back(label_img);

	if (run_mode_ == "train")
	{
		Resize(vectImg, vectImg, train_w, train_h, 1);
		HorizontalFlip(vectImg, vectImg, 0);
		VerticalFlip(vectImg, vectImg, 0);
		RandomScaleRotate(vectImg, vectImg, 0, 45.0, 0.1, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	}
	else
	{
		Resize(vectImg, vectImg, val_w, val_h, 1);
	}
	img = vectImg[0];
	label_img = vectImg[1];

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).contiguous();		// change to {Channels, height, width)
	img_tensor = img_tensor.toType(torch::kFloat32);				// change to float
	auto img_tensor_tmp = img_tensor.clone().contiguous();

	// convert label image from {h, w} to {num_class, h, w}
	torch::Tensor color_label_tensor = torch::from_blob(label_img.data, { label_img.rows, label_img.cols, label_img.channels() }, torch::kByte);
	torch::Tensor label_tensor = torch::zeros({ label_img.rows, label_img.cols });
	torch::Tensor label_for_each_class = torch::zeros({ num_class_, label_img.rows, label_img.cols });

	for (int i = 0; i < color_list.size(); i++)
	{
		cv::Scalar color = color_list[i];
		torch::Tensor color_tensor = torch::tensor({ color.val[0], color.val[1], color[2] });
		label_tensor = torch::all(color_label_tensor == color_tensor, -1) * i;
		label_for_each_class[i] = label_tensor.clone();
	}

	return { img_tensor_tmp.clone(), label_for_each_class.clone() };
}

torch::optional<size_t> VOCSegDataset::size() const
{
	return image_files_.size();
}


void VOCSegDataset::Resize(std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, int width, int height, float probability)
{
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) 
	{
		for(int i = 0; i < src.size(); i++)
		{
			cv::resize(src[i], dst[i], cv::Size(width, height));
		}
	}
}

void VOCSegDataset::HorizontalFlip(std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, float probability)
{
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) 
	{
		for (int i = 0; i < src.size(); i++)
		{
			cv::flip(src[i], dst[i], 1);
		}

	}
}

void VOCSegDataset::VerticalFlip(std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, float probability)
{
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability)
	{
		for (int i = 0; i < src.size(); i++)
		{
			cv::flip(src[i], dst[i], 0);
		}
	}
}

void VOCSegDataset::RandomScaleRotate(std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, float probability, float rotate_limit,
	float scale_limit, int interpolation, int boder_mode)
{
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) 
	{
		float angle = RandomNum<float>(-rotate_limit, rotate_limit);
		float scale = RandomNum<float>(-scale_limit, scale_limit);
		for (int i = 0; i < src.size(); i++)
		{
			dst[i] = RotateImage(src[i], angle, scale, interpolation, boder_mode);
		}
	}
}
