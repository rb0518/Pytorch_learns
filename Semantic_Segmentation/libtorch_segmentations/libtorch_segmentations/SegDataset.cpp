#include "SegDataset.h"
#include "Augmentations.h"

std::vector<cv::Scalar> get_color_list() {
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
	return color_list;
}

SegDataset::SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
	std::vector<std::string> list_labels, std::vector<std::string> name_list, trainTricks trichs, bool isTrain /* = false */)
	: tricks_(trichs), resize_width_(resize_width), resize_height_(resize_height), list_names_(name_list),
	list_images_(list_images), list_labels_(list_labels)
{
	for (int i = 0; i < name_list.size(); i++)
	{
		name2color_.insert(std::pair<std::string, int>(name_list[i], i));
	}
	std::vector<cv::Scalar> color_list = get_color_list();
	CHECK(name_list.size() <= color_list.size()) << "Number of classes exceeds define color list";
	for (int i = 0; i < name_list.size(); i++)
	{
		name2color_.insert(std::pair<std::string, cv::Scalar>(name_list[i], color_list[i]));
	}
}

torch::data::Example<> SegDataset::get(size_t index) {
	std::string image_path = list_images_.at(index);
	std::string label_path = list_labels_.at(index);
	cv::Mat image = cv::imread(image_path);
	cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);

	draw_mask(label_path, mask);

	auto m_data = Data(image, mask);
	if (is_train_)
	{
		m_data = Augmentations::Resize(m_data, resize_width_, resize_height_, 1);
		m_data = Augmentations::HorizontalFlip(m_data, tricks_.horizontal_flip_prob);
		m_data = Augmentations::VerticalFlip(m_data, tricks_.vertical_flip_prob);
		m_data = Augmentations::RandomScaleRotate(m_data, tricks_.scale_rotate_prob, \
			tricks_.rotate_limit, tricks_.scale_limit, \
			tricks_.interpolation, tricks_.border_mode);
	}
	else
	{
		m_data = Augmentations::Resize(m_data, resize_width_, resize_height_, 1);
	}

	torch::Tensor img_tensor = torch::from_blob(m_data.image.data, 
		{ m_data.image.rows, m_data.image.cols, m_data.image.channels() }, torch::kByte).permute({ 2, 0, 1 });	// CHW
	torch::Tensor colorful_label_tensor = torch::from_blob(m_data.mask.data,
		{ m_data.mask.rows, m_data.mask.cols, m_data.mask.channels() }, torch::kByte);
	torch::Tensor label_tensor = torch::zeros({ m_data.image.rows, m_data.image.cols });
	for (int i = 0; i < list_names_.size(); i++)
	{
		cv::Scalar color = name2color_[list_names_[i]];
		torch::Tensor color_tensor = torch::tensor({ color.val[0], color.val[1], color.val[2] });
		label_tensor = label_tensor + torch::all(colorful_label_tensor == color_tensor, -1) * i;	// 所有相等的点
	}
	label_tensor = label_tensor.unsqueeze(0);	// CHW==>NCHW
	return { img_tensor.clone(), label_tensor.clone() };
}
#include <fstream>
#include "json.hpp"
void SegDataset::draw_mask(std::string json_path, cv::Mat& mask)
{
	std::ifstream jfile(json_path);
	nlohmann::json j;
	jfile >> j;
	size_t  num_blobs = j["shapes"].size();		// 找到所有shapes节点
	for (int i = 0; i < num_blobs; i++)
	{
		std::string name = j["shapes"][i]["label"];	//目标的类名称
		size_t points_len = j["shapes"][i]["points"].size();

		std::vector<cv::Point> contour = {};
		for (int t = 0; t < points_len; t++)
		{
			int x = round(double(j["shapes"][i]["points"][t][0]));
			int y = round(double(j["shapes"][i]["points"][t][1]));

			contour.push_back(cv::Point(x, y));
		}

		const cv::Point* ppt[1] = { contour.data() };
		int npt[] = { int(contour.size()) };
		cv::fillPoly(mask, ppt, npt, 1, name2color_[name]);
	}
}