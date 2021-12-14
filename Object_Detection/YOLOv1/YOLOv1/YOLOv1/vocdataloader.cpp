#include "vocdataloader.hpp"
#include "utils.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <glog/logging.h>

DataLoader::VOCDataLoader::VOCDataLoader(std::string root, VOC_DATA_TYPE datatype, RUN_MODE runmode, 
	std::vector<transforms_Compose>& transformBB, std::vector<transforms_Compose>& transformI,
	bool need_create_voc_label/* = true*/) : root_(root), data_type_(datatype), run_mode_(),
	transformBB_(transformBB), transformI_(transformI)
{
	classnames_filename_ = std::filesystem::path(root).append("clasnames.txt").string();
	std::filesystem::path dirtemp(root);
	if (datatype == VOC_2007)
	{
		dirtemp = dirtemp.append("VOC2007");
	}
	else
	{
		dirtemp = dirtemp.append("VOC2012");
	}
	root_ = dirtemp.string();

	annotations_path_ = dirtemp.string() + "\\Annotations";
	images_path_ = dirtemp.string() + "\\JPEGImages";
	labels_path_ = dirtemp.string() + "\\labels";
	LOG(INFO) << "dataset root: " << root_;
	LOG(INFO) << "class_names_file: " << classnames_filename_;
	LOG(INFO) << "Annotations path: " << annotations_path_;
	LOG(INFO) << "JPEGImages: " << images_path_;

	loadclassnames();
	loadfilenames();

	std::filesystem::create_directories(labels_path_);
	create_voc_labels(need_create_voc_label);

}

bool DataLoader::VOCDataLoader::loadfilenames()
{
	std::filesystem::path dirtemp(root_);
	std::string datapath = dirtemp.append("ImageSets").append("Main").string();
	std::string datafilename;
	if (run_mode_ == _TRAIN_)
	{
		datafilename = datapath + "\\train.txt";
	}
	else
	{
		datafilename = datapath + "\\trainval.txt";
	}
	LOG(INFO) << "Load data file names from file: " << datafilename;

	read_lines_from_file(datafilename, filenames_);

	return true;
}

bool DataLoader::VOCDataLoader::loadclassnames()
{
	if (!std::filesystem::exists(std::filesystem::path(classnames_filename_)))
	{
		return false;
	}

	read_lines_from_file(classnames_filename_, class_names_);
	class_nameindex_map_.clear();
	for (int i = 0; i < class_names_.size(); i++)
	{
		class_nameindex_map_.insert(std::pair <std::string, int>(class_names_[i], i));
	}

	return true;
}
#include <iostream>
#include <fstream>
void DataLoader::VOCDataLoader::create_voc_labels(bool create_voc_file)
{
	std::ofstream of;
	if (create_voc_file)
	{
		of.open(labels_path_ + "//voc2012.txt");
	}
	samples_.clear();
	for (auto filename : filenames_)
	{
		SamplesDef sampletemp;
		sampletemp.filename = images_path_ + "//" + filename + ".jpg";
		convert_annotation(annotations_path_, labels_path_, filename, sampletemp.labels, sampletemp.bboxes);
		samples_.push_back(sampletemp);

		if (create_voc_file)
		{
			of << sampletemp.filename << ".jpg ";		// path not have empty char 
			for (int i = 0; i < sampletemp.labels.size(); i++)
			{
				of << sampletemp.bboxes[i][0] << " " << sampletemp.bboxes[i][1] << " " << sampletemp.bboxes[i][2] 
					<< " " << sampletemp.bboxes[i][3] << " " << sampletemp.labels[i] << " ";
			}
			of << std::endl;
		}

	}
	if (create_voc_file)
	{
		of.close();
	}
}

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/foreach.hpp>
using namespace boost::property_tree;
void DataLoader::VOCDataLoader::convert_annotation(std::string annot_path, std::string out_path, std::string image_id,
												std::vector<int>& labels, std::vector<std::vector<int>>& bboxes)
{
	std::string in_filename = annot_path + "\\" + image_id + ".xml";
	std::string out_filename = out_path + "\\" + image_id + ".txt";

	CHECK(std::filesystem::exists(std::filesystem::path(in_filename))) << "file " << in_filename << "not exists!";

	ptree rootTree;

	try {
		xml_parser::read_xml(in_filename, rootTree);
		int width = rootTree.get<int>("annotation.size.width");
		int height = rootTree.get<int>("annotation.size.height");

		BOOST_FOREACH(ptree::value_type& v1, rootTree.get_child("annotation")) 
		{		// 遍循查找整个annotation节点，查找所有的object子节点
			if (v1.first == "object")
			{
				std::string objname = v1.second.get<std::string>("name");
				int classindex = get_classindex(objname);
				auto bboxtree = v1.second.get_child("bndbox");
				int xmin = bboxtree.get<int>("xmin");
				int ymin = bboxtree.get<int>("ymin");
				int xmax = bboxtree.get<int>("xmax");
				int ymax = bboxtree.get<int>("ymax");

				//std::cout << "bbox: " << xmin << " " << ymin << " - " << xmax << " " << ymax << std::endl;

				std::vector<int> tmp = { xmin, ymin, xmax, ymax };
				bboxes.push_back(tmp);
				labels.push_back(classindex);
			}
		}
	}
	catch (std::exception& e)
	{
		LOG(ERROR) << "open xml: " << in_filename << "  error!";
		return;
	}
 }


#include <opencv2/opencv.hpp>
void  DataLoader::VOCDataLoader::load(const size_t index, std::tuple<torch::Tensor,
	std::tuple<torch::Tensor, torch::Tensor>,
	std::string, std::string>& data)
{
	// 1 load image from file
	cv::Mat image_Mat_BGR, image_Mat_RGB;
	std::string imagefilename = images_path_ + "//" + filenames_[index] + ".jpg";
	image_Mat_BGR = cv::imread(imagefilename, cv::IMREAD_COLOR);
	cv::cvtColor(image_Mat_BGR, image_Mat_RGB, cv::COLOR_BGR2RGB);

	int imagewidth = image_Mat_RGB.cols;
	int imageheight = image_Mat_RGB.rows;

	// 2 load bbox and class_id from vector samples_
	float bbox_cx, bbox_cy;				// bbox 中心点坐标
	float bbox_width, bbox_height;		// bbox 宽、高

	float objectid;

	torch::Tensor id, cx, cy, w, h, coord;
	torch::Tensor ids, coords;
	std::tuple<torch::Tensor, torch::Tensor> BBoxs;

	int num_bbox = samples_[index].labels.size();		
	for (int i = 0; i < num_bbox; i++)
	{
		float x1 = float(samples_[index].bboxes[i][0]);
		float y1 = float(samples_[index].bboxes[i][1]);
		float x2 = float(samples_[index].bboxes[i][2]);
		float y2 = float(samples_[index].bboxes[i][3]);

		bbox_cx = (x1 + x2) / 2;
		bbox_cy = (y1 + y2) / 2;
		bbox_width = x2 - x1;
		bbox_height = y2 - y1;

		objectid = float(samples_[index].labels[i]);

		id = torch::full({ 1, 1 }, objectid, torch::TensorOptions().dtype(torch::kFloat));
		cx = torch::full({ 1, 1 }, bbox_cx, torch::TensorOptions().dtype(torch::kFloat));
		cy = torch::full({ 1,1 }, bbox_cy, torch::TensorOptions().dtype(torch::kFloat));
		w = torch::full({ 1, 1 }, bbox_width, torch::TensorOptions().dtype(torch::kFloat));
		h = torch::full({ 1,1 }, bbox_height, torch::TensorOptions().dtype(torch::kFloat));
		coord = torch::cat({ cx, cy, w, h }, 1);

		if (i == 0)
		{
			ids = id;
			coords = coord;
		}
		else
		{
			ids = torch::cat({ ids, id }, 0);
			coords = torch::cat({ coords, coord }, 0);
		}
	}

	if (ids.numel() > 0)
	{
		ids = ids.contiguous().detach().clone();
		coords = coords.contiguous().detach().clone();
	}

	BBoxs = { ids, coords };

	cv::Mat image_Mat_mid;
	std::tuple<torch::Tensor, torch::Tensor> BBs_mid;
	for (size_t i = 0; i < transformBB_.size(); i++)
	{
		this->deepcopy(image_Mat_RGB, BBoxs, image_Mat_mid, BBs_mid);
		this->transformBB_.at(i)->forward(image_Mat_mid, BBs_mid, image_Mat_RGB, BBoxs);
	}

	torch::Tensor image = transforms::apply(transformI_, image_Mat_RGB);

	data = { image.detach().clone(), BBoxs, filenames_[index], filenames_[index] };
}

bool DataLoader::VOCDataLoader::loadbatch(const size_t startindex, const size_t batchsize, std::tuple<torch::Tensor,
	std::vector<std::tuple<torch::Tensor, torch::Tensor>>,
	std::vector<std::string>, std::vector<std::string>>& batchdata)
{
	if (samples_.size() < (startindex + batchsize))
		return false;

	torch::Tensor data1, tensor1;
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> data2;
	std::vector<std::string> data3, data4;
	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>, std::string, std::string> *batchdata_temp;

	CHECK(batchsize >= 2) << "batch size must more than 1";
	batchdata_temp = new std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>, std::string, std::string>[batchsize];
	CHECK(nullptr != batchdata_temp) << "allocate memory error!";
	
	CHECK(samples_.size() > 0) << "samples.size == 0";

	for (int i = 0; i < batchsize; i++)
	{
		load(i % samples_.size(), batchdata_temp[i]);
	}

	data1 = std::get<0>(batchdata_temp[0]);
	data1 = torch::unsqueeze(data1, 0);
	data2.push_back(std::get<1>(batchdata_temp[0]));
	data3.push_back(std::get<2>(batchdata_temp[0]));
	data4.push_back(std::get<3>(batchdata_temp[0]));
	for (int i = 1; i < batchsize; i++)
	{
		tensor1 = std::get<0>(batchdata_temp[i]);
		tensor1 = torch::unsqueeze(tensor1, 0);
		data1 = torch::cat({ data1, tensor1 }, 0);
		data2.push_back(std::get<1>(batchdata_temp[i]));
		data2.push_back(std::get<1>(batchdata_temp[i]));
		data2.push_back(std::get<1>(batchdata_temp[i]));
	}
	data1 = data1.contiguous().detach().clone();
	batchdata = { data1, data2, data3, data4 };
	delete[] batchdata_temp;

	return true;
}


void DataLoader::VOCDataLoader::deepcopy(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, 
			cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2) 
{
	data_in1.copyTo(data_out1);
	if (std::get<0>(data_in2).numel() > 0) {
		data_out2 = { std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone() };
	}
	else {
		data_out2 = { torch::Tensor(), torch::Tensor() };
	}
	return;
}
