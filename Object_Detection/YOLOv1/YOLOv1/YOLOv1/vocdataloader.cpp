#include "vocdataloader.hpp"
#include "utils.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <glog/logging.h>

DataLoader::VOCDataLoader::VOCDataLoader(std::string root, VOC_DATA_TYPE datatype, RUN_MODE runmode, 
	bool need_create_voc_label/* = true*/) : root_(root), data_type_(datatype), run_mode_()
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
