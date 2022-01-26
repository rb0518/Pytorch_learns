#pragma once

#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <map>
#include <fstream>
#include "json.hpp"

struct trainTricks {
	unsigned int freeze_epochs = 0;
	std::vector<unsigned int> decay_epochs = { 0 };
	float dice_ce_ratio = 0.5;

	float horizontal_flip_prob = 0;
	float vertical_flip_prob = 0;
	float scale_rotate_prob = 0;

	float scale_limit = 0.1f;
	float rotate_limit = 45.0f;
	int interpolation = cv::INTER_LINEAR;
	int border_mode = cv::BORDER_CONSTANT;
};


class SegDataset : public torch::data::Dataset<SegDataset>
{
public:
	SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
		std::vector<std::string> list_labels, std::vector<std::string> name_list, 
		trainTricks trichs, bool isTrain = false);

	torch::optional<size_t> size() const override {
		return list_labels_.size();
	};

	torch::data::Example<> get(size_t index) override;
private:
	void draw_mask(std::string json_path, cv::Mat& mask);
	std::vector<std::string> list_images_;
	std::vector<std::string> list_labels_;
	std::vector<std::string> list_names_;
	std::map<std::string, int> name2index_;
	std::map<std::string, cv::Scalar> name2color_;
	int resize_width_;
	int resize_height_;
	bool is_train_;
	trainTricks tricks_;
};

namespace COCODATA {
	/*
	// JSON file such as following strust.
	{"info": 
		{	
			"description": "COCO 2017 Dataset", 
			"url" : "http://cocodataset.org", 
			"version" : "1.0", 
			"year" : 2017, 
			"contributor" : "COCO Consortium", 
			"date_created" : "2017/09/01"
		}, 
	"licenses" : 
	[
		{
			"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", 
			"id" : 1, 
			"name" : "Attribution-NonCommercial-ShareAlike License"
		}, 
		{ 
			"url": "http://creativecommons.org/licenses/by-nc/2.0/",
			"id" : 2,
			"name" : "Attribution-NonCommercial License" 
		}, 
		{ 
			"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
			"id" : 3,
			"name" : "Attribution-NonCommercial-NoDerivs License" 
		},
		{ 
			"url": "http://creativecommons.org/licenses/by/2.0/",
			"id" : 4,
			"name" : "Attribution License" 
		}, 
		{ 
			"url": "http://creativecommons.org/licenses/by-sa/2.0/",
			"id" : 5,
			"name" : "Attribution-ShareAlike License" 
		}, 
		{ 
			"url": "http://creativecommons.org/licenses/by-nd/2.0/",
			"id" : 6,
			"name" : "Attribution-NoDerivs License" 
		}, 
		{ 
			"url": "http://flickr.com/commons/usage/",
			"id" : 7,
			"name" : "No known copyright restrictions" 
		}, 
		{ 
			"url": "http://www.usa.gov/copyright.shtml",
			"id" : 8,
			"name" : "United States Government Work" 
		}
	] , 
	"images" : 
	[
		{
			"license": 4, 
			"file_name" : "000000397133.jpg", 
			"coco_url" : "http://images.cocodataset.org/val2017/000000397133.jpg", 
			"height" : 427, 
			"width" : 640, 
			"date_captured" : "2013-11-14 17:02:52", 
			"flickr_url" : "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg", 
			"id" : 397133
		}, 
		{ 
			"license": 1,
			"file_name" : "000000037777.jpg",
			"coco_url" : "http://images.cocodataset.org/val2017/000000037777.jpg",
			"height" : 230,
			"width" : 352,
			"date_captured" : "2013-11-14 20:55:31",
			"flickr_url" : "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
			"id" : 37777 
		}, 
		{ 
			"license": 4,
			"file_name" : "000000252219.jpg",
			"coco_url" : "http://images.cocodataset.org/val2017/000000252219.jpg",
			"height" : 428,
			"width" : 640,
			"date_captured" : "2013-11-14 22:32:02",
			"flickr_url" : "http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg",
			"id" : 252219 
		},
		......

		"annotations": 
		[
			{
				"segmentation": 
				[
					[
						510.66,423.01,511.72,420.03,510.45,416.0,510.34,413.02,510.77,410.26,510.77,407.5,510.34,405.16,511.51,402.83,511.41,400.49,510.24,398.16,509.39,397.31,504.61,399.22,502.17,399.64,500.89,401.66,500.47,402.08,499.09,401.87,495.79,401.98,490.59,401.77,488.79,401.77,485.39,398.58,483.9,397.31,481.56,396.35,478.48,395.93,476.68,396.03,475.4,396.77,473.92,398.79,473.28,399.96,473.49,401.87,474.56,403.47,473.07,405.59,473.39,407.71,476.68,409.41,479.23,409.73,481.56,410.69,480.4,411.85,481.35,414.93,479.86,418.65,477.32,420.03,476.04,422.58,479.02,422.58,480.29,423.01,483.79,419.93,486.66,416.21,490.06,415.57,492.18,416.85,491.65,420.24,492.82,422.9,493.56,424.39,496.43,424.6,498.02,423.01,498.13,421.31,497.07,420.03,497.07,415.15,496.33,414.51,501.1,411.96,502.06,411.32,503.02,415.04,503.33,418.12,501.1,420.24,498.98,421.63,500.47,424.39,505.03,423.32,506.2,421.31,507.69,419.5,506.31,423.32,510.03,423.01,510.45,423.01
					]
				],
				"area": 702.1057499999998,
				"iscrowd": 0,
				"image_id": 289343,
				"bbox": 
				[
					473.07,395.93,38.65,28.67
				],
				"category_id": 18,
				"id": 1768
			},
			{
				"segmentation": 
				[
		......
		"categories": 
		[
			{
				"supercategory": "person",		// person vehicle outdoor animal indoor accessory sports 
				"id": 1,
				"name": "person"
			},
			{	
				"supercategory": "vehicle",
				"id": 2,
				"name": "bicycle"
			},
			......
			{
				"supercategory": "indoor",
				"id": 90,
				"name": "toothbrush"
			}
		]
	}
	*/

	typedef struct tagJSONFileInfo {
		std::string description;		
		std::string url;
		std::string version;
		int year;
		std::string contributor;
		std::string date_created;
	}JSONFileInfo;

	typedef struct tagJSONFileLicense {
		std::string url;
		int id;
		std::string name;
	}JSONFileLicense;

	typedef struct tagCOCOImage {
		int license;
		std::string file_name;
		int height;
		int width;
		std::string date_captured;
		std::string flickr_url;
		int id;
	}COCOImage;

	typedef struct tagCOCOAnnotations
	{
		std::vector<float> segmentation;
		float area;
		int iscrowd;
		int image_id;
		float bbox[4];
		int category_id;
		int id;
		int width;
		int height;
	}COCOAnnotations;

	typedef struct tagCOCOCategories {
		std::string supercategory;
		int id;
		std::string name;
	}COCOCategories;

	typedef struct tagCOCO_JSONFile {
		JSONFileInfo info;
		std::vector<JSONFileLicense> licenses;
		std::vector<COCOImage> images;
		std::vector<COCOAnnotations> annotations;
		std::vector<COCOCategories> categories;

		void read_info(nlohmann::json j)
		{
			info.description = j.at("description");
			info.url = j.at("url");
			info.version = j.at("version");
			info.year = j.at("year");
			info.contributor = j.at("contributor");
			info.date_created = j.at("date_created");
		}

		void read_licenses(nlohmann::json j)
		{
			licenses.clear();
			for (int i = 0; i < j["licenses"].size(); i++)
			{
				JSONFileLicense license;
				license.url = j["licenses"][i]["url"];
				license.id = j["licenses"][i]["id"];
				license.name = j["licenses"][i]["name"];
				licenses.push_back(license);
			}

			std::cout << "load licenses cout: " << licenses.size() << std::endl;
		}

		void read_images(nlohmann::json j)
		{
			images.clear();
			COCOImage image;
			size_t totalimages = j.size();
			size_t i;
			for (i = 0; i < totalimages; i++)
			{
				try
				{
					image.license = j[i]["license"];
					image.file_name = j[i]["file_name"];
					image.height = j[i]["height"];
					image.width = j[i]["width"];
					image.date_captured = j[i]["date_captured"];
					image.flickr_url = j[i]["flickr_url"];
					image.id = j[i]["id"];

					images.push_back(image);
				}
				catch (const std::exception& e)
				{
					std::cout << "id " << i <<" : " << totalimages << e.what() << std::endl;
					for (int k = 0; k < j[i].size(); k++)
						std::cout << j[i][k] << "  ";
					std::cout << std::endl;
				}
			}
			std::cout << "load images count: " << images.size() << std::endl;
		}

		void read_annotations(nlohmann::json j)
		{
			annotations.clear();
			int totalsize = j.size();
			for (int i = 0; i < totalsize; i++)
			{
				COCOAnnotations anno;
				anno.segmentation.clear();

				if (j[i]["segmentation"].count("counts") > 0)
				{
					std::cout << "id: " << i << " " << j[i]["segmentation"]["counts"].size() << std::endl;
					for (int k = 0; k < j[i]["segmentation"]["counts"].size(); k++)
					{
						anno.segmentation.push_back(float(j[i]["segmentation"]["counts"][k]));
					}
					anno.width = j[i]["segmentation"]["size"][1];
					anno.height = j[i]["segmentation"]["size"][0];
				}
				else
				{
					for (int k = 0; k < j[i]["segmentation"][0].size(); k++)
					{
						anno.segmentation.push_back(j[i]["segmentation"][0][k]);
					}
					anno.width = -1;
					anno.height = -1;
				}


				anno.area = j[i]["area"];
				anno.iscrowd = j[i]["iscrowd"];
				anno.image_id = j[i]["image_id"];

				for(int k = 0; k < 4; k++)
					anno.bbox[k] = j[i]["bbox"][k];

				anno.category_id = j[i]["category_id"];
				anno.id = j[i]["id"];

				annotations.push_back(anno);
			}

			std::cout << "read label objects count: " << annotations.size() << std::endl;
		}

		void read_categories(nlohmann::json j)
		{
			categories.clear();
			for (int i = 0; i < j.size(); i++)
			{
				COCOCategories category;

				category.supercategory = j[i]["supercategory"];
				category.id = j[i]["id"];
				category.name = j[i]["name"];
			}
		}

		void read_cocojsonfile(std::string filename)
		{
			std::ifstream jfile(filename);
			nlohmann::json j;
			jfile >> j;

			read_info(j["info"]);
			read_licenses(j);
			read_images(j["images"]);
			read_annotations(j["annotations"]);
			read_categories(j["categories"]);
		}
	}COCO_JSONFILE;
};		// end namespace COCODATA