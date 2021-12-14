#pragma once
#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>

#include "augmentation.hpp"

namespace DataLoader {
	class VOCDataLoader
	{
	public:
		enum VOC_DATA_TYPE {
			VOC_2007 = 0,
			VOC_2012
		};

		enum RUN_MODE
		{
			_TRAIN_ = 0,
			_TEST_
		};

		typedef struct tagSamplesDef
		{
			std::string filename;
			std::vector<int> labels;
			std::vector<std::vector<int>> bboxes;
		}SamplesDef;


	public:
		/*!
		* @param root: VOC data path
		*/
		VOCDataLoader(std::string root, VOC_DATA_TYPE datatype, RUN_MODE runmode, 
			std::vector<transforms_Compose>& transformBB, std::vector<transforms_Compose>& transformI, 
			bool need_create_voc_label = true);

		std::string get_root() { return root_; }
		std::vector<std::string> get_classnames() { return class_names_; }
		int get_classindex(std::string classname) { return class_nameindex_map_.at(classname); }

		size_t get_samples_count() { return samples_.size(); }
		void reset_index() { current_index_ = 0; }
		int get_current_indexid() { return current_index_; }
		void set_current_indexid(size_t newvalue) { current_index_ = newvalue; }

		bool loadbatch(const size_t startindex, const size_t batchsize, std::tuple<torch::Tensor,
							std::vector<std::tuple<torch::Tensor, torch::Tensor>>,
							std::vector<std::string>, std::vector<std::string>>& batchdata);
		void load(const size_t index, std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>, std::string, std::string>& data);
		void deepcopy(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2,
			cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	private:
		bool loadfilenames();
		bool loadclassnames();		


//		void readoneimage(int imageid, cv::Mat& image, int)
		void create_voc_labels(bool create_voc_file);
		void convert_annotation(std::string annot_path, std::string out_path, std::string image_id,
							std::vector<int>& labels, std::vector<std::vector<int>>& bboxes);
	private:
		std::string root_;						// such as D:/VOCdevkit/VOC2012
		std::string classnames_filename_;		// such as D:/VOCdevkit/VOC2012/voc2012.txt
		std::string images_path_;				// such as D:/VOCdevkit/VOC2012/JPEGImages
		std::string annotations_path_;			// such as D:/VOCdevkit/VOC2012/Annotations
		std::string labels_path_;
		
		VOC_DATA_TYPE data_type_;
		RUN_MODE run_mode_;

		std::vector<std::string> class_names_;
		std::map<std::string, int> class_nameindex_map_;

		std::vector<std::string> filenames_; // store in ImageSets/Main/train.txt or val.txt

		std::vector<SamplesDef> samples_;

		int current_index_;

		std::vector<transforms_Compose> transformBB_, transformI_;
	};
}

