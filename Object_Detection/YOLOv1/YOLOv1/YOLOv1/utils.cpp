#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <glog/logging.h>


bool load_class_names(std::string root, std::string filename, std::vector<std::string>& classnames)
{
	std::filesystem::path filepath(root);
	filepath.append(filename);
	if (!std::filesystem::exists(filepath))
	{
		LOG(ERROR) << "file: " << filepath.string() << " not exists";
		return false;
	}

	std::ifstream infile(filepath.string());
	std::string readline;
	while (std::getline(infile, readline))
	{
		classnames.push_back(readline);
	}
	std::cout << "load class names:" << std::endl;
	for (size_t i = 0; i < classnames.size(); i++)
	{
		std::cout << classnames.at(i) << " ";
	}
	std::cout << "| total " << classnames.size() <<  std::endl;

	infile.close();
	return true;
}

