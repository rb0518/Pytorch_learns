#include "myutils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
//#include <glog/logging.h>


bool MyUtils::read_lines_from_file(std::string filepathname, std::vector<std::string>& datas, bool appendflag /*= false*/)
{
	if (false == appendflag)
	{
		datas.clear();
	}

	std::ifstream infile(filepathname);
	std::string readline;
	int read_count = 0;
	while (std::getline(infile, readline))
	{
		read_count++;
		datas.push_back(readline);
	}
//	LOG(INFO) << "Read " << read_count << " lines from file: " << filepathname;
	infile.close();
	return true;
}

