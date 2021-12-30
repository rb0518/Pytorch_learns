#pragma once
#include <vector>
#include <string>


namespace MyUtils {
	bool read_lines_from_file(std::string filepathname, std::vector<std::string>& datas, bool appendflag = false);
}

