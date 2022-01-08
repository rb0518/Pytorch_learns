#pragma once
//#include <functional>
#include <algorithm>

#include <vector>

#include <glog/logging.h>
#include <filesystem>       // g++8.0 support std::filesytem

// �ж��Ƿ����ļ���
// ��filesytem�����޸���ԭ���룬һ���ܹ���windows��linux
inline bool is_folder(const char* dir_name)
{
	auto dir = std::filesystem::path(std::string(dir_name));
	return std::filesystem::is_directory(dir) && std::filesystem::exists(dir);
}

inline bool is_folder(const std::string& dir_name)
{
	auto dir = std::filesystem::path(dir_name);
	return std::filesystem::is_directory(dir) && std::filesystem::exists(dir);
}
using file_filter_type = std::function<bool(const char*, const char*)>;
/*
 * �г�ָ��Ŀ¼�������ļ�(������Ŀ¼)ִ�У���ÿ���ļ�ִ��filter��������
 * filter����trueʱ���ļ���ȫ·������std::vector
 * subΪtrueʱΪĿ¼�ݹ�
 * ����ÿ���ļ���ȫ·����
*/
#if 0
static  std::vector<std::string> for_each_file(const std::string& dir_name, file_filter_type filter, bool sub = false)
#else
static  std::vector<std::string> for_each_file(const std::string& dir_name, const std::string& filter, bool sub = false)
#endif
{
    std::vector<std::string> v;
#if 0
    auto dir = opendir(dir_name.data());
    struct dirent* ent;
    if (dir) {
        while ((ent = readdir(dir)) != nullptr) {
            auto p = std::string(dir_name).append({ file_sepator() }).append(ent->d_name);
            if (sub) {
                if (0 == strcmp(ent->d_name, "..") || 0 == strcmp(ent->d_name, ".")) {
                    continue;
                }
                else if (is_folder(p)) {
                    auto r = for_each_file(p, filter, sub);
                    v.insert(v.end(), r.begin(), r.end());
                    continue;
                }
            }
            if (sub || !is_folder(p))//������ļ�������ù�����filter
                if (filter(dir_name.data(), ent->d_name))
                    v.emplace_back(p);
        }
        closedir(dir);
    }
#else
    for (auto &dir_itr : std::filesystem::directory_iterator(dir_name))
    {
        if (std::filesystem::is_directory(dir_itr.status()) && sub)
        {
            auto ret = for_each_file(dir_itr.path().string(), filter, sub);
            v.insert(v.end(), ret.begin(), ret.end());
        }
        else
        {
            auto filetype = dir_itr.path().extension().string();
            if (filetype == filter)
            {
                v.push_back(dir_itr.path().string());
            }
        }
    }
#endif
    return v;
}

//�ַ�����Сдת��
inline std::string tolower1(const std::string& src) {
	auto dst = src;
	std::transform(src.begin(), src.end(), dst.begin(), ::tolower);
	return dst;
}
// �ж�src�Ƿ���ָ�����ַ���(suffix)��β
inline bool end_with(const std::string& src, const std::string& suffix) {
	return src.substr(src.size() - suffix.size()) == suffix;
}

