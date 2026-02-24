#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct CSVTable {
    std::vector<std::string> cols;
    std::vector<std::unordered_map<std::string, std::string>> rows;
};

CSVTable read_csv(const std::string& path);