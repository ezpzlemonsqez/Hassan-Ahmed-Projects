#include "csv.hpp"
#include <fstream>
#include <stdexcept>

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    out.push_back(cur);
    return out;
}

CSVTable read_csv(const std::string& path) {
    CSVTable t;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open CSV: " + path);

    std::string line;
    if (!std::getline(f, line)) return t;
    t.cols = split_csv_line(line);

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto fields = split_csv_line(line);
        std::unordered_map<std::string, std::string> row;
        row.reserve(t.cols.size());
        for (size_t i = 0; i < t.cols.size(); ++i) {
            row[t.cols[i]] = (i < fields.size()) ? fields[i] : "";
        }
        t.rows.push_back(std::move(row));
    }
    return t;
}