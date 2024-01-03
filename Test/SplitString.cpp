#include <vector>
#include <iostream>
#include <string>
#include <cctype>
#include <string_view>

using i32 = int32_t;
using c8 = char;

std::vector<std::string> SplitStringsFromWhitespace(const std::string& str) {
    std::vector<std::string> ret;

    i32 start = 0;
    while (start < str.size()) {
        // skip leading whitespace
        while (start < str.size() && std::isspace(str[start])) {
            ++start;
        }

        // find the end of the current word
        auto end = start;
        while (end < str.size() && !std::isspace(str[end])) {
            ++end;
        }

        ret.push_back(str.substr(start, end - start));
        start = end;
    }

    return ret;
}

std::vector<std::string> SplitString(const std::string& str, c8 ch) {
    std::vector<std::string> strings;

    if (str.empty()) {
        return strings;
    }

    i32 start = 0;
    while (start < str.size()) {
        auto end = str.find(ch, start);
        if (end == std::string::npos) {
            strings.push_back(str.substr(start));
            break;
        } else {
            strings.push_back(str.substr(start, end - start));
            start = end + 1;
        }
    }

    return strings;
}

int main() {
    std::vector<std::string> strings;

    std::string testStr = "Hello World! Test Split String 12 $$ @#@ *** ## // ??";
    strings = SplitString(testStr, 'T');

    for (const auto& str : strings) {
        std::cout << str << std::endl;
    }
}
