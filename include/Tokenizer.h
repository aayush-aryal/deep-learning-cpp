#pragma once 
#include <string>
#include <vector>
#include <unordered_map>


class Tokenizer{

    public:
        Tokenizer(const std::string& text);
        std::vector<int> encode(const std::string& text)const;
        std::string decode(const std::vector<int>& encoded) const;
        int vocab_size() const;

    private:
    std::unordered_map<char, int> char_to_int_;
    std::unordered_map<int,char> int_to_char_;

};