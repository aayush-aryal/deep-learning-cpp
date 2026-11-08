#include "Tokenizer.h"
#include <set>

Tokenizer::Tokenizer(const std::string& text){
    std::set<char> chars(text.begin(), text.end());

    int i=0;
    for (char c:chars){
        int_to_char_[i]=c;
        char_to_int_[c]=i;
        i++;
    }
}


 std::vector<int> Tokenizer::encode(const std::string& text) const{
    std::vector<int> res;

    for (char c:text){
        res.push_back(char_to_int_.at(c));
    }
    return res;
 }


 std::string Tokenizer::decode(const std::vector<int>& encoded) const{
    std::string res;

    for (int e:encoded){
        res.push_back(int_to_char_.at(e));
    }
    return res;
 }

 int Tokenizer::vocab_size() const {
    return static_cast<int>(int_to_char_.size());
}