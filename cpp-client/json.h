// json.h —— 极简 JSON 解析器，够用就行
// 只实现本项目需要的部分：对象、数组、字符串、数字、布尔、null
#pragma once

#include <map>
#include <string>
#include <vector>

namespace json {

struct Value {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;

    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    bool is_null() const { return type == NUL; }

    // 取对象字段，不存在返回一个空 Value（不抛异常）
    const Value& operator[](const std::string& key) const {
        static const Value empty;
        auto it = obj.find(key);
        return it == obj.end() ? empty : it->second;
    }

    // 取数组元素，越界返回空 Value
    const Value& at(size_t i) const {
        static const Value empty;
        return i < arr.size() ? arr[i] : empty;
    }

    size_t size() const { return type == ARR ? arr.size() : obj.size(); }
};

Value parse(const std::string& text);

}  // namespace json