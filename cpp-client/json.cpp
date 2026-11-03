// json.cpp —— 递归下降解析
#include "json.h"

#include <cstdlib>
#include <stdexcept>

namespace json {
namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    explicit Parser(const std::string& text) : s(text) {}

    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' ||
                                s[i] == '\r' || s[i] == '\t'))
            ++i;
    }

    char peek() {
        skip_ws();
        return i < s.size() ? s[i] : '\0';
    }

    void expect(char c) {
        skip_ws();
        if (i >= s.size() || s[i] != c)
            throw std::runtime_error(std::string("JSON 期望 '") + c + "'");
        ++i;
    }

    // 解析字符串，处理转义和 \uXXXX（含代理对）
    std::string parse_string() {
        expect('"');
        std::string out;
        while (i < s.size() && s[i] != '"') {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c != '\\') {
                out.push_back(s[i++]);
                continue;
            }
            ++i;  // 跳过反斜杠
            if (i >= s.size()) break;
            char e = s[i++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned cp = parse_hex4();
                    // 代理对：两个 \uXXXX 合成一个字符
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() &&
                        s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        unsigned lo = parse_hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: out.push_back(e); break;
            }
        }
        expect('"');
        return out;
    }

    unsigned parse_hex4() {
        if (i + 4 > s.size()) throw std::runtime_error("JSON \\u 不完整");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else throw std::runtime_error("JSON \\u 非法字符");
        }
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    Value parse_value() {
        char c = peek();
        Value v;
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') { v.type = Value::STR; v.str = parse_string(); return v; }
        if (c == 't') { expect_word("true"); v.type = Value::BOOL; v.b = true; return v; }
        if (c == 'f') { expect_word("false"); v.type = Value::BOOL; v.b = false; return v; }
        if (c == 'n') { expect_word("null"); return v; }
        // 数字必须以负号或数字开头，否则是非法 JSON（比如 {"a": }）
        if (!(c == '-' || (c >= '0' && c <= '9')))
            throw std::runtime_error("JSON 非法字符");
        v.type = Value::NUM;
        v.num = parse_number();
        return v;
    }

    void expect_word(const char* w) {
        for (const char* p = w; *p; ++p) {
            if (i >= s.size() || s[i] != *p)
                throw std::runtime_error("JSON 字面量不匹配");
            ++i;
        }
    }

    double parse_number() {
        size_t start = i;
        while (i < s.size() &&
               (isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' ||
                s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
            ++i;
        return std::strtod(s.substr(start, i - start).c_str(), nullptr);
    }

    Value parse_object() {
        Value v;
        v.type = Value::OBJ;
        expect('{');
        if (peek() == '}') { ++i; return v; }
        while (true) {
            std::string key = parse_string();
            expect(':');
            v.obj[key] = parse_value();
            char c = peek();
            if (c == ',') { ++i; continue; }
            expect('}');
            break;
        }
        return v;
    }

    Value parse_array() {
        Value v;
        v.type = Value::ARR;
        expect('[');
        if (peek() == ']') { ++i; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            char c = peek();
            if (c == ',') { ++i; continue; }
            expect(']');
            break;
        }
        return v;
    }
};

}  // namespace

Value parse(const std::string& text) {
    Parser p(text);
    return p.parse_value();
}

}  // namespace json