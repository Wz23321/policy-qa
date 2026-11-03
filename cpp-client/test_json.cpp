// test_json.cpp —— JSON 解析器的单元测试
// 编译: g++ -std=c++17 -o test_json test_json.cpp json.cpp
// 运行: ./test_json

#include <cstdio>
#include <string>

#include "json.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (cond) {                                                     \
            ++g_pass;                                                   \
        } else {                                                        \
            ++g_fail;                                                   \
            std::printf("  FAIL  第 %d 行: %s\n", __LINE__, #cond);      \
        }                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                  \
    do {                                                                \
        auto _a = (a);                                                  \
        auto _b = (b);                                                  \
        if (_a == _b) {                                                 \
            ++g_pass;                                                   \
        } else {                                                        \
            ++g_fail;                                                   \
            std::printf("  FAIL  第 %d 行: %s != %s\n", __LINE__, #a, #b); \
        }                                                               \
    } while (0)

void test_object() {
    auto v = json::parse(R"({"answer": "你好", "used_tokens": 812})");
    CHECK_EQ(v["answer"].str, std::string("你好"));
    CHECK(v["used_tokens"].type == json::Value::NUM);
    CHECK_EQ(static_cast<int>(v["used_tokens"].num), 812);
}

void test_array() {
    auto v = json::parse(R"({"sources": [{"doc_no": "A1"}, {"doc_no": "B2"}]})");
    CHECK(v["sources"].type == json::Value::ARR);
    CHECK_EQ(v["sources"].size(), size_t(2));
    CHECK_EQ(v["sources"].at(0)["doc_no"].str, std::string("A1"));
    CHECK_EQ(v["sources"].at(1)["doc_no"].str, std::string("B2"));
}

void test_escapes() {
    auto v = json::parse(R"({"s": "换行\n引号\"反斜杠\\制表\t"})");
    CHECK_EQ(v["s"].str, std::string("换行\n引号\"反斜杠\\制表\t"));
}

void test_unicode_escape() {
    // 你好 = 你好
    auto v = json::parse(R"({"s": "你好"})");
    CHECK_EQ(v["s"].str, std::string("你好"));
}

void test_unicode_surrogate() {
    // 😀 = 😀 (U+1F600)
    auto v = json::parse(R"({"s": "😀"})");
    CHECK_EQ(v["s"].str, std::string("\xF0\x9F\x98\x80"));
}

void test_literals() {
    auto v = json::parse(R"({"t": true, "f": false, "n": null})");
    CHECK(v["t"].type == json::Value::BOOL && v["t"].b);
    CHECK(v["f"].type == json::Value::BOOL && !v["f"].b);
    CHECK(v["n"].is_null());
}

void test_missing_key() {
    auto v = json::parse(R"({"a": 1})");
    CHECK(v["not_exist"].is_null());      // 取不存在的键返回空值，不崩溃
    CHECK(v["a"]["deep"].is_null());      // 对非对象取键也不崩溃
}

void test_empty_containers() {
    auto v = json::parse(R"({"o": {}, "a": []})");
    CHECK(v["o"].type == json::Value::OBJ);
    CHECK_EQ(v["o"].size(), size_t(0));
    CHECK(v["a"].type == json::Value::ARR);
    CHECK_EQ(v["a"].size(), size_t(0));
}

void test_nested() {
    auto v = json::parse(R"({"a": {"b": {"c": [1, 2, {"d": "深层"}]}}})");
    CHECK_EQ(v["a"]["b"]["c"].at(2)["d"].str, std::string("深层"));
}

void test_real_response() {
    // 真实 /agent 响应的结构
    const char* body = R"({
      "answer": "可以申请 30 万元",
      "steps": [{"tool": "search_policy", "args": {"query": "创业贷款"}}],
      "used_tokens": 2400,
      "session_id": "abc123"
    })";
    auto v = json::parse(body);
    CHECK_EQ(v["answer"].str, std::string("可以申请 30 万元"));
    CHECK_EQ(v["steps"].at(0)["tool"].str, std::string("search_policy"));
    CHECK_EQ(v["steps"].at(0)["args"]["query"].str, std::string("创业贷款"));
    CHECK_EQ(v["session_id"].str, std::string("abc123"));
}

void test_bad_json_throws() {
    bool threw = false;
    try {
        json::parse(R"({"a": })");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    std::printf("JSON 解析器测试\n");
    test_object();
    test_array();
    test_escapes();
    test_unicode_escape();
    test_unicode_surrogate();
    test_literals();
    test_missing_key();
    test_empty_containers();
    test_nested();
    test_real_response();
    test_bad_json_throws();

    std::printf("\n%d 通过, %d 失败\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}