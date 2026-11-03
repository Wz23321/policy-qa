// http.h —— 极简 HTTP/1.1 客户端（只支持 POST + JSON）
#pragma once

#include <string>

struct HttpResponse {
    int status = 0;
    std::string body;
    std::string error;   // 非空表示失败

    bool ok() const { return error.empty() && status >= 200 && status < 300; }
};

// 向 host:port 发一个 POST，path 如 "/ask"，body 是 JSON 字符串
HttpResponse http_post(const std::string& host, int port,
                       const std::string& path, const std::string& body,
                       int timeout_sec = 120);

// 把一段文本转成 JSON 字符串字面量（含首尾引号，内部做转义）
std::string json_escape(const std::string& s);