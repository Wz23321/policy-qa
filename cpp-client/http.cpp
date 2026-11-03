// http.cpp
#include "http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// 从 "HTTP/1.1 200 OK" 里取 200
int parse_status(const std::string& line) {
    size_t sp = line.find(' ');
    if (sp == std::string::npos) return 0;
    return std::atoi(line.c_str() + sp + 1);
}

// 在头部里找 Content-Length，找不到返回 -1
long parse_content_length(const std::string& headers) {
    const std::string key = "content-length:";
    std::string lower;
    lower.reserve(headers.size());
    for (char c : headers) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    size_t pos = lower.find(key);
    if (pos == std::string::npos) return -1;
    pos += key.size();
    while (pos < headers.size() && (headers[pos] == ' ' || headers[pos] == '\t')) ++pos;

    long v = 0;
    bool any = false;
    while (pos < headers.size() && isdigit(static_cast<unsigned char>(headers[pos]))) {
        v = v * 10 + (headers[pos] - '0');
        ++pos;
        any = true;
    }
    return any ? v : -1;
}

bool is_chunked(const std::string& headers) {
    std::string lower;
    for (char c : headers) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return lower.find("transfer-encoding: chunked") != std::string::npos;
}

// 解 chunked 编码：每块前面是十六进制的长度
std::string decode_chunked(const std::string& in) {
    std::string out;
    size_t i = 0;
    while (i < in.size()) {
        size_t line_end = in.find("\r\n", i);
        if (line_end == std::string::npos) break;

        std::string hex = in.substr(i, line_end - i);
        long len = std::strtol(hex.c_str(), nullptr, 16);
        if (len <= 0) break;                       // 0 表示结束

        size_t data_start = line_end + 2;
        if (data_start + len > in.size()) break;
        out.append(in, data_start, len);
        i = data_start + len + 2;                  // 跳过结尾的 \r\n
    }
    return out;
}

}  // namespace

std::string json_escape(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out += "\"";
    return out;
}

HttpResponse http_post(const std::string& host, int port,
                       const std::string& path, const std::string& body,
                       int timeout_sec) {
    HttpResponse res;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { res.error = "socket 创建失败"; return res; }

    // 设读写超时，避免服务没响应时一直卡住
    struct timeval tv{timeout_sec, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        res.error = "地址非法: " + host;
        close(fd);
        return res;
    }

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        res.error = "连接失败: " + host + ":" + std::to_string(port) +
                    "（服务起了吗？）";
        close(fd);
        return res;
    }

    std::string req =
        "POST " + path + " HTTP/1.1\r\n"
        "Host: " + host + ":" + std::to_string(port) + "\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "\r\n" + body;

    // 一次 send 可能发不全，循环发完
    size_t sent = 0;
    while (sent < req.size()) {
        ssize_t n = send(fd, req.data() + sent, req.size() - sent, 0);
        if (n <= 0) { res.error = "发送请求失败"; close(fd); return res; }
        sent += static_cast<size_t>(n);
    }

    std::string raw;
    char buf[8192];
    while (true) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, static_cast<size_t>(n));
    }
    close(fd);

    size_t head_end = raw.find("\r\n\r\n");
    if (head_end == std::string::npos) {
        res.error = "响应格式异常";
        return res;
    }

    std::string head = raw.substr(0, head_end);
    std::string body_raw = raw.substr(head_end + 4);

    size_t line_end = head.find("\r\n");
    res.status = parse_status(head.substr(0, line_end));

    std::string headers = line_end == std::string::npos ? "" : head.substr(line_end);
    if (is_chunked(headers)) {
        res.body = decode_chunked(body_raw);
    } else {
        long cl = parse_content_length(headers);
        if (cl >= 0 && static_cast<size_t>(cl) <= body_raw.size())
            res.body = body_raw.substr(0, static_cast<size_t>(cl));
        else
            res.body = body_raw;
    }

    return res;
}