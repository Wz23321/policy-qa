// main.cpp —— 政策问答服务的 C++ 客户端
//
// 用法:
//   ./policy-cli "创业担保贷款最多能贷多少？"
//   ./policy-cli -a "想跟同学合伙开公司，启动资金不够怎么办？"   # Agent 模式
//   ./policy-cli -h 127.0.0.1 -p 8100 "问题"
//   ./policy-cli                                               # 交互模式

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "http.h"
#include "json.h"

constexpr const char* kDefaultHost = "127.0.0.1";
constexpr int kDefaultPort = 8100;

void usage() {
    std::printf(
        "用法:\n"
        "  policy-cli [选项] <问题>\n"
        "  policy-cli                       进入交互模式\n\n"
        "选项:\n"
        "  -a, --agent      Agent 模式（模型自己调工具，慢一些）\n"
        "  -h, --host IP    服务地址，默认 %s\n"
        "  -p, --port N     服务端口，默认 %d\n",
        kDefaultHost, kDefaultPort);
}

// 发一个问题，打印结果
bool ask_once(const std::string& host, int port, const std::string& question,
              bool use_agent) {
    std::string body = "{\"question\": " + json_escape(question) + "}";
    const char* path = use_agent ? "/agent" : "/ask";

    HttpResponse res = http_post(host, port, path, body);
    if (!res.ok()) {
        if (!res.error.empty())
            std::fprintf(stderr, "请求失败: %s\n", res.error.c_str());
        else
            std::fprintf(stderr, "服务返回 HTTP %d\n%s\n", res.status,
                         res.body.substr(0, 300).c_str());
        return false;
    }

    json::Value root;
    try {
        root = json::parse(res.body);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "解析响应失败: %s\n", e.what());
        return false;
    }

    const json::Value& answer = root["answer"];
    if (answer.type != json::Value::STR) {
        std::fprintf(stderr, "响应里没有 answer 字段\n");
        return false;
    }

    std::printf("\n%s\n\n", answer.str.c_str());

    // Agent 模式：打印调用链
    const json::Value& steps = root["steps"];
    if (steps.type == json::Value::ARR && steps.size() > 0) {
        std::printf("调用记录:\n");
        for (size_t i = 0; i < steps.size(); ++i) {
            const json::Value& s = steps.at(i);
            std::printf("  %zu. %s\n", i + 1, s["tool"].str.c_str());
        }
        std::printf("\n");
    }

    // 引用来源
    const json::Value& sources = root["sources"];
    if (sources.type == json::Value::ARR && sources.size() > 0) {
        std::printf("来源:\n");
        for (size_t i = 0; i < sources.size(); ++i) {
            const json::Value& s = sources.at(i);
            std::printf("  [%zu] %s  %s\n", i + 1,
                        s["doc_no"].str.c_str(), s["doc"].str.c_str());
        }
        std::printf("\n");
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::string host = kDefaultHost;
    int port = kDefaultPort;
    bool use_agent = false;
    std::vector<std::string> words;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-a" || a == "--agent") {
            use_agent = true;
        } else if ((a == "-h" || a == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((a == "-p" || a == "--port") && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (a == "--help" || a == "-?") {
            usage();
            return 0;
        } else {
            words.push_back(a);
        }
    }

    // 有参数：问一次就退出
    if (!words.empty()) {
        std::string question;
        for (size_t i = 0; i < words.size(); ++i) {
            if (i) question += " ";
            question += words[i];
        }
        return ask_once(host, port, question, use_agent) ? 0 : 1;
    }

    // 没参数：交互模式
    std::printf("政策问答（输入 q 退出）%s\n", use_agent ? " [Agent 模式]" : "");
    std::string line;
    while (true) {
        std::printf("问: ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "q" || line == "Q" || line.empty()) break;
        ask_once(host, port, line, use_agent);
    }
    return 0;
}