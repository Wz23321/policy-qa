@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动政策问答服务...
echo 浏览器打开 http://127.0.0.1:8100
python -m uvicorn server.app:app --host 127.0.0.1 --port 8100