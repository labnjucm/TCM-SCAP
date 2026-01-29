#!/bin/bash
#
# ChemHub 停止脚本
# 停止所有服务
#

set -e

echo "🛑 ChemHub 停止脚本"
echo "===================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 停止服务函数
stop_service() {
    local name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo "🛑 停止 $name (PID: $PID)..."
            kill $PID
            
            # 等待进程结束
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ $name 已停止${NC}"
                    rm -f "$pid_file"
                    return 0
                fi
                sleep 0.5
            done
            
            # 如果还没停止，强制终止
            echo -e "${YELLOW}⚠️  强制终止 $name${NC}"
            kill -9 $PID 2>/dev/null || true
            rm -f "$pid_file"
        else
            echo -e "${YELLOW}⚠️  $name 进程不存在 (PID: $PID)${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${YELLOW}⚠️  未找到 $name 的 PID 文件${NC}"
    fi
}

# 按端口停止进程
stop_by_port() {
    local name=$1
    local port=$2
    
    PID=$(lsof -ti:$port 2>/dev/null || echo "")
    if [ ! -z "$PID" ]; then
        echo "🛑 停止占用端口 $port 的进程 (PID: $PID)..."
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
        echo -e "${GREEN}✓ $name 已停止${NC}"
    fi
}

# 主函数
main() {
    # 通过 PID 文件停止
    if [ -d "logs" ]; then
        stop_service "前端" "logs/frontend.pid"
        stop_service "分子对接应用" "logs/docking.pid"
        stop_service "分子动力学应用" "logs/md.pid"
        stop_service "ORCA 应用" "logs/orca.pid"
    fi
    
    echo ""
    echo "🔍 检查端口占用..."
    
    # 通过端口停止（备用方案）
    if command -v lsof &> /dev/null; then
        stop_by_port "前端 (5173)" 5173
        stop_by_port "分子对接 (7861)" 7861
        stop_by_port "分子动力学 (7862)" 7862
        stop_by_port "ORCA (7863)" 7863
    else
        echo -e "${YELLOW}⚠️  未安装 lsof，无法检查端口占用${NC}"
    fi
    
    echo ""
    echo "===================="
    echo -e "${GREEN}✅ 所有服务已停止${NC}"
    echo ""
}

# 运行主函数
main

