#!/bin/bash
#
# ChemHub 一键启动脚本
# 启动前端和三个 Gradio 示例应用
#

set -e

echo "🚀 ChemHub 启动脚本"
echo "===================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查依赖
check_dependencies() {
    echo "📦 检查依赖..."
    
    # 检查 Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ 未找到 Node.js，请先安装 Node.js${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Node.js $(node -v)${NC}"
    
    # 检查 npm
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}❌ 未找到 npm${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ npm $(npm -v)${NC}"
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ 未找到 Python 3${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python $(python3 --version)${NC}"
    
    # 检查 Gradio
    if ! python3 -c "import gradio" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  未找到 Gradio，尝试安装...${NC}"
        pip3 install gradio
    else
        echo -e "${GREEN}✓ Gradio$(NC}"
    fi
    
    echo ""
}

# 安装前端依赖
install_frontend() {
    if [ ! -d "frontend/node_modules" ]; then
        echo "📥 安装前端依赖..."
        cd frontend
        npm install
        cd ..
        echo -e "${GREEN}✓ 前端依赖安装完成${NC}"
        echo ""
    else
        echo -e "${GREEN}✓ 前端依赖已安装${NC}"
        echo ""
    fi
}

# 创建环境变量文件
create_env() {
    if [ ! -f "frontend/.env.local" ]; then
        echo "📝 创建环境变量文件..."
        cat > frontend/.env.local << 'EOF'
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)
NEXT_PUBLIC_DOCKING_PATH=/apps/docking/
NEXT_PUBLIC_MD_PATH=/apps/md/
NEXT_PUBLIC_ORCA_PATH=/apps/orca/
NEXT_PUBLIC_SWISSADME_PATH=/embed/swissadme/
NEXT_PUBLIC_PREADMET_PATH=/embed/preadmet/
EOF
        echo -e "${GREEN}✓ 已创建 frontend/.env.local${NC}"
        echo ""
    fi
}

# 启动前端
start_frontend() {
    echo "🌐 启动前端 (端口 5173)..."
    cd frontend
    
    # 构建前端（如果需要）
    if [ ! -d ".next" ]; then
        echo "🔨 首次运行，构建前端..."
        npm run build
    fi
    
    # 启动前端
    nohup npm run start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/frontend.pid
    cd ..
    echo -e "${GREEN}✓ 前端已启动 (PID: $FRONTEND_PID)${NC}"
    echo ""
}

# 启动 Gradio 应用
start_gradio() {
    echo "🧬 启动 Gradio 应用..."
    
    # 分子对接应用
    nohup python3 examples/docking_app.py > logs/docking.log 2>&1 &
    DOCKING_PID=$!
    echo $DOCKING_PID > logs/docking.pid
    echo -e "${GREEN}✓ 分子对接应用已启动 (端口 7861, PID: $DOCKING_PID)${NC}"
    
    # 分子动力学应用
    nohup python3 examples/md_app.py > logs/md.log 2>&1 &
    MD_PID=$!
    echo $MD_PID > logs/md.pid
    echo -e "${GREEN}✓ 分子动力学应用已启动 (端口 7862, PID: $MD_PID)${NC}"
    
    # ORCA 应用
    nohup python3 examples/orca_app.py > logs/orca.log 2>&1 &
    ORCA_PID=$!
    echo $ORCA_PID > logs/orca.pid
    echo -e "${GREEN}✓ ORCA 应用已启动 (端口 7863, PID: $ORCA_PID)${NC}"
    
    echo ""
}

# 等待服务启动
wait_for_services() {
    echo "⏳ 等待服务启动..."
    sleep 5
    echo ""
}

# 检查服务状态
check_services() {
    echo "🔍 检查服务状态..."
    
    # 检查前端
    if curl -s http://localhost:5173 > /dev/null; then
        echo -e "${GREEN}✓ 前端运行正常 (http://localhost:5173)${NC}"
    else
        echo -e "${RED}✗ 前端未响应${NC}"
    fi
    
    # 检查 Gradio 应用
    if curl -s http://localhost:7861 > /dev/null; then
        echo -e "${GREEN}✓ 分子对接应用运行正常 (http://localhost:7861)${NC}"
    else
        echo -e "${RED}✗ 分子对接应用未响应${NC}"
    fi
    
    if curl -s http://localhost:7862 > /dev/null; then
        echo -e "${GREEN}✓ 分子动力学应用运行正常 (http://localhost:7862)${NC}"
    else
        echo -e "${RED}✗ 分子动力学应用未响应${NC}"
    fi
    
    if curl -s http://localhost:7863 > /dev/null; then
        echo -e "${GREEN}✓ ORCA 应用运行正常 (http://localhost:7863)${NC}"
    else
        echo -e "${RED}✗ ORCA 应用未响应${NC}"
    fi
    
    echo ""
}

# 主函数
main() {
    # 创建日志目录
    mkdir -p logs
    
    check_dependencies
    install_frontend
    create_env
    start_frontend
    start_gradio
    wait_for_services
    check_services
    
    echo "===================="
    echo -e "${GREEN}✅ ChemHub 启动完成！${NC}"
    echo ""
    echo "📱 访问地址："
    echo "   主界面: http://localhost:5173"
    echo "   分子对接: http://localhost:7861"
    echo "   分子动力学: http://localhost:7862"
    echo "   ORCA 计算: http://localhost:7863"
    echo ""
    echo "📝 日志文件："
    echo "   前端: logs/frontend.log"
    echo "   分子对接: logs/docking.log"
    echo "   分子动力学: logs/md.log"
    echo "   ORCA: logs/orca.log"
    echo ""
    echo "🛑 停止服务："
    echo "   运行: ./stop-all.sh"
    echo ""
    echo "💡 提示："
    echo "   - 如需使用 Nginx 统一入口，请参考 reverse-proxy/README.md"
    echo "   - 如需自定义配置，请编辑 frontend/app/config/catalog.ts"
    echo ""
}

# 运行主函数
main

