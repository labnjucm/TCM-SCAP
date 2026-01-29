#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║        🚀 ChemHub 启动脚本（系统 MySQL）                    ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""



# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# MySQL 配置
MYSQL_USER="root"
MYSQL_PASS="pass"
DB_NAME="chemhub"

# 检查 MySQL 是否运行
echo "🔍 检查 MySQL 服务..."
if ! mysqladmin ping -h localhost -u $MYSQL_USER -p$MYSQL_PASS &> /dev/null; then
    echo -e "${RED}❌ MySQL 未运行或密码错误${NC}"
    echo ""
    echo "请执行以下操作之一："
    echo "1. 启动 MySQL：sudo systemctl start mysql"
    echo "2. 检查密码是否为 'pass'"
    echo "3. 如果密码不同，编辑本脚本第17行修改密码"
    exit 1
fi

echo -e "${GREEN}✅ MySQL 正在运行${NC}"

# 检查数据库是否存在
echo "🗄️  检查数据库..."
if ! mysql -u $MYSQL_USER -p$MYSQL_PASS -e "USE $DB_NAME" &> /dev/null; then
    echo -e "${YELLOW}⚠️  数据库 $DB_NAME 不存在${NC}"
    echo "正在创建数据库..."
    
    if mysql -u $MYSQL_USER -p$MYSQL_PASS -e "CREATE DATABASE $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"; then
        echo -e "${GREEN}✅ 数据库已创建${NC}"
    else
        echo -e "${RED}❌ 数据库创建失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ 数据库已存在${NC}"
fi

# cd components
# cd Comchemistry
# conda run -n kaifa python orca_gradio_app.py
# cd ..
# cd Modynamics
# conda run -n kaifa python gmx_gui_runner.py
# cd ..
# cd docking
# conda run -n test1_bat1 python app/main.py
# cd ..
# cd ..

# 进入前端目录
cd frontend || exit 1

# 检查 .env.local 是否存在
if [ ! -f ".env.local" ]; then
    echo "📝 创建配置文件..."
    cat > .env.local << 'EOF'
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)
NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:7861
NEXT_PUBLIC_MD_PATH=http://127.0.0.1:7862
NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:7863
DATABASE_URL="mysql://root:pass@localhost:3306/chemhub"
JWT_SECRET="chemhub_jwt_secret_change_in_production_2024"
EOF
    echo -e "${GREEN}✅ 配置文件已创建${NC}"
fi

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖（首次运行需要 1-3 分钟）..."
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 依赖安装失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 依赖安装完成${NC}"
else
    echo -e "${GREEN}✅ 依赖已安装${NC}"
fi

# 初始化 Prisma
if [ ! -d "node_modules/.prisma" ]; then
    echo "🔧 生成 Prisma Client..."
    npx prisma generate
    echo -e "${GREEN}✅ Prisma Client 已生成${NC}"
fi

# 检查是否需要迁移
echo "🔄 检查数据库迁移..."
TABLE_COUNT=$(mysql -u $MYSQL_USER -p$MYSQL_PASS $DB_NAME -se "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '$DB_NAME';")

if [ "$TABLE_COUNT" -eq "0" ]; then
    echo "🗄️  运行数据库迁移（创建 users 表）..."
    npx prisma migrate dev --name init
    echo -e "${GREEN}✅ 数据库表已创建${NC}"
else
    echo -e "${GREEN}✅ 数据库表已存在${NC}"
    # 确保最新
    npx prisma migrate deploy &> /dev/null || true
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ 所有准备工作完成！${NC}"
echo ""
echo "▶️  启动前端服务..."
echo ""
echo "访问地址："
echo "  • 主界面：http://localhost:5173"
echo "  • 数据库管理：npx prisma studio"
echo ""
echo "测试新功能："
echo "  1. 点击右下角齿轮按钮 ⚙️  注册/登录"
echo "  2. 点击左侧资源 → 查看详细说明"
echo ""
echo "停止服务：按 Ctrl+C"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# 启动前端
npm run dev

