#!/bin/bash

echo "🗄️  ChemHub 数据库初始化脚本"
echo "================================"
echo ""

MYSQL_USER="root"
MYSQL_PASS="pass"
DB_NAME="chemhub"

# 检查 MySQL
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL 未安装"
    echo "请先安装 MySQL：sudo apt install mysql-server"
    exit 1
fi

echo "✅ MySQL 已安装"

# 测试连接
echo "🔐 测试 MySQL 连接..."
if ! mysql -u $MYSQL_USER -p$MYSQL_PASS -e "SELECT 1;" &> /dev/null; then
    echo "❌ MySQL 连接失败"
    echo ""
    echo "请检查："
    echo "  1. MySQL 是否运行：sudo systemctl status mysql"
    echo "  2. 密码是否正确（当前设置：$MYSQL_PASS）"
    echo ""
    echo "如果密码不是 'pass'，请："
    echo "  1. 编辑 start-with-mysql.sh 修改第17行"
    echo "  2. 编辑 frontend/.env.local 修改 DATABASE_URL"
    exit 1
fi

echo "✅ MySQL 连接成功"

# 创建数据库
echo "🗄️  创建数据库 $DB_NAME..."
mysql -u $MYSQL_USER -p$MYSQL_PASS << EOF
CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
SHOW DATABASES LIKE '$DB_NAME';
EOF

echo ""
echo "✅ 数据库初始化完成！"
echo ""
echo "数据库信息："
echo "  • 数据库名：$DB_NAME"
echo "  • 用户：$MYSQL_USER"
echo "  • 密码：$MYSQL_PASS"
echo "  • 地址：localhost:3306"
echo ""
echo "下一步："
echo "  ./start-with-mysql.sh"
echo ""
