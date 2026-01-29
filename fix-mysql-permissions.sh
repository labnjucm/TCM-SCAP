#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║        🔧 MySQL 权限自动修复脚本                            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# MySQL 配置
MYSQL_USER="root"
MYSQL_PASS="pass"
DB_NAME="chemhub"

# 获取当前机器的 IP 地址
echo "🔍 检测当前服务器 IP 地址..."
SERVER_IP=$(hostname -I | awk '{print $1}')
echo -e "${BLUE}当前服务器 IP: $SERVER_IP${NC}"
echo ""

# 检查 MySQL 是否运行
echo "🔍 检查 MySQL 服务状态..."
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
echo ""

# 显示当前用户权限
echo "📋 当前 root 用户的访问权限："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mysql -u $MYSQL_USER -p$MYSQL_PASS -e "SELECT host, user FROM mysql.user WHERE user='root';" 2>/dev/null
echo ""

# 询问用户选择修复方案
echo "请选择修复方案："
echo ""
echo "1️⃣  允许从特定 IP 连接（推荐用于生产环境）"
echo "   - 将授权 root@$SERVER_IP 访问"
echo ""
echo "2️⃣  允许从任何 IP 连接（开发环境）"
echo "   - 将授权 root@% 访问（任意主机）"
echo "   - ⚠️  降低安全性，仅用于开发/测试"
echo ""
echo "3️⃣  修改为使用 localhost 连接"
echo "   - 修改 .env.local 使用 localhost"
echo "   - 适用于应用和数据库在同一服务器"
echo ""
echo "4️⃣  创建专用数据库用户（最安全）"
echo "   - 创建 chemhub 用户，仅授权 chemhub 数据库"
echo ""
echo "5️⃣  退出"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔧 方案 1: 授权特定 IP ($SERVER_IP)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # 可以手动输入 IP
        read -p "使用检测到的 IP ($SERVER_IP)? [Y/n]: " use_detected_ip
        if [[ $use_detected_ip == "n" || $use_detected_ip == "N" ]]; then
            read -p "请输入要授权的 IP 地址: " CUSTOM_IP
            SERVER_IP=$CUSTOM_IP
        fi
        
        echo "正在授权 root@$SERVER_IP..."
        mysql -u $MYSQL_USER -p$MYSQL_PASS << EOF
CREATE USER IF NOT EXISTS 'root'@'$SERVER_IP' IDENTIFIED BY '$MYSQL_PASS';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO 'root'@'$SERVER_IP';
FLUSH PRIVILEGES;
SELECT host, user FROM mysql.user WHERE user='root';
EOF
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ 权限授予成功！${NC}"
            echo ""
            echo "已授权："
            echo "  • 用户: root@$SERVER_IP"
            echo "  • 数据库: $DB_NAME"
            echo ""
        else
            echo -e "${RED}❌ 权限授予失败${NC}"
            exit 1
        fi
        ;;
        
    2)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔧 方案 2: 允许从任何 IP 连接"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo -e "${YELLOW}⚠️  警告：此方法会降低安全性${NC}"
        read -p "确认继续? [y/N]: " confirm
        
        if [[ $confirm != "y" && $confirm != "Y" ]]; then
            echo "已取消"
            exit 0
        fi
        
        echo "正在授权 root@% ..."
        mysql -u $MYSQL_USER -p$MYSQL_PASS << EOF
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY '$MYSQL_PASS';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO 'root'@'%';
FLUSH PRIVILEGES;
SELECT host, user FROM mysql.user WHERE user='root';
EOF
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ 权限授予成功！${NC}"
            echo ""
            echo "已授权："
            echo "  • 用户: root@% (任意主机)"
            echo "  • 数据库: $DB_NAME"
            echo ""
        else
            echo -e "${RED}❌ 权限授予失败${NC}"
            exit 1
        fi
        ;;
        
    3)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔧 方案 3: 修改为 localhost 连接"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        ENV_FILE="frontend/.env.local"
        
        if [ ! -f "$ENV_FILE" ]; then
            echo -e "${RED}❌ 文件 $ENV_FILE 不存在${NC}"
            exit 1
        fi
        
        # 备份配置文件
        cp "$ENV_FILE" "$ENV_FILE.backup"
        echo -e "${GREEN}✅ 已备份到 $ENV_FILE.backup${NC}"
        
        # 修改 DATABASE_URL
        sed -i 's|DATABASE_URL=.*|DATABASE_URL="mysql://root:pass@localhost:3306/chemhub"|' "$ENV_FILE"
        
        echo ""
        echo -e "${GREEN}✅ 配置文件已更新${NC}"
        echo ""
        echo "新的数据库连接："
        echo "  DATABASE_URL=\"mysql://root:pass@localhost:3306/chemhub\""
        echo ""
        ;;
        
    4)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔧 方案 4: 创建专用数据库用户"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        CHEMHUB_USER="chemhub"
        read -p "为新用户设置密码（留空使用 'chemhub_pass'）: " NEW_PASS
        if [ -z "$NEW_PASS" ]; then
            NEW_PASS="chemhub_pass"
        fi
        
        echo "正在创建用户 $CHEMHUB_USER..."
        mysql -u $MYSQL_USER -p$MYSQL_PASS << EOF
CREATE USER IF NOT EXISTS '$CHEMHUB_USER'@'localhost' IDENTIFIED BY '$NEW_PASS';
CREATE USER IF NOT EXISTS '$CHEMHUB_USER'@'$SERVER_IP' IDENTIFIED BY '$NEW_PASS';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$CHEMHUB_USER'@'localhost';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$CHEMHUB_USER'@'$SERVER_IP';
FLUSH PRIVILEGES;
SELECT host, user FROM mysql.user WHERE user='$CHEMHUB_USER';
EOF
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ 用户创建成功！${NC}"
            echo ""
            echo "新用户信息："
            echo "  • 用户名: $CHEMHUB_USER"
            echo "  • 密码: $NEW_PASS"
            echo "  • 授权主机: localhost, $SERVER_IP"
            echo "  • 数据库: $DB_NAME"
            echo ""
            
            # 更新 .env.local
            ENV_FILE="frontend/.env.local"
            if [ -f "$ENV_FILE" ]; then
                cp "$ENV_FILE" "$ENV_FILE.backup"
                echo -e "${GREEN}✅ 已备份到 $ENV_FILE.backup${NC}"
                
                sed -i "s|DATABASE_URL=.*|DATABASE_URL=\"mysql://$CHEMHUB_USER:$NEW_PASS@localhost:3306/$DB_NAME\"|" "$ENV_FILE"
                echo -e "${GREEN}✅ 已更新 $ENV_FILE${NC}"
                echo ""
            fi
        else
            echo -e "${RED}❌ 用户创建失败${NC}"
            exit 1
        fi
        ;;
        
    5)
        echo "已退出"
        exit 0
        ;;
        
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 下一步操作："
echo ""
echo "1️⃣  检查 MySQL 绑定地址（如需远程访问）："
echo "   sudo grep bind-address /etc/mysql/mysql.conf.d/mysqld.cnf"
echo "   # 如果是 127.0.0.1，改为 0.0.0.0，然后重启："
echo "   # sudo systemctl restart mysql"
echo ""
echo "2️⃣  重新生成 Prisma Client："
echo "   cd frontend"
echo "   npx prisma generate"
echo ""
echo "3️⃣  运行数据库迁移（如果需要）："
echo "   cd frontend"
echo "   npx prisma migrate deploy"
echo ""
echo "4️⃣  重启应用："
echo "   ./stop-all.sh"
echo "   ./start-with-mysql.sh"
echo ""
echo "5️⃣  测试登录/注册功能："
echo "   打开 http://localhost:5173"
echo "   点击右下角齿轮按钮测试"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ 修复完成！${NC}"
echo ""
echo "详细文档请查看: MYSQL_CONNECTION_FIX.md"
echo ""

