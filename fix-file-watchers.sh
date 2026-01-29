#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "  修复文件监视器限制问题"
echo "════════════════════════════════════════════════════════════"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查当前限制
CURRENT=$(cat /proc/sys/fs/inotify/max_user_watches)
echo "当前文件监视器限制: $CURRENT"
echo ""

# 新的限制值（推荐值）
NEW_LIMIT=524288

echo "正在增加文件监视器限制到: $NEW_LIMIT"
echo ""

# 临时增加限制（立即生效，重启后失效）
echo "1. 应用临时修复（立即生效）..."
sudo sysctl -w fs.inotify.max_user_watches=$NEW_LIMIT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 临时修复已应用${NC}"
else
    echo -e "${RED}❌ 临时修复失败${NC}"
    exit 1
fi

echo ""

# 永久增加限制（重启后仍然生效）
echo "2. 应用永久修复（重启后仍有效）..."

# 检查配置文件是否存在
if grep -q "fs.inotify.max_user_watches" /etc/sysctl.conf 2>/dev/null; then
    echo "   配置已存在，正在更新..."
    sudo sed -i "s/fs.inotify.max_user_watches=.*/fs.inotify.max_user_watches=$NEW_LIMIT/" /etc/sysctl.conf
else
    echo "   正在添加配置..."
    echo "fs.inotify.max_user_watches=$NEW_LIMIT" | sudo tee -a /etc/sysctl.conf > /dev/null
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 永久修复已应用${NC}"
else
    echo -e "${RED}❌ 永久修复失败${NC}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ 所有修复完成！${NC}"
echo ""
echo "新的文件监视器限制: $(cat /proc/sys/fs/inotify/max_user_watches)"
echo ""
echo "现在可以运行: ./start-with-mysql.sh"
echo ""

