#!/bin/bash
# ORCA 配置测试脚本

echo "=================================="
echo "ORCA 配置测试"
echo "=================================="
echo ""

# 测试1: 检查 orca 是否在 PATH 中
echo "1. 检查 PATH 中的 orca..."
if command -v orca &> /dev/null; then
    echo "   ✅ 找到 orca: $(which orca)"
    echo "   版本信息:"
    orca 2>&1 | head -n 5 || echo "   无法获取版本信息"
else
    echo "   ❌ orca 不在 PATH 中"
fi
echo ""

# 测试2: 检查指定路径
ORCA_PATH="/home/zyb/apps/orca-6.0.1/orca"
echo "2. 检查指定路径: $ORCA_PATH"
if [ -f "$ORCA_PATH" ]; then
    echo "   ✅ 文件存在"
    if [ -x "$ORCA_PATH" ]; then
        echo "   ✅ 可执行"
        echo "   版本信息:"
        "$ORCA_PATH" 2>&1 | head -n 5 || echo "   无法获取版本信息"
    else
        echo "   ❌ 文件不可执行，尝试添加执行权限:"
        echo "      chmod +x $ORCA_PATH"
    fi
else
    echo "   ❌ 文件不存在"
    echo "   请检查 ORCA 安装目录"
fi
echo ""

# 测试3: 检查 orca_run 目录
echo "3. 检查运行目录..."
if [ -d "orca_run" ]; then
    echo "   ✅ orca_run 目录存在"
    echo "   目录内容:"
    ls -lh orca_run/ 2>/dev/null || echo "   目录为空"
else
    echo "   ℹ️  orca_run 目录不存在（首次运行时会自动创建）"
fi
echo ""

echo "=================================="
echo "测试完成"
echo "=================================="

