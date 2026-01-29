#!/bin/bash

# 创建所有剩余的MD文件

for file in autodock-vina gnina gromacs openmm namd swissadme preadmet orca gaussian q-chem; do
  if [ ! -f "${file}.md" ]; then
    cat > "${file}.md" << EOF
# ${file^} - 工具说明

## 简介
**${file^}** 是重要的计算化学工具。[占位内容 - 详细文档待补充]

## 典型用例
1. 基础计算任务
2. 高级分析
3. 结果可视化

## 输入/输出
**输入**：配置文件、结构文件  
**输出**：计算结果、日志文件

## 快速上手
基本使用流程：
1. 准备输入文件
2. 运行计算
3. 分析结果

### 示例代码
\`\`\`bash
# 基本命令示例
echo "示例命令"
\`\`\`

## 注意事项与常见坑
1. 注意输入文件格式
2. 检查计算资源
3. 验证结果合理性

## 官方链接与引用
- 官方网站
- 文档链接
- 引用信息
EOF
    echo "创建: ${file}.md"
  fi
done

echo "所有MD文件创建完成"
