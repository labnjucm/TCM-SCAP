#!/usr/bin/env python3
"""
Slim Guard - 防止训练依赖回退的守护脚本

此脚本用于静态扫描代码，确保 src/ 和 app/ 目录中没有引入训练相关的库。
可以作为 pre-commit hook 或 CI 检查使用。
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# 禁止的训练相关关键词
FORBIDDEN_IMPORTS = {
    'pytorch_lightning',
    'lightning',
    'pl.Trainer',
    'wandb',
    'tensorboard',
    'torch.utils.tensorboard',
    'deepspeed',
    'accelerate',
    'bitsandbytes',
    'torchmetrics',
}

# 禁止的训练相关函数/方法调用
FORBIDDEN_PATTERNS = [
    r'\.backward\(',           # 反向传播
    r'\.zero_grad\(',          # 梯度清零
    r'optimizer\.',            # 优化器
    r'\.fit\(',                # 训练fit方法
    r'Trainer\(',              # Trainer类
    r'wandb\.',                # WandB
    r'lr_scheduler',           # 学习率调度器
]

# 需要检查的目录
CHECK_DIRS = ['src/', 'app/gradio_app.py', 'app/runtime_config.yaml']

# 豁免文件（允许包含训练关键词）
EXEMPT_FILES = {
    'slim_guard.py',
    'TRAINING_COMPONENTS_INVENTORY.md',
    'README-SLIM.md',
    'SLIM_REPORT.md',
}


class SlimGuard:
    """守护检查器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: List[Tuple[str, int, str, str]] = []
    
    def check_file(self, file_path: Path) -> bool:
        """
        检查单个文件
        
        Returns:
            True if clean, False if violations found
        """
        # 跳过豁免文件
        if file_path.name in EXEMPT_FILES:
            return True
        
        # 跳过非Python文件（YAML等配置文件也检查）
        if file_path.suffix not in ['.py', '.yaml', '.yml']:
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️  无法读取文件 {file_path}: {e}")
            return True
        
        file_clean = True
        
        for line_no, line in enumerate(lines, start=1):
            # 跳过注释行
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            
            # 检查禁止的import
            for forbidden in FORBIDDEN_IMPORTS:
                if f'import {forbidden}' in line or f'from {forbidden}' in line:
                    self.violations.append((
                        str(file_path.relative_to(self.project_root)),
                        line_no,
                        f"禁止的import: {forbidden}",
                        line.strip()
                    ))
                    file_clean = False
            
            # 检查禁止的模式
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    self.violations.append((
                        str(file_path.relative_to(self.project_root)),
                        line_no,
                        f"禁止的模式: {pattern}",
                        line.strip()
                    ))
                    file_clean = False
        
        return file_clean
    
    def scan_directory(self, directory: Path) -> int:
        """
        扫描目录
        
        Returns:
            违规文件数量
        """
        violation_count = 0
        
        if directory.is_file():
            # 如果是单个文件
            if not self.check_file(directory):
                violation_count += 1
        else:
            # 扫描目录
            for py_file in directory.rglob('*.py'):
                if not self.check_file(py_file):
                    violation_count += 1
            
            # 也检查YAML配置文件
            for yaml_file in directory.rglob('*.yaml'):
                if not self.check_file(yaml_file):
                    violation_count += 1
            
            for yml_file in directory.rglob('*.yml'):
                if not self.check_file(yml_file):
                    violation_count += 1
        
        return violation_count
    
    def run(self) -> bool:
        """
        运行完整扫描
        
        Returns:
            True if all clean, False if violations found
        """
        print("=" * 70)
        print("🛡️  Slim Guard - 训练依赖守护检查")
        print("=" * 70)
        print()
        
        total_violations = 0
        
        for check_path_str in CHECK_DIRS:
            check_path = self.project_root / check_path_str
            
            if not check_path.exists():
                print(f"⚠️  路径不存在，跳过: {check_path_str}")
                continue
            
            print(f"🔍 扫描: {check_path_str}")
            violations = self.scan_directory(check_path)
            total_violations += violations
            
            if violations == 0:
                print(f"   ✅ 无违规\n")
            else:
                print(f"   ❌ 发现 {violations} 个违规文件\n")
        
        # 输出详细违规信息
        if self.violations:
            print("=" * 70)
            print("❌ 发现以下违规:")
            print("=" * 70)
            print()
            
            current_file = None
            for file_path, line_no, reason, line_content in self.violations:
                if file_path != current_file:
                    print(f"\n📄 {file_path}")
                    current_file = file_path
                
                print(f"   第 {line_no} 行: {reason}")
                print(f"   代码: {line_content}")
                print()
        
        # 总结
        print("=" * 70)
        if total_violations == 0:
            print("✅ 检查通过！没有发现训练依赖。")
            print("=" * 70)
            return True
        else:
            print(f"❌ 检查失败！发现 {len(self.violations)} 处违规。")
            print("=" * 70)
            print("\n⚠️  请移除上述训练相关的import和代码。")
            print("提示：如果这些代码是必需的，请考虑：")
            print("  1. 将其移动到 archive_training/ 目录")
            print("  2. 重构代码以避免训练依赖")
            print("  3. 如果是误报，将文件添加到 EXEMPT_FILES\n")
            return False
    
    def generate_report(self) -> str:
        """生成检查报告"""
        report_lines = []
        report_lines.append("# Slim Guard 检查报告\n")
        report_lines.append(f"检查目录: {', '.join(CHECK_DIRS)}\n")
        report_lines.append(f"发现违规: {len(self.violations)} 处\n")
        
        if self.violations:
            report_lines.append("\n## 违规详情\n")
            for file_path, line_no, reason, line_content in self.violations:
                report_lines.append(f"- **{file_path}:{line_no}** - {reason}")
                report_lines.append(f"  ```python")
                report_lines.append(f"  {line_content}")
                report_lines.append(f"  ```\n")
        else:
            report_lines.append("\n✅ 无违规\n")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 创建守护器
    guard = SlimGuard(project_root)
    
    # 运行检查
    success = guard.run()
    
    # 可选：保存报告
    if '--save-report' in sys.argv:
        report = guard.generate_report()
        report_path = project_root / 'slim_guard_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📝 报告已保存到: {report_path}")
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

