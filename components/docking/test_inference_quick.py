#!/usr/bin/env python3
"""
快速测试推理功能是否正常
"""
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.inference import DiffDockRuntime

print("=" * 60)
print("快速推理测试")
print("=" * 60)

# 加载配置
print("\n1. 加载配置...")
with open('app/runtime_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"   ✓ 配置加载成功")
print(f"   - inference_steps: {config.get('inference_steps')}")
print(f"   - actual_steps: {config.get('actual_steps')}")
print(f"   - samples_per_complex: {config.get('samples_per_complex')}")

# 创建 Runtime
print("\n2. 创建 Runtime...")
try:
    runtime = DiffDockRuntime(config)
    print("   ✓ Runtime 创建成功")
except Exception as e:
    print(f"   ✗ Runtime 创建失败: {e}")
    sys.exit(1)

# 加载模型
print("\n3. 加载模型（这可能需要几分钟）...")
try:
    runtime.load()
    print("   ✓ 模型加载成功")
except Exception as e:
    print(f"   ✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查示例文件
print("\n4. 检查示例文件...")
example_protein = PROJECT_ROOT / "examples" / "6w70.pdb"
if not example_protein.exists():
    print(f"   ⚠ 示例文件不存在: {example_protein}")
    print("   测试将跳过推理步骤")
    print("\n✓ 基础测试通过！Runtime 可以正常初始化和加载模型。")
    sys.exit(0)

print(f"   ✓ 示例文件存在: {example_protein.name}")

# 执行推理（使用简化参数）
print("\n5. 执行推理（生成2个样本）...")
try:
    # 临时修改配置，减少样本数以加快测试
    runtime.config['samples_per_complex'] = 2
    runtime.config['inference_steps'] = 10
    
    result = runtime.predict(
        protein_path=str(example_protein),
        ligand_description='COc1ccc(cc1)n2c3c(c(n2)C(=O)N)CCN(C3=O)c4ccc(cc4)N5CCCCC5=O',
        complex_name='test_6w70',
        save_visualisation=False
    )
    
    if result['success']:
        print("   ✓ 推理成功！")
        print(f"   - 输出目录: {result['output_dir']}")
        print(f"   - 生成文件数: {len(result.get('files', []))}")
        if result.get('confidences'):
            print(f"   - 最佳置信度: {result['confidences'][0]:.3f}")
    else:
        print(f"   ✗ 推理失败: {result.get('error')}")
        if 'traceback' in result:
            print("\n详细错误信息:")
            print(result['traceback'])
        sys.exit(1)
    
except Exception as e:
    print(f"   ✗ 推理异常: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！推理功能正常工作。")
print("=" * 60)
print("\n现在可以启动 Gradio 界面:")
print("  python app/gradio_app.py")
print("=" * 60)

