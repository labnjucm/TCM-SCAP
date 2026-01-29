#!/usr/bin/env python3
"""
详细的推理调试脚本 - 捕获完整错误信息
"""
import sys
import traceback
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.inference import DiffDockRuntime

print("=" * 70)
print("详细推理调试")
print("=" * 70)

# 加载配置
print("\n步骤 1: 加载配置...")
with open('app/runtime_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("✓ 配置加载成功")
print(f"  - model_dir: {config['model_dir']}")
print(f"  - samples_per_complex: {config['samples_per_complex']}")
print(f"  - inference_steps: {config['inference_steps']}")
print(f"  - actual_steps: {config['actual_steps']}")

# 创建 Runtime
print("\n步骤 2: 创建 Runtime...")
try:
    runtime = DiffDockRuntime(config)
    print("✓ Runtime 创建成功")
except Exception as e:
    print(f"✗ 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 加载模型
print("\n步骤 3: 加载模型...")
try:
    runtime.load()
    print("✓ 模型加载成功")
    print(f"  - 设备: {runtime.device}")
    print(f"  - 评分模型: {runtime.model is not None}")
    print(f"  - 置信度模型: {runtime.confidence_model is not None}")
except Exception as e:
    print(f"✗ 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 检查模型参数
print("\n步骤 4: 检查模型参数...")
try:
    args = runtime.score_model_args
    print(f"  - no_torsion: {args.no_torsion}")
    print(f"  - tr_sigma_max: {args.tr_sigma_max}")
    print(f"  - receptor_radius: {args.receptor_radius}")
    print(f"  - remove_hs: {args.remove_hs}")
    print(f"  - all_atoms: {args.all_atoms}")
    print("✓ 模型参数正常")
except Exception as e:
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 执行推理
print("\n步骤 5: 执行推理...")
example_protein = PROJECT_ROOT / "examples" / "6w70.pdb"
example_smiles = 'COc1ccc(cc1)n2c3c(c(n2)C(=O)N)CCN(C3=O)c4ccc(cc4)N5CCCCC5=O'

if not example_protein.exists():
    print(f"⚠ 示例文件不存在: {example_protein}")
    sys.exit(0)

print(f"  使用示例: {example_protein.name}")
print(f"  SMILES: {example_smiles[:50]}...")
print(f"  生成样本数: 2 (简化测试)")

# 修改配置以加快测试
runtime.config['samples_per_complex'] = 2
runtime.config['inference_steps'] = 10

try:
    result = runtime.predict(
        protein_path=str(example_protein),
        ligand_description=example_smiles,
        complex_name='debug_test',
        save_visualisation=False
    )
    
    if result['success']:
        print("\n" + "=" * 70)
        print("✅ 推理成功！")
        print("=" * 70)
        print(f"  输出目录: {result['output_dir']}")
        print(f"  生成文件数: {len(result.get('files', []))}")
        if result.get('confidences'):
            print(f"  置信度: {result['confidences']}")
    else:
        print("\n" + "=" * 70)
        print("❌ 推理失败")
        print("=" * 70)
        print(f"  错误: {result.get('error')}")
        if 'traceback' in result:
            print("\n详细错误:")
            print(result['traceback'])

except Exception as e:
    print("\n" + "=" * 70)
    print("❌ 推理异常")
    print("=" * 70)
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    print("\n完整堆栈:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("调试完成")
print("=" * 70)

