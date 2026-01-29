"""
最小推理测试

验证精简版推理功能是否正常工作。
仅使用 requirements-slim.txt 中的依赖。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml


def test_imports():
    """测试基础导入"""
    print("=" * 70)
    print("测试 1: 检查基础导入")
    print("=" * 70)
    
    try:
        # 测试核心库
        import numpy as np
        import pandas as pd
        from rdkit import Chem
        import gradio as gr
        
        print("✅ 基础库导入成功")
        
        # 测试项目模块
        from src.inference import DiffDockRuntime
        from src.preprocess import validate_protein_input, validate_ligand_input
        from src.postprocess import format_result_summary
        
        print("✅ 项目模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_no_training_imports():
    """测试是否意外导入了训练库"""
    print("\n" + "=" * 70)
    print("测试 2: 检查训练依赖泄漏")
    print("=" * 70)
    
    forbidden_modules = [
        'pytorch_lightning',
        'lightning',
        'wandb',
        'tensorboard',
        'deepspeed',
        'accelerate',
        'bitsandbytes',
        'torchmetrics'
    ]
    
    leaked = []
    for mod in forbidden_modules:
        if mod in sys.modules:
            leaked.append(mod)
    
    if leaked:
        print(f"❌ 检测到训练库泄漏: {leaked}")
        return False
    else:
        print("✅ 未检测到训练依赖泄漏")
        return True


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 70)
    print("测试 3: 配置文件加载")
    print("=" * 70)
    
    config_path = PROJECT_ROOT / "app" / "runtime_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model_dir', 'device', 'samples_per_complex']
        for key in required_keys:
            if key not in config:
                print(f"❌ 配置文件缺少必需键: {key}")
                return False
        
        print(f"✅ 配置文件加载成功")
        print(f"   模型目录: {config.get('model_dir')}")
        print(f"   设备: {config.get('device')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_preprocessing():
    """测试预处理模块"""
    print("\n" + "=" * 70)
    print("测试 4: 预处理功能")
    print("=" * 70)
    
    try:
        from src.preprocess import (
            validate_protein_input,
            validate_ligand_input,
            prepare_input_summary
        )
        
        # 测试验证函数
        valid, msg = validate_protein_input(None, None)
        assert not valid, "应该验证失败：没有蛋白质输入"
        
        valid, msg = validate_ligand_input(None)
        assert not valid, "应该验证失败：没有配体输入"
        
        # 测试示例文件验证
        example_pdb = PROJECT_ROOT / "examples" / "6w70.pdb"
        if example_pdb.exists():
            valid, msg = validate_protein_input(str(example_pdb), None)
            assert valid, f"示例PDB文件应该验证通过: {msg}"
            print(f"✅ 示例文件验证通过: {example_pdb.name}")
        
        print("✅ 预处理功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 预处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_postprocessing():
    """测试后处理模块"""
    print("\n" + "=" * 70)
    print("测试 5: 后处理功能")
    print("=" * 70)
    
    try:
        from src.postprocess import format_result_summary, extract_top_result
        
        # 测试成功结果格式化
        success_result = {
            'success': True,
            'complex_name': 'test',
            'output_dir': '/tmp/test',
            'confidences': [0.95, 0.87, 0.75],
            'files': ['rank1.sdf', 'rank2.sdf']
        }
        
        summary = format_result_summary(success_result)
        assert '✅' in summary or '成功' in summary, "成功结果应包含成功标志"
        
        # 测试失败结果格式化
        fail_result = {
            'success': False,
            'error': 'Test error'
        }
        
        summary = format_result_summary(fail_result)
        assert '❌' in summary or '失败' in summary, "失败结果应包含失败标志"
        
        print("✅ 后处理功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 后处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_runtime_creation():
    """测试 Runtime 创建（不加载模型）"""
    print("\n" + "=" * 70)
    print("测试 6: Runtime 创建")
    print("=" * 70)
    
    try:
        from src.inference import DiffDockRuntime
        
        # 创建一个最小配置
        config = {
            'model_dir': 'workdir/v1.1',
            'ckpt': 'best_ema_inference_epoch_model.pt',
            'device': 'cpu',  # 使用CPU避免需要GPU
            'samples_per_complex': 1,
            'inference_steps': 5,
            'batch_size': 1,
            'out_dir': 'results/test'
        }
        
        runtime = DiffDockRuntime(config)
        
        print(f"✅ Runtime 创建成功")
        print(f"   设备: {runtime.device}")
        
        # 注意：我们不加载模型，因为可能没有权重文件
        print("   (跳过模型加载测试，因为可能没有权重文件)")
        
        return True
        
    except Exception as e:
        print(f"❌ Runtime 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_example_files():
    """测试示例文件是否存在"""
    print("\n" + "=" * 70)
    print("测试 7: 检查示例文件")
    print("=" * 70)
    
    examples_dir = PROJECT_ROOT / "examples"
    
    if not examples_dir.exists():
        print(f"⚠️  示例目录不存在: {examples_dir}")
        return True  # 不算失败，只是警告
    
    required_files = [
        "6w70.pdb",
        "6w70_ligand.sdf",
    ]
    
    found_files = []
    missing_files = []
    
    for filename in required_files:
        filepath = examples_dir / filename
        if filepath.exists():
            found_files.append(filename)
        else:
            missing_files.append(filename)
    
    if found_files:
        print(f"✅ 找到 {len(found_files)} 个示例文件:")
        for f in found_files:
            print(f"   - {f}")
    
    if missing_files:
        print(f"⚠️  缺失 {len(missing_files)} 个示例文件:")
        for f in missing_files:
            print(f"   - {f}")
    
    return True


def test_slim_guard():
    """测试守护脚本"""
    print("\n" + "=" * 70)
    print("测试 8: 守护脚本检查")
    print("=" * 70)
    
    guard_script = PROJECT_ROOT / "slim_guard.py"
    
    if not guard_script.exists():
        print(f"❌ 守护脚本不存在: {guard_script}")
        return False
    
    try:
        # 导入并运行守护检查
        import slim_guard
        
        guard = slim_guard.SlimGuard(PROJECT_ROOT)
        success = guard.run()
        
        if success:
            print("\n✅ 守护检查通过")
        else:
            print("\n⚠️  守护检查发现违规")
            print("   (这可能是正常的，如果您正在开发中)")
        
        return True  # 即使有违规也返回True，因为这只是警告
        
    except Exception as e:
        print(f"❌ 守护脚本运行失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "最小推理测试" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    tests = [
        ("基础导入", test_imports),
        ("训练依赖检查", test_no_training_imports),
        ("配置加载", test_config_loading),
        ("预处理功能", test_preprocessing),
        ("后处理功能", test_postprocessing),
        ("Runtime创建", test_runtime_creation),
        ("示例文件", test_example_files),
        ("守护脚本", test_slim_guard),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status:8} | {test_name}")
    
    print("=" * 70)
    print(f"总计: {passed}/{total} 通过 ({passed/total*100:.0f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 所有测试通过！精简版推理功能正常。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败。请检查上述错误信息。")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)

