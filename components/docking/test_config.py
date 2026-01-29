#!/usr/bin/env python3
"""
快速测试配置是否正确
"""
import os
import yaml

# 读取配置
with open('app/runtime_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("配置文件检查")
print("=" * 60)

# 检查评分模型
model_dir = config['model_dir']
ckpt = config['ckpt']
model_path = os.path.join(model_dir, ckpt)
params_path = os.path.join(model_dir, 'model_parameters.yml')

print(f"\n评分模型:")
print(f"  目录: {model_dir}")
print(f"  存在: {'✓' if os.path.exists(model_dir) else '✗'}")
print(f"  检查点: {ckpt}")
print(f"  检查点存在: {'✓' if os.path.exists(model_path) else '✗'}")
print(f"  参数文件存在: {'✓' if os.path.exists(params_path) else '✗'}")

# 检查置信度模型
conf_dir = config.get('confidence_model_dir')
if conf_dir:
    conf_ckpt = config.get('confidence_ckpt')
    conf_path = os.path.join(conf_dir, conf_ckpt)
    conf_params = os.path.join(conf_dir, 'model_parameters.yml')
    
    print(f"\n置信度模型:")
    print(f"  目录: {conf_dir}")
    print(f"  存在: {'✓' if os.path.exists(conf_dir) else '✗'}")
    print(f"  检查点: {conf_ckpt}")
    print(f"  检查点存在: {'✓' if os.path.exists(conf_path) else '✗'}")
    print(f"  参数文件存在: {'✓' if os.path.exists(conf_params) else '✗'}")

print("\n" + "=" * 60)

all_exist = (
    os.path.exists(model_dir) and 
    os.path.exists(model_path) and 
    os.path.exists(params_path)
)

if all_exist:
    print("✅ 配置正确！可以启动 Gradio 界面了")
    print("\n运行: python app/gradio_app.py")
else:
    print("❌ 配置有误，请检查上述路径")

print("=" * 60)

