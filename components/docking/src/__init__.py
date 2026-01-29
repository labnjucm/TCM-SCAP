"""
HerbDock Inference-Only Package

这个包仅包含推理相关的功能，所有训练代码已被移除。
"""

__version__ = "1.0.0-slim"
__author__ = "HerbDock Team (Inference-only refactor)"

# 确保不导入任何训练库
import sys

# 禁止导入的训练库列表
FORBIDDEN_MODULES = [
    'pytorch_lightning',
    'lightning', 
    'wandb',
    'tensorboard',
    'deepspeed',
    'accelerate',
    'bitsandbytes'
]

# 在导入时检查
def _check_imports():
    """检查是否意外导入了训练库"""
    for mod_name in FORBIDDEN_MODULES:
        if mod_name in sys.modules:
            import warnings
            warnings.warn(
                f"警告: 检测到训练库 '{mod_name}' 已被导入。"
                f"这可能导致不必要的依赖。请检查代码。",
                RuntimeWarning
            )

# 启动时进行检查（可选，仅在开发时启用）
# _check_imports()

