# (Slim Inference Mode)

> 专注于高效、稳定的分子对接推理

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)

## 📋 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [FAQ](#faq)
- [贡献与支持](#贡献与支持)

---

## 🎯 概述

HerbDock 是一个基于扩散模型的分子对接工具，可以预测小分子配体与蛋白质的结合姿态。

**本版特点:**
- ✅ **仅推理** - 移除了所有训练代码和依赖
- ✅ **统一API** - 提供简洁的 Python API 接口
- ✅ **Gradio 界面** - 友好的 Web UI，即开即用
- ✅ **轻量级** - 依赖精简，安装快速
- ✅ **易维护** - 代码结构清晰，文档完善

### 内容

- ✅ 推理脚本：`inference.py` (原始版本)
- ✅ 推理 API：`src/inference.py` (新封装)
- ✅ 模型定义：`models/` (仅推理相关)
- ✅ 推理工具：`utils/inference_utils.py`, `utils/sampling.py` 等
- ✅ Gradio 界面：`app/gradio_app.py` (全新设计)
- ✅ 示例数据：`examples/` (用于测试)

### 新增的内容

- ➕ `src/inference.py` - 统一的推理 API 封装
- ➕ `src/preprocess.py` - 输入预处理模块
- ➕ `src/postprocess.py` - 输出后处理模块
- ➕ `app/gradio_app.py` - 全新的 Gradio Web 界面
- ➕ `app/runtime_config.yaml` - 运行时配置
- ➕ `requirements-slim.txt` - 精简依赖列表
- ➕ `slim_guard.py` - 防止训练依赖回退的守护脚本
- ➕ `archive_training/` - 归档的训练代码

---

## 🚀 快速开始

### 系统要求

- **Python**: 3.9+
- **CUDA**: 11.7+ (推荐，用于 GPU 加速)
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间（用于模型权重）

### 1. 安装依赖

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖（分步安装，避免冲突）
# 步骤1: 先安装 PyTorch
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 步骤2: 安装其他依赖
pip install -r requirements-slim.txt
```

**注意**: 如果 `openfold` 安装失败，请参考 [FAQ](#faq) 中的解决方案。

### 2. 下载模型权重

模型会在首次运行时自动下载。您也可以手动下载：

```bash
# 模型将保存到 workdir/v1.1/ 目录
# 首次运行 inference.py 或 gradio_app.py 时会自动下载
```

### 3. 启动 Gradio 界面

```bash
python app/gradio_app.py
```

然后在浏览器中打开 `http://localhost:7860`

**首次使用**: 点击 "🚀 初始化/加载模型" 按钮加载模型（需要几分钟）。


# 执行推理
result = runtime.predict(
    protein_path='examples/6w70.pdb',
    ligand_description='COc1ccc(cc1)n2c3c(c(n2)C(=O)N)CCN(C3=O)c4ccc(cc4)N5CCCCC5=O',
    complex_name='6w70',
    save_visualisation=False
)

print(result)
# 输出目录: results/my_inference/6w70/
# 包含 rank1.sdf, rank2_confidence0.85.sdf, ...
```

---

## 📖 使用方法

### 方法 1: Gradio Web 界面

**优点**: 直观易用，适合快速测试和演示

1. 启动界面：`python app/gradio_app.py`
2. 点击 "初始化/加载模型"
3. 在输入区填写：
   - **蛋白质 PDB 文件路径**: 例如 `examples/6w70.pdb`
   - **配体描述**: SMILES 字符串或 SDF 文件路径
   - **复合物名称** (可选): 用于命名输出
4. 调整参数（可选）：
   - 生成样本数: 1-50
   - 计算设备: auto/cuda/cpu
5. 点击 "▶️ 运行推理"
6. 查看结果和历史记录

### 方法 2: 命令行 (原始 inference.py)

**优点**: 适合批量处理和脚本化

```bash
python inference.py \
    --protein_path examples/6w70.pdb \
    --ligand_description "your_smiles_string" \
    --out_dir results/my_output \
    --samples_per_complex 10 \
    --inference_steps 20
```

**批量推理** (使用 CSV 文件):

```bash
python inference.py \
    --protein_ligand_csv my_inputs.csv \
    --out_dir results/batch_output
```

CSV 格式示例:
```csv
complex_name,protein_path,ligand_description
6w70,examples/6w70.pdb,COc1ccc(cc1)n2c3c...
6moa,examples/6moa_protein_processed.pdb,examples/6moa_ligand.sdf
```


```

---

## 📁 项目结构

```
HerbDock/
├── app/                          # Gradio 应用
│   ├── gradio_app.py            # 统一的 Web 界面 (新)
│   ├── runtime_config.yaml      # 运行时配置 (新)
│   ├── main.py                  # 原始 Gradio 界面 (保留)
│   └── ...
├── src/                          # 推理核心模块 (新)
│   ├── __init__.py
│   ├── inference.py             # 统一推理 API
│   ├── preprocess.py            # 输入预处理
│   └── postprocess.py           # 输出后处理
├── models/                       # 模型定义
│   ├── aa_model.py
│   ├── cg_model.py
│   └── ...
├── utils/                        # 推理工具函数
│   ├── inference_utils.py
│   ├── sampling.py
│   ├── diffusion_utils.py
│   └── ...
├── datasets/                     # 数据处理 (仅推理相关)
│   ├── process_mols.py
│   └── ...
├── examples/                     # 示例数据
│   ├── 6w70.pdb
│   ├── 6w70_ligand.sdf
│   └── ...
├── archive_training/             # 训练代码归档 (不参与运行)
│   ├── train.py
│   ├── confidence_train.py
│   └── ...
├── weights/                      # 模型权重目录 (占位)
├── workdir/                      # 默认模型权重位置
│   └── v1.1/
│       ├── best_ema_inference_epoch_model.pt
│       └── model_parameters.yml
├── inference.py                  # 原始推理脚本 (保留)
├── requirements-slim.txt         # 精简依赖 (新)
├── requirements.txt              # 原始依赖 (保留，但不推荐用)
├── slim_guard.py                 # 守护脚本 (新)
├── README-SLIM.md               # 本文档 (新)
├── README.md                     # 原始 README (保留)
└── TRAINING_COMPONENTS_INVENTORY.md  # 训练组件清单 (新)
```

---

## ⚙️ 配置说明

### 运行时配置 (`app/runtime_config.yaml`)

```yaml
# 模型路径
model_dir: "workdir/v1.1"
ckpt: "best_ema_inference_epoch_model.pt"
confidence_model_dir: null  # 可选

# 设备
device: "auto"  # auto | cuda | cpu

# 推理参数
samples_per_complex: 10
inference_steps: 20
batch_size: 10

# 输出
out_dir: "results/inference_output"
save_visualisation: false

# 高级参数
temp_sampling_tr: 1.0
temp_sampling_rot: 1.0
temp_sampling_tor: 1.0
# ... 更多参数见文件
```

可以在 Gradio 界面的"高级设置"中编辑配置并重新加载。

---

## 🧪 测试

### 运行守护检查

确保没有引入训练依赖：

```bash
python slim_guard.py
```

### 最小推理测试

```bash
python tests/test_infer_minimal.py
```

预期输出: 成功生成对接结果文件。

---

## 💡 FAQ

### Q1: 安装 `openfold` 失败怎么办？

**A**: OpenFold 的安装可能比较复杂。建议：

1. 先安装 PyTorch:
   ```bash
   pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

2. 如果还是失败，尝试：
   ```bash
   # 确保有 GCC 和 CUDA 编译环境
   sudo apt install build-essential  # Ubuntu
   ```

3. 如果完全无法安装，可以注释掉 `requirements-slim.txt` 中的 OpenFold 行。
   - **注意**: 这会影响蛋白质嵌入功能，但如果只使用预处理好的 PDB 文件则无影响。


### Q2: 推理速度慢怎么办？

**A**: 几个优化建议：

1. **使用 GPU**: 确保 CUDA 可用，设置 `device: "cuda"`
2. **减少样本数**: 将 `samples_per_complex` 降到 5 或更少
3. **减少推理步骤**: 将 `inference_steps` 降到 10-15（可能略微降低精度）
4. **增大批量**: 如果 GPU 内存充足，增大 `batch_size`

### Q4: 如何恢复训练功能？

**A**: 如果需要训练：

1. 查看 `archive_training/README.md`
2. 将需要的训练脚本复制回主目录
3. 安装原始的 `requirements.txt` 依赖
4. 按照原始 `README.md` 进行训练

### Q5: 输出文件在哪里？

**A**: 默认输出在 `results/inference_output/复合物名称/` 目录下：

```
results/inference_output/my_complex/
├── rank1.sdf                      # 最佳结果
├── rank1_confidence0.95.sdf       # 带置信度的最佳结果
├── rank2_confidence0.87.sdf       # 第二名
└── ...
```

### Q6: 可以使用自己的蛋白质序列吗？

**A**: 可以！支持从序列生成结构（通过 ESMFold）：

```python
result = runtime.predict(
    protein_sequence='MKTAYIAKQRQ...',  # 不提供 protein_path
    ligand_description='your_smiles',
    complex_name='my_protein'
)
```

或在 Gradio 界面中：留空 "蛋白质文件路径"，在高级设置中提供序列。

---

## 🔒 守护机制

为防止意外引入训练依赖，本项目提供了 `slim_guard.py` 守护脚本：

```bash
# 手动运行检查
python slim_guard.py

# 保存检查报告
python slim_guard.py --save-report
```

**CI/CD 集成**: 可以在 CI 流程中添加：

```yaml
# .github/workflows/slim-check.yml
- name: Check training dependencies
  run: python slim_guard.py
```

---

## 🤝 贡献与支持

### 贡献

欢迎贡献！请遵循以下原则：

- ✅ 仅推理相关的改进和优化
- ✅ 保持代码简洁，避免引入训练依赖
- ✅ 更新文档和测试

提交 PR 前请运行：
```bash
python slim_guard.py  # 确保无训练依赖
python tests/test_infer_minimal.py  # 确保功能正常
```


### 许可

本项目遵循 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 📞 联系

- 问题反馈: zhangshiyu654@gmail.com

---

**祝您使用愉快！** 🎉

如有问题，请查看 [FAQ](#faq) 或提交 Issue。

