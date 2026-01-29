# 训练代码归档目录

⚠️ **注意**: 本目录包含的所有训练相关代码已从主运行路径中移除。

## 目录说明

此目录包含 HerbDock 项目的训练相关代码，这些代码在"仅推理"(inference-only) 模式下**不会被使用**。

### 归档文件列表

- `train.py` - 主训练脚本
- `confidence_train.py` - 置信度模型训练脚本  
- `evaluate.py` - 模型评估脚本
- `training.py` - 训练工具函数（epoch训练、损失计算等）
- `*.sh` - 各种训练Shell脚本

## 如何启用训练功能

如果您需要重新启用训练功能：

1. **恢复训练代码**
   ```bash
   # 将需要的训练脚本复制回主目录
   cp archive_training/train.py ../
   cp archive_training/confidence/confidence_train.py ../confidence/
   ```

2. **安装完整依赖**
   ```bash
   # 使用原始的 requirements.txt 而不是 requirements-slim.txt
   pip install -r requirements-full.txt  # 如果存在
   # 或手动安装训练依赖
   pip install wandb pytorch-lightning torchmetrics
   ```

3. **准备训练数据**
   - 按照原项目 README.md 准备 PDBBind 数据集
   - 配置数据路径和分割文件

4. **运行训练**
   ```bash
   python train.py --config your_config.yaml
   ```

## 推理模式

在推理模式下，请使用：
```bash
# 启动 Gradio 界面
python app/gradio_app.py

# 或使用命令行推理
python inference.py --config default_inference_args.yaml \
    --protein_path examples/6w70.pdb \
    --ligand_description "your_smiles_string"
```

详细说明请参考主目录的 `README-SLIM.md`。

## 参考

- 原始完整功能文档: `../README.md`
- 精简版文档: `../README-SLIM.md`
- 训练组件清单: `../TRAINING_COMPONENTS_INVENTORY.md`

