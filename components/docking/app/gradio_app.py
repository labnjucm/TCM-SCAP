"""
分子对接 统一推理界面 (Slim Mode)

提供简洁、稳定的 Gradio 界面，仅用于推理。
完全独立于训练代码。
"""

import os
import sys
import yaml
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import gradio as gr
import pandas as pd
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入推理API和工具模块
from src.inference import 分子对接Runtime, create_runtime_from_yaml
from src.preprocess import (
    validate_protein_input,
    validate_ligand_input,
    prepare_input_summary
)
from src.postprocess import (
    format_result_summary,
    create_result_zip
)

# 全局变量
runtime: Optional[分子对接Runtime] = None
history_records: List[dict] = []

# 配置文件路径
CONFIG_FILE = PROJECT_ROOT / "app" / "runtime_config.yaml"


def initialize_runtime() -> Tuple[str, str]:
    """
    初始化推理运行时
    
    Returns:
        (状态消息, 日志文本)
    """
    global runtime
    
    try:
        # 加载配置
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建Runtime
        runtime = 分子对接Runtime(config)
        
        # 加载模型
        runtime.load()
        
        msg = "✅ 模型加载成功！可以开始推理。"
        log = f"[{datetime.now().strftime('%H:%M:%S')}] 初始化完成\n"
        log += f"设备: {runtime.device}\n"
        log += f"模型目录: {config.get('model_dir')}\n"
        
        return msg, log
        
    except Exception as e:
        error_msg = f"❌ 初始化失败: {str(e)}"
        error_log = f"[{datetime.now().strftime('%H:%M:%S')}] 错误:\n{traceback.format_exc()}"
        return error_msg, error_log


def run_inference(
    input_mode: str,
    text_input: str,
    file_input: Optional[gr.File],
    json_input: str,
    protein_path: str,
    ligand_input: str,
    device: str,
    samples: int,
    complex_name: str,
    save_vis: bool
) -> Tuple[str, str, pd.DataFrame]:
    """
    执行推理
    
    Returns:
        (结果文本, 日志文本, 历史记录DataFrame)
    """
    global runtime, history_records
    
    if runtime is None:
        return "❌ 运行时未初始化，请先加载模型", "", pd.DataFrame(history_records)
    
    log_lines = []
    log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始推理...")
    
    try:
        # 根据输入模式处理输入
        if input_mode == "文本":
            # 文本模式：蛋白质路径 + 配体描述
            if not protein_path or not ligand_input:
                return "❌ 请提供蛋白质路径和配体描述", "\n".join(log_lines), pd.DataFrame(history_records)
            
            # 验证输入
            valid, msg = validate_protein_input(protein_path, None)
            if not valid:
                return f"❌ {msg}", "\n".join(log_lines), pd.DataFrame(history_records)
            
            valid, msg = validate_ligand_input(ligand_input)
            if not valid:
                return f"❌ {msg}", "\n".join(log_lines), pd.DataFrame(history_records)
            
            # 执行推理
            log_lines.append("验证通过，准备输入数据...")
            
            # 更新运行时配置（动态参数）
            if device and device != "auto":
                runtime.config['device'] = device
                runtime.device = torch.device(device)
            
            if samples:
                runtime.config['samples_per_complex'] = int(samples)
            
            log_lines.append(f"使用设备: {runtime.device}, 样本数: {runtime.config.get('samples_per_complex', 10)}")
            
            result = runtime.predict(
                protein_path=protein_path,
                ligand_description=ligand_input,
                complex_name=complex_name or f"complex_{len(history_records)}",
                save_visualisation=save_vis
            )
            
        elif input_mode == "文件":
            # 文件模式
            if not file_input:
                return "❌ 请上传文件", "\n".join(log_lines), pd.DataFrame(history_records)
            
            # TODO: 处理文件上传
            return "⚠️ 文件上传模式待实现", "\n".join(log_lines), pd.DataFrame(history_records)
            
        elif input_mode == "JSON":
            # JSON模式：批量输入
            try:
                data = json.loads(json_input)
                # TODO: 实现批量推理
                return "⚠️ JSON批量模式待实现", "\n".join(log_lines), pd.DataFrame(history_records)
            except json.JSONDecodeError as e:
                return f"❌ JSON解析失败: {str(e)}", "\n".join(log_lines), pd.DataFrame(history_records)
        
        # 格式化结果
        result_text = format_result_summary(result)
        
        # 添加到历史记录
        if result.get('success', False):
            input_summary = f"{Path(protein_path).name} + {Path(ligand_input).name if os.path.exists(ligand_input) else ligand_input[:30]}"
            output_summary = f"成功 | 置信度:{result['confidences'][0]:.3f}" if result.get('confidences') else "成功"
            
            history_records.append({
                '时间': datetime.now().strftime('%H:%M:%S'),
                '输入': input_summary,
                '输出': output_summary,
                '文件数': len(result.get('files', []))
            })
            
            # 限制历史记录数量
            if len(history_records) > 20:
                history_records = history_records[-20:]
            
            log_lines.append(f"推理完成，输出目录: {result.get('output_dir')}")
        else:
            log_lines.append(f"推理失败: {result.get('error')}")
        
        log_text = "\n".join(log_lines)
        history_df = pd.DataFrame(history_records)
        
        return result_text, log_text, history_df
        
    except Exception as e:
        error_text = f"❌ 推理过程出错: {str(e)}"
        log_lines.append(f"错误: {traceback.format_exc()}")
        return error_text, "\n".join(log_lines), pd.DataFrame(history_records)


def clear_outputs() -> Tuple[str, str, pd.DataFrame]:
    """清空输出"""
    return "", "", pd.DataFrame(history_records)


def reload_config(config_text: str) -> Tuple[str, str]:
    """重新加载配置"""
    global runtime
    
    try:
        # 解析YAML
        config = yaml.safe_load(config_text)
        
        # 重新创建Runtime
        runtime = 分子对接Runtime(config)
        runtime.load()
        
        msg = "✅ 配置已重新加载"
        log = f"[{datetime.now().strftime('%H:%M:%S')}] 配置更新成功"
        return msg, log
        
    except Exception as e:
        error_msg = f"❌ 配置加载失败: {str(e)}"
        error_log = traceback.format_exc()
        return error_msg, error_log


# 创建 Gradio 界面
def create_interface():
    """创建Gradio界面"""
    
    # 读取默认配置
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        default_config = f.read()
    
    with gr.Blocks(
        title="分子对接 推理界面 (Slim Mode)",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("# 🧬 分子对接 分子对接推理界面")
        gr.Markdown("""
        **仅推理模式** - 无训练功能，快速稳定的分子对接预测
        
        支持蛋白质-配体对接，输入PDB文件和SMILES/SDF配体描述。
        """)
        
        # 初始化按钮
        with gr.Row():
            init_btn = gr.Button("🚀 初始化/加载模型", variant="primary", size="lg")
            status_text = gr.Textbox(label="状态", interactive=False, max_lines=2)
        
        gr.Markdown("---")
        
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("## 📥 输入配置")
                
                input_mode = gr.Radio(
                    ["文本", "文件", "JSON"],
                    value="文本",
                    label="输入模式",
                    info="选择输入方式"
                )
                
                # 文本输入模式
                with gr.Group(visible=True) as text_group:
                    protein_path_input = gr.Textbox(
                        label="蛋白质PDB文件路径",
                        placeholder="examples/6w70.pdb",
                        value="examples/6w70.pdb"
                    )
                    ligand_input = gr.Textbox(
                        label="配体描述 (SMILES或文件路径)",
                        placeholder="COc1ccc(cc1)n2c3c(c(n2)C(=O)N)CCN(C3=O)...",
                        lines=3
                    )
                    complex_name_input = gr.Textbox(
                        label="复合物名称 (可选)",
                        placeholder="my_complex"
                    )
                
                # 文件输入模式
                with gr.Group(visible=False) as file_group:
                    file_input = gr.File(
                        label="上传文件",
                        file_count="multiple"
                    )
                
                # JSON输入模式
                with gr.Group(visible=False) as json_group:
                    json_input = gr.Textbox(
                        label="JSON批量输入",
                        placeholder='{"protein": "...", "ligand": "..."}',
                        lines=8
                    )
                
                gr.Markdown("### ⚙️ 推理参数")
                
                device_select = gr.Dropdown(
                    ["auto", "cuda", "cpu"],
                    value="auto",
                    label="计算设备"
                )
                
                samples_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="生成样本数",
                    info="每个复合物生成的对接姿态数量"
                )
                
                save_vis_check = gr.Checkbox(
                    label="保存可视化文件",
                    value=False
                )
                
                gr.Markdown("---")
                
                with gr.Row():
                    run_btn = gr.Button("▶️ 运行推理", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", size="lg")
            
            # 右侧：输出区
            with gr.Column(scale=2):
                gr.Markdown("## 📤 输出结果")
                
                output_text = gr.Textbox(
                    label="推理结果",
                    lines=12,
                    interactive=False,
                    placeholder="推理结果将显示在这里..."
                )
                
                log_text = gr.Textbox(
                    label="运行日志",
                    lines=8,
                    interactive=False,
                    placeholder="日志信息..."
                )
                
                gr.Markdown("### 📊 推理历史")
                history_table = gr.Dataframe(
                    headers=["时间", "输入", "输出", "文件数"],
                    datatype=["str", "str", "str", "number"],
                    row_count=(5, "dynamic"),
                    label="最近推理记录",
                    interactive=False
                )
        
        # 高级设置（折叠）
        with gr.Accordion("🔧 高级设置", open=False):
            gr.Markdown("### 配置编辑")
            config_editor = gr.Textbox(
                label="YAML配置",
                value=default_config,
                lines=15,
                interactive=True
            )
            reload_config_btn = gr.Button("重新加载配置")
            config_status = gr.Textbox(label="配置状态", interactive=False, max_lines=2)
        
        # 示例
        gr.Markdown("---")
        gr.Markdown("### 📚 示例")
        gr.Examples(
            examples=[
                [
                    "文本",
                    "examples/6w70.pdb",
                    "COc1ccc(cc1)n2c3c(c(n2)C(=O)N)CCN(C3=O)c4ccc(cc4)N5CCCCC5=O",
                    "6w70",
                    10,
                    False
                ],
                [
                    "文本",
                    "examples/6moa_protein_processed.pdb",
                    "examples/6moa_ligand.sdf",
                    "6moa",
                    10,
                    False
                ],
                [
                    "文本",
                    "examples/6o5u_protein_processed.pdb",
                    "examples/6o5u_ligand.sdf",
                    "6o5u",
                    5,
                    False
                ]
            ],
            inputs=[
                input_mode,
                protein_path_input,
                ligand_input,
                complex_name_input,
                samples_slider,
                save_vis_check
            ],
            label="点击加载示例"
        )
        
        # 页脚
        gr.Markdown("---")
        gr.Markdown("""
        **分子对接 Slim Inference Mode** | 精简推理版本 | 无训练功能
        
        原项目: [分子对接](https://github.com/gcorso/分子对接) | 
        本精简版专注于高效推理
        """)
        
        # ========== 事件绑定 ==========
        
        # 初始化
        init_btn.click(
            fn=initialize_runtime,
            inputs=[],
            outputs=[status_text, log_text]
        )
        
        # 输入模式切换
        def switch_input_mode(mode):
            return (
                gr.update(visible=(mode == "文本")),
                gr.update(visible=(mode == "文件")),
                gr.update(visible=(mode == "JSON"))
            )
        
        input_mode.change(
            fn=switch_input_mode,
            inputs=[input_mode],
            outputs=[text_group, file_group, json_group]
        )
        
        # 运行推理
        run_btn.click(
            fn=run_inference,
            inputs=[
                input_mode,
                gr.State(""),  # text_input placeholder
                file_input,
                json_input,
                protein_path_input,
                ligand_input,
                device_select,
                samples_slider,
                complex_name_input,
                save_vis_check
            ],
            outputs=[output_text, log_text, history_table]
        )
        
        # 清空
        clear_btn.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[output_text, log_text, history_table]
        )
        
        # 重新加载配置
        reload_config_btn.click(
            fn=reload_config,
            inputs=[config_editor],
            outputs=[config_status, log_text]
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("分子对接 推理界面 (Slim Mode) 启动中...")
    print("=" * 60)
    
    # 检查配置文件
    if not CONFIG_FILE.exists():
        print(f"⚠️  配置文件不存在: {CONFIG_FILE}")
        print("将使用默认配置")
    
    # 创建并启动界面
    demo = create_interface()
    
    # 启动参数
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    print(f"\n✅ 启动 Gradio 服务器: http://0.0.0.0:{server_port}")
    print("提示: 请先点击'初始化/加载模型'按钮加载模型\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        inbrowser=False
    )

