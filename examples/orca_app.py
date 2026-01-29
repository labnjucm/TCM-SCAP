#!/usr/bin/env python3
"""
ChemHub - ORCA 量化计算 Gradio 示例应用
端口：7863
子路径：/apps/orca/
"""

import gradio as gr

def orca_calculation(xyz_coords, calculation_type, basis_set):
    """
    模拟 ORCA 量化计算
    
    Args:
        xyz_coords: XYZ 格式分子坐标
        calculation_type: 计算类型
        basis_set: 基组选择
    
    Returns:
        计算结果信息
    """
    if not xyz_coords or xyz_coords.strip() == "":
        return "❌ 请输入分子坐标（XYZ 格式）"
    
    result = f"""
✅ ORCA 量化计算完成！

⚗️ 计算类型: {calculation_type}
📐 基组: {basis_set}

📊 计算结果:
- 总能量: -115.234567 Hartree
- HOMO 能级: -0.245 a.u.
- LUMO 能级: 0.089 a.u.
- 能隙: 9.08 eV
- 偶极矩: 1.85 Debye

🔬 优化几何:
{xyz_coords[:200]}...

📈 振动频率分析:
- 最低频率: 125 cm⁻¹
- 最高频率: 3450 cm⁻¹
- 零点能: 45.2 kcal/mol

⚠️ 注意：这是一个示例界面，实际计算需要安装 ORCA 软件包。
    """
    return result

# 示例分子坐标（水分子）
EXAMPLE_XYZ = """3

O    0.000000    0.000000    0.119262
H    0.000000    0.763239   -0.477047
H    0.000000   -0.763239   -0.477047"""

# 创建 Gradio 界面
with gr.Blocks(title="ORCA 量化计算") as demo:
    gr.Markdown("""
    # 🔬 ORCA 量化计算工具
    
    输入分子坐标，选择计算方法和基组，运行量子化学计算
    """)
    
    with gr.Row():
        with gr.Column():
            xyz_input = gr.Textbox(
                label="分子坐标 (XYZ 格式)",
                lines=10,
                placeholder="输入 XYZ 格式坐标...\n例如：\n3\n\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0",
                value=EXAMPLE_XYZ
            )
            
            calculation_type = gr.Dropdown(
                choices=[
                    "单点能计算 (SP)",
                    "几何优化 (OPT)",
                    "频率分析 (FREQ)",
                    "激发态 (TD-DFT)",
                    "NMR 化学位移"
                ],
                value="几何优化 (OPT)",
                label="计算类型"
            )
            
            basis_set = gr.Dropdown(
                choices=[
                    "def2-SVP",
                    "def2-TZVP",
                    "def2-QZVP",
                    "6-31G(d)",
                    "6-311++G(d,p)",
                    "cc-pVDZ",
                    "cc-pVTZ"
                ],
                value="def2-TZVP",
                label="基组"
            )
            
            with gr.Accordion("高级选项", open=False):
                functional = gr.Dropdown(
                    choices=["B3LYP", "PBE0", "M06-2X", "wB97X-D3"],
                    value="B3LYP",
                    label="泛函"
                )
                solvent = gr.Dropdown(
                    choices=["真空", "水", "DMSO", "氯仿", "甲醇"],
                    value="真空",
                    label="溶剂模型"
                )
            
            run_btn = gr.Button("🚀 开始计算", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="计算结果",
                lines=25,
                placeholder="结果将显示在这里..."
            )
    
    run_btn.click(
        fn=orca_calculation,
        inputs=[xyz_input, calculation_type, basis_set],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ### 📚 使用说明
    
    1. 输入分子的 XYZ 坐标（原子序号 + 三维坐标）
    2. 选择计算类型（单点能、优化、频率等）
    3. 选择合适的基组（精度 vs 计算成本）
    4. 可选：配置泛函和溶剂模型
    5. 点击"开始计算"按钮
    
    ### 🔗 相关资源
    
    - [ORCA 官方论坛](https://orcaforum.kofo.mpg.de/)
    - [Gaussian 官网](https://gaussian.com/)
    - [Q-Chem 官网](https://www.q-chem.com/)
    
    ### 💡 提示
    
    - **def2-SVP**：快速计算，适合初步筛选
    - **def2-TZVP**：平衡精度与速度
    - **def2-QZVP**：高精度，计算量大
    - **B3LYP**：最常用的 DFT 泛函
    - 大分子建议先用小基组优化，再用大基组算单点能
    
    ### 📖 XYZ 格式示例
    
    ```
    3
    水分子
    O    0.000000    0.000000    0.119262
    H    0.000000    0.763239   -0.477047
    H    0.000000   -0.763239   -0.477047
    ```
    """)

if __name__ == "__main__":
    # 直接启动，不使用子路径（无需 Nginx）
    demo.launch(
        server_name="127.0.0.1",  # 本地访问
        server_port=7863,
        show_error=True,
        share=False
    )

