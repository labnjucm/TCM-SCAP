#!/usr/bin/env python3
"""
ChemHub - 分子动力学模拟 Gradio 示例应用
端口：7862
子路径：/apps/md/
"""

import gradio as gr

def md_simulation(structure_file, force_field, simulation_time, temperature):
    """
    模拟分子动力学计算
    
    Args:
        structure_file: 分子结构文件
        force_field: 力场选择
        simulation_time: 模拟时长 (ns)
        temperature: 温度 (K)
    
    Returns:
        模拟结果信息
    """
    if structure_file is None:
        return "❌ 请上传分子结构文件"
    
    result = f"""
✅ 分子动力学模拟完成！

📁 输入文件: {structure_file.name}
⚗️ 力场: {force_field}
⏱️ 模拟时长: {simulation_time} ns
🌡️ 温度: {temperature} K

📊 模拟结果:
- 总能量: -45678.3 kJ/mol
- RMSD: 2.3 Å
- Rg (回旋半径): 1.8 nm
- 氢键数量: 145
- 溶剂可及表面积: 185 nm²

📈 生成文件:
- trajectory.xtc (轨迹文件)
- energy.xvg (能量曲线)
- rmsd.xvg (RMSD 曲线)

⚠️ 注意：这是一个示例界面，实际计算需要集成 GROMACS/OpenMM 等工具。
    """
    return result

# 创建 Gradio 界面
with gr.Blocks(title="分子动力学模拟") as demo:
    gr.Markdown("""
    # ⚛️ 分子动力学模拟工具
    
    上传分子结构，配置模拟参数，运行 MD 模拟（GROMACS/OpenMM）
    """)
    
    with gr.Row():
        with gr.Column():
            structure_input = gr.File(
                label="分子结构文件 (PDB/GRO)",
                file_types=[".pdb", ".gro"]
            )
            force_field = gr.Dropdown(
                choices=[
                    "AMBER99SB-ILDN",
                    "CHARMM36",
                    "GROMOS96 54a7",
                    "OPLS-AA/L"
                ],
                value="AMBER99SB-ILDN",
                label="力场选择"
            )
            simulation_time = gr.Slider(
                minimum=0.1,
                maximum=100,
                value=10,
                step=0.1,
                label="模拟时长 (ns)"
            )
            temperature = gr.Slider(
                minimum=250,
                maximum=400,
                value=300,
                step=10,
                label="温度 (K)"
            )
            run_btn = gr.Button("🚀 开始模拟", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="模拟结果",
                lines=20,
                placeholder="结果将显示在这里..."
            )
    
    run_btn.click(
        fn=md_simulation,
        inputs=[structure_input, force_field, simulation_time, temperature],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ### 📚 使用说明
    
    1. 准备分子结构文件（PDB 或 GRO 格式）
    2. 选择合适的力场
    3. 设置模拟时长和温度
    4. 点击"开始模拟"按钮
    
    ### 🔗 相关资源
    
    - [GROMACS 官网](https://www.gromacs.org/)
    - [OpenMM 官网](http://openmm.org/)
    - [NAMD 官网](https://www.ks.uiuc.edu/Research/namd/)
    
    ### 💡 提示
    
    - 短时间模拟（<1 ns）适合快速测试
    - 蛋白质折叠模拟通常需要 100+ ns
    - 使用 GPU 可显著加速计算
    """)

if __name__ == "__main__":
    # 直接启动，不使用子路径（无需 Nginx）
    demo.launch(
        server_name="127.0.0.1",  # 本地访问
        server_port=7862,
        show_error=True,
        share=False
    )

