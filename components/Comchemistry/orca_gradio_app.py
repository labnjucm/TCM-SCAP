#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCA Gradio Application
一个基于 Gradio 的 ORCA 量子化学计算前端界面
支持上传 .inp 文件、一键运行 ORCA、查看结果
"""

import os
import time
import subprocess
import shutil
import signal
import threading
import queue
from pathlib import Path
from typing import Tuple, Optional, Iterator

import gradio as gr


# ============================================================================
# 全局进程管理
# ============================================================================

class ProcessManager:
    """管理 ORCA 计算进程，支持终止操作"""
    def __init__(self):
        self.current_process = None
        self.lock = threading.Lock()
    
    def set_process(self, process):
        with self.lock:
            self.current_process = process
    
    def terminate_process(self):
        with self.lock:
            if self.current_process and self.current_process.poll() is None:
                try:
                    # 尝试优雅终止
                    self.current_process.terminate()
                    # 等待 5 秒
                    try:
                        self.current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # 强制终止
                        self.current_process.kill()
                        self.current_process.wait()
                    return True
                except Exception as e:
                    print(f"终止进程失败: {e}")
                    return False
            return False
    
    def clear_process(self):
        with self.lock:
            self.current_process = None


# 全局进程管理器实例
process_manager = ProcessManager()


# ============================================================================
# ORCA 简介与格式说明（Markdown 文本）
# ============================================================================

ORCA_INTRO_MD = """
# ORCA 量子化学计算工具

## ORCA 简介

**ORCA** 是由德国马克斯·普朗克煤炭研究所 Frank Neese 教授团队开发的现代量子化学软件包。
该软件对学术界免费授权使用，已成为全球计算化学研究者的重要工具之一。

ORCA 支持广泛的量子化学计算任务，包括但不限于：
- 单点能计算（Single Point Energy）
- 几何结构优化（Geometry Optimization）
- 振动频率分析（Frequency Analysis）
- 激发态与光谱性质（TDDFT, EOM-CC）
- 自然键轨道分析（NBO）
- 能量分解分析（EDA）

软件特点：
- 🔬 **丰富的功能模块**：涵盖 DFT、半经验、从头算、多参考、相对论等多种方法
- 📊 **现代方法与基组**：支持最新的泛函、色散校正（D3BJ, D4）、大型基组
- ⚡ **高效并行计算**：通过 `%pal` 指令方便地设置多核并行
- 🔧 **良好的可扩展性**：易于与 Multiwfn、VMD 等分析工具配合使用

使用提示：
1. 使用前需从 [ORCA 官网](https://orcaforum.kofo.mpg.de/) 申请并安装
2. 确保 `orca` 可执行文件在系统 PATH 环境变量中，或在下方指定完整路径
3. 合理设置并行线程数（`%pal nprocs`）与内存（`%maxcore`）以充分利用硬件资源

---

## .inp 文件格式说明

ORCA 的输入文件（`.inp`）采用简洁的关键字+坐标结构，主要包含以下部分：

### 1. 关键字行
以 `!` 开头，指定计算方法、基组和任务类型：
```
! B3LYP D3BJ def2-SVP Opt TightSCF
```
- **方法**：`B3LYP`, `PBE0`, `CCSD(T)`, `MP2` 等
- **色散校正**：`D3BJ`, `D4` 等
- **基组**：`def2-SVP`, `def2-TZVP`, `cc-pVTZ` 等
- **任务**：`Opt`（优化）, `Freq`（频率）, `SinglePoint`（单点）, `TDDFT` 等
- **收敛选项**：`TightSCF`, `VeryTightSCF` 等

### 2. 并行与内存设置（可选）
```
%pal nprocs 8 end           # 使用 8 个 CPU 核心
%maxcore 4096               # 每个核心分配 4096 MB 内存
```

### 3. 分子结构
使用 `* xyz <电荷> <多重度>` 定义笛卡尔坐标：
```
* xyz 0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.757160   0.586260
H   0.000000  -0.757160   0.586260
*
```
- `*` 作为坐标块的起止标记
- 坐标单位为埃（Å）
- 电荷与多重度需与体系匹配（如：中性单重态水分子为 `0 1`）

### 完整示例：水分子几何优化
```
! B3LYP D3BJ def2-SVP Opt TightSCF

%pal nprocs 4 end
%maxcore 2048

* xyz 0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.757160   0.586260
H   0.000000  -0.757160   0.586260
*
```

### 常见任务类型
- `Opt`：几何优化
- `Freq`：频率分析（需在优化后的结构基础上）
- `Opt Freq`：优化+频率一步完成
- `SinglePoint`：单点能计算
- `TDDFT`：激发态计算

---
"""


# ============================================================================
# 辅助函数
# ============================================================================

def read_output_file(file_path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    """
    读取输出文件内容。
    如果文件 > max_bytes（默认 4MB），则截断显示前后各 100KB。
    
    Args:
        file_path: 输出文件路径
        max_bytes: 最大字节数阈值
        
    Returns:
        文件内容字符串
    """
    if not file_path.exists():
        return "⚠️ 输出文件不存在"
    
    file_size = file_path.stat().st_size
    
    if file_size == 0:
        return "⚠️ 输出文件为空"
    
    # 如果文件小于阈值，直接读取全部
    if file_size <= max_bytes:
        try:
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return f"❌ 读取文件失败: {e}"
    
    # 大文件：读取前 100KB + 后 100KB
    try:
        with open(file_path, 'rb') as f:
            head = f.read(100 * 1024).decode('utf-8', errors='ignore')
            f.seek(-100 * 1024, 2)  # 从文件末尾往前 100KB
            tail = f.read().decode('utf-8', errors='ignore')
        
        return (
            f"ℹ️ 文件过大 ({file_size / (1024*1024):.2f} MB)，已截断显示\n"
            f"显示前 100 KB 和后 100 KB\n\n"
            f"{'='*70}\n"
            f"{head}\n\n"
            f"{'='*70}\n"
            f"...[中间部分已截断]...\n"
            f"{'='*70}\n\n"
            f"{tail}"
        )
    except Exception as e:
        return f"❌ 读取文件失败: {e}"


def sanitize_filename(filename: str) -> Tuple[str, bool]:
    """
    检查并修正文件名中的逗号（容错处理）。
    
    Args:
        filename: 原始文件名
        
    Returns:
        (修正后的文件名, 是否进行了修正)
    """
    if ',' in filename:
        corrected = filename.replace(',', '.')
        return corrected, True
    return filename, False


# ============================================================================
# 核心执行函数
# ============================================================================

def run_orca_calculation(
    inp_file: Optional[str],
    orca_bin: str,
    run_dir: str
) -> Iterator[Tuple[str, Optional[str], str, str]]:
    """
    执行 ORCA 计算的主函数（生成器版本，支持实时更新）。
    
    Args:
        inp_file: 上传的 .inp 文件路径（由 Gradio File 组件提供）
        orca_bin: ORCA 可执行文件路径
        run_dir: 运行目录
        
    Yields:
        (日志文本, test.out 路径, test.out 内容, 链接 Markdown)
    """
    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("🚀 ORCA 计算任务开始")
    log_lines.append("=" * 70)
    log_lines.append(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 链接信息（始终返回）
    links_md = """
---
## 进一步分析工具

计算完成后，您可以使用以下工具进行进一步分析：

- **[Multiwfn](http://sobereva.com/multiwfn/)**: 强大的波函数分析程序，支持电荷分析、轨道分析、弱相互作用分析等
- **[VMD](https://www.ks.uiuc.edu/Research/vmd/)**: 分子可视化与动力学分析工具，可查看结构、轨道、振动模式等
"""
    
    # 1. 校验输入文件
    if inp_file is None:
        log_lines.append("❌ 错误：未上传 .inp 文件！")
        log_lines.append("请先上传 ORCA 输入文件后再运行。")
        yield "\n".join(log_lines), None, "", links_md
        return
    
    log_lines.append(f"📄 检测到上传文件: {Path(inp_file).name}")
    yield "\n".join(log_lines), None, "", links_md
    
    # 2. 创建运行目录
    run_path = Path(run_dir).resolve()
    try:
        run_path.mkdir(parents=True, exist_ok=True)
        log_lines.append(f"📁 运行目录: {run_path}")
        yield "\n".join(log_lines), None, "", links_md
    except Exception as e:
        log_lines.append(f"❌ 创建运行目录失败: {e}")
        yield "\n".join(log_lines), None, "", links_md
        return
    
    # 3. 复制输入文件到运行目录
    inp_src = Path(inp_file)
    original_name = inp_src.name
    inp_dest = run_path / original_name
    
    try:
        shutil.copy2(inp_src, inp_dest)
        log_lines.append(f"✅ 已复制输入文件到运行目录: {original_name}")
        yield "\n".join(log_lines), None, "", links_md
    except Exception as e:
        log_lines.append(f"❌ 复制文件失败: {e}")
        yield "\n".join(log_lines), None, "", links_md
        return
    
    # 4. 文件名容错处理（逗号 -> 点）
    runtime_name = original_name
    corrected_name, was_corrected = sanitize_filename(original_name)
    
    if was_corrected:
        log_lines.append(f"⚠️  检测到文件名包含逗号: {original_name}")
        log_lines.append(f"🔧 自动修正为: {corrected_name}")
        # 重命名文件
        corrected_dest = run_path / corrected_name
        try:
            inp_dest.rename(corrected_dest)
            runtime_name = corrected_name
            log_lines.append(f"✅ 文件已重命名为: {corrected_name}")
        except Exception as e:
            log_lines.append(f"⚠️  重命名失败: {e}，将使用原文件名")
            runtime_name = original_name
    else:
        log_lines.append(f"✅ 使用输入文件名: {runtime_name}")
    
    yield "\n".join(log_lines), None, "", links_md
    
    # 5. 构建并执行 ORCA 命令
    output_file = run_path / "test.out"
    cmd_input = runtime_name
    
    # 命令格式: orca <实际文件名> > test.out
    log_lines.append(f"🔧 ORCA 可执行文件: {orca_bin}")
    log_lines.append(f"📝 执行命令: {orca_bin} {cmd_input} > test.out")
    log_lines.append("-" * 70)
    log_lines.append("⏳ 计算进行中，可随时点击'终止计算'按钮停止...")
    yield "\n".join(log_lines), None, "", links_md
    
    start_time = time.time()
    last_update_time = start_time
    
    try:
        # 使用 shell=True 执行重定向命令
        cmd = f'"{orca_bin}" "{cmd_input}" > test.out'
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(run_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 注册进程到管理器
        process_manager.set_process(process)
        
        log_lines.append("🔄 ORCA 进程已启动，等待计算完成...")
        log_lines.append(f"   进程 PID: {process.pid}")
        yield "\n".join(log_lines), None, "", links_md
        
        # 等待进程完成，周期性更新状态
        while process.poll() is None:
            time.sleep(2)  # 每2秒检查一次
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 每5秒更新一次日志
            if current_time - last_update_time >= 5:
                # 尝试读取 test.out 的最后几行
                status_info = f"   运行中... 已耗时: {elapsed:.0f} 秒"
                if output_file.exists():
                    try:
                        file_size = output_file.stat().st_size
                        status_info += f" | test.out 大小: {file_size / 1024:.1f} KB"
                    except:
                        pass
                
                # 更新日志的最后一行或添加新行
                if log_lines[-1].startswith("   运行中..."):
                    log_lines[-1] = status_info
                else:
                    log_lines.append(status_info)
                
                yield "\n".join(log_lines), None, "", links_md
                last_update_time = current_time
            
            # 超时检查（1小时）
            if elapsed > 3600:
                process.kill()
                process.wait()
                log_lines.append(f"❌ 计算超时（超过 1 小时），已终止")
                process_manager.clear_process()
                yield "\n".join(log_lines), None, "", links_md
                return
        
        # 进程结束，获取返回码
        returncode = process.returncode
        process_manager.clear_process()
        
        elapsed_time = time.time() - start_time
        
        # 清除运行状态行
        if log_lines[-1].startswith("   运行中..."):
            log_lines.pop()
        
        log_lines.append("-" * 70)
        
        # 检查返回码
        if returncode == 0:
            log_lines.append(f"✅ ORCA 计算成功完成！")
        elif returncode < 0:
            log_lines.append(f"⚠️  进程被信号终止: {-returncode} (可能是手动终止)")
        else:
            log_lines.append(f"⚠️  ORCA 返回非零退出码: {returncode}")
            log_lines.append(f"   可能存在错误，请检查输出文件。")
        
        log_lines.append(f"⏱️  计算耗时: {elapsed_time:.2f} 秒")
        log_lines.append(f"⏰ 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        yield "\n".join(log_lines), None, "", links_md
        
    except Exception as e:
        process_manager.clear_process()
        log_lines.append(f"❌ 执行失败: {e}")
        yield "\n".join(log_lines), None, "", links_md
        return
    
    # 6. 读取并返回输出文件
    log_lines.append("=" * 70)
    
    if output_file.exists():
        log_lines.append(f"📊 输出文件已生成: {output_file.name}")
        log_lines.append(f"📦 文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        
        output_content = read_output_file(output_file)
        
        yield (
            "\n".join(log_lines),
            str(output_file),
            output_content,
            links_md
        )
    else:
        log_lines.append("❌ 未找到输出文件 test.out")
        yield "\n".join(log_lines), None, "⚠️ 输出文件不存在", links_md


def terminate_calculation() -> str:
    """
    终止当前正在运行的 ORCA 计算。
    
    Returns:
        终止操作的结果消息
    """
    if process_manager.terminate_process():
        return "🛑 已发送终止信号，计算进程正在停止..."
    else:
        return "ℹ️  当前没有正在运行的计算任务"


# ============================================================================
# Gradio 界面构建
# ============================================================================

def build_interface() -> gr.Blocks:
    """构建 Gradio Blocks 界面"""
    
    with gr.Blocks(
        title="ORCA Quantum Chemistry Tool",
        theme=gr.themes.Soft()
    ) as demo:
        
        # 顶部：ORCA 简介与格式说明
        gr.Markdown(ORCA_INTRO_MD)
        
        # 主体：两列布局
        with gr.Row():
            # 左列：输入与设置
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 计算设置")
                
                inp_file = gr.File(
                    label="📂 上传 ORCA 输入文件 (.inp)",
                    file_types=[".inp"],
                    file_count="single"
                )
                
                orca_bin = gr.Textbox(
                    label="🔧 ORCA 可执行文件路径",
                    value="orca",
                    placeholder="例如: /home/zyb/apps/orca-6.0.1/orca",
                    info="默认使用 PATH 中的 'orca'，或指定完整路径（需包含可执行文件名）"
                )
                
                run_dir = gr.Textbox(
                    label="📁 运行目录",
                    value="./orca_run",
                    placeholder="./orca_run",
                    info="计算文件将保存在此目录"
                )
                
                with gr.Row():
                    run_btn = gr.Button(
                        "▶️  Run ORCA",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
                    stop_btn = gr.Button(
                        "🛑 终止计算",
                        variant="stop",
                        size="lg",
                        scale=1
                    )
                
                stop_status = gr.Textbox(
                    label="终止状态",
                    interactive=False,
                    visible=True,
                    lines=1
                )
            
            # 右列：输出与结果
            with gr.Column(scale=1):
                gr.Markdown("## 📊 计算结果")
                
                log_output = gr.Textbox(
                    label="📋 运行日志",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    placeholder="点击 'Run ORCA' 开始计算..."
                )
                
                out_file = gr.File(
                    label="💾 下载 test.out",
                    interactive=False
                )
                
                out_view = gr.Textbox(
                    label="📄 查看 test.out 内容",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
                
                links_output = gr.Markdown("")
        
        # 按钮事件绑定
        run_btn.click(
            fn=run_orca_calculation,
            inputs=[inp_file, orca_bin, run_dir],
            outputs=[log_output, out_file, out_view, links_output]
        )
        
        stop_btn.click(
            fn=terminate_calculation,
            inputs=[],
            outputs=[stop_status]
        )
        
        # 页面底部额外说明
        gr.Markdown("""
---
### 💡 使用提示

1. **ORCA 路径配置**：
   - 如果 `orca` 在环境变量中，直接使用默认值 `orca` 即可
   - 否则需要指定完整路径，例如：`/home/zyb/apps/orca-6.0.1/orca`
   - 注意：路径需包含可执行文件名，而非仅目录
   
2. **实时状态更新**：
   - 运行日志每5秒自动更新，显示运行时间和输出文件大小
   - 进程 PID 会在启动后显示，可用于系统监控
   - 您可以用 `htop` 或 `top` 命令查看 ORCA 进程的 CPU 使用情况

3. **文件名处理**：程序会使用您上传的实际文件名进行计算，如果文件名包含逗号（如 `test,inp`），会自动修正为 `test.inp`

4. **终止计算**：如果计算时间过长或需要修改参数，可随时点击"终止计算"按钮停止运行

5. **并行计算**：在 `.inp` 文件中使用 `%pal nprocs N end` 设置线程数，建议不超过物理核心数

6. **内存设置**：使用 `%maxcore M` 设置每核内存（单位 MB），确保总内存不超过系统可用量

7. **大文件处理**：输出文件 > 4 MB 时会自动截断显示，完整内容可通过下载按钮获取

### ⚠️  注意事项

- 确保系统已正确安装 ORCA 并配置环境变量（或指定完整路径）
- 长时间计算（> 1 小时）将被自动终止
- 界面会实时更新状态，无需刷新页面
- 输出文件始终保存为 `test.out`，方便统一管理

### 🔍 故障排查

如果点击运行后没有反应或 CPU 占用很低：
1. 检查 ORCA 路径是否正确（运行日志会显示执行的命令）
2. 在终端运行相同命令测试：`cd orca_run && /path/to/orca yourfile.inp > test.out`
3. 查看进程 PID，用 `ps aux | grep <PID>` 确认进程是否存在
4. 检查 test.out 文件是否在生成（运行日志会显示文件大小变化）
""")
    
    return demo


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 确保默认运行目录存在
    os.makedirs("./orca_run", exist_ok=True)
    
    # 构建并启动应用
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        inbrowser=False,
        share=False
    )

