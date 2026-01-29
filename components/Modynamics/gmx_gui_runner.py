#!/usr/bin/env python3
"""
GROMACS GUI Runner with Hook Support
单文件 Gradio 应用，在同一 bash 进程中执行脚本，并在 gmx pdb2gmx 后自动暂停执行 Hook
"""

import os
import re
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List
import json
import signal

import gradio as gr


# ============================================================================
# 脚本解析模块：逻辑行解析和 gmx pdb2gmx 定位
# ============================================================================

def normalize_line_endings(text: str) -> str:
    """CRLF -> LF 规范化"""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def parse_logical_lines(script_text: str) -> List[str]:
    """
    将脚本解析为逻辑行（处理反斜杠续行）
    返回逻辑行列表，每个逻辑行是完整的命令
    """
    normalized = normalize_line_endings(script_text)
    lines = normalized.split('\n')
    
    logical_lines = []
    current_logical = []
    
    for line in lines:
        # 检查是否以反斜杠结尾（续行）
        if line.rstrip().endswith('\\'):
            # 去掉反斜杠，添加到当前逻辑行
            current_logical.append(line.rstrip()[:-1])
        else:
            # 完整的逻辑行
            current_logical.append(line)
            logical_lines.append('\n'.join(current_logical))
            current_logical = []
    
    # 处理最后可能未完成的逻辑行
    if current_logical:
        logical_lines.append('\n'.join(current_logical))
    
    return logical_lines


def find_pdb2gmx_line(logical_lines: List[str]) -> Optional[int]:
    """
    找到第一个包含 'gmx pdb2gmx' 的逻辑行索引
    允许前面有管道、括号等，如 (echo "6"; echo "5") | gmx pdb2gmx ...
    """
    pattern = re.compile(r'\bgmx\s+pdb2gmx\b', re.IGNORECASE)
    
    for idx, line in enumerate(logical_lines):
        if pattern.search(line):
            return idx
    
    return None


def split_script_at_pdb2gmx(script_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    将脚本分为三部分：prefix, target, suffix
    返回 (prefix, target, suffix) 或 (None, None, None) 如果找不到 pdb2gmx
    """
    logical_lines = parse_logical_lines(script_text)
    pdb2gmx_idx = find_pdb2gmx_line(logical_lines)
    
    if pdb2gmx_idx is None:
        return None, None, None
    
    prefix_lines = logical_lines[:pdb2gmx_idx]
    target_line = logical_lines[pdb2gmx_idx]
    suffix_lines = logical_lines[pdb2gmx_idx + 1:]
    
    prefix = '\n'.join(prefix_lines) if prefix_lines else ''
    suffix = '\n'.join(suffix_lines) if suffix_lines else ''
    
    return prefix, target_line, suffix


# ============================================================================
# Python Hook 函数：文件修改
# ============================================================================

def merge_gro(protein_processed_gro: str, ligand_gro: str, output_gro: str) -> None:
    """
    合并蛋白和配体的 .gro 文件
    - 读取两个文件
    - 将配体坐标追加到蛋白坐标后
    - 更新原子计数为两者之和
    - 保留蛋白文件的 box 行
    """
    # 读取蛋白文件
    with open(protein_processed_gro, 'r') as f:
        protein_lines = f.readlines()
    
    # 读取配体文件
    with open(ligand_gro, 'r') as f:
        ligand_lines = f.readlines()
    
    if len(protein_lines) < 3 or len(ligand_lines) < 3:
        raise ValueError("GRO 文件格式不正确，至少需要 3 行")
    
    # 解析蛋白文件
    protein_title = protein_lines[0]
    protein_natoms = int(protein_lines[1].strip())
    protein_coords = protein_lines[2:-1]  # 去掉标题、计数和 box 行
    protein_box = protein_lines[-1]
    
    # 解析配体文件
    ligand_natoms = int(ligand_lines[1].strip())
    ligand_coords = ligand_lines[2:-1]  # 去掉标题、计数和 box 行
    
    # 宽容处理：如果声明的原子数与实际不符，使用实际行数
    actual_protein = len(protein_coords)
    actual_ligand = len(ligand_coords)
    
    if actual_protein < protein_natoms:
        protein_coords = protein_coords[:actual_protein]
        protein_natoms = actual_protein
    
    if actual_ligand < ligand_natoms:
        ligand_coords = ligand_coords[:actual_ligand]
        ligand_natoms = actual_ligand
    
    # 合并
    total_atoms = protein_natoms + ligand_natoms
    merged_coords = protein_coords + ligand_coords
    
    # 写入输出文件
    with open(output_gro, 'w') as f:
        f.write(protein_title)
        f.write(f"{total_atoms:5d}\n")
        for coord_line in merged_coords:
            f.write(coord_line)
        f.write(protein_box)


def patch_topol_top(topol_path: str, ligand_itp_path: str) -> None:
    """
    修改 topol.top 文件
    1. 将 ligand.itp 的第 3 行起的内容插入到 #include "...forcefield.itp" 之后
    2. 确保 [ molecules ] 段落存在且包含 'MOL    1'
    """
    # 读取 ligand.itp
    with open(ligand_itp_path, 'r') as f:
        ligand_lines = f.readlines()
    
    # 从第 3 行起提取内容（索引 2 开始）
    if len(ligand_lines) <= 2:
        ligand_content = []
    else:
        ligand_content = ligand_lines[2:]
    
    # 读取 topol.top
    with open(topol_path, 'r') as f:
        topol_lines = f.readlines()
    
    # 找到 forcefield.itp 的 include 行
    forcefield_idx = None
    first_include_idx = None
    
    for idx, line in enumerate(topol_lines):
        if '#include' in line.lower():
            if first_include_idx is None:
                first_include_idx = idx
            if 'forcefield.itp' in line.lower():
                forcefield_idx = idx
                break
    
    # 确定插入位置
    if forcefield_idx is not None:
        insert_idx = forcefield_idx + 1
    elif first_include_idx is not None:
        insert_idx = first_include_idx + 1
    else:
        insert_idx = 0
    
    # 插入 ligand.itp 内容（前后各留一空行）
    insertion = ['\n'] + ligand_content + ['\n']
    topol_lines = topol_lines[:insert_idx] + insertion + topol_lines[insert_idx:]
    
    # 确保 [ molecules ] 段落存在且包含 MOL 1
    molecules_idx = None
    for idx, line in enumerate(topol_lines):
        if re.match(r'^\s*\[\s*molecules\s*\]', line, re.IGNORECASE):
            molecules_idx = idx
            break
    
    if molecules_idx is None:
        # 创建 [ molecules ] 段落
        topol_lines.append('\n')
        topol_lines.append('[ molecules ]\n')
        topol_lines.append('; Compound        #mols\n')
        topol_lines.append('MOL    1\n')
    else:
        # 检查是否已有 MOL 条目
        has_mol = False
        for idx in range(molecules_idx + 1, len(topol_lines)):
            line = topol_lines[idx]
            # 遇到下一个段落就停止
            if re.match(r'^\s*\[', line):
                break
            if re.match(r'^\s*MOL\s+', line, re.IGNORECASE):
                has_mol = True
                break
        
        if not has_mol:
            # 在 [ molecules ] 段落后添加 MOL 1
            # 找到段落结束或下一个段落的位置
            insert_mol_idx = molecules_idx + 1
            for idx in range(molecules_idx + 1, len(topol_lines)):
                if re.match(r'^\s*\[', topol_lines[idx]):
                    insert_mol_idx = idx
                    break
                insert_mol_idx = idx + 1
            
            topol_lines.insert(insert_mol_idx, 'MOL    1\n')
    
    # 写回 topol.top
    with open(topol_path, 'w') as f:
        f.writelines(topol_lines)


def execute_hook(workdir: str, log_callback) -> bool:
    """
    执行 Python Hook：调用 merge_gro 和 patch_topol_top
    返回是否成功
    """
    try:
        log_callback("\n" + "="*60 + "\n")
        log_callback("🔧 执行 Python Hook: 修改本地文件\n")
        log_callback("="*60 + "\n")
        
        # 文件路径
        protein_gro = os.path.join(workdir, 'protein_processed.gro')
        ligand_gro = os.path.join(workdir, 'ligand.gro')
        topol_top = os.path.join(workdir, 'topol.top')
        ligand_itp = os.path.join(workdir, 'ligand.itp')
        
        # 执行 merge_gro
        log_callback(f"📝 合并 GRO 文件: {protein_gro} + {ligand_gro}\n")
        merge_gro(protein_gro, ligand_gro, protein_gro)
        log_callback("✓ merge_gro 完成\n")
        
        # 执行 patch_topol_top
        log_callback(f"📝 修补 topol.top: {topol_top}\n")
        patch_topol_top(topol_top, ligand_itp)
        log_callback("✓ patch_topol_top 完成\n")
        
        log_callback("="*60 + "\n")
        log_callback("✓ Hook 执行成功，继续执行脚本剩余部分\n")
        log_callback("="*60 + "\n\n")
        
        return True
        
    except Exception as e:
        log_callback(f"\n❌ Hook 执行失败: {str(e)}\n\n")
        return False


# ============================================================================
# Bash 进程管理：单进程执行、哨兵检测、流式输出
# ============================================================================

class BashRunner:
    """管理单一 bash 进程的执行"""
    
    def __init__(self, workdir: str, env: dict):
        self.workdir = workdir
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.log_lines = []
        self.log_file = None
        self.is_running = False
        self.sentinel = "__AFTER_PDB2GMX__"
        
    def start_process(self):
        """启动 bash 进程"""
        self.process = subprocess.Popen(
            ["/bin/bash", "-Eeuo", "pipefail"],
            cwd=self.workdir,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.is_running = True
        
        # 打开日志文件
        log_path = os.path.join(self.workdir, 'run.log')
        self.log_file = open(log_path, 'w', buffering=1)
        
    def write_command(self, command: str):
        """向 bash 进程写入命令"""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
    
    def read_until_sentinel(self, log_callback) -> bool:
        """读取输出直到遇到哨兵行，返回是否成功"""
        if not self.process or not self.process.stdout:
            return False
        
        try:
            for line in self.process.stdout:
                line = line.rstrip('\n')
                
                # 检查是否是哨兵行
                if self.sentinel in line:
                    # 不输出哨兵行本身
                    return True
                
                # 输出其他行
                log_callback(line + '\n')
                self.log_lines.append(line)
                if self.log_file:
                    self.log_file.write(line + '\n')
                
                # 检查进程是否已退出
                if self.process.poll() is not None:
                    break
            
            return False
            
        except Exception as e:
            log_callback(f"\n❌ 读取输出错误: {str(e)}\n")
            return False
    
    def read_remaining(self, log_callback):
        """读取剩余的所有输出"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            for line in self.process.stdout:
                line = line.rstrip('\n')
                log_callback(line + '\n')
                self.log_lines.append(line)
                if self.log_file:
                    self.log_file.write(line + '\n')
        except Exception as e:
            log_callback(f"\n❌ 读取输出错误: {str(e)}\n")
    
    def stop(self, log_callback):
        """停止 bash 进程"""
        if not self.process:
            return
        
        try:
            log_callback("\n⚠️  正在停止进程...\n")
            
            # 先发送 SIGINT
            self.process.send_signal(signal.SIGINT)
            
            # 等待 5 秒
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 超时则发送 SIGTERM
                log_callback("⚠️  SIGINT 超时，发送 SIGTERM...\n")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 再超时则强制杀死
                    log_callback("⚠️  SIGTERM 超时，强制杀死进程...\n")
                    self.process.kill()
                    self.process.wait()
            
            log_callback("✓ 进程已停止\n")
            
        except Exception as e:
            log_callback(f"❌ 停止进程错误: {str(e)}\n")
        
        finally:
            self.is_running = False
            if self.log_file:
                self.log_file.close()
                self.log_file = None
    
    def wait_and_close(self, log_callback):
        """等待进程结束并关闭"""
        if not self.process:
            return 0
        
        try:
            # 关闭 stdin 以通知进程输入结束
            if self.process.stdin:
                self.process.stdin.close()
            
            # 等待进程结束
            return_code = self.process.wait()
            
            # 记录返回码
            if return_code == 0:
                log_callback(f"\n✓ 脚本执行完成 (exit code: {return_code})\n")
            else:
                log_callback(f"\n❌ 脚本执行失败 FAILED (exit {return_code})\n")
            
            return return_code
            
        except Exception as e:
            log_callback(f"\n❌ 等待进程错误: {str(e)}\n")
            return -1
        
        finally:
            self.is_running = False
            if self.log_file:
                self.log_file.close()
                self.log_file = None


def run_script_with_hook(
    script_text: str,
    workdir: str,
    env: dict,
    log_callback,
    runner_state: dict
) -> bool:
    """
    运行脚本并在 pdb2gmx 后执行 Hook
    返回是否成功
    """
    # 分割脚本
    prefix, target, suffix = split_script_at_pdb2gmx(script_text)
    
    if prefix is None:
        log_callback("⚠️  警告: 脚本中未找到 'gmx pdb2gmx' 命令，将直接执行整个脚本\n\n")
        # 直接执行整个脚本
        runner = BashRunner(workdir, env)
        runner_state['runner'] = runner
        
        try:
            runner.start_process()
            runner.write_command(script_text)
            runner.read_remaining(log_callback)
            runner.wait_and_close(log_callback)
            return True
        except Exception as e:
            log_callback(f"\n❌ 执行脚本错误: {str(e)}\n")
            return False
    
    # 找到了 pdb2gmx，执行分段流程
    log_callback(f"✓ 检测到 gmx pdb2gmx 命令，将在其后执行 Hook\n\n")
    
    runner = BashRunner(workdir, env)
    runner_state['runner'] = runner
    
    try:
        # 启动进程
        runner.start_process()
        
        # 执行 prefix 部分
        if prefix.strip():
            log_callback("="*60 + "\n")
            log_callback("阶段 1: 执行 pdb2gmx 之前的命令\n")
            log_callback("="*60 + "\n")
            runner.write_command(prefix)
        
        # 执行 target 行并插入哨兵
        log_callback("\n" + "="*60 + "\n")
        log_callback("阶段 2: 执行 gmx pdb2gmx 命令\n")
        log_callback("="*60 + "\n")
        runner.write_command(target)
        runner.write_command(f'echo "{runner.sentinel}"')
        
        # 读取直到哨兵
        if not runner.read_until_sentinel(log_callback):
            log_callback("\n❌ 未检测到哨兵行，pdb2gmx 可能未成功执行\n")
            runner.wait_and_close(log_callback)
            return False
        
        # 执行 Hook
        if not execute_hook(workdir, log_callback):
            log_callback("\n❌ Hook 执行失败，中止脚本执行\n")
            runner.stop(log_callback)
            return False
        
        # 执行 suffix 部分
        if suffix.strip():
            log_callback("="*60 + "\n")
            log_callback("阶段 3: 执行 pdb2gmx 之后的命令\n")
            log_callback("="*60 + "\n")
            runner.write_command(suffix)
        
        # 读取剩余输出并等待结束
        runner.read_remaining(log_callback)
        runner.wait_and_close(log_callback)
        
        return True
        
    except Exception as e:
        log_callback(f"\n❌ 执行脚本错误: {str(e)}\n")
        runner.stop(log_callback)
        return False


# ============================================================================
# 文件管理：打包、关键产物展示
# ============================================================================

def create_workspace_zip(workdir: str, output_zip: str) -> bool:
    """打包工作目录为 ZIP"""
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            workdir_path = Path(workdir)
            for file_path in workdir_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(workdir_path.parent)
                    zipf.write(file_path, arcname)
        return True
    except Exception as e:
        print(f"打包失败: {e}")
        return False


def get_key_output_files(workdir: str) -> List[str]:
    """获取关键产物文件列表"""
    workdir_path = Path(workdir)
    found_files = []
    
    # 单个文件模式
    single_patterns = [
        'topol.top',
        'index.ndx',
        'em.gro',
        'nvt.gro',
        'npt.gro',
        'protein_processed.gro'
    ]
    
    # 匹配单个文件
    for pattern in single_patterns:
        matches = list(workdir_path.glob(pattern))
        found_files.extend([str(f) for f in matches if f.is_file()])
    
    # 匹配所有 md_0_1 开头的文件（不同后缀）
    for file_path in workdir_path.iterdir():
        if file_path.is_file() and file_path.name.startswith('md_0_1'):
            found_files.append(str(file_path))
    
    return sorted(found_files)


# ============================================================================
# Gradio 界面
# ============================================================================

# 全局状态
REQUIRED_FILES = [
    'protein.pdb',
    'ligand.itp',
    'ligand.gro',
    'ions.mdp',
    'em1.mdp',
    'em2.mdp',
    'nvt.mdp',
    'npt.mdp',
    'md.mdp'
]

OPTIONAL_FILES = [
    'ligand.top'
]


def validate_files(file_dict: dict) -> Tuple[bool, List[str]]:
    """
    验证必需文件是否都已上传
    返回 (是否全部上传, 缺失文件列表)
    """
    missing = []
    for req_file in REQUIRED_FILES:
        if req_file not in file_dict or file_dict[req_file] is None:
            missing.append(req_file)
    return len(missing) == 0, missing


def prepare_workspace(file_dict: dict, script_file, workdir: str) -> Tuple[bool, str]:
    """
    准备工作目录
    返回 (是否成功, 消息)
    """
    try:
        # 创建工作目录
        os.makedirs(workdir, exist_ok=True)
        
        # 复制所有文件
        for filename, filepath in file_dict.items():
            if filepath is not None:
                import shutil
                dest = os.path.join(workdir, filename)
                shutil.copy2(filepath, dest)
        
        # 处理脚本文件
        if script_file is None:
            return False, "未上传脚本文件"
        
        with open(script_file, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # CRLF -> LF 规范化
        script_content = normalize_line_endings(script_content)
        
        # 写入 run.sh
        run_sh_path = os.path.join(workdir, 'run.sh')
        with open(run_sh_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # chmod +x
        os.chmod(run_sh_path, 0o755)
        
        return True, "工作目录准备完成"
        
    except Exception as e:
        return False, f"准备工作目录失败: {str(e)}"


def run_button_click(
    protein_pdb, ligand_itp, ligand_gro,
    ions_mdp, em1_mdp, em2_mdp, nvt_mdp, npt_mdp, md_mdp,
    ligand_top, script_file, env_json,
    runner_state
):
    """Run 按钮点击处理"""
    
    # 检查是否已有进程在运行
    if runner_state.get('runner') and runner_state['runner'].is_running:
        yield "⚠️  已有脚本正在运行，请等待完成或先停止\n", None, None
        return
    
    # 构建文件字典
    file_dict = {
        'protein.pdb': protein_pdb,
        'ligand.itp': ligand_itp,
        'ligand.gro': ligand_gro,
        'ions.mdp': ions_mdp,
        'em1.mdp': em1_mdp,
        'em2.mdp': em2_mdp,
        'nvt.mdp': nvt_mdp,
        'npt.mdp': npt_mdp,
        'md.mdp': md_mdp,
        'ligand.top': ligand_top
    }
    
    # 验证必需文件
    is_valid, missing_files = validate_files(file_dict)
    if not is_valid:
        msg = "❌ 缺失必需文件:\n" + "\n".join(f"  - {f}" for f in missing_files) + "\n\n请上传所有必需文件后重试\n"
        yield msg, None, None
        return
    
    # 解析环境变量
    env = os.environ.copy()
    if env_json and env_json.strip():
        try:
            custom_env = json.loads(env_json)
            if not isinstance(custom_env, dict):
                yield "❌ 环境变量 JSON 格式错误: 必须是对象\n", None, None
                return
            env.update(custom_env)
            yield f"✓ 已加载自定义环境变量: {list(custom_env.keys())}\n\n", None, None
        except json.JSONDecodeError as e:
            yield f"❌ 环境变量 JSON 解析失败: {str(e)}\n", None, None
            return
    
    # 准备工作目录
    workdir = './gmx_run'
    success, msg = prepare_workspace(file_dict, script_file, workdir)
    if not success:
        yield f"❌ {msg}\n", None, None
        return
    
    yield f"✓ {msg}\n\n", None, None
    
    # 读取脚本内容
    run_sh_path = os.path.join(workdir, 'run.sh')
    with open(run_sh_path, 'r') as f:
        script_text = f.read()
    
    # 累积日志
    log_buffer = [f"✓ {msg}\n\n"]
    
    def log_callback(line):
        log_buffer.append(line)
    
    # 执行脚本
    yield "开始执行脚本...\n\n", None, None
    
    # 在线程中执行，以便实时更新
    result_container = {'success': False}
    
    def run_thread():
        result_container['success'] = run_script_with_hook(
            script_text, workdir, env, log_callback, runner_state
        )
    
    thread = threading.Thread(target=run_thread)
    thread.start()
    
    # 实时输出日志
    last_len = 0
    while thread.is_alive():
        if len(log_buffer) > last_len:
            full_log = ''.join(log_buffer)
            yield full_log, None, None
            last_len = len(log_buffer)
        time.sleep(0.1)
    
    thread.join()
    
    # 最后一次输出完整日志
    full_log = ''.join(log_buffer)
    
    # 清理运行状态，允许下次执行
    if 'runner' in runner_state:
        runner_state['runner'] = None
    
    full_log += "\n" + "="*60 + "\n"
    full_log += "✓ 进程已终止，状态已刷新，可以再次运行\n"
    full_log += "="*60 + "\n"
    
    # 打包 workspace
    zip_path = os.path.join(workdir, 'workspace.zip')
    if create_workspace_zip(workdir, zip_path):
        full_log += "\n✓ 已生成 workspace.zip\n"
    
    # 获取关键产物文件
    key_files = get_key_output_files(workdir)
    
    yield full_log, zip_path, key_files if key_files else None


def stop_button_click(runner_state):
    """Stop 按钮点击处理"""
    runner = runner_state.get('runner')
    if not runner or not runner.is_running:
        return "⚠️  没有正在运行的进程\n"
    
    log_buffer = []
    def log_callback(line):
        log_buffer.append(line)
    
    runner.stop(log_callback)
    
    # 清理运行状态
    runner_state['runner'] = None
    log_buffer.append("\n✓ 进程已终止，状态已刷新\n")
    
    return ''.join(log_buffer)


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="GROMACS GUI Runner", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧬 分子动力学模拟模块")
        gr.Markdown(
            "在同一 bash 进程中执行 GROMACS 脚本，并在 `gmx pdb2gmx` 后自动执行 Python Hook 修改文件"
        )
        
        # 全局状态
        runner_state = gr.State(value={})
        
        with gr.Row():
            # 左列：文件上传和参数
            with gr.Column(scale=1):
                gr.Markdown("### 📁 必需文件")
                
                protein_pdb = gr.File(label="protein.pdb", file_types=[".pdb"])
                ligand_itp = gr.File(label="ligand.itp", file_types=[".itp"])
                ligand_gro = gr.File(label="ligand.gro", file_types=[".gro"])
                
                with gr.Row():
                    ions_mdp = gr.File(label="ions.mdp", file_types=[".mdp"])
                    em1_mdp = gr.File(label="em1.mdp", file_types=[".mdp"])
                
                with gr.Row():
                    em2_mdp = gr.File(label="em2.mdp", file_types=[".mdp"])
                    nvt_mdp = gr.File(label="nvt.mdp", file_types=[".mdp"])
                
                with gr.Row():
                    npt_mdp = gr.File(label="npt.mdp", file_types=[".mdp"])
                    md_mdp = gr.File(label="md.mdp", file_types=[".mdp"])
                
                gr.Markdown("### 📄 可选文件")
                ligand_top = gr.File(label="ligand.top (可选)", file_types=[".top"])
                
                gr.Markdown("### 📜 Shell 脚本")
                script_file = gr.File(label="Shell 脚本 (任意名)", file_types=[".sh", ".bash"])
                
                gr.Markdown("### ⚙️ 环境变量 (JSON)")
                env_json = gr.Textbox(
                    label="环境变量",
                    placeholder='{"GMX_GPU_ID": "0"}',
                    lines=3
                )
                
                gr.Markdown("### 🎮 控制")
                with gr.Row():
                    run_btn = gr.Button("▶️  Run Script", variant="primary")
                    stop_btn = gr.Button("⏹️  Stop", variant="stop")
            
            # 右列：日志和输出
            with gr.Column(scale=2):
                gr.Markdown("### 📊 实时日志")
                log_output = gr.Textbox(
                    label="执行日志",
                    lines=25,
                    max_lines=25,
                    interactive=False,
                    autoscroll=True
                )
                
                gr.Markdown("### 📦 下载")
                zip_output = gr.File(label="workspace.zip")
                
                gr.Markdown("### 🎯 关键产物")
                key_files_output = gr.Files(label="关键产物文件")
        
        # 按钮事件
        run_btn.click(
            fn=run_button_click,
            inputs=[
                protein_pdb, ligand_itp, ligand_gro,
                ions_mdp, em1_mdp, em2_mdp, nvt_mdp, npt_mdp, md_mdp,
                ligand_top, script_file, env_json,
                runner_state
            ],
            outputs=[log_output, zip_output, key_files_output]
        )
        
        stop_btn.click(
            fn=stop_button_click,
            inputs=[runner_state],
            outputs=[log_output]
        )
    
    return demo


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import socket
    
    # 获取本机 IP 地址
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    local_ip = get_local_ip()
    
    print("="*60)
    print("🧬 GROMACS GUI Runner 启动中...")
    print("="*60)
    print(f"本地访问: http://127.0.0.1:7862")
    print(f"局域网访问: http://{local_ip}:7862")
    print(f"外网访问: 确保防火墙允许端口 7860")
    print("="*60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # 监听所有网络接口，允许外部访问
        server_port=7862,
        share=False,  # 如需 Gradio 公网链接，设为 True
        show_error=True,
        inbrowser=False  # 不自动打开浏览器
    )

