# Gradio 示例应用

本目录包含三个 Gradio 应用示例，用于演示如何将计算化学工具集成到 ChemHub 平台。

## 📦 安装依赖

```bash
pip install gradio
```

## 🚀 启动应用

### 方法 1：单独启动（适合开发调试）

```bash
# 终端 1：分子对接应用
python docking_app.py

# 终端 2：分子动力学应用
python md_app.py

# 终端 3：ORCA 量化计算应用
python orca_app.py
```

### 方法 2：使用后台运行

```bash
# Linux/Mac
nohup python docking_app.py > docking.log 2>&1 &
nohup python md_app.py > md.log 2>&1 &
nohup python orca_app.py > orca.log 2>&1 &

# 查看进程
ps aux | grep python

# 停止进程
pkill -f docking_app.py
```

### 方法 3：使用 tmux/screen（推荐）

```bash
# 创建会话
tmux new -s chemhub

# 在 tmux 中启动应用
python docking_app.py

# 退出 tmux（应用继续运行）
# 按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t chemhub

# 列出所有会话
tmux ls
```

## 📋 应用详情

### 1. docking_app.py - 分子对接工具

- **端口**：7861
- **子路径**：`/apps/docking/`
- **功能**：模拟 AutoDock Vina 分子对接流程
- **输入**：蛋白质 PDB 文件 + 配体文件
- **输出**：对接结果（结合能、配体效率等）

**直接访问**：http://localhost:7861

**通过 Nginx 访问**：http://localhost/apps/docking/

### 2. md_app.py - 分子动力学模拟

- **端口**：7862
- **子路径**：`/apps/md/`
- **功能**：模拟 GROMACS/OpenMM 分子动力学模拟
- **输入**：分子结构 + 力场 + 模拟参数
- **输出**：轨迹文件、能量曲线、RMSD 等

**直接访问**：http://localhost:7862

**通过 Nginx 访问**：http://localhost/apps/md/

### 3. orca_app.py - ORCA 量化计算

- **端口**：7863
- **子路径**：`/apps/orca/`
- **功能**：模拟 ORCA 量子化学计算
- **输入**：XYZ 分子坐标 + 计算方法 + 基组
- **输出**：能量、轨道、频率等

**直接访问**：http://localhost:7863

**通过 Nginx 访问**：http://localhost/apps/orca/

## 🔑 关键配置

所有应用都必须设置 `root_path` 参数：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    root_path="/apps/docking"  # ← 关键！
)
```

这确保了：
- 静态资源（JS/CSS）正确加载
- WebSocket 连接路径正确
- 在 Nginx 反向代理后正常工作

## 🔄 集成真实计算引擎

这些示例应用只返回模拟数据。要集成真实计算，需要：

### AutoDock Vina 集成示例

```python
import subprocess

def run_vina(protein_file, ligand_file):
    cmd = [
        "vina",
        "--receptor", protein_file,
        "--ligand", ligand_file,
        "--out", "output.pdbqt",
        "--log", "log.txt"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout
```

### GROMACS 集成示例

```python
import subprocess

def run_gromacs(structure_file, simulation_time):
    # 能量最小化
    subprocess.run(["gmx", "grompp", "-f", "minim.mdp", ...])
    subprocess.run(["gmx", "mdrun", "-deffnm", "em"])
    
    # MD 模拟
    subprocess.run(["gmx", "grompp", "-f", "md.mdp", ...])
    subprocess.run(["gmx", "mdrun", "-deffnm", "md"])
    
    return "模拟完成"
```

### ORCA 集成示例

```python
import subprocess

def run_orca(xyz_coords, method="B3LYP", basis="def2-TZVP"):
    # 生成 ORCA 输入文件
    with open("molecule.inp", "w") as f:
        f.write(f"! {method} {basis} OPT\n")
        f.write("* xyz 0 1\n")
        f.write(xyz_coords)
        f.write("\n*\n")
    
    # 运行 ORCA
    result = subprocess.run(
        ["orca", "molecule.inp"],
        capture_output=True,
        text=True
    )
    
    # 解析输出
    with open("molecule.out", "r") as f:
        return f.read()
```

## 🐛 故障排查

### 应用启动失败

```bash
# 检查端口是否被占用
lsof -i :7861
lsof -i :7862
lsof -i :7863

# 如果被占用，终止进程
kill -9 <PID>
```

### 在 ChemHub 中显示空白

**可能原因**：
1. 应用未启动
2. `root_path` 配置错误
3. Nginx 未配置或未重启

**检查步骤**：
```bash
# 1. 验证应用是否运行
curl http://localhost:7861
curl http://localhost:7862
curl http://localhost:7863

# 2. 验证 Nginx 反代
curl http://localhost/apps/docking/
curl http://localhost/apps/md/
curl http://localhost/apps/orca/

# 3. 查看浏览器控制台错误
# 按 F12 打开开发者工具
```

### WebSocket 连接失败

确保 Nginx 配置了 WebSocket 支持：

```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

## 📝 开发建议

### 添加文件上传功能

```python
def process_file(file):
    if file is None:
        return "未选择文件"
    
    # 读取文件内容
    content = file.read()
    
    # 或获取文件路径
    file_path = file.name
    
    return f"处理文件: {file_path}"

gr.File(label="上传文件", file_types=[".pdb", ".xyz"])
```

### 添加进度条

```python
import time

def long_calculation(input_data, progress=gr.Progress()):
    progress(0, desc="初始化...")
    time.sleep(1)
    
    progress(0.3, desc="读取文件...")
    time.sleep(1)
    
    progress(0.6, desc="计算中...")
    time.sleep(2)
    
    progress(1.0, desc="完成！")
    return "计算结果"
```

### 添加可视化

```python
import matplotlib.pyplot as plt

def plot_results(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy (kJ/mol)")
    return fig

gr.Plot(label="能量曲线")
```

## 🔗 相关资源

- [Gradio 官方文档](https://gradio.app/docs/)
- [Gradio 示例库](https://gradio.app/demos/)
- [Gradio Blocks 教程](https://gradio.app/docs/#blocks)

---

**需要帮助？** 查看主项目 [README.md](../README.md) 或提交 Issue。

