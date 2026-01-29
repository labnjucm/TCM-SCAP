# GROMACS GUI Runner

一个基于 Gradio 的 GUI 应用，用于在同一 bash 进程中执行 GROMACS 脚本，并在 `gmx pdb2gmx` 命令后自动暂停执行 Python Hook 来修改本地文件。

## 功能特性

- ✅ **单进程执行**: 在同一个 bash 会话中执行整个脚本，保持环境变量和上下文
- ✅ **智能暂停**: 自动检测 `gmx pdb2gmx` 命令，执行后暂停并运行 Hook
- ✅ **文件修改 Hook**: 自动合并 GRO 文件并修补 topol.top
- ✅ **实时日志**: 流式显示 stdout/stderr 输出
- ✅ **进程控制**: 支持停止运行中的脚本
- ✅ **自动打包**: 执行完成后自动打包工作目录为 ZIP

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动应用：

```bash
python gmx_gui_runner.py
```

启动后会显示访问地址：
- **本地访问**: `http://127.0.0.1:7860`
- **局域网访问**: `http://<本机IP>:7860` (同网络设备可访问)
- **外网访问**: 需要配置防火墙和端口转发

2. 在浏览器中打开对应的地址

3. 上传必需文件：
   - protein.pdb
   - ligand.itp
   - ligand.gro
   - ions.mdp, em1.mdp, em2.mdp, nvt.mdp, npt.mdp, md.mdp
   - Shell 脚本 (如 run.sh)

4. （可选）设置环境变量 JSON，例如：
   ```json
   {"GMX_GPU_ID": "0"}
   ```

5. 点击 "Run Script" 开始执行

## Hook 功能

应用会在 `gmx pdb2gmx` 命令执行完成后自动执行以下操作：

1. **merge_gro**: 合并 `protein_processed.gro` 和 `ligand.gro`
   - 将配体坐标追加到蛋白坐标后
   - 更新原子计数
   - 保留蛋白的 box 信息

2. **patch_topol_top**: 修改 `topol.top`
   - 将 `ligand.itp` 内容插入到 forcefield.itp 之后
   - 确保 `[ molecules ]` 段落包含 `MOL 1`

## 技术实现

- 使用逻辑行解析处理反斜杠续行
- 通过哨兵行 `__AFTER_PDB2GMX__` 检测命令执行完成
- 单一 bash 进程，通过 stdin 逐行输入命令
- 实时读取 stdout/stderr 并显示

## 注意事项

- 脚本中的交互式输入应通过管道提供，例如：`(echo "6"; echo "5") | gmx pdb2gmx ...`
- 工作目录为 `./gmx_run/`
- 执行日志保存在 `gmx_run/run.log`
- 同一时间只能运行一个脚本
- 脚本执行完成后会自动终止进程并刷新状态，可以立即再次运行
- 关键产物文件会自动识别所有 `md_0_1*` 文件（包括 .tpr, .log, .edr, .trr, .xtc, .cpt 等）

