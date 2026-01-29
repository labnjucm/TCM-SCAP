# Smina 详细使用指南

## 目录
1. [什么是 Smina?](#1-什么是-smina)
2. [Smina vs. AutoDock Vina: 主要优势](#2-smina-vs-autodock-vina-主要优势)
3. [安装](#3-安装)
4. [基础使用 (与 Vina 兼容)](#4-基础使用-与-vina-兼容)
    - [4.1 输入文件准备 (PDBQT)](#41-输入文件准备-pdbqt)
    - [4.2 运行对接](#42-运行对接)
5. [Smina 的核心高级功能](#5-smina-的核心高级功能)
    - [5.1 自定义打分函数 (`--custom_scoring`)](#51-自定义打分函数---custom_scoring)
    - [5.2 基于配体的对接盒子自动生成 (`--autobox_ligand`)](#52-基于配体的对接盒子自动生成---autobox_ligand)
    - [53 能量最小化 (`--minimize`)](#53-能量最小化---minimize)
    - [5.4 原子贡献得分注释 (`--atom_terms`)](#54-原子贡献得分注释---atom_terms)
    - [5.5 创建自定义打分项 (`--custom_terms`)](#55-创建自定义打分项---custom_terms)
6. [完整流程示例：使用 Smina 的高级功能](#6-完整流程示例使用-smina-的高级功能)
    - [步骤一：准备文件](#步骤一准备文件)
    - [步骤二：创建自定义打分函数文件 (可选)](#步骤二创建自定义打分函数文件-可选)
    - [步骤三：使用命令行运行 Smina](#步骤三使用命令行运行-smina)
    - [步骤四：分析结果](#步骤四分析结果)
7. [结果解读](#7-结果解读)
    - [7.1 日志文件 (Log File)](#71-日志文件-log-file)
    - [7.2 输出结构文件 (`out.pdbqt`)](#72-输出结构文件-outpdbqt)
    - [7.3 分析原子贡献得分](#73-分析原子贡献得分)
8. [常用命令行参数速查表](#8-常用命令行参数速查表)
9. [常见问题与解决方案 (FAQ)](#9-常见问题与解决方案-faq)

---

## 1. 什么是 Smina?

**Smina** 是由 David Koes 开发的分子对接程序，它是著名的 **AutoDock Vina** 的一个分支（fork）。Smina 不仅完整保留了 Vina 的所有功能和高性能，还在此基础上进行了大量改进和功能扩展，使其在灵活性、可定制性和易用性方面更胜一筹。

你可以将 Smina 视为 **"Vina 的超集"** 或 **"Vina Pro"**。对于 Vina 用户来说，迁移到 Smina 是无缝的，因为它可以作为 Vina 的**直接替代品 (drop-in replacement)**。

## 2. Smina vs. AutoDock Vina: 主要优势

Smina 继承了 Vina 的速度和准确性，并增加了以下关键特性：

-   **自定义打分函数 (Custom Scoring Functions)**：允许用户通过简单的文本文件修改、加权甚至创建全新的打分项，这对于开发针对特定靶标家族的打分函数非常有用。
-   **改进的最小化算法**：提供更强大的局部能量最小化功能，可用于优化对接后的构象或任何给定的配体姿态。
-   **对接盒子自动生成**：可以根据一个参考配体文件自动生成对接盒子（search box），极大地简化了对接前的设置工作。
-   **原子能量贡献注释**：能够将每个原子的能量贡献值写入输出的 PDBQT 文件的 B-factor 列，方便进行可视化分析，快速识别关键的相互作用原子。
-   **支持创建自定义打分项**：除了修改现有项，还可以引入基于原子属性的新打分项。
-   **性能优化**：在某些情况下，Smina 的运行速度比原始 Vina 更快。

## 3. 安装

推荐使用 `conda` 进行安装，这是最简单快捷的方式。

```bash
conda install -c conda-forge smina
```

安装完成后，你就可以在命令行中直接使用 `smina` 命令了。

你也可以从其 GitHub 仓库下载源码并自行编译：
> [https://github.com/dkoes/smina]

## 4. 基础使用 (与 Vina 兼容)

Smina 的基础用法与 Vina 完全相同。如果你熟悉 Vina，可以跳过此节。

### 4.1 输入文件准备 (PDBQT)

Smina 同样使用 **PDBQT** 格式的输入文件。你需要使用 **MGLTools (ADT)** 或 **Open Babel** 等工具来准备受体（receptor）和配体（ligand）的 PDBQT 文件。

-   **受体 (`receptor.pdbqt`)**: 刚性部分。
-   **配体 (`ligand.pdbqt`)**: 柔性部分。

这个过程与为 Vina 准备文件的过程完全一致。

### 4.2 运行对接

你可以像使用 Vina 一样，通过配置文件或纯命令行参数来运行 Smina。

#### 方法一：使用配置文件 (与 Vina 相同)

创建 `config.txt`:
```ini
receptor = receptor.pdbqt
ligand = ligand.pdbqt
out = docked_poses.pdbqt

center_x = 10.5
center_y = 20.2
center_z = -5.8

size_x = 25
size_y = 25
size_z = 25

exhaustiveness = 8
```

然后运行：
```bash
smina --config config.txt --log results.log
```

#### 方法二：使用纯命令行参数

Smina 非常适合在命令行中直接指定所有参数，这在编写脚本时特别方便。

```bash
smina -r receptor.pdbqt -l ligand.pdbqt --center_x 10.5 --center_y 20.2 --center_z -5.8 --size_x 25 --size_y 25 --size_z 25 --exhaustiveness 8 -o docked_poses.pdbqt --log results.log
```

## 5. Smina 的核心高级功能

这是 Smina 真正强大的地方。

### 5.1 自定义打分函数 (`--custom_scoring`)

你可以创建一个文本文件（例如 `my_sf.txt`），在其中定义打分函数的权重。Smina 默认的打分函数（与 Vina 类似）包含以下几项：

-   `gauss1` (高斯吸引项)
-   `gauss2` (高斯吸引项)
-   `repulsion` (排斥项)
-   `hydrophobic` (疏水项)
-   `hydrogen` (氢键项)
-   `num_rotors` (配体可旋转键的惩罚项)

一个自定义打分函数文件的例子 `my_sf.txt`：
```text
# 这是一个示例，将疏水项的权重加倍，并稍微增加氢键权重
# 格式: weight term_name [optional_parameters]
weight_gauss1 -0.035579
weight_gauss2 -0.005156
weight_repulsion 0.840245
weight_hydrophobic 0.110352 # 原始值是 -0.055176，我们乘以-2
weight_hydrogen -0.65 # 原始值是 -0.587439，我们稍微增强
weight_rot -0.035069
```
**使用方法**:
```bash
smina -r receptor.pdbqt -l ligand.pdbqt ... --custom_scoring my_sf.txt
```

### 5.2 基于配体的对接盒子自动生成 (`--autobox_ligand`)

如果你有一个已知的结合物（例如，PDB 中的共晶配体），你不再需要手动测量盒子中心和尺寸。Smina 可以为你自动完成。

假设 `reference_ligand.pdbqt` 是你的参考配体。

```bash
smina -r receptor.pdbqt -l new_ligand.pdbqt --autobox_ligand reference_ligand.pdbqt --autobox_add 8 -o output.pdbqt
```

-   `--autobox_ligand reference_ligand.pdbqt`: Smina 会计算 `reference_ligand.pdbqt` 的几何中心作为对接盒子的中心，并设置一个刚好能包裹住它的盒子尺寸。
-   `--autobox_add 8`: 在自动生成的盒子尺寸的每个维度上再增加 8 埃（Å）。这为新的、可能更大的配体提供了足够的搜索空间。推荐设置此项。

### 5.3 能量最小化 (`--minimize`)

Smina 可以对给定的配体构象进行局部能量最小化，以优化其几何构象并获得更精确的打分。这对于优化实验结构或其他程序生成的对接姿态非常有用。

```bash
smina -r receptor.pdbqt -l initial_pose.pdbqt --minimize -o minimized_pose.pdbqt --log minimize.log
```
-   `--minimize`: 激活最小化模式。
-   `initial_pose.pdbqt`: 包含你想要优化的配体初始构象的文件。

### 5.4 原子贡献得分注释 (`--atom_terms`)

这是一个强大的分析功能。Smina 可以计算每个配体原子对总结合能的贡献，并将这些值写入输出 PDBQT 文件的 B-factor（温度因子）列。

```bash
smina -r receptor.pdbqt -l ligand.pdbqt ... -o annotated_poses.pdbqt --atom_terms all
```

-   `--atom_terms all`: 计算所有能量项的贡献总和。
-   你也可以指定特定项，如 `--atom_terms hydrophobic`，来只看疏水作用的贡献。

之后，你可以在 PyMOL 等可视化软件中，根据 B-factor 对配体原子进行着色，从而直观地看到哪些原子是“能量热点”（贡献大），哪些是“能量冷点”（贡献小或有冲突）。

### 5.5 创建自定义打分项 (`--custom_terms`)

除了修改现有项，你还可以添加自己的打分项。例如，你可以创建一个基于特定原子属性（如 `AD_TYPE`）的奖励或惩罚项。

自定义项文件 `my_term.txt`:
```text
# 如果配体原子是氯(Cl)或溴(Br)，则给予一个能量奖励
# 格式: atom_property value weight
AD_TYPE Cl -1.5
AD_TYPE Br -1.5
```
**使用方法**:
```bash
smina ... --custom_terms my_term.txt
```
这在处理卤键或特定金属相互作用时非常有用。

## 6. 完整流程示例：使用 Smina 的高级功能

**目标**：将一个新的配体 `new_ligand.pdbqt` 对接到受体 `receptor.pdbqt` 上。我们有一个参考配体 `ref_ligand.pdbqt` 用于定义活性位点，并且我们想增强疏水作用的权重。

### 步骤一：准备文件
确保 `receptor.pdbqt`, `new_ligand.pdbqt`, 和 `ref_ligand.pdbqt` 文件已准备好。

### 步骤二：创建自定义打分函数文件 (可选)
创建一个名为 `hydrophobic_boost.txt` 的文件，内容如下：

```text
# 增强疏水作用
weight_hydrophobic -0.11
```

### 步骤三：使用命令行运行 Smina

在终端中执行以下命令：

```bash
smina -r receptor.pdbqt \
      -l new_ligand.pdbqt \
      --autobox_ligand ref_ligand.pdbqt \
      --autobox_add 8 \
      --custom_scoring hydrophobic_boost.txt \
      --exhaustiveness 16 \
      --num_modes 20 \
      --cpu 4 \
      -o docked_hydro_boost.pdbqt \
      --log log_hydro_boost.txt
```
**命令解释**:
-   `-r`, `-l`: 指定受体和配体。
-   `--autobox_ligand`, `--autobox_add`: 使用参考配体自动设置对接盒子，并扩大8Å。
-   `--custom_scoring`: 使用我们自定义的打分函数。
-   `--exhaustiveness`, `--num_modes`: 标准的 Vina 参数，设置搜索深度和输出模式数。
-   `--cpu 4`: 使用 4 个 CPU核心进行并行计算（如果你的机器支持）。
-   `-o`, `--log`: 指定输出文件和日志文件。

### 步骤四：分析结果
检查 `log_hydro_boost.txt` 获取结合能打分，并使用可视化软件打开 `receptor.pdbqt` 和 `docked_hydro_boost.pdbqt` 进行分析。

## 7. 结果解读

### 7.1 日志文件 (Log File)
与 Vina 相同，日志文件会列出每个构象模式（Mode）的结合亲和力（Affinity, kcal/mol）。**分值越低（负得越多），结合越好**。

### 7.2 输出结构文件 (`out.pdbqt`)
包含所有对接构象的多模型 PDBQT 文件。每个 `MODEL` 块对应一个构象。

### 7.3 分析原子贡献得分
如果你使用了 `--atom_terms` 参数，你可以用 PyMOL 进行可视化分析：

1.  加载受体和输出的配体文件 (`annotated_poses.pdbqt`)。
2.  选择你感兴趣的配体构象。
3.  在 PyMOL 命令行中输入：
    ```pymol
    # 对 B-factor 进行着色，蓝色=负贡献(有利)，红色=正贡献(不利)
    spectrum b, blue_white_red, minimum=-1, maximum=1
    ```
    这将让你一目了然地看到配体的哪个部分对结合最有利。

## 8. 常用命令行参数速查表

| 参数 | 别名 | 描述 |
| :--- | :--- | :--- |
| `--receptor` | `-r` | 指定受体 PDBQT 文件。 |
| `--ligand` | `-l` | 指定配体 PDBQT 文件。 |
| `--config` | | 指定配置文件。 |
| `--out` | `-o` | 指定输出对接构象的 PDBQT 文件。 |
| `--log` | | 指定输出日志文件。 |
| `--center_x/y/z` | | 手动设置盒子中心坐标。 |
| `--size_x/y/z` | | 手动设置盒子尺寸。 |
| **`--autobox_ligand`** | | **Smina 特有**: 指定一个参考配体文件以自动生成盒子。 |
| **`--autobox_add`** | | **Smina 特有**: 在自动生成的盒子上增加的尺寸。 |
| `--exhaustiveness` | | 搜索穷尽度，默认为 8。 |
| `--num_modes` | | 输出的最大构象数，默认为 9。 |
| `--cpu` | | 使用的 CPU 核心数。 |
| **`--minimize`** | | **Smina 特有**: 激活能量最小化模式。 |
| **`--custom_scoring`** | | **Smina 特有**: 指定自定义打分函数文件。 |
| **`--atom_terms`** | | **Smina 特有**: 将原子能量贡献写入 B-factor 列。 |
| `--score_only` | | 仅对输入构象打分，不进行对接。 |
| `--local_only` | | 仅在配体初始位置附近进行局部搜索。 |

## 9. 常见问题与解决方案 (FAQ)

1.  **Smina 和 Vina 的结果有差异吗？**
    -   即使使用默认参数，由于算法的细微改进和不同的默认随机种子，结果可能会有微小差异。如果你需要与 Vina 完全一致的结果，请使用 Vina。Smina 的价值在于其扩展功能。

2.  **我的自定义打分函数似乎不起作用。**
    -   **检查语法**：确保你的 `.txt` 文件中每行的格式正确 (`weight_termname value`)。
    -   **检查项名称**：确保你使用的打分项名称是 Smina 支持的（`gauss1`, `gauss2`, `repulsion`, `hydrophobic`, `hydrogen`, `rot`）。
    -   **路径问题**：确保 `--custom_scoring` 后面的文件路径正确。

3.  **`--autobox_ligand` 生成的盒子位置奇怪。**
    -   检查你的参考配体文件 (`reference_ligand.pdbqt`)。它的坐标是否正确？它是否位于你期望的活性位点？有时从 PDB 文件中提取配体时可能会丢失坐标变换信息。

4.  **Smina 是不是总是比 Vina 好？**
    -   不一定。"好"取决于你的需求。如果你只需要标准的、可靠的分子对接，Vina 是久经考验的黄金标准。如果你需要进行方法学研究、开发针对特定体系的打分函数，或者希望简化工作流程（如使用 `autobox`），那么 Smina 是一个更优秀的选择。

---
```