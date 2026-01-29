# AutoDock Vina 详细使用指南

## 目录
1. [什么是 AutoDock Vina?](#1-什么是-autodock-vina)
2. [核心概念](#2-核心概念)
3. [安装](#3-安装)
4. [准备工作：输入文件的创建](#4-准备工作输入文件的创建)
    - [4.1 准备受体 (Receptor)](#41-准备受体-receptor)
    - [4.2 准备配体 (Ligand)](#42-准备配体-ligand)
5. [使用方法：运行 Vina](#5-使用方法运行-vina)
    - [5.1 配置文件 `config.txt` 详解](#51-配置文件-configtxt-详解)
    - [5.2 运行命令](#52-运行命令)
6. [完整流程示例](#6-完整流程示例)
    - [步骤一：获取分子结构](#步骤一获取分子结构)
    - [步骤二：使用 ADT 准备受体和配体](#步骤二使用-adt-准备受体和配体)
    - [步骤三：确定对接盒子 (Binding Box)](#步骤三确定对接盒子-binding-box)
    - [步骤四：创建配置文件](#步骤四创建配置文件)
    - [步骤五：运行 Vina 对接](#步骤五运行-vina-对接)
7. [结果解读](#7-结果解读)
    - [7.1 日志文件 (Log File)](#71-日志文件-log-file)
    - [7.2 输出结构文件 (`out.pdbqt`)](#72-输出结构文件-outpdbqt)
8. [进阶技巧与最佳实践](#8-进阶技巧与最佳实践)
    - [8.1 `exhaustiveness` 参数的权衡](#81-exhaustiveness-参数的权衡)
    - [8.2 `seed` 参数用于结果复现](#82-seed-参数用于结果复现)
    - [8.3 柔性对接 (Flexible Docking)](#83-柔性对接-flexible-docking)
    - [8.4 虚拟筛选 (Virtual Screening)](#84-虚拟筛选-virtual-screening)
9. [常见问题与解决方案 (FAQ)](#9-常见问题与解决方案-faq)

---

## 1. 什么是 AutoDock Vina?

**AutoDock Vina**（通常简称为 Vina）是一款由 Scripps 研究所的 Oleg Trott 博士开发的开源分子对接程序。它被广泛用于计算化学、药物设计和结构生物学领域，主要任务是预测小分子（配体）如何与大分子（通常是蛋白质或核酸等受体）结合。

**主要特点**:
- **速度快**：相比于其前身 AutoDock 4，Vina 的计算速度有数量级的提升。
- **准确性高**：在多个基准测试中表现出优秀的预测能力。
- **易于使用**：通过简化的配置文件和命令行接口，大大降低了使用门槛。
- **免费开源**：可免费用于学术和商业用途。

分子对接的核心目标是找到配体在受体活性口袋中的最佳结合构象（Pose），并评估其结合亲和力（Binding Affinity），通常以能量单位（kcal/mol）表示。

## 2. 核心概念

在开始使用 Vina 之前，需要理解以下几个基本概念：

- **受体 (Receptor)**：通常是蛋白质、DNA 等大分子，我们希望研究其与小分子的相互作用。在对接中，受体通常被处理为**刚性**的（除了特定情况下的柔性对接）。
- **配体 (Ligand)**：通常是药物分子、天然产物等小分子，我们希望预测它如何与受体结合。在对接中，配体被处理为**柔性**的，其构象可以自由变化。
- **对接盒子 (Binding Box / Search Space)**：一个三维的长方体空间，Vina 会在这个指定的区域内搜索配体的最佳结合位置和构象。正确设置对接盒子至关重要。
- **结合构象 (Pose)**：配体在受体活性位点中的一个特定空间位置和方向。Vina 会输出多个得分最高的构象。
- **结合亲和力 (Binding Affinity)**：衡量配体与受体结合紧密程度的指标。Vina 会为每个构象计算一个打分，该分值是结合自由能的近似值。**分值越低（负得越多），表示结合能力越强**。

## 3. 安装

### 从官网下载
最直接的方式是从 Vina 官网下载预编译好的可执行文件：
> [http://vina.scripps.edu/download/](http://vina.scripps.edu/download/)

根据你的操作系统（Windows, macOS, Linux）下载对应的版本，解压后即可在命令行中使用。建议将可执行文件所在路径添加到系统的环境变量 `PATH` 中，以便在任何目录下都能直接调用 `vina` 命令。

### 使用 Conda 安装
如果你使用 `conda` 包管理器，安装会更加方便：
```bash
conda install -c conda-forge autodock-vina
```

## 4. 准备工作：输入文件的创建

Vina 需要特定格式的输入文件：**PDBQT**。这个格式在标准 PDB 格式的基础上，增加了原子类型、部分电荷等信息。你需要使用辅助工具来将常见的 PDB, MOL2, SDF 等文件转换为 PDBQT 格式。

最常用的工具是 **MGLTools**（包含了 **AutoDockTools** 或简称 ADT）：
> [http://mgltools.scripps.edu/downloads](http://mgltools.scripps.edu/downloads)

另一个强大的命令行工具是 **Open Babel**，也可以用于格式转换。

### 4.1 准备受体 (Receptor)

以从 PDB 数据库下载的蛋白质结构（例如 `protein.pdb`）为例，使用 ADT 的基本步骤如下：

1.  **启动 ADT** (`adt` 命令)。
2.  **加载 PDB 文件**：`File -> Read Molecule`。
3.  **清理结构**：
    -   删除水分子 (`Edit -> Hydrogens -> Remove Waters`)。
    -   修复缺失的原子（如果需要）。
4.  **加氢**：`Edit -> Hydrogens -> Add`，选择 `Polar Only`（只给极性原子加氢）通常足够。
5.  **计算电荷**：`Edit -> Charges -> Compute Gasteiger`。
6.  **设置原子类型**：`Grid -> Macromolecule -> Choose`，选择该分子作为受体，ADT 会自动合并非极性氢并设置原子类型。
7.  **保存为 PDBQT**：`File -> Save -> Write PDBQT`，保存为 `receptor.pdbqt`。

> **注意**：ADT 会检查是否有不完整的残基或未分配电荷的原子。在保存前确保所有问题都已解决。

### 4.2 准备配体 (Ligand)

以 SDF 或 MOL2 格式的配体文件（例如 `ligand.mol2`）为例：

1.  **启动 ADT**。
2.  **加载配体文件**：`Ligand -> Input -> Open`。
3.  **设置电荷和可旋转键**：
    -   ADT 通常会自动计算 Gasteiger 电荷。
    -   ADT 会自动检测可旋转键（Torsions）。你可以通过 `Ligand -> Torsion Tree -> Choose Torsions` 来检查或修改。
4.  **保存为 PDBQT**：`Ligand -> Output -> Save as PDBQT`，保存为 `ligand.pdbqt`。

**使用 Open Babel (命令行)**:
这是一个更快速的自动化方法。
```bash
obabel ligand.mol2 -O ligand.pdbqt --partialcharge gasteiger
```

## 5. 使用方法：运行 Vina

Vina 的运行依赖于一个配置文件，通常命名为 `config.txt`。

### 5.1 配置文件 `config.txt` 详解

这是一个典型的 `config.txt` 文件示例：

```ini
# ===================================================
#          INPUT and OUTPUT
# ===================================================
receptor = receptor.pdbqt
ligand = ligand.pdbqt
out = all_poses.pdbqt
log = results.log

# ===================================================
#          SEARCH SPACE
# ===================================================
center_x = 10.5
center_y = 20.2
center_z = -5.8

size_x = 25
size_y = 25
size_z = 25

# ===================================================
#          OTHER OPTIONS
# ===================================================
exhaustiveness = 8
num_modes = 10
energy_range = 3
seed = 12345
```

**参数说明**：

- **`receptor`**: 受体 PDBQT 文件路径。
- **`ligand`**: 配体 PDBQT 文件路径。
- **`out`**: 输出文件路径，包含所有对接构象，也是 PDBQT 格式。
- **`log`**: 日志文件路径，包含结合能打分等信息。

- **`center_x`, `center_y`, `center_z`**: 对接盒子的中心点坐标（单位：埃, Å）。
- **`size_x`, `size_y`, `size_z`**: 对接盒子在三个维度上的尺寸（单位：埃, Å）。

- **`exhaustiveness`**: 穷尽度/搜索深度（整数）。值越高，搜索越彻底，结果可能越准，但耗时也越长。默认值为 `8`。对于快速筛选，可以设为 `4`；对于最终的精确计算，可以设为 `16` 或更高。
- **`num_modes`**: 输出的结合构象数量上限。Vina 会按能量从低到高排序，输出不多于此数量的构象。默认值为 `9`。
- **`energy_range`**: 输出构象的能量范围（单位：kcal/mol）。只输出能量在最佳构象能量 `energy_range` 范围内的构象。例如，如果最佳能量是 -9.0，`energy_range` 是 3，则只会输出能量在 [-9.0, -6.0] 范围内的构象。默认值为 `3`。
- **`seed`**: 随机数种子。设置一个固定的种子可以确保每次运行得到完全相同的结果，方便复现。

### 5.2 运行命令

在终端中，确保 `vina` 可执行文件在你的 `PATH` 中，或者提供其完整路径。然后执行：

```bash
vina --config config.txt
```

Vina 会开始计算，并在屏幕上显示进度。计算完成后，你将得到 `all_poses.pdbqt` 和 `results.log` 两个文件。

## 6. 完整流程示例

### 步骤一：获取分子结构
- **受体**：从 PDB 数据库 (rcsb.org) 下载一个蛋白质结构，例如 `1KE7`。
- **配体**：`1KE7` 结构中包含一个共结晶配体 `STI`。我们可以提取该配体作为对接的参考，或者从 PubChem/ZINC 数据库下载一个全新的小分子。

### 步骤二：使用 ADT 准备受体和配体
按照 [章节 4](#4-准备工作输入文件的创建) 的描述，将 `1KE7.pdb` 处理成 `1KE7_protein.pdbqt`，并将配体 `STI` 处理成 `sti_ligand.pdbqt`。

### 步骤三：确定对接盒子 (Binding Box)

确定对接盒子的位置和大小是关键。一个常用的方法是**以共结晶配体为中心**来定义盒子。

1.  在 PyMOL 或 Chimera 等分子可视化软件中同时打开受体和共结晶配体。
2.  找到配体所占据的空间范围。
3.  **确定中心**：找到配体的几何中心坐标。很多软件有直接计算几何中心的功能。
4.  **确定尺寸**：设置一个足够大的长方体，确保能完全包裹住配体，并留出一些额外空间（例如，各方向延伸 5-10 Å），以便配体可以自由旋转和平移。盒子尺寸一般建议在 20-30 Å 之间。盒子过大会极大地增加计算时间。

假设我们通过软件测量得到中心坐标为 `(15.0, 55.0, 96.0)`，并且我们决定使用一个 `24x24x24` Å 的盒子。

### 步骤四：创建配置文件
创建一个 `config.txt` 文件，内容如下：

```ini
receptor = 1KE7_protein.pdbqt
ligand = sti_ligand.pdbqt

out = sti_docked_poses.pdbqt
log = sti_docking_log.txt

center_x = 15.0
center_y = 55.0
center_z = 96.0

size_x = 24
size_y = 24
size_z = 24

exhaustiveness = 16
num_modes = 20
```

### 步骤五：运行 Vina 对接
在终端中运行：
```bash
vina --config config.txt
```

## 7. 结果解读

### 7.1 日志文件 (Log File)
打开 `sti_docking_log.txt`，你会看到类似下面的内容：

```
Scoring function : vina
Ligand: sti_ligand.pdbqt
Receptor: 1KE7_protein.pdbqt
Exhaustiveness: 16

-----+------------+----------+----------
Mode | Affinity   | RMSD_lb  | RMSD_ub
     | (kcal/mol) |          |
-----+------------+----------+----------
   1 |      -11.2 |    0.000 |    0.000
   2 |      -10.5 |    1.853 |    2.451
   3 |      -10.1 |    2.109 |    3.017
   4 |       -9.8 |    6.780 |    8.912
   ...
```

- **`Mode`**: 构象的编号。
- **`Affinity (kcal/mol)`**: 结合亲和力打分。**数值越低（负得越多），代表结合越稳定，是 Vina 对接最重要的输出结果**。Mode 1 的打分是最佳预测结果。
- **`RMSD_lb` (Root Mean Square Deviation, lower bound)**: 与最佳构象（Mode 1）相比的均方根偏差（仅考虑原子位置，忽略对称性），是构象多样性的一个指标。
- **`RMSD_ub` (Root Mean Square Deviation, upper bound)**: 同上，但考虑了对称性，是更准确的差异性度量。

**核心结论**：Mode 1 的构象是 Vina 预测的最佳结合模式，其结合能为 -11.2 kcal/mol。

### 7.2 输出结构文件 (`out.pdbqt`)
`sti_docked_poses.pdbqt` 是一个多模型的 PDBQT 文件。你可以用文本编辑器打开它，会看到：

```pdb
MODEL 1
REMARK VINA RESULT:      -11.2      0.000      0.000
... (原子坐标) ...
ENDMDL
MODEL 2
REMARK VINA RESULT:      -10.5      1.853      2.451
... (原子坐标) ...
ENDMDL
...
```

- 每个 `MODEL`...`ENDMDL` 块代表一个结合构象，其编号与日志文件中的 `Mode` 对应。
- `REMARK VINA RESULT` 行记录了该构象的能量和 RMSD 值。
- 你可以使用 PyMOL, Chimera, VMD 等软件打开这个文件，将所有构象与受体一起可视化，从而直观地分析配体与受体氨基酸残基的相互作用（如氢键、疏水作用等）。

## 8. 进阶技巧与最佳实践

### 8.1 `exhaustiveness` 参数的权衡
- **快速筛选**：对数千个分子的库进行初步筛选时，可设为 `4` 或 `8`，以在可接受的时间内获得初步结果。
- **精确计算**：对少数几个候选分子进行精细对接时，可设为 `16`, `32` 甚至更高，以获得更可靠的结合构象和能量。

### 8.2 `seed` 参数用于结果复现
Vina 的算法包含随机性。如果你想让你的研究结果可以被他人或自己精确复现，务必在配置文件中设置一个固定的 `seed` 值。

### 8.3 柔性对接 (Flexible Docking)
Vina 支持受体侧链的柔性对接，这能更真实地模拟结合过程中的“诱导契合”效应。

1.  **准备柔性受体**：在 ADT 中，选择你希望设为柔性的氨基酸残基（例如，活性位点附近的关键残基）。
2.  **保存文件**：ADT 会将受体分为两部分：刚性部分 (`receptor_rigid.pdbqt`) 和柔性部分 (`receptor_flex.pdbqt`)。
3.  **修改配置文件**：
    ```ini
    receptor = receptor_rigid.pdbqt
    flex = receptor_flex.pdbqt
    ligand = ...
    ...
    ```
> **注意**：柔性对接会显著增加计算的复杂度和时间。建议只选择少数几个关键残基作为柔性部分。

### 8.4 虚拟筛选 (Virtual Screening)
Vina 可以结合 Shell 脚本或 Python 脚本，对包含成千上万个分子的库进行自动化对接。

Vina 提供了一些有用的命令行参数来支持这一点：
- `--score_only`: 对已经存在的构象只进行打分，而不进行对接搜索。
- `--local_only`: 只在配体初始位置附近进行局部优化，而不进行全局搜索。

一个典型的虚拟筛选脚本流程：
1.  循环遍历所有配体文件。
2.  为每个配体运行 Vina。
3.  从日志文件中提取最佳结合能。
4.  将所有配体的名称和得分汇总到一个结果文件中。
5.  根据得分对配体进行排序，筛选出高潜力的候选分子。

## 9. 常见问题与解决方案 (FAQ)

1.  **错误：`FATAL ERROR: cannot open "receptor.pdbqt"`**
    - **原因**：文件路径不正确或文件不存在。
    - **解决**：检查 `config.txt` 中的文件名和路径是否正确，并确保文件与 `config.txt` 在同一目录，或使用绝对路径。

2.  **错误：`ValueError: ATOM syntax incorrect` 或关于原子类型的警告**
    - **原因**：PDBQT 文件制备不当，可能缺少电荷、原子类型，或格式有误。
    - **解决**：返回 ADT 或 Open Babel，严格按照流程重新制备 PDBQT 文件，确保所有原子都有正确的电荷和原子类型。

3.  **对接结果很差，能量很高（接近 0 或为正）**
    - **原因**：
        -   对接盒子设置不当，没有覆盖真实的活性位点。
        -   配体或受体的初始结构质量差（例如，不合理的质子化状态、错误的化学结构）。
        -   该配体本身与该受体就没有好的结合能力。
    - **解决**：
        -   使用可视化软件仔细检查并重新调整对接盒子。
        -   检查并修正分子的化学结构和质子化状态。
        -   尝试增加 `exhaustiveness`。

4.  **Vina 运行时间过长**
    - **原因**：
        -   对接盒子太大。
        -   `exhaustiveness` 设置得太高。
        -   配体可旋转键过多。
        -   进行了柔性对接。
    - **解决**：
        -   缩小对接盒子，使其仅覆盖必要的区域。
        -   适当降低 `exhaustiveness`。
        -   在配体制备时，考虑将一些不重要的键（如甲基旋转）设为非旋转。

