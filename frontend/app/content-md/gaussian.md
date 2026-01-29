# Gaussian 详细使用指南：从入门到实践

## 目录
1. [什么是 Gaussian?](#1-什么是-gaussian)
2. [核心功能与应用](#2-核心功能与应用)
3. [Gaussian 输入文件 (`.gjf` / `.com`) 详解](#3-gaussian-输入文件-gjf--com-详解)
    - [3.1 输入文件结构图](#31-输入文件结构图)
    - [3.2 各部分详细说明](#32-各部分详细说明)
4. [核心计算类型与常用关键词](#4-核心计算类型与常用关键词)
    - [4.1 单点能计算 (`SP`)](#41-单点能计算-sp)
    - [4.2 几何优化 (`Opt`)](#42-几何优化-opt)
    - [4.3 频率计算 (`Freq`)](#43-频率计算-freq)
    - [4.4 过渡态搜索 (`Opt=TS`)](#44-过渡态搜索-optts)
    - [4.5 其他常见计算](#45-其他常见计算)
5. [运行 Gaussian 计算](#5-运行-gaussian-计算)
6. [输出文件解读](#6-输出文件解读)
    - [6.1 日志文件 (`.log`)](#61-日志文件-log)
    - [6.2 检查点文件 (`.chk` / `.fchk`)](#62-检查点文件-chk--fchk)
7. [完整流程示例：水分子的优化与频率分析](#7-完整流程示例水分子的优化与频率分析)
    - [步骤一：创建输入文件 `water_opt_freq.gjf`](#步骤一创建输入文件-water_opt_freqgjf)
    - [步骤二：运行计算](#步骤二运行计算)
    - [步骤三：解读关键输出](#步骤三解读关键输出)
8. [最佳实践与高级技巧](#8-最佳实践与高级技巧)
9. [常见问题与解决方案 (FAQ)](#9-常见问题与解决方案-faq)

---

## 1. 什么是 Gaussian?

**Gaussian** 是一款功能极其强大的商业量子化学计算软件，是计算化学领域的黄金标准之一。它由诺贝尔化学奖得主 John Pople 教授及其团队开发，并不断发展至今（最新版本为 Gaussian 16）。

Gaussian 的核心是通过求解近似的薛定谔方程（使用从头算 `ab initio` 方法或密度泛函理论 `DFT` 等）来预测分子和化学反应的各种性质。它不是一个可视化软件，而是一个在后台运行的命令行程序，通过处理文本输入文件并生成文本输出文件来完成计算。

## 2. 核心功能与应用

Gaussian 可以计算和预测分子的广泛性质，包括：

-   **能量与结构**:
    -   单点能、分子轨道能量
    -   几何优化（寻找能量最低的稳定构象）
    -   过渡态结构搜索（寻找反应的能垒）
-   **光谱性质**:
    -   红外（IR）和拉曼（Raman）光谱
    -   核磁共振（NMR）化学位移
    -   紫外-可见（UV-Vis）光谱
    -   圆二色谱（CD）
-   **热力学性质**:
    -   零点振动能（ZPVE）
    -   焓、熵、吉布斯自由能
-   **反应路径**:
    -   内禀反应坐标（IRC）计算，连接过渡态与反应物/产物。
-   **其他性质**:
    -   偶极矩、旋光度
    -   原子电荷、静电势
    -   极化率、超极化率

## 3. Gaussian 输入文件 (`.gjf` / `.com`) 详解

Gaussian 的所有操作都由一个结构化的文本输入文件驱动。这个文件的结构必须严格遵守格式。

### 3.1 输入文件结构图

下面是一个典型的 Gaussian 输入文件结构示意图，它由几个被**空行**严格分隔的部分组成。

<style>
.gaussian-input-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    border: 2px solid #0969da;
    border-radius: 8px;
    padding: 15px;
    background-color: #f6f8fa;
    max-width: 700px;
    margin: 20px auto;
}
.input-section {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    margin-bottom: 10px;
    padding: 10px;
    position: relative;
}
.input-section .label {
    position: absolute;
    top: -12px;
    left: 10px;
    background-color: #f6f8fa;
    padding: 0 5px;
    font-weight: bold;
    color: #0969da;
    font-size: 0.9em;
}
.input-section pre {
    margin: 0;
    padding: 5px;
    background-color: #f6f8fa;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.blank-line {
    text-align: center;
    color: #8b949e;
    font-style: italic;
    font-size: 0.9em;
    margin: 5px 0;
}
</style>

<div class="gaussian-input-container">
    <div class="input-section">
        <span class="label">Link 0 Commands (可选)</span>
        <pre>%nprocshared=8
%mem=16GB
%chk=my_calc.chk</pre>
    </div>
    <div class="blank-line">(空行)</div>
    <div class="input-section">
        <span class="label">Route Section (#行)</span>
        <pre># B3LYP/6-31G(d) Opt Freq</pre>
    </div>
    <div class="blank-line">(空行)</div>
    <div class="input-section">
        <span class="label">Title Section</span>
        <pre>Optimization of Formaldehyde</pre>
    </div>
    <div class="blank-line">(空行)</div>
    <div class="input-section">
        <span class="label">Molecule Specification</span>
        <pre>0 1
 C   0.000000    0.000000    0.000000
 O   0.000000    0.000000    1.200000
 H   0.000000    0.940000   -0.590000
 H   0.000000   -0.940000   -0.590000</pre>
    </div>
    <div class="blank-line">(空行, 文件末尾可能需要一个或多个空行)</div>
</div>

### 3.2 各部分详细说明

1.  **Link 0 Commands (链接0命令)**:
    -   以 `%` 开头，用于设置计算资源和文件。
    -   `%nprocshared=N`: 设置使用的 CPU 核心数。
    -   `%mem=XGB` 或 `XMB`: 设置分配的内存大小。
    -   `%chk=filename.chk`: 指定检查点文件的名称，**强烈建议总是设置此项**，用于重启计算和后续分析。

2.  **Route Section (路径部分)**:
    -   以 `#` 开头，是输入文件的**核心**，告诉 Gaussian 要做什么以及怎么做。
    -   格式为 `# Method/BasisSet Keyword1 Keyword2 ...`
    -   **Method (方法)**: 指定量子化学方法，如 `HF` (Hartree-Fock), `MP2` (Møller-Plesset), 或 `B3LYP`, `wB97XD` (DFT 泛函)。
    -   **Basis Set (基组)**: 指定描述原子轨道的数学函数集，如 `6-31G(d)`, `def2-SVP`, `aug-cc-pVDZ`。
    -   **Keywords (关键词)**: 指定计算任务，如 `Opt` (几何优化), `Freq` (频率计算), `NMR` (核磁计算)等。

3.  **Title Section (标题部分)**:
    -   一个简短的、人类可读的标题，用于描述此计算。Gaussian 会将其原样打印到输出文件中。

4.  **Molecule Specification (分子定义)**:
    -   **Charge and Multiplicity (电荷和多重度)**:
        -   第一行是两个整数，分别是分子的**总电荷**和**自旋多重度**。
        -   **电荷**: 中性分子为 0，阴离子为 -1, -2...，阳离子为 +1, +2...。
        -   **多重度**: `2S + 1`，其中 `S` 是总自旋量子数。对于所有电子都成对的闭壳层分子，`S=0`，多重度为 `1` (单重态)。对于有一个单电子的自由基，`S=1/2`，多重度为 `2` (双重态)。
    -   **Atomic Coordinates (原子坐标)**:
        -   最常用的格式是笛卡尔坐标（Cartesian Coordinates）。
        -   每行格式为 `AtomSymbol X Y Z`。
        -   `AtomSymbol` 是元素符号，`X, Y, Z` 是其在三维空间中的坐标（单位：埃, Å）。

## 4. 核心计算类型与常用关键词

### 4.1 单点能计算 (`SP`)
-   **目的**: 在一个**固定**的几何结构下，计算分子的能量、波函数和相关性质。
-   **关键词**: `SP`
-   **用途**:
    -   获取一个特定构象的能量。
    -   作为更复杂计算（如 NMR）的基础。
-   **示例**: `# B3LYP/6-31G(d) SP`

### 4.2 几何优化 (`Opt`)
-   **目的**: 从一个初始猜测结构出发，通过迭代计算，找到势能面上能量最低的稳定点（局部最小值）。
-   **关键词**: `Opt`
-   **用途**:
    -   获得分子的平衡几何构象。
    -   是几乎所有其他计算（如频率、NMR）的**必要前序步骤**。
-   **示例**: `# B3LYP/6-31G(d) Opt`

### 4.3 频率计算 (`Freq`)
-   **目的**: 计算分子振动的简正模式及其频率。
-   **关键词**: `Freq`
-   **用途**:
    1.  **结构验证**: 优化得到的稳定结构，其振动频率应**全部为正值**。如果出现一个或多个**虚频**（输出中显示为负值），说明该结构不是能量最低点，而是一个鞍点（如过渡态）。
    2.  **热力学校正**: 计算零点振动能（ZPVE）、焓、熵和吉布斯自由能，从而将电子能（0K下的能量）校正到标准温度下的热力学能量。
    3.  **光谱预测**: 预测分子的红外（IR）和拉曼（Raman）光谱。
-   **黄金组合**: `Opt Freq`，表示先进行几何优化，然后在优化得到的结构上进行频率计算。
-   **示例**: `# B3LYP/6-31G(d) Opt Freq`

### 4.4 过渡态搜索 (`Opt=TS`)
-   **目的**: 在势能面上寻找连接反应物和产物的能量最高点，即鞍点（过渡态）。
-   **关键词**: `Opt=(TS, CalcFC)` 或 `Opt=(TS, NoEigenTest)`
    -   `TS`: 表明要搜索过渡态。
    -   `CalcFC`: 在计算开始时计算完整的力常数矩阵（Hessian），这会增加计算成本，但大大提高收敛成功率。
-   **验证**: 成功的过渡态计算后，频率分析应得到**且仅得到一个虚频**。
-   **示例**: `# B3LYP/6-31G(d) Opt=(TS, CalcFC) Freq`

### 4.5 其他常见计算
-   **NMR**: `# B3LYP/6-311+G(2d,p) NMR` (通常需要更大的基组)
-   **UV-Vis**: `# B3LYP/6-31G(d) TD(NStates=10)` (计算前10个激发态)
-   **IRC**: `# B3LYP/6-31G(d) IRC` (从过渡态出发，沿着反应路径走向反应物和产物)

## 5. 运行 Gaussian 计算

Gaussian 是一个命令行程序。在安装好 Gaussian 的服务器或工作站上，通常使用以下命令格式：

```bash
g16 < input_file.gjf > output_file.log
```
-   `g16`: Gaussian 16 的可执行文件名（可能是 `g09` 等，取决于版本）。
-   `< input_file.gjf`: 标准输入重定向，将输入文件的内容“喂”给 Gaussian 程序。
-   `> output_file.log`: 标准输出重定向，将屏幕上所有的输出信息保存到日志文件中。

对于耗时较长的计算，建议在后台运行，并将标准错误也重定向：
```bash
nohup g16 < input.gjf > output.log 2>&1 &
```
-   `nohup`: 防止终端关闭时计算被中断。
-   `&`: 在后台运行。

## 6. 输出文件解读

### 6.1 日志文件 (`.log`)
这是**最重要**的输出文件，包含了计算过程和所有结果的详细信息。由于文件可能非常长，你需要学会快速定位关键信息：

-   **正常开始**: 搜索 `Entering Gaussian`。
-   **几何优化过程**:
    -   搜索 `SCF Done:`，可以看到每一步优化的能量值。
    -   搜索 `Converged?`，当看到连续四个 `YES` 时，表示这一步的几何结构收敛了。
    -   搜索 `Optimization completed`，表示整个优化过程完成。
-   **频率计算结果**:
    -   搜索 `Frequencies --`，后面会列出所有振动频率。**检查是否有负值（虚频）**。
    -   搜索 `Zero-point correction`，可以找到零点能。
    -   搜索 `Sum of electronic and thermal Free Energies`，可以找到最终的吉布斯自由能。
-   **计算结束**:
    -   文件末尾的 `Normal termination of Gaussian` 表示计算成功完成。
    -   如果出现 `Error termination`，则表示计算失败，需要向上查找错误信息。
-   **最终能量**: 搜索 `SCF Done:  E(RB3LYP)` (方法名会变)，在优化收敛后的最后一次出现，就是最终的电子能。

### 6.2 检查点文件 (`.chk` / `.fchk`)
-   `.chk` (Checkpoint File): 二进制文件，包含了波函数、分子结构等所有关键信息。
    -   可用于**可视化**（使用 GaussView 等软件）。
    -   可用于**重启**失败的计算 (`Opt=Restart`)。
    -   可用于从一个计算结果开始新的计算（例如，在优化后的结构上计算 NMR）。
-   `.fchk` (Formatted Checkpoint File): `.chk` 文件的文本格式版本，更具可移植性，方便与其他程序交换数据。可以使用 `formchk` 工具生成：`formchk my_calc.chk`。

## 7. 完整流程示例：水分子的优化与频率分析

### 步骤一：创建输入文件 `water_opt_freq.gjf`
```text
%nprocshared=4
%mem=4GB
%chk=water.chk

# B3LYP/6-31G(d) Opt Freq

Water Optimization and Frequencies

0 1
O   0.000000    0.000000    0.117300
H   0.000000    0.757200   -0.469200
H   0.000000   -0.757200   -0.469200

```

### 步骤二：运行计算
```bash
g16 < water_opt_freq.gjf > water_opt_freq.log
```

### 步骤三：解读关键输出
在 `water_opt_freq.log` 文件中：

1.  **确认优化收敛**: 搜索 `Optimization completed.`
2.  **查看最终能量**: 在优化收敛后，找到最后一个 `SCF Done:  E(RB3LYP) =  -76.4196173045     A.U.`
3.  **检查频率**: 搜索 `Frequencies --`，你会看到类似：
    ```
    Frequencies --  1696.5881               3831.3316               3938.8367
    ```
    所有频率都是正值，说明这是一个真实的能量最低点。
4.  **获取热力学校正**: 搜索 `Sum of electronic and thermal Free Energies`，你会看到：
    ```
    Sum of electronic and thermal Free Energies=           -76.398935
    ```
    这个值 `-76.398935` Hartree 就是水分子的吉布斯自由能。

## 8. 最佳实践与高级技巧

-   **总是先优化，再计算性质**: 除非特殊需要，否则任何性质（NMR, TD-SCF 等）都应在优化后的稳定结构上计算。
-   **总是进行频率分析**: `Opt` 之后紧跟 `Freq` 是黄金法则，用以验证结构并获得热力学数据。
-   **从小处着手**: 对于复杂的分子，先用较小的方法/基组（如 `B3LYP/6-31G(d)`）进行初步优化，然后以其结果作为更高级别计算（如 `wB97XD/def2-TZVP`）的初始结构。
-   **可视化是你的朋友**: 使用 GaussView, Avogadro, VMD 等软件打开 `.chk` 或 `.log` 文件，直观地检查你的分子结构是否合理，动画显示振动模式等。
-   **善用检查点文件**: 对于耗时长的计算，设置 `%chk` 是必须的。如果计算意外中断，你可以使用 `Opt=Restart` 从断点处继续，节省大量时间。

## 9. 常见问题与解决方案 (FAQ)

1.  **计算一开始就报错 (`Error termination`)**
    - **原因**: 绝大多数情况是输入文件语法错误。
    - **解决**: 仔细检查空行、电荷与多重度、关键词拼写、原子坐标格式。

2.  **几何优化不收敛**
    - **原因**: 初始结构太差；势能面非常平坦；分子非常柔性。
    - **解决**:
        -   用分子力学或可视化软件预优化一个更合理的初始结构。
        -   尝试 `Opt=Loose` 进行粗略优化，再用默认标准优化。
        -   对于非常困难的情况，尝试 `Opt=CalcAll`，它在每一步都计算精确的 Hessian 矩阵，虽然慢但非常稳健。

3.  **优化后频率计算出现虚频**
    - **原因**: 优化收敛到了一个鞍点（过渡态）而不是能量最低点。
    - **解决**:
        -   使用 GaussView 等软件将虚频振动模式进行动画显示，观察原子如何运动。
        -   手动沿着该振动方向微调原子坐标，破坏对称性，然后以此为新结构重新优化。

4.  **我应该选择哪种方法和基组？**
    - **答案**: "视情况而定"。这是一个贯穿计算化学的复杂问题。
    - **入门建议**: `B3LYP/6-31G(d)` 是一个不错的起点，适用于大多数有机分子的结构和频率计算。
    - **更高精度**: 对于非共价相互作用，推荐使用包含色散校正的泛函，如 `wB97XD` 或 `B3LYP-D3`。基组可选用 `def2-SVP` 或 `def2-TZVP`。

---
```