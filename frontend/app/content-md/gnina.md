# GNINA 详细使用指南：结合机器学习的下一代分子对接

## 目录
1. [什么是 GNINA?](#1-什么是-gnina)
2. [核心优势：为什么选择 GNINA?](#2-核心优势为什么选择-gnina)
3. [安装](#3-安装)
4. [核心概念：机器学习打分函数](#4-核心概念机器学习打分函数)
5. [使用方法](#5-使用方法)
    - [5.1 基础对接 (作为 Smina/Vina 的替代品)](#51-基础对接-作为-sminasvina-的替代品)
    - [5.2 使用 CNN 打分函数进行对接](#52-使用-cnn-打分函数进行对接)
    - [5.3 对接后重新打分 (Re-scoring)](#53-对接后重新打分-re-scoring)
    - [5.4 利用 GPU 加速](#54-利用-gpu-加速)
6. [完整流程示例：使用 CNN 模型进行对接](#6-完整流程示例使用-cnn-模型进行对接)
    - [步骤一：准备输入文件 (PDBQT)](#步骤一准备输入文件-pdbqt)
    - [步骤二：运行 GNINA 对接](#步骤二运行-gnina-对接)
    - [步骤三：解读输出结果](#步骤三解读输出结果)
7. [结果解读](#7-结果解读)
    - [7.1 日志文件 (Log File)](#71-日志文件-log-file)
    - [7.2 输出结构文件 (`out.pdbqt`)](#72-输出结构文件-outpdbqt)
    - [7.3 如何选择最终构象?](#73-如何选择最终构象)
8. [高级技巧与最佳实践](#8-高级技巧与最佳实践)
    - [8.1 选择合适的 CNN 模型](#81-选择合适的-cnn-模型)
    - [8.2 虚拟筛选策略](#82-虚拟筛选策略)
    - [8.3 继承 Smina 的所有高级功能](#83-继承-smina-的所有高级功能)
9. [常见问题与解决方案 (FAQ)](#9-常见问题与解决方案-faq)

---

## 1. 什么是 GNINA?

**GNINA** (General-purpose Interaction Neural Network-based Affinity) 是一个基于深度学习的分子对接程序。它构建于 **Smina** 之上，而 Smina 本身又是 **AutoDock Vina** 的一个分支。因此，GNINA 不仅继承了 Smina 和 Vina 的全部功能和高效的构象搜索算法，还集成了一系列**卷积神经网络 (Convolutional Neural Networks, CNNs)** 模型，用于对生成的结合构象进行打分和排序。

您可以这样理解它们的演进关系：

<style>
.hierarchy-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    padding: 20px;
    background-color: #f6f8fa;
    border-radius: 8px;
    border: 1px solid #d0d7de;
    max-width: 600px;
    margin: 20px auto;
}
.hierarchy-level {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.hierarchy-level h3 {
    margin: 0 0 5px 0;
    color: #0969da;
}
.hierarchy-level p {
    margin: 0;
    font-size: 0.9em;
    color: #57606a;
}
.arrow-down {
    text-align: center;
    font-size: 24px;
    color: #8b949e;
    margin: -5px 0;
}
</style>
<div class="hierarchy-container">
    <div class="hierarchy-level">
        <h3>AutoDock Vina</h3>
        <p>经典、快速的分子对接程序，使用经验打分函数。</p>
    </div>
    <div class="arrow-down">↓</div>
    <div class="hierarchy-level">
        <h3>Smina</h3>
        <p>Vina 的分支，增加了自定义打分函数、自动盒子等高级功能。</p>
    </div>
    <div class="arrow-down">↓</div>
    <div class="hierarchy-level" style="border-color: #1a7f37;">
        <h3 style="color: #1a7f37;">GNINA</h3>
        <p>Smina 的分支，<strong>集成了 CNN 机器学习模型</strong>，提供更准确的打分和亲和力预测。</p>
    </div>
</div>

简而言之，GNINA 将传统的物理化学打分函数与最先进的深度学习模型相结合，旨在提供更准确的结合姿态预测（Pose Prediction）和结合亲和力预测（Affinity Prediction）。

## 2. 核心优势：为什么选择 GNINA?

-   **更高的准确性**：大量研究表明，GNINA 的 CNN 模型在区分正确和错误的结合构象（“打分”能力）以及预测结合亲和力方面，通常优于传统的经验打分函数。
-   **双重打分系统**：GNINA 同时提供传统的 Vina/Smina 打分 (`Affinity`) 和多种 CNN 打分 (`CNNscore`, `CNNaffinity`)，为用户提供多维度的决策依据。
-   **继承全部功能**：作为 Smina 的直接替代品，它拥有 Smina 的所有优点，如自动对接盒子生成 (`--autobox_ligand`)、自定义打分函数等。
-   **GPU 加速**：CNN 计算是计算密集型的，GNINA 支持使用 NVIDIA GPU 进行大规模加速，这对于处理大型化合物库至关重要。
-   **持续开发**：GNINA 是一个活跃的开源项目，不断有新的模型和功能被集成进来。

## 3. 安装

强烈推荐使用 `conda` 进行安装，因为它可以方便地管理复杂的依赖项，尤其是 GPU 相关的库。

#### CPU 版本
如果你的机器没有 NVIDIA GPU，或者只是想进行小规模测试，可以安装 CPU 版本：
```bash
conda install -c conda-forge gnina
```

#### GPU 版本
为了获得最佳性能，强烈建议安装 GPU 版本。你需要先确保已经安装了合适的 NVIDIA 驱动。

```bash
# conda 会自动处理 cudatoolkit 的安装
conda install -c conda-forge gnina-gpu
```

安装完成后，在终端输入 `gnina --help`，如果能看到帮助信息，则表示安装成功。

## 4. 核心概念：机器学习打分函数

GNINA 的核心是其 CNN 模型。当 Smina 的构象搜索引擎生成一个候选结合构象后，GNINA 会执行以下操作：

1.  将受体和配体构象转化为一个 3D 网格（Grid）。
2.  网格中的每个体素（voxel）都包含了原子类型等信息。
3.  这个 3D 网格就像一张“3D 图片”，被输入到预训练好的 CNN 模型中。
4.  CNN 模型输出预测值。

<style>
.cnn-process-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex;
    align-items: center;
    justify-content: space-around;
    flex-wrap: wrap;
    padding: 20px;
    background-color: #f6f8fa;
    border-radius: 8px;
    border: 1px solid #d0d7de;
    margin: 20px auto;
}
.cnn-box {
    text-align: center;
    padding: 10px;
    margin: 10px;
}
.cnn-box-label {
    font-weight: bold;
    margin-bottom: 10px;
    color: #24292f;
}
.cnn-item {
    border: 2px solid #8b949e;
    border-radius: 6px;
    padding: 20px;
    background: #fff;
    min-width: 150px;
}
.cnn-arrow {
    font-size: 2.5em;
    color: #57606a;
    margin: 10px;
}
.cnn-model-box {
    border: 2px dashed #0969da;
    background: #ddf4ff;
}
.cnn-output-box {
    border: 2px solid #1a7f37;
    background: #e6ffec;
}
.cnn-output-box ul {
    list-style-type: none;
    padding: 0;
    text-align: left;
    font-size: 0.9em;
}
.cnn-output-box li {
    margin-bottom: 5px;
}
.cnn-output-box code {
    background-color: rgba(0,0,0,0.1);
    padding: 2px 4px;
    border-radius: 3px;
}

@media (max-width: 600px) {
    .cnn-process-container {
        flex-direction: column;
    }
    .cnn-arrow {
        transform: rotate(90deg);
    }
}
</style>
<div class="cnn-process-container">
    <div class="cnn-box">
        <div class="cnn-box-label">输入</div>
        <div class="cnn-item">蛋白质 + 配体构象</div>
    </div>
    <div class="cnn-arrow">→</div>
    <div class="cnn-box">
        <div class="cnn-box-label">处理</div>
        <div class="cnn-item cnn-model-box">
            <strong>CNN 模型</strong>
            <p style="font-size:0.8em; color:#57606a;">(e.g., dense, default)</p>
        </div>
    </div>
    <div class="cnn-arrow">→</div>
    <div class="cnn-box">
        <div class="cnn-box-label">输出</div>
        <div class="cnn-item cnn-output-box">
            <ul>
                <li><code>CNNscore</code>: 0-1 的概率值</li>
                <li><code>CNNaffinity</code>: 结合能 (kcal/mol)</li>
                <li><code>Affinity</code>: 经典打分 (kcal/mol)</li>
            </ul>
        </div>
    </div>
</div>

**GNINA 的主要输出值**:

-   `Affinity` (kcal/mol): 这是传统的 **Smina/Vina 打分**。数值越低越好。
-   `CNNscore` (0 到 1): 这是 CNN 模型预测该构象是**“好构象”（接近晶体结构）的概率**。数值越接近 1 越好。这是评估姿态质量的主要指标。
-   `CNNaffinity` (kcal/mol): 这是 CNN 模型预测的**结合亲和力**。数值越低越好。

## 5. 使用方法

GNINA 的命令行接口与 Smina 完全兼容，只是增加了与 CNN 模型相关的参数。

### 5.1 基础对接 (作为 Smina/Vina 的替代品)

如果你不使用任何 CNN 相关参数，GNINA 的行为就和 Smina 完全一样。

```bash
gnina -r receptor.pdbqt -l ligand.pdbqt --autobox_ligand ref_ligand.pdbqt -o output.pdbqt
```
在这种模式下，你只会得到经典的 `Affinity` 打分。

### 5.2 使用 CNN 打分函数进行对接

这是 GNINA 的标准用法。使用 `--cnn` 参数来指定要使用的 CNN 模型。

```bash
gnina -r receptor.pdbqt -l ligand.pdbqt \
--autobox_ligand ref_ligand.pdbqt --autobox_add 4 \
--cnn default \
-o cnn_docked.pdbqt --log cnn_log.txt
```
-   `--cnn default`: 使用默认的 CNN 模型 (`crossdock_default_v1.1`) 对生成的每个构象进行打分。GNINA 会根据 `CNNscore` 对构象进行排序。

### 5.3 对接后重新打分 (Re-scoring)

这是一个非常高效的策略。你可以先用一个快速的方法（如 Smina 或不带 CNN 的 GNINA）生成构象，然后用 GNINA 的 CNN 模型对这些构象进行重新打分。

```bash
# 假设 docked_from_smina.pdbqt 是 Smina 的输出文件
gnina -r receptor.pdbqt --score_only -l docked_from_smina.pdbqt \
--cnn dense \
-o rescore_output.pdbqt --log rescore_log.txt
```
-   `--score_only`: 告诉 GNINA 不要进行新的构象搜索，只需对输入配体文件 (`-l`) 中的构象进行打分。
-   `--cnn dense`: 使用 `dense` 模型（一个计算量更大但通常更准确的模型）进行重新打分。

### 5.4 利用 GPU 加速

如果安装了 `gnina-gpu` 并且有可用的 NVIDIA GPU，只需添加 `--gpu` 标志。

```bash
gnina -r receptor.pdbqt -l ligand.pdbqt --cnn default --gpu -o gpu_docked.pdbqt
```
**注意**: 只有 CNN 模型的计算部分会在 GPU 上运行。构象搜索等其他步骤仍然在 CPU 上运行。因此，你通常可以同时指定 `--cpu` 和 `--gpu`。

```bash
# 使用 8 个 CPU 核心进行构象搜索，使用 GPU 进行 CNN 打分
gnina ... --cnn default --cpu 8 --gpu
```

## 6. 完整流程示例：使用 CNN 模型进行对接

**目标**：将配体 `new_ligand.pdbqt` 对接到受体 `receptor.pdbqt`，使用参考配体 `ref_ligand.pdbqt` 定义活性位点，并利用 `dense` CNN 模型和 GPU 进行计算。

### 步骤一：准备输入文件 (PDBQT)
确保 `receptor.pdbqt`, `new_ligand.pdbqt`, 和 `ref_ligand.pdbqt` 文件已通过 ADT 或 OpenBabel 正确制备。

### 步骤二：运行 GNINA 对接

在终端中执行以下命令：
```bash
gnina -r receptor.pdbqt \
      -l new_ligand.pdbqt \
      --autobox_ligand ref_ligand.pdbqt \
      --autobox_add 8 \
      --cnn dense \
      --exhaustiveness 16 \
      --num_modes 20 \
      --cpu 8 \
      --gpu \
      -o gnina_dense_output.pdbqt \
      --log gnina_dense_log.txt
```
**命令解释**:
-   `--autobox_ligand ... --autobox_add 8`: 继承自 Smina 的功能，自动定义搜索空间。
-   `--cnn dense`: 使用 `dense` CNN 模型进行打分和排序。
-   `--exhaustiveness 16`: 增加构象搜索的穷尽度。
-   `--cpu 8 --gpu`: 分配 8 个 CPU 核心用于搜索，并使用 GPU 加速 CNN 计算。
-   `-o` 和 `--log`: 指定输出文件。

### 步骤三：解读输出结果
分析生成的 `gnina_dense_log.txt` 和 `gnina_dense_output.pdbqt` 文件。

## 7. 结果解读

### 7.1 日志文件 (Log File)
打开 `gnina_dense_log.txt`，你会看到一个包含额外列的表格：

```
CNN models: ['dense']
-----+------------+----------+----------+----------+-------------
Mode | Affinity   | CNNscore | CNNaffinity | RMSD_lb  | RMSD_ub
     | (kcal/mol) |          | (kcal/mol)  |          |
-----+------------+----------+----------+----------+-------------
   1 |       -7.2 |    0.985 |       -8.9 |    0.000 |    0.000
   2 |       -7.1 |    0.972 |       -8.7 |    1.543 |    2.101
   3 |       -6.5 |    0.850 |       -7.5 |    2.011 |    2.890
   4 |       -7.0 |    0.451 |       -6.2 |    5.432 |    7.813
...
```
-   **排序依据**: 默认情况下，输出的构象（Mode）是**按 `CNNscore` 从高到低排序的**。Mode 1 是 CNN 模型认为最可能是正确姿态的构象。
-   `Affinity`: 经典的 Vina/Smina 打分。
-   `CNNscore`: 姿态置信度。Mode 1 的 `0.985` 表示模型有 98.5% 的把握认为这是一个高质量的结合姿态。
-   `CNNaffinity`: CNN 预测的结合能。

### 7.2 输出结构文件 (`out.pdbqt`)
输出的 PDBQT 文件在 `REMARK` 行中包含了所有三个关键打分值：

```pdbqt
MODEL 1
REMARK VINA RESULT:       -7.2      0.000      0.000
REMARK CNNscore: 0.985
REMARK CNNaffinity: -8.9
... (原子坐标) ...
ENDMDL
MODEL 2
REMARK VINA RESULT:       -7.1      1.543      2.101
REMARK CNNscore: 0.972
REMARK CNNaffinity: -8.7
... (原子坐标) ...
ENDMDL
```
这使得在可视化软件中检查每个构象的得分变得非常方便。

### 7.3 如何选择最终构象?

理想情况下，你应该寻找一个**三者兼优**的构象：
1.  **高 `CNNscore`** (e.g., > 0.8): 表明这是一个高质量、可信的结合姿态。
2.  **低 `CNNaffinity`**: 表明 CNN 模型预测其结合力强。
3.  **低 `Affinity`**: 表明经典打分函数也认为它是一个好的构象。

当 `CNNscore` 和经典 `Affinity` 的排序不一致时（例如，Mode 4 的 `Affinity` 比 Mode 3 好，但 `CNNscore` 低得多），通常**更应该相信 `CNNscore` 的排序**，因为这正是 GNINA 的优势所在。

## 8. 高级技巧与最佳实践

### 8.1 选择合适的 CNN 模型

GNINA 提供了多个预训练模型，可以通过 `--cnn` 参数指定。一些常用的模型包括：
-   `default` (`crossdock_default_v1.1`): 通用模型，速度和精度的良好平衡。
-   `dense`: 更大、更深的模型，计算量更大，但通常在亲和力预测上更准确。适合用于最终的精细打分或 re-scoring。
-   `redock_default`: 专门在 PDBbind 的 redocking 任务上训练的模型，可能在姿态预测上表现更好。

你甚至可以进行**模型集成 (ensemble)**，即同时使用多个模型打分：
```bash
gnina ... --cnn dense default
```
日志中会为每个模型提供单独的列。

### 8.2 虚拟筛选策略

由于 CNN 计算成本较高，对超大型库进行全流程 GNINA 对接可能不现实。一个高效的分层策略是：

1.  **初筛 (Smina/Vina)**: 使用快速的 Smina 对整个化合物库进行对接，保留每个分子得分最好的前 N 个构象（或保留得分高于某个阈值的分子）。
2.  **精筛 (GNINA Re-scoring)**: 对初筛中得到的“潜力股”分子及其构象，使用 GNINA 的 `dense` 模型和 `--score_only` 模式进行重新打分。
3.  **最终排序**: 根据 `CNNscore` 和 `CNNaffinity` 对精筛结果进行最终排序，选出最佳候选分子进行后续实验验证。

### 8.3 继承 Smina 的所有高级功能

不要忘记，GNINA 就是 Smina。所有 Smina 的高级功能都可以无缝使用：
-   `--custom_scoring`: 修改经典打分函数的权重。
-   `--atom_terms`: 分析每个原子对经典打分函数的贡献。
-   `--minimize`: 对输入构象进行能量最小化。

## 9. 常见问题与解决方案 (FAQ)

1.  **GNINA 运行得非常慢怎么办？**
    -   这是正常的，尤其是与 Vina 相比。CNN 计算是密集型的。
    -   **解决方案**：确保你使用的是 GPU 版本 (`gnina-gpu`) 并添加了 `--gpu` 标志。如果没有 GPU，可以减小 `--exhaustiveness`，或者采用分层筛选策略。

2.  **我应该相信哪个分数：`Affinity`, `CNNscore`, 还是 `CNNaffinity`?**
    -   `CNNscore` 是评估**姿态可信度**的最佳指标。
    -   `CNNaffinity` 是评估**结合强度**的最佳指标。
    -   `Affinity` 是一个有用的参考，特别是当它与 CNN 的结果趋势一致时。
    -   **首选 `CNNscore` 来排序和选择最佳姿态，然后参考 `CNNaffinity` 来评估结合强度。**

3.  **运行 GPU 版本时出错，提示 CUDA 错误。**
    -   **原因**：NVIDIA 驱动版本、CUDA toolkit 版本或 PyTorch 与 GPU 的兼容性问题。
    -   **解决方案**：最可靠的方法是使用 `conda` 安装 `gnina-gpu`，它会自动处理 `cudatoolkit` 的依赖。确保你的 NVIDIA 驱动是最新的。如果问题持续，检查 GNINA 的 GitHub Issues 页面看是否有类似问题。

4.  **`CNNscore` 很高，但 `CNNaffinity` 也很高（不利），怎么办？**
    -   这表示模型认为这是一个**几何上合理但能量上不利**的构象。这可能是真实的物理化学情况（例如，一个去溶剂化惩罚很大的构象），也可能是模型的预测偏差。在分析时应持谨慎态度，并与其他构象进行比较。

---
```