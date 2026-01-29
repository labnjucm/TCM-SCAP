# NAMD 详细使用指南：可扩展的分子动力学模拟

## 目录
1. [什么是 NAMD?](#1-什么是-namd)
2. [核心概念：PSF/PDB 分离体系](#2-核心概念psfpdb-分离体系)
3. [安装 NAMD](#3-安装-namd)
4. [NAMD 标准工作流程](#4-namd-标准工作流程)
5. [NAMD 配置文件 (`.conf`) 详解](#5-namd-配置文件-conf-详解)
    - [5.1 输入/输出 (I/O)](#51-输入输出-io)
    - [5.2 模拟参数](#52-模拟参数)
    - [53 温度与压力控制](#53-温度与压力控制)
    - [5.4 约束与固定](#54-约束与固定)
6. [运行 NAMD](#6-运行-namd)
7. [完整流程实战：一个蛋白质在水中的模拟](#7-完整流程实战一个蛋白质在水中的模拟)
    - [步骤 0：系统准备 (使用 VMD/psfgen)](#步骤-0系统准备-使用-vmdpsfgen)
    - [步骤 1：能量最小化](#步骤-1能量最小化)
    - [步骤 2：平衡 (Heating & Equilibration)](#步骤-2平衡-heating--equilibration)
    - [步骤 3：生产 MD (Production)](#步骤-3生产-md-production)
8. [输出文件解读与分析](#8-输出文件解读与分析)
    - [8.1 日志文件 (`.log`)](#81-日志文件-log)
    - [8.2 轨迹文件 (`.dcd`)](#82-轨迹文件-dcd)
    - [8.3 使用 VMD 进行分析](#83-使用-vmd-进行分析)
9. [NAMD vs. GROMACS: 简要对比](#9-namd-vs-gromacs-简要对比)
10. [常见问题与解决方案 (FAQ)](#10-常见问题与解决方案-faq)

---

## 1. 什么是 NAMD?

**NAMD** (Not Another Molecular Dynamics program) 是一款为大规模并行计算设计的高性能、开源分子动力学模拟软件。它由伊利诺伊大学厄巴纳-香槟分校 (UIUC) 的理论与计算生物物理学小组开发，与可视化软件 **VMD** 和 **CHARMM 力场** 构成了紧密结合的生态系统。

**核心优势**:
-   **卓越的可扩展性**: NAMD 的设计初衷就是为了在从单台工作站到拥有数十万核心的超级计算机上高效运行。
-   **与 VMD 无缝集成**: 作为姊妹软件，VMD 是准备 NAMD 输入文件和分析其输出结果的最强大工具。
-   **支持多种力场**: 虽然与 CHARMM 关系最密切，但它同样支持 AMBER, OPLS 等力场。
-   **丰富的功能**: 除了标准的 MD 模拟，还支持自由能计算 (FEP)、拉伸分子动力学 (SMD) 等高级功能。

## 2. 核心概念：PSF/PDB 分离体系

理解 NAMD 的第一步是理解其处理分子结构的方式，这与 GROMACS 的 `topol.top` 有所不同。NAMD 沿用了 CHARMM 的文件体系，将**拓扑**和**坐标**信息分离在两个文件中。

<style>
.concept-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 20px 0;
}
.concept-card {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 20px;
}
.concept-card h3 {
    margin-top: 0;
    color: #0969da;
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 5px;
}
.concept-card p {
    color: #24292f;
}
.concept-card code {
    background-color: #afb8c133;
    padding: 2px 5px;
    border-radius: 4px;
}
@media (max-width: 600px) {
    .concept-grid {
        grid-template-columns: 1fr;
    }
}
</style>

<div class="concept-grid">
    <div class="concept-card">
        <h3>PSF 文件 (Protein Structure File)</h3>
        <p><strong>作用：分子拓扑的“蓝图”。</strong></p>
        <p>这个文件描述了分子的化学结构：原子类型、质量、电荷，以及哪些原子通过键、角、二面角连接在一起。<strong>它不包含任何三维坐标信息。</strong></p>
        <p>文件名通常为 <code>system.psf</code>。</p>
    </div>
    <div class="concept-card">
        <h3>PDB 文件 (Protein Data Bank)</h3>
        <p><strong>作用：原子坐标的“快照”。</strong></p>
        <p>这个文件只包含每个原子的三维坐标 (X, Y, Z)。它告诉 NAMD 分子在模拟开始时的空间位置。NAMD 会严格按照 PSF 文件中的原子顺序来读取 PDB 文件中的坐标。</p>
        <p>文件名通常为 <code>system.pdb</code>。</p>
    </div>
</div>

**关键点**: NAMD 在运行时，会将 `PSF` 文件中的拓扑信息和 `PDB` 文件中的坐标信息“合并”，从而构建一个完整的、可进行模拟的分子体系。

## 3. 安装 NAMD

最简单的方式是从 NAMD 官网下载预编译好的二进制版本：
> [https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD](https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD)

根据你的操作系统（Linux, Windows, macOS）和是否拥有 NVIDIA GPU (CUDA 版本) 选择合适的版本下载。下载解压后，`namd2` (或 `namd3`) 就是可执行文件。

## 4. NAMD 标准工作流程

与 GROMACS 类似，NAMD 的模拟流程也是一个多步骤的过程，旨在逐步将系统带入稳定状态。

<style>
.workflow-container-namd {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
}
.workflow-step-namd {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 10px 15px;
    margin: 8px 0;
    width: 90%;
    max-width: 600px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: relative;
}
.workflow-step-namd .step-number-namd {
    position: absolute;
    top: 50%;
    left: -15px;
    transform: translateY(-50%);
    background-color: #6f42c1;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
}
.workflow-step-namd h4 { margin: 5px 0; color: #6f42c1; }
.workflow-step-namd p { margin: 5px 0; font-size: 0.9em; color: #57606a; }
.workflow-arrow-namd { font-size: 24px; color: #8b949e; margin: -5px 0; }
</style>

<div class="workflow-container-namd">
    <div class="workflow-step-namd">
        <div class="step-number-namd">0</div>
        <h4>系统准备 (VMD/psfgen)</h4>
        <p>构建 PSF/PDB 文件，包括溶剂化和离子化。</p>
    </div>
    <div class="workflow-arrow-namd">↓</div>
    <div class="workflow-step-namd">
        <div class="step-number-namd">1</div>
        <h4>能量最小化</h4>
        <p>移除原子间的冲突，使系统达到一个局部能量最低点。</p>
    </div>
    <div class="workflow-arrow-namd">↓</div>
    <div class="workflow-step-namd">
        <div class="step-number-namd">2</div>
        <h4>平衡 (升温 & NPT)</h4>
        <p>逐步将系统加热到目标温度，然后在恒温恒压下使盒子密度稳定。</p>
    </div>
    <div class="workflow-arrow-namd">↓</div>
    <div class="workflow-step-namd" style="border-color: #1a7f37;">
        <div class="step-number-namd" style="background-color: #1a7f37;">3</div>
        <h4 style="color: #1a7f37;">生产 MD</h4>
        <p>在稳定的 NPT 系综下长时间运行，采集用于分析的轨迹数据。</p>
    </div>
    <div class="workflow-arrow-namd">↓</div>
    <div class="workflow-step-namd">
        <div class="step-number-namd">4</div>
        <h4>分析 (VMD)</h4>
        <p>计算 RMSD、RMSF、径向分布函数等性质。</p>
    </div>
</div>

## 5. NAMD 配置文件 (`.conf`) 详解

与 GROMACS 的 `.mdp` 文件类似，NAMD 的所有模拟参数都定义在一个配置文件（通常以 `.conf` 结尾）中。这是一个简单的文本文件。

### 5.1 输入/输出 (I/O)
```tcl
# --- Input Files ---
structure          system.psf      # PSF 拓扑文件
coordinates        system.pdb      # PDB 坐标文件
parameters         par_all36_prot.prm  # 力场参数文件 (可以有多个)
parameters         par_all36_na.prm
parameters         toppar_water_ions.str

# --- Output Files ---
outputName         output_prefix   # 输出文件的前缀 (e.g., output_prefix.dcd, output_prefix.log)
dcdfreq            5000            # 每 5000 步保存一次轨迹 (.dcd)
xstfreq            5000            # 每 5000 步保存一次盒子信息 (.xst)
outputEnergies     1000            # 每 1000 步在日志中打印一次能量
restartfreq        5000            # 每 5000 步保存一次重启文件
```

### 5.2 模拟参数
```tcl
# --- Simulation Parameters ---
firsttimestep      0               # 初始时间步
numsteps           500000          # 总模拟步数 (500,000 steps * 2 fs/step = 1 ns)
timestep           2.0             # 时间步长 (fs)
integrator         Langevin        # 积分器类型

# --- Force Field Parameters ---
exclude            scaled1-4       # 1-4 相互作用缩放
1-4scaling         1.0
switching          on              # 开启切换函数
switchdist         10.0            # 切换函数开始距离 (Å)
cutoff             12.0            # 截断距离 (Å)
pairlistdist       14.0            # 对列表生成距离 (Å)
PME                yes             # 使用 PME 计算长程静电
PMEGridSizeX       100             # PME 网格大小 (大致与盒子尺寸相当)
PMEGridSizeY       100
PMEGridSizeZ       120
```

### 5.3 温度与压力控制
```tcl
# --- Temperature Control ---
langevin           on              # 开启 Langevin 动力学
langevinDamping    1.0             # 阻尼系数 (1/ps)
langevinTemp       310.0           # 目标温度 (K)
langevinHydrogen   off             # 不对氢原子施加 Langevin 动力学

# --- Pressure Control ---
langevinPiston     on              # 开启 Langevin 活塞压力控制
langevinPistonTarget  1.01325      # 目标压力 (bar)
langevinPistonPeriod  200.0        # 活塞振荡周期 (fs)
langevinPistonDecay   100.0        # 活塞衰减时间 (fs)
useGroupPressure   yes             # 使用组压力计算 (推荐)
useFlexibleCell    no              # 盒子各向同性变化
```

### 5.4 约束与固定
```tcl
# --- Constraints ---
rigidBonds         all             # 约束所有含氢的键 (允许 2fs 时间步长)
# 或者使用 SHAKE
# consref            initial.pdb
# conkfile           initial.pdb
# conkcol            B
# constraints        on

# --- Fixed Atoms ---
# fixedAtoms         on              # 固定原子开关
# fixedAtomsFile     fixed_atoms.pdb # 指定哪些原子被固定
# fixedAtomsCol      B               # PDB 文件中 B-factor 列非零的原子被固定
```

## 6. 运行 NAMD

NAMD 通过命令行启动。

#### 单核运行
```bash
namd2 my_simulation.conf > my_simulation.log
```

#### 多核并行运行 (推荐)
NAMD 使用 `charmrun` 工具进行并行。
```bash
# 在本地机器上使用 8 个核心运行
charmrun +p8 namd2 my_simulation.conf > my_simulation.log
```
- `+pN`: 指定使用 N 个处理器核心。

## 7. 完整流程实战：一个蛋白质在水中的模拟

### 步骤 0：系统准备 (使用 VMD/psfgen)
NAMD 本身不负责构建系统。这个过程通常在 VMD 中使用 `psfgen` 包完成。
1.  **加载 PDB**: `package require psfgen; topology top_all36_prot.rtf; segment PRO {pdb protein.pdb}`
2.  **猜测缺失坐标**: `guesscoord`
3.  **溶剂化**: 使用 VMD 的 `solvate` 插件。
4.  **离子化**: 使用 VMD 的 `autoionize` 插件。
5.  **生成文件**: `writepsf system.psf; writepdb system.pdb`

**这一步完成后，你将得到 `system.psf`, `system.pdb` 和力场参数文件。**

### 步骤 1：能量最小化
创建一个 `minim.conf` 文件。
```tcl
# --- minim.conf ---
structure          system.psf
coordinates        system.pdb
parameters         par_all36_prot.prm
...
outputName         min_out

# --- Simulation ---
temperature        0
minimize           10000   # 运行 10000 步能量最小化
... (其他 I/O 和力场参数)
```
**运行**: `namd2 minim.conf > minim.log`

### 步骤 2：平衡 (Heating & Equilibration)
创建一个 `equil.conf` 文件。这一步通常会固定或约束蛋白质主链，让水分子先弛豫。
```tcl
# --- equil.conf ---
structure          system.psf
coordinates        min_out.coor  # 从最小化结束时的坐标开始
parameters         ...
outputName         equil_out

# --- Restart from minimization ---
binCoordinates     min_out.coor
binVelocities      min_out.vel   # 可选

# --- Simulation ---
numsteps           50000         # 运行 100 ps
timestep           2.0

# --- Temperature Control ---
langevin           on
langevinTemp       310.0
# ... 其他温度和压力参数

# --- Constraints (restrain protein backbone) ---
constraints        on
consref            system.pdb   # 参考结构
conkfile           system.pdb   # 指定约束力常数的文件
conkcol            B            # 使用 PDB 的 B-factor 列作为力常数
```
**运行**: `charmrun +p8 namd2 equil.conf > equil.log`

### 步骤 3：生产 MD (Production)
创建一个 `prod.conf` 文件，它与 `equil.conf` 非常相似，但通常会移除约束，并延长模拟时间。
```tcl
# --- prod.conf ---
# ... 与 equil.conf 大部分相同 ...
coordinates        equil_out.coor
binCoordinates     equil_out.coor
binVelocities      equil_out.vel
outputName         prod_out

# --- Simulation ---
numsteps           5000000       # 运行 10 ns

# --- Remove constraints ---
constraints        off
# ...
```
**运行**: `charmrun +p8 namd2 prod.conf > prod.log`

## 8. 输出文件解读与分析

### 8.1 日志文件 (`.log`)
-   **能量信息**: 搜索 `ETITLE:`，你会看到能量项的表头，下面是每一步的能量数据。
    ```
    ETITLE:      TS           BOND          ANGLE          DIHED          IMPRP        ELECT            VDW       BOUNDARY           MISC        KINETIC ...
    ENERGY:       0   1305.3582     1896.7628      3336.5682       198.8145   -155829.4310     16027.6718      0.0000 ...
    ```
-   **性能信息**: 在文件末尾，`TIMING:` 部分显示了每个核心的运行时间和性能（ns/day）。

### 8.2 轨迹文件 (`.dcd`)
这是一个二进制文件，包含了模拟过程中原子坐标的“电影”。它本身不可读，需要使用 VMD 等软件进行可视化和分析。

### 8.3 使用 VMD 进行分析
VMD 是分析 NAMD 轨迹的最佳工具。
1.  **启动 VMD**。
2.  在 VMD 的 Tk Console 中输入：
    ```tcl
    # 加载拓扑结构
    mol new system.psf
    
    # 加载轨迹文件到该分子
    mol addfile prod_out.dcd waitfor all
    ```
3.  现在你可以在 VMD 主窗口中播放模拟动画。
4.  使用 VMD 的分析工具（如 `Extensions > Analysis > RMSD Trajectory Tool`）来计算各种性质。

## 9. NAMD vs. GROMACS: 简要对比

| 特性 | NAMD | GROMACS |
| :--- | :--- | :--- |
| **核心优势** | 极致的并行可扩展性，适合超算 | 单节点/单 GPU 性能极高，内置丰富的分析工具 |
| **生态系统** | 与 VMD 紧密集成，偏向 CHARMM 力场 | 自成体系，支持多种力场，社区教程丰富 |
| **文件系统** | PSF/PDB 分离体系 | 拓扑 (`.top`) 和坐标 (`.gro`) 结合 |
| **配置** | 单一 `.conf` 文件 | 多个 `.mdp` 文件用于不同阶段 |
| **学习曲线** | 依赖 VMD 进行系统构建，初学者可能感觉步骤分散 | 工作流更线性，`gmx` 命令一体化程度高 |

**结论**: 两者都是顶级的 MD 软件。如果你的主要目标是在超级计算机上运行超大规模体系，NAMD 是不二之选。如果你的工作主要在单台工作站或小型集群上，GROMACS 可能提供更快的单点性能和更便捷的分析体验。

## 10. 常见问题与解决方案 (FAQ)

1.  **模拟“爆炸”了 (Simulation blew up!)**
    -   **现象**: 能量值突然变成 `NaN`，程序崩溃。
    -   **原因**:
        -   能量最小化不充分，初始结构有原子冲突。
        -   时间步长 (`timestep`) 对于你的体系来说太大了。
        -   平衡不充分，系统在进入生产阶段时仍不稳定。
    -   **解决**: 增加最小化步数；减小时间步长（如从 2.0fs 降到 1.0fs）；延长平衡时间。

2.  **如何重启一个中断的计算？**
    -   NAMD 会自动生成 `.restart.coor`, `.restart.vel`, `.restart.xsc` 文件。在你的新 `.conf` 文件中，指定这些文件即可从断点处继续：
        ```tcl
        binCoordinates   my_sim.restart.coor
        binVelocities    my_sim.restart.vel
        extendedSystem   my_sim.restart.xsc
        firsttimestep    [last_step_from_log]
        ```

3.  **为什么我的模拟盒子变形了？**
    -   如果你使用了 `useFlexibleCell yes`，盒子可以在各个维度上独立变化，可能导致变形。对于水盒子中的球蛋白，通常使用 `useFlexibleCell no` 保持盒子形状。

---