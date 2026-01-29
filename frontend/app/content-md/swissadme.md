# SwissADME 详细使用指南：药物发现的智能向导

## 目录
1. [什么是 SwissADME?](#1-什么是-swissadme)
2. [核心优势：为什么选择 SwissADME?](#2-核心优势为什么选择-swissadme)
3. [使用方法：简单三步，洞悉分子特性](#3-使用方法简单三步洞悉分子特性)
4. [结果解读：关键参数详解](#4-结果解读关键参数详解)
    - [4.1 物理化学性质 (Physicochemical Properties)](#41-物理化学性质-physicochemical-properties)
    - [4.2 亲脂性 (Lipophilicity)](#42-亲脂性-lipophilicity)
    - [4.3 水溶性 (Water Solubility)](#43-水溶性-water-solubility)
    - [4.4 药代动力学 (Pharmacokinetics)](#44-药代动力学-pharmacokinetics)
    - [4.5 类药性 (Drug-likeness)](#45-类药性-drug-likeness)
    - [4.6 药化学友好性 (Medicinal Chemistry)](#46-药化学友好性-medicinal-chemistry)
5. [特色可视化工具解读](#5-特色可视化工具解读)
    - [5.1 煮蛋图 (BOILED-Egg Plot)](#51-煮蛋图-boiled-egg-plot)
    - [5.2 生物利用度雷达图 (Bioavailability Radar)](#52-生物利用度雷达图-bioavailability-radar)
6. [最佳实践与局限性](#6-最佳实践与局限性)
7. [常见问题 (FAQ)](#7-常见问题-faq)

---

## 1. 什么是 SwissADME?

**SwissADME** 是由瑞士生物信息学研究所 (SIB) 开发的一款功能强大、界面友好的免费在线工具。它旨在帮助化学家和药理学家快速评估小分子的**药代动力学 (Pharmacokinetics)**、**类药性 (Drug-likeness)** 和**药化学友好性 (Medicinal Chemistry Friendliness)**。

与 PREADMET 类似，SwissADME 也是一个 ADMET 预测工具，但它更侧重于物理化学性质、类药性规则和一些独特的、直观的可视化方法，使其成为药物发现早期阶段进行分子设计和筛选的理想伴侣。

## 2. 核心优势：为什么选择 SwissADME?

<style>
.advantage-box-swiss {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background-color: #fffbe6;
    border-left: 5px solid #faad14;
    padding: 20px;
    margin: 20px auto;
    max-width: 700px;
    border-radius: 0 8px 8px 0;
}
.advantage-box-swiss h3 {
    margin-top: 0;
    color: #faad14;
}
.advantage-box-swiss ul {
    list-style-type: none;
    padding-left: 0;
}
.advantage-box-swiss li {
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
}
.advantage-box-swiss .icon {
    color: #faad14;
    font-size: 1.2em;
    margin-right: 10px;
    flex-shrink: 0;
}
</style>

<div class="advantage-box-swiss">
    <h3>SwissADME 的独特魅力</h3>
    <ul>
        <li><span class="icon">🎨</span><div><strong>直观的可视化</strong>: 独创的“煮蛋图”(BOILED-Egg) 和“生物利用度雷达图”将复杂的 ADMET 数据转化为一目了然的图形，极大地方便了决策。</div></li>
        <li><span class="icon">🚀</span><div><strong>快速且免费</strong>: 无需注册，完全免费。输入分子后几乎可以立即得到结果，支持批量处理。</div></li>
        <li><span class="icon">🧠</span><div><strong>全面的类药性评估</strong>: 集成了多种经典的类药性规则（如 Lipinski, Ghose, Veber 等），并提供了自定义的“合成可及性分数”(Synthetic Accessibility)。</div></li>
        <li><span class="icon">💊</span><div><strong>关注药化友好性</strong>: 提供了 PAINS (泛分析干扰化合物) 和 Brenk 结构警报筛选，帮助研究者避开已知的“麻烦”分子结构。</div></li>
    </ul>
</div>

## 3. 使用方法：简单三步，洞悉分子特性

SwissADME 的使用流程极其简单。

1.  **访问官网**: 打开 SwissADME 网站：[http://www.swissadme.ch/](http://www.swissadme.ch/)
2.  **输入分子**:
    -   在左侧的大输入框中，直接粘贴一个或多个分子的 **SMILES** 字符串（每行一个）。
    -   或者，使用页面上方的分子编辑器绘制结构，然后点击 "Paste SMILES in the list" 将其添加到列表中。
3.  **运行计算**: 点击 "Run" 按钮。几秒钟后，结果就会显示在页面下方。如果有多个分子，你可以点击每个分子的结果行来查看其详细报告。

## 4. 结果解读：关键参数详解

SwissADME 的结果页面信息量巨大，分为几个主要部分。

### 4.1 物理化学性质 (Physicochemical Properties)
-   **Formula, MW**: 分子式和分子量 (g/mol)。
-   **#Heavy atoms, #Arom. heavy atoms**: 重原子数和芳香重原子数。
-   **Fraction Csp3**: sp3 杂化碳原子占总碳原子数的比例。**数值越高，分子的三维构象越丰富，通常被认为是好的性质**。
-   **#Rotatable bonds**: 可旋转键的数量。通常认为少于 10 个更好，以降低构象熵的损失。
-   **#H-bond acceptors/donors**: 氢键受体和供体的数量。
-   **Molar Refractivity**: 摩尔折射率，与分子的体积和极化性有关。
-   **TPSA**: **拓扑极性表面积 (Topological Polar Surface Area)**。这是一个非常重要的参数，与分子的被动转运（如肠道吸收和血脑屏障穿透）密切相关。**TPSA < 140 Å² 通常意味着良好的口服生物利用度**。

### 4.2 亲脂性 (Lipophilicity)
-   **Log P<sub>o/w</sub> (iLOGP, XLOGP3, WLOGP, MLOGP, SILICOS-IT)**: 正辛醇/水分配系数的对数，是衡量分子亲脂性的最重要指标。SwissADME 会提供多个不同算法计算的共识值。**理想的 logP 值通常在 1 到 3 之间，不应超过 5**。

### 4.3 水溶性 (Water Solubility)
-   **Log S (ESOL, Ali, SILICOS-IT)**: 预测分子的水溶性。
-   **Solubility**: 将 LogS 转换为更直观的溶解度单位 (mg/ml 或 mol/l)。
-   **Solubility Class**: 对溶解度进行分类（e.g., poorly, moderately, soluble）。**良好的水溶性是成药的关键因素之一**。

### 4.4 药代动力学 (Pharmacokinetics)
-   **GI absorption**: 预测人体肠道吸收 (HIA) 的能力，分为 `High` 和 `Low`。
-   **BBB permeant**: 预测是否能穿透血脑屏障 (BBB)。对于中枢神经系统药物是必需的，对于外周药物则应避免。
-   **P-gp substrate**: 预测是否是 P-糖蛋白 (P-gp) 的底物。如果是，分子可能会被主动泵出细胞，导致肠道吸收率降低或无法穿透血脑屏障。
-   **CYP Inhibitor**: 预测是否会抑制主要的细胞色素 P450 酶（1A2, 2C19, 2C9, 2D6, 3A4）。**抑制 CYP 酶是导致药物-药物相互作用的主要原因，应尽量避免**。
-   **Skin Permeation (Log Kp)**: 预测皮肤渗透性，对透皮给药的药物很重要。

### 4.5 类药性 (Drug-likeness)
-   **Lipinski (Pfizer) rule**: 经典的“五规则”。违反项越多，口服生物利用度差的可能性越大。
-   **Ghose (Amgen), Veber (GSK), Egan (Pharmacia), Muegge (Bayer)**: 其他制药公司提出的类似规则，从不同角度评估类药性。
-   **Bioavailability Score**: 基于多个参数计算的一个综合分数（0.17, 0.55, 0.56, 0.85），用于快速评估口服生物利用度的可能性。**0.55 或更高是比较理想的**。

### 4.6 药化学友好性 (Medicinal Chemistry)
-   **PAINS**: 筛选分子是否包含泛分析干扰化合物 (Pan-Assay Interference Compounds) 的结构片段。这些片段常在多种高通量筛选中产生假阳性信号。**强烈建议避开带有 PAINS 警报的分子**。
-   **Brenk**: 筛选分子是否包含已知的有毒、化学不稳定或代谢不稳定的结构片段。
-   **Lead-likeness**: 评估分子是否符合“先导化合物”的性质（通常比最终药物分子更小、亲脂性更低）。
-   **Synthetic accessibility**: 合成可及性分数 (1-10)。**分数越低，代表分子在化学上越容易合成**。

## 5. 特色可视化工具解读

### 5.1 煮蛋图 (BOILED-Egg Plot)

<style>
.egg-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    position: relative;
    width: 350px;
    height: 250px;
    margin: 20px auto;
    border: 2px solid #ccc;
    border-radius: 10px;
    background: #f9f9f9;
}
.egg-yolk {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 180px;
    height: 180px;
    background-color: #ffc;
    border: 2px dashed #f90;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    font-weight: bold;
    color: #f90;
}
.egg-white {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 300px;
    height: 220px;
    background-color: rgba(255, 255, 255, 0.8);
    border: 2px dashed #aaa;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    font-weight: bold;
    color: #555;
}
.egg-label { position: absolute; font-size: 0.8em; }
.egg-xlabel { bottom: 5px; left: 50%; transform: translateX(-50%); }
.egg-ylabel { top: 50%; left: 5px; transform: translateY(-50%) rotate(-90deg); }
.egg-point-blue { position: absolute; width: 10px; height: 10px; background: blue; border-radius: 50%; }
.egg-point-red { position: absolute; width: 10px; height: 10px; background: red; border-radius: 50%; }
</style>

<div class="egg-container">
    <div class="egg-white">
        <div class="egg-yolk">高概率穿透 BBB</div>
        高概率被动肠道吸收
    </div>
    <div class="egg-label egg-xlabel">WLOGP (亲脂性) →</div>
    <div class="egg-label egg-ylabel">TPSA (极性) →</div>
    <!-- 示例点 -->
    <div class="egg-point-blue" style="bottom: 80px; left: 180px;" title="在蛋黄区：高HIA, 高BBB穿透"></div>
    <div class="egg-point-blue" style="bottom: 100px; left: 100px;" title="在蛋白区：高HIA, 低BBB穿透"></div>
    <div class="egg-point-red" style="bottom: 50px; left: 280px;" title="灰色区，且是P-gp底物：低HIA, 低BBB穿透"></div>
</div>

这是 SwissADME 的标志性工具。它将两个关键参数——**亲脂性 (WLOGP)** 和**极性 (TPSA)**——映射到一个二维图中。
-   **蛋黄区 (Yolk)**: 代表具有高概率**穿透血脑屏障 (BBB)** 的化学空间。
-   **蛋白区 (White)**: 代表具有高概率**被动肠道吸收 (HIA)** 的化学空间。
-   **灰色区域**: 代表两种被动吸收能力都较差的化学空间。
-   **点的颜色**:
    -   **蓝色点**: 预测为非 P-gp 底物。
    -   **红色点**: 预测为 P-gp 底物 (被动吸收会受阻)。

**如何使用**: 一个理想的口服外周药物应该落在**蛋白区**（高 HIA，低 BBB），并且是**蓝色**的（非 P-gp 底物）。一个理想的口服中枢神经系统药物应该落在**蛋黄区**，并且是**蓝色**的。

### 5.2 生物利用度雷达图 (Bioavailability Radar)

这是一个六边形的雷达图，展示了分子在六个关键物理化学性质上的表现是否符合口服生物利用度的要求。
-   **六个轴**:
    -   **LIPO**: 亲脂性 (XLOGP3, -0.7 到 +5.0)
    -   **SIZE**: 分子量 (MW, 150 到 500 g/mol)
    -   **POLAR**: 极性 (TPSA, 20 到 130 Å²)
    -   **INSOLU**: 水溶性 (Log S, 不超过 6)
    -   **INSATU**: 不饱和度 (sp3 碳比例, 不低于 0.25)
    -   **FLEX**: 柔性 (可旋转键, 不超过 9)
-   **如何解读**: 粉红色的区域代表了理想的参数范围。一个好的候选药物，其雷达图（粉色区域）应该尽可能地**饱满**，表示它在所有六个方面都表现良好。

## 6. 最佳实践与局限性

-   **快速迭代**: SwissADME 是为快速反馈而设计的。在设计分子时，可以随时将其 SMILES 粘贴进去，检查其性质是否改善。
-   **多维度考量**: 不要仅仅因为一个参数不理想就放弃一个分子。综合所有数据，特别是“煮蛋图”和“雷达图”，进行整体评估。
-   **模型局限性**: 所有预测都是基于模型的，它们无法预测所有情况，尤其是对于新颖的化学骨架。实验验证永远是金标准。

## 7. 常见问题 (FAQ)

1.  **SwissADME 和 PREADMET 有什么区别？**
    -   两者功能有重叠，但侧重点不同。SwissADME 强于物理化学性质、类药性规则和直观的可视化。PREADMET 则提供了更广泛的毒理学终点预测（如 Ames, hERG）和代谢（CYP 底物/抑制剂）模型。两者可以**互为补充**，结合使用。

2.  **我的分子在“煮蛋图”的灰色区域，是不是就没用了？**
    -   不一定。这只意味着它**被动吸收**的可能性较低。它仍然可能通过主动转运等其他机制被吸收。但这确实是一个警示信号，表明口服给药可能会遇到挑战。

3.  **什么是 PAINS 警报，为什么它很重要？**
    -   PAINS (泛分析干扰化合物) 是一些特定的化学结构，它们在多种生物检测中都倾向于产生假阳性结果，例如通过聚集、发生氧化还原反应或与蛋白质非特异性结合。在药物发现早期排除这些分子可以节省大量的时间和资源，避免追逐错误的“活性”信号。

---