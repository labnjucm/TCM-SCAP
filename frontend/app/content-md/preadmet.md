# PREADMET 详细使用指南：药物发现的 ADMET 预测利器

## 目录
1. [什么是 PREADMET?](#1-什么是-preadmet)
2. [核心优势：为什么使用 PREADMET?](#2-核心优势为什么使用-preadmet)
3. [访问与注册](#3-访问与注册)
4. [使用方法：一步一步教你预测](#4-使用方法一步一步教你预测)
    - [4.1 工作流程图](#41-工作流程图)
    - [4.2 详细步骤](#42-详细步骤)
5. [结果解读：关键 ADMET 参数详解](#5-结果解读关键-admet-参数详解)
    - [5.1 吸收 (Absorption)](#51-吸收-absorption)
    - [5.2 分布 (Distribution)](#52-分布-distribution)
    - [5.3 代谢 (Metabolism)](#53-代谢-metabolism)
    - [5.4 毒性 (Toxicity)](#54-毒性-toxicity)
6. [可视化报告：雷达图与其他](#6-可视化报告雷达图与其他)
7. [最佳实践与局限性](#7-最佳实践与局限性)
8. [常见问题 (FAQ)](#8-常见问题-faq)

---

## 1. 什么是 PREADMET?

**PREADMET** 是一个广泛使用的、基于网络的 ADMET 性质预测服务器。ADMET 是药物发现领域中的一个关键概念，代表了药物在体内的四个主要过程及其潜在毒性：

-   **A**bsorption (吸收)
-   **D**istribution (分布)
-   **M**etabolism (代谢)
-   **E**xcretion (排泄)
-   **T**oxicity (毒性)

PREADMET 利用机器学习和 QSAR (定量构效关系) 模型，根据你输入的分子结构，快速预测其一系列关键的 ADMET 性质。这使得研究人员能够在药物发现的早期阶段，以零成本、零耗时的方式筛选和优化候选化合物，从而避免在后期开发中因不良的 ADMET 性质而导致失败。

## 2. 核心优势：为什么使用 PREADMET?

在众多预测工具中，PREADMET 因其以下优点而备受欢迎：

<style>
.advantage-box {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background-color: #e6f7ff;
    border-left: 5px solid #1890ff;
    padding: 20px;
    margin: 20px auto;
    max-width: 700px;
    border-radius: 0 8px 8px 0;
}
.advantage-box h3 {
    margin-top: 0;
    color: #1890ff;
}
.advantage-box ul {
    list-style-type: none;
    padding-left: 0;
}
.advantage-box li {
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
}
.advantage-box .icon {
    color: #1890ff;
    font-size: 1.2em;
    margin-right: 10px;
    flex-shrink: 0;
}
</style>

<div class="advantage-box">
    <h3>PREADMET 的核心价值</h3>
    <ul>
        <li><span class="icon">✓</span><div><strong>易于使用</strong>: 无需安装任何软件，只需一个浏览器即可访问。直观的界面使得非计算化学专业的研究人员也能轻松上手。</div></li>
        <li><span class="icon">✓</span><div><strong>免费访问</strong>: 对于学术用户完全免费，极大地降低了早期药物筛选的门槛。</div></li>
        <li><span class="icon">✓</span><div><strong>全面覆盖</strong>: 提供涵盖 ADMET 各个方面的多种预测模型，包括肠道吸收、血脑屏障穿透、CYP450 代谢和多种毒性终点。</div></li>
        <li><span class="icon">✓</span><div><strong>快速反馈</strong>: 提交分子后，通常在几秒到几分钟内即可获得详细的预测报告，加速了“设计-预测-再设计”的循环。</div></li>
    </ul>
</div>

## 3. 访问与注册

1.  **访问官网**: 在浏览器中打开 PREADMET 的官方网站。由于网址可能会变更，建议通过搜索引擎搜索 "PREADMET" 来找到最新的官方链接。一个常用的链接是：[http://preadmet.bmdrc.kr/](http://preadmet.bmdrc.kr/)
2.  **注册**: PREADMET 通常需要用户注册一个免费的学术账号才能使用。点击 "Sign Up" 或 "Register"，使用你的机构/学术邮箱进行注册。

## 4. 使用方法：一步一步教你预测

### 4.1 工作流程图

<style>
.workflow-container-preadmet {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 15px;
    padding: 20px;
    background-color: #f6f8fa;
    border-radius: 8px;
    margin: 20px auto;
}
.workflow-step-preadmet {
    background-color: #fff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 15px;
    width: 180px;
    height: 150px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.workflow-step-preadmet h4 { margin-top: 0; color: #0366d6; font-size: 1em; }
.workflow-step-preadmet p { font-size: 0.85em; color: #586069; margin: 0; }
.workflow-arrow-preadmet { font-size: 2.5em; color: #586069; }

@media (max-width: 600px) {
    .workflow-container-preadmet { flex-direction: column; }
    .workflow-arrow-preadmet { transform: rotate(90deg); margin: 0; }
}
</style>

<div class="workflow-container-preadmet">
    <div class="workflow-step-preadmet">
        <h4>第一步: 输入分子</h4>
        <p>绘制结构或粘贴 SMILES/SDF 文件</p>
    </div>
    <div class="workflow-arrow-preadmet">→</div>
    <div class="workflow-step-preadmet">
        <h4>第二步: 选择模型</h4>
        <p>勾选你想要预测的 ADMET 属性</p>
    </div>
    <div class="workflow-arrow-preadmet">→</div>
    <div class="workflow-step-preadmet">
        <h4>第三步: 运行预测</h4>
        <p>点击 "Prediction Start" 按钮</p>
    </div>
    <div class="workflow-arrow-preadmet">→</div>
    <div class="workflow-step-preadmet" style="border-color: #28a745;">
        <h4 style="color:#28a745;">第四步: 查看结果</h4>
        <p>分析生成的详细报告和图表</p>
    </div>
</div>

### 4.2 详细步骤

1.  **登录**: 登录你的 PREADMET 账号。
2.  **输入分子**: 你有多种方式输入分子结构：
    -   **绘制结构**: 使用网页内嵌的分子编辑器（如 Marvin JS）直接绘制化学结构。
    -   **粘贴 SMILES**: 在指定的文本框中粘贴分子的 SMILES 字符串，这是最快捷的方式。
    -   **上传文件**: 上传包含分子结构的 `.sdf` 或 `.mol` 文件。
3.  **选择预测模型**: 在页面上，你会看到一个包含所有可用模型的列表，按 ADMET 分类。你可以勾选所有模型，或只选择你当前关心的几个。
4.  **开始预测**: 点击 "Prediction Start" 或类似的按钮。
5.  **查看结果**: 稍等片刻，页面会自动跳转到结果报告页。报告通常包含一个汇总表、每个模型的详细预测值以及可视化图表。

## 5. 结果解读：关键 ADMET 参数详解

这是使用 PREADMET 最重要的一步。理解每个参数的含义至关重要。

<style>
.results-table-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 20px auto;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    overflow-x: auto;
}
.results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}
.results-table th, .results-table td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid #e1e4e8;
}
.results-table th {
    background-color: #f6f8fa;
    font-weight: 600;
}
.results-table .category-header {
    background-color: #0366d6;
    color: white;
    font-size: 1.1em;
}
.results-table .good { color: #28a745; font-weight: bold; }
.results-table .bad { color: #d73a49; font-weight: bold; }
.results-table .neutral { color: #e36209; }
</style>

<div class="results-table-container">
<table class="results-table">
    <thead>
        <tr><th colspan="3" class="category-header">吸收 (Absorption)</th></tr>
        <tr>
            <th>参数</th>
            <th>描述</th>
            <th>理想范围/解读</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Caco-2</td>
            <td>模拟人体小肠上皮细胞的体外模型，预测肠道吸收能力。单位：nm/sec。</td>
            <td>&lt; 4: <span class="bad">低</span> | 4-70: <span class="neutral">中</span> | &gt; 70: <span class="good">高</span></td>
        </tr>
        <tr>
            <td>HIA (Human Intestinal Absorption)</td>
            <td>预测口服药物被人体肠道吸收的百分比。</td>
            <td>&lt; 30%: <span class="bad">吸收差</span> | &gt; 80%: <span class="good">吸收好</span></td>
        </tr>
        <tr>
            <td>MDCK</td>
            <td>另一种细胞模型，常用于评估化合物是否为 P-糖蛋白 (P-gp) 的底物。</td>
            <td>数值越高，通透性越好。</td>
        </tr>
    </tbody>
    <thead>
        <tr><th colspan="3" class="category-header">分布 (Distribution)</th></tr>
    </thead>
    <tbody>
        <tr>
            <td>BBB (Blood-Brain Barrier)</td>
            <td>预测化合物穿透血脑屏障的能力。常以 logBB 表示。</td>
            <td>&gt; 0: <span class="good">易穿透 (中枢神经系统药物)</span><br>&lt; -1: <span class="good">不易穿透 (外周药物)</span></td>
        </tr>
        <tr>
            <td>PPB (Plasma Protein Binding)</td>
            <td>预测药物与血浆蛋白（主要是白蛋白）结合的百分比。</td>
            <td>&lt; 90%: <span class="good">结合弱 (更多游离药物)</span><br>&gt; 90%: <span class="bad">结合强 (可能影响药效)</span></td>
        </tr>
    </tbody>
    <thead>
        <tr><th colspan="3" class="category-header">代谢 (Metabolism)</th></tr>
    </thead>
    <tbody>
        <tr>
            <td>CYP Inhibition / Substrate</td>
            <td>预测化合物是否会抑制或成为细胞色素P450 (CYP) 酶的底物 (e.g., 3A4, 2D6, 2C9)。</td>
            <td><span class="bad">抑制剂</span>: 可能导致药物-药物相互作用。<br><span class="good">非抑制剂</span>: 风险较低。</td>
        </tr>
    </tbody>
    <thead>
        <tr><th colspan="3" class="category-header">毒性 (Toxicity)</th></tr>
    </thead>
    <tbody>
        <tr>
            <td>Ames Test</td>
            <td>预测化合物是否具有致突变性。</td>
            <td><span class="bad">Mutagen</span>: 潜在致癌风险。<br><span class="good">Non-mutagen</span>: 安全性更高。</td>
        </tr>
        <tr>
            <td>hERG Inhibition</td>
            <td>预测化合物是否会抑制 hERG 钾离子通道，这与潜在的心脏毒性（长QT综合征）相关。</td>
            <td><span class="bad">抑制剂</span>: 高心脏毒性风险。<br><span class="good">非抑制剂</span>: 风险较低。</td>
        </tr>
        <tr>
            <td>Carcinogenicity</td>
            <td>预测化合物对小鼠或大鼠的致癌性。</td>
            <td><span class="bad">Carcinogen</span>: 潜在致癌物。<br><span class="good">Non-carcinogen</span>: 安全性更高。</td>
        </tr>
    </tbody>
</table>
</div>

## 6. 可视化报告：雷达图与其他

PREADMET 通常会提供一个**雷达图 (Radar Chart)** 来汇总关键的类药性 (Drug-likeness) 和 ADMET 风险。

-   **如何解读**: 雷达图的每个轴代表一个性质。理想的候选药物其数据点应尽可能地**落在绿色的“安全/理想”区域内**。如果某个点延伸到黄色或红色区域，表示该性质存在潜在风险，需要特别关注。

这个图表提供了一个非常直观的、一目了然的分子“体检报告”。

## 7. 最佳实践与局限性

-   **用于比较，而非绝对**: PREADMET 的预测值是基于模型的，存在误差。它最强大的用途是在一系列类似化合物中进行**横向比较**，以筛选出性质相对更优的分子。
-   **注意适用域 (Applicability Domain)**: 模型的预测精度对于与训练集结构差异巨大的新颖分子可能会下降。如果你的分子结构非常独特，应对预测结果持谨慎态度。
-   **早期筛选工具**: PREADMET 是一个**早期发现**的工具，用于指导化学合成和优化。它不能替代任何体外或体内实验。预测结果好的化合物仍需通过实验进行验证。
-   **综合评估**: 不要只盯着一个参数。一个好的候选药物需要在吸收、分布、代谢、毒性等多个方面取得平衡。

## 8. 常见问题 (FAQ)

1.  **PREADMET 的预测结果可靠吗？**
    -   在模型的适用域内，其预测的准确性（通常以 Q², AUC 等指标衡量）在 70%-90% 之间，这对于早期筛选来说是相当可靠的。但它绝不能替代实验数据。

2.  **我可以一次提交多个分子吗？**
    -   可以。PREADMET 支持上传包含多个分子的 SDF 文件，进行批量预测。这对于虚拟筛选非常有用。

3.  **预测结果显示我的分子有毒性风险，我该怎么办？**
    -   这是计算化学指导药物设计的价值所在。分析你的分子结构，看看是哪个官能团或结构片段可能导致了毒性（例如，某些基团是已知的结构警报）。然后尝试对分子进行化学修饰，以消除或减弱该风险，再进行新一轮的预测。

4.  **网站上有很多模型版本，我应该用哪个？**
    -   通常，使用最新版本的模型是最佳选择，因为它们通常基于更大、更新的数据集进行训练，性能可能更好。

---