# AlphaFold DB - AI 预测的蛋白质结构数据库

## 简介

**AlphaFold Database** 是由 DeepMind 和 EMBL-EBI 合作构建的蛋白质结构预测数据库，包含超过 2 亿个蛋白质结构预测。基于 DeepMind 的 AlphaFold 2 深度学习模型，该数据库为尚未通过实验解析结构的蛋白质提供高精度的三维结构预测。

**核心特点**：
- 覆盖 UniProt 数据库中的大部分蛋白质序列
- 提供预测可信度评分（pLDDT）
- 免费开放访问
- 与 UniProt、PDB 等数据库集成

**应用场景**：
- 当 PDB 中无实验结构时的替代方案
- 结构生物学研究的起点
- 药物设计中的靶标结构预测
- 进化分析和功能预测

## 典型用例

### 1. 获取未解析结构的蛋白
研究新的疾病靶标，PDB 中无实验结构，使用 AlphaFold 预测结构进行初步分析。

### 2. 多结构域蛋白分析
预测全长蛋白结构，识别结构域边界和相互作用。

### 3. 突变体建模
基于野生型预测结构，进行点突变分析。

### 4. 蛋白-蛋白相互作用预测
使用 AlphaFold-Multimer 预测复合物结构。

## 输入/输出

### 输入
- **UniProt ID**：如 `P12345`、`Q9Y6K9`
- **基因名**：如 `TP53`、`EGFR`
- **蛋白名称**：如 "Tumor protein p53"
- **序列**：FASTA 格式（用于本地预测）

### 输出
- **PDB 格式**：预测的三维坐标
- **mmCIF 格式**：包含更多元数据
- **PAE（Predicted Aligned Error）**：残基对之间的位置误差预测
- **pLDDT（per-residue confidence）**：每个残基的可信度评分（0-100）
  - >90: 高可信度
  - 70-90: 良好
  - 50-70: 低可信度
  - <50: 不可信

## 快速上手

### Web 界面

1. **访问**: https://alphafold.ebi.ac.uk/
2. **搜索**：输入 UniProt ID、基因名或蛋白名
3. **查看结构**：
   - 3D 查看器（Mol*）
   - 按 pLDDT 着色（蓝色=高可信度，红色=低可信度）
   - 查看 PAE 图（评估结构域间相对位置可信度）
4. **下载**：
   - PDB/mmCIF 文件
   - PAE JSON 文件
   - pLDDT CSV 文件

### 与 PDB 的差异

| 特性 | PDB | AlphaFold DB |
|------|-----|--------------|
| 数据来源 | 实验解析 | AI 预测 |
| 精度 | 高（~2Å） | 视可信度而定 |
| 覆盖率 | ~20万结构 | >2亿蛋白 |
| 配体 | 包含 | 不包含 |
| 构象多样性 | 单一晶体构象 | 单一预测构象 |
| 柔性区域 | 可能缺失 | 低可信度 |
| 多聚体 | 包含 | 需单独预测 |

## 命令/API 示例

### 1. 直接下载 PDB 文件

```bash
# 通过 UniProt ID 下载
UNIPROT_ID="P12345"
curl -o ${UNIPROT_ID}.pdb \
  "https://alphafold.ebi.ac.uk/files/AF-${UNIPROT_ID}-F1-model_v4.pdb"

# 下载 mmCIF 格式
curl -o ${UNIPROT_ID}.cif \
  "https://alphafold.ebi.ac.uk/files/AF-${UNIPROT_ID}-F1-model_v4.cif"

# 下载 PAE JSON
curl -o ${UNIPROT_ID}_pae.json \
  "https://alphafold.ebi.ac.uk/files/AF-${UNIPROT_ID}-F1-predicted_aligned_error_v4.json"
```

### 2. Python 批量下载

```python
import requests
from pathlib import Path
import time

def download_alphafold(uniprot_id, output_dir='./alphafold_structures'):
    """
    从 AlphaFold DB 下载蛋白质结构
    
    Args:
        uniprot_id: UniProt ID (如 'P12345')
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    base_url = "https://alphafold.ebi.ac.uk/files"
    file_prefix = f"AF-{uniprot_id}-F1"
    
    files_to_download = {
        'pdb': f"{file_prefix}-model_v4.pdb",
        'cif': f"{file_prefix}-model_v4.cif",
        'pae': f"{file_prefix}-predicted_aligned_error_v4.json"
    }
    
    results = {}
    for file_type, filename in files_to_download.items():
        url = f"{base_url}/{filename}"
        response = requests.get(url)
        
        if response.status_code == 200:
            output_path = Path(output_dir) / filename
            output_path.write_bytes(response.content)
            results[file_type] = str(output_path)
            print(f"✓ 已下载: {filename}")
        else:
            print(f"✗ 下载失败: {filename} (HTTP {response.status_code})")
            results[file_type] = None
        
        time.sleep(0.1)  # 避免请求过快
    
    return results

# 批量下载
uniprot_ids = ['P04637', 'P00533', 'P53779']  # TP53, EGFR, MAPK6
for uid in uniprot_ids:
    print(f"\n下载 {uid}...")
    download_alphafold(uid)
```

### 3. 解析 pLDDT 评分

```python
from Bio import PDB
import numpy as np

def parse_plddt(pdb_file):
    """
    从 AlphaFold PDB 文件中提取 pLDDT 评分
    pLDDT 存储在 B-factor 列
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    plddts = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # 标准氨基酸
                    ca = residue['CA']
                    plddt = ca.get_bfactor()
                    plddts.append(plddt)
                    residues.append(f"{residue.get_resname()}{residue.id[1]}")
    
    return residues, plddts

# 分析可信度
residues, plddts = parse_plddt('AF-P12345-F1-model_v4.pdb')
avg_plddt = np.mean(plddts)

print(f"平均 pLDDT: {avg_plddt:.2f}")
print(f"高可信度残基 (>90): {sum(p > 90 for p in plddts)}")
print(f"低可信度残基 (<70): {sum(p < 70 for p in plddts)}")

# 找出低可信度区域
low_conf_regions = [
    (res, plddt) for res, plddt in zip(residues, plddts) if plddt < 70
]
if low_conf_regions:
    print("\n低可信度区域：")
    for res, plddt in low_conf_regions[:10]:
        print(f"  {res}: {plddt:.1f}")
```

### 4. 批量搜索 API

```python
import requests

def search_alphafold(query, limit=10):
    """
    搜索 AlphaFold 数据库
    
    Args:
        query: 搜索词（基因名、蛋白名等）
        limit: 返回结果数量
    """
    url = "https://www.alphafold.ebi.ac.uk/api/search"
    params = {
        "q": query,
        "size": limit
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        return results
    return None

# 搜索示例
results = search_alphafold("kinase human", limit=5)
if results:
    for hit in results['hits']['hits']:
        source = hit['_source']
        print(f"UniProt: {source['uniprotAccession']}")
        print(f"基因: {source.get('geneName', 'N/A')}")
        print(f"蛋白: {source.get('uniprotDescription', 'N/A')}")
        print(f"长度: {source.get('sequenceLength', 'N/A')} aa")
        print("---")
```

### 5. 本地运行 AlphaFold（需要 GPU）

```bash
# 使用 Docker 运行 AlphaFold
# 注意：需要下载 ~2.2 TB 的数据库

# 1. 拉取镜像
docker pull alphafold/alphafold:latest

# 2. 运行预测
docker run --gpus all \
  -v /path/to/databases:/data \
  -v /path/to/output:/output \
  -v /path/to/fasta:/fasta \
  alphafold/alphafold:latest \
  --fasta_paths=/fasta/query.fasta \
  --output_dir=/output \
  --data_dir=/data \
  --model_preset=monomer \
  --max_template_date=2024-01-01
```

## 注意事项与常见坑

### 1. pLDDT 不等于原子精度
- pLDDT >90 表示高可信度，但不等于 RMSD <1Å
- 柔性环区、无序区域通常 pLDDT 较低

### 2. 缺少配体和辅因子
- AlphaFold 只预测蛋白质主链和侧链
- 不包含配体、金属离子、辅因子
- 解决：需手动对接配体或使用 AlphaFold-Multimer

### 3. 单一构象
- AlphaFold 输出单一静态构象
- 实际蛋白可能有多种构象（开/闭状态）
- 解决：结合 MD 模拟探索构象空间

### 4. 多聚体预测限制
- 标准 AlphaFold DB 是单体预测
- 使用 AlphaFold-Multimer 预测复合物（需本地运行）

### 5. 跨膜蛋白预测
- 跨膜区域预测通常可信度较高
- 但膜内侧链方向可能不准确
- 解决：结合实验数据或使用专门的跨膜蛋白预测工具

### 6. PAE 图解读
- PAE（Predicted Aligned Error）反映残基对之间相对位置的不确定性
- 对角线外的大块高 PAE 区域表示结构域间相对位置不确定
- 解决：独立对待每个结构域

### 7. 版本更新
- AlphaFold 模型持续更新（v1 → v2 → v3 → v4）
- 旧版预测可能已被更高质量预测取代
- 注意文件名中的版本号（`_v4.pdb`）

### 8. 序列长度限制
- Web 接口通常限制 ~2700 残基
- 超长蛋白需分段预测或本地运行

## 官方链接与引用

### 官方资源
- **主站**: https://alphafold.ebi.ac.uk/
- **文档**: https://alphafold.ebi.ac.uk/faq
- **GitHub**: https://github.com/deepmind/alphafold
- **Colab**: https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb
- **论文**: https://www.nature.com/articles/s41586-021-03819-2

### 引用方式

**AlphaFold 2 算法**：
```
Jumper, J., Evans, R., Pritzel, A. et al. 
Highly accurate protein structure prediction with AlphaFold. 
Nature 596, 583–589 (2021).
doi: 10.1038/s41586-021-03819-2
```

**AlphaFold Database**：
```
Varadi, M., Anyango, S., Deshpande, M. et al. 
AlphaFold Protein Structure Database: massively expanding 
the structural coverage of protein-sequence space with 
high-accuracy models. Nucleic Acids Research (2022).
doi: 10.1093/nar/gkab1061
```

### 相关工具
- **ColabFold**: 简化版 AlphaFold（更快、更易用）
- **ESMFold**: Meta 的替代模型（更快但略低精度）
- **AlphaFold-Multimer**: 多链复合物预测
- **ChimeraX**: 可视化 AlphaFold 结构和 pLDDT


