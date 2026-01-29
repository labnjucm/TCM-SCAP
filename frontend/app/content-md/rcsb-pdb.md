# RCSB PDB - 蛋白质数据银行

## 简介

**RCSB PDB（Research Collaboratory for Structural Bioinformatics Protein Data Bank）**是世界上最权威的生物大分子三维结构数据库，由美国 RCSB 联盟维护。数据库包含超过 20 万个经 X-射线晶体学、核磁共振（NMR）、冷冻电镜（Cryo-EM）等实验方法解析的蛋白质、核酸及复合物结构。

RCSB PDB 是结构生物学、药物设计、分子对接、分子动力学模拟等领域的基础数据源。每个结构都有唯一的 **PDB ID**（4 位字符，如 1ABC），并包含原子坐标、实验方法、分辨率、生物学组装等元数据。

**核心价值**：
- 提供高质量、经同行评审的实验结构数据
- 支持多种格式下载（PDB、mmCIF、PDBML）
- 丰富的 API 接口（REST、GraphQL）
- 强大的搜索与分析工具

## 典型用例

### 1. 分子对接研究
下载靶标蛋白结构（如 SARS-CoV-2 主蛋白酶 7BQY），用于虚拟筛选和对接研究。

### 2. 同源建模
搜索与目标蛋白序列相似的已解析结构，作为同源建模的模板。

### 3. 结构比对与分析
下载多个同一蛋白家族的结构，进行结构叠合、保守性分析。

### 4. 分子动力学模拟
获取蛋白-配体复合物结构，作为 MD 模拟的初始构象。

### 5. 教学与可视化
下载经典结构（如血红蛋白 1HHO）用于教学演示。

## 输入/输出

### 输入
- **PDB ID**：4 位字符（如 `1ABC`、`7BQY`）
- **UniProt ID**：蛋白质序列数据库 ID（可关联到 PDB）
- **搜索关键词**：蛋白名称、基因名、配体名、作者等
- **序列**：FASTA 格式序列（BLAST 搜索）
- **结构**：上传结构文件进行相似性搜索

### 输出
- **PDB 格式**（传统）：原子坐标、HEADER、ATOM、HETATM 记录
- **mmCIF 格式**（推荐）：更完整的元数据，支持大分子
- **PDBML 格式**：XML 格式
- **元数据**：JSON/XML，包含实验方法、分辨率、作者、引用等
- **序列**：FASTA 格式

## 快速上手

### Web 界面检索

1. **访问**: https://www.rcsb.org/
2. **搜索栏输入**：PDB ID、蛋白名称、UniProt ID
3. **高级搜索**：
   - 按分辨率筛选（如 ≤2.0 Å）
   - 按实验方法筛选（X-ray、NMR、EM）
   - 按配体筛选（如含 ATP 的结构）
   - 按序列相似性搜索（BLAST）

4. **下载结构**：
   - 点击 **Download Files** → 选择格式（PDB/mmCIF）
   - 生物学组装（Biological Assembly）vs 不对称单元（Asymmetric Unit）

### 检索技巧

**按分辨率搜索高质量结构**：
```
Resolution: [0 TO 2.0] AND Method: X-RAY DIFFRACTION
```

**搜索特定配体的复合物**：
```
Ligand Name: ATP
```

**搜索特定物种的结构**：
```
Organism: Homo sapiens
```

**按发布日期筛选**：
```
Release Date: [2023-01-01 TO 2024-12-31]
```

## 命令/API 示例

### 1. REST API - 下载 PDB 文件

```bash
# 下载 PDB 格式（传统）
curl -o 1abc.pdb https://files.rcsb.org/download/1ABC.pdb

# 下载 mmCIF 格式（推荐）
curl -o 1abc.cif https://files.rcsb.org/download/1ABC.cif

# 下载生物学组装
curl -o 1abc_assembly1.cif https://files.rcsb.org/download/1ABC-assembly1.cif
```

### 2. REST API - 获取元数据

```bash
# 获取完整元数据（JSON）
curl https://data.rcsb.org/rest/v1/core/entry/1ABC

# 获取实验方法和分辨率
curl https://data.rcsb.org/rest/v1/core/entry/1ABC | jq '.exptl, .refine'

# 获取引用信息
curl https://data.rcsb.org/rest/v1/core/entry/1ABC | jq '.citation'
```

### 3. Python - 批量下载

```python
import requests
from pathlib import Path

def download_pdb(pdb_id, format='cif', output_dir='./structures'):
    """
    下载 PDB 结构文件
    
    Args:
        pdb_id: PDB ID（如 '1ABC'）
        format: 'pdb' 或 'cif'
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.{format}"
    
    response = requests.get(url)
    if response.status_code == 200:
        output_path = Path(output_dir) / f"{pdb_id}.{format}"
        output_path.write_text(response.text)
        print(f"✓ 已下载: {pdb_id}.{format}")
        return str(output_path)
    else:
        print(f"✗ 下载失败: {pdb_id} (HTTP {response.status_code})")
        return None

# 批量下载
pdb_ids = ['1ABC', '7BQY', '6LU7', '1HHO']
for pdb_id in pdb_ids:
    download_pdb(pdb_id, format='cif')
```

### 4. Python - 搜索 API

```python
import requests
import json

def search_pdb(query, rows=10):
    """
    使用 RCSB Search API 搜索结构
    
    Args:
        query: 搜索关键词
        rows: 返回结果数量
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    search_query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "value": query
            }
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "pager": {"start": 0, "rows": rows}
        }
    }
    
    response = requests.post(url, json=search_query)
    if response.status_code == 200:
        results = response.json()
        pdb_ids = [item['identifier'] for item in results.get('result_set', [])]
        return pdb_ids
    else:
        print(f"搜索失败: HTTP {response.status_code}")
        return []

# 示例：搜索 SARS-CoV-2 主蛋白酶结构
results = search_pdb("SARS-CoV-2 main protease")
print(f"找到 {len(results)} 个结构:")
for pdb_id in results[:5]:
    print(f"  - {pdb_id}")
```

### 5. GraphQL API 示例

```python
import requests

def graphql_query(pdb_id):
    """使用 GraphQL 获取详细信息"""
    url = "https://data.rcsb.org/graphql"
    
    query = """
    query ($id: String!) {
      entry(entry_id: $id) {
        struct {
          title
        }
        exptl {
          method
        }
        rcsb_entry_info {
          resolution_combined
          deposited_atom_count
        }
        polymer_entities {
          rcsb_polymer_entity {
            pdbx_description
          }
          entity_poly {
            pdbx_seq_one_letter_code_can
          }
        }
      }
    }
    """
    
    response = requests.post(
        url,
        json={"query": query, "variables": {"id": pdb_id}}
    )
    
    if response.status_code == 200:
        data = response.json()
        return data['data']['entry']
    return None

# 示例
info = graphql_query("7BQY")
if info:
    print(f"标题: {info['struct']['title']}")
    print(f"方法: {info['exptl'][0]['method']}")
    print(f"分辨率: {info['rcsb_entry_info']['resolution_combined']} Å")
```

## 注意事项与常见坑

### 1. PDB vs mmCIF 格式
- **PDB 格式**：传统格式，列宽固定，**不支持超过 99,999 原子或 9,999 残基**
- **mmCIF 格式**：现代格式，无长度限制，**推荐用于大分子（如核糖体、病毒衣壳）**
- 解决：优先使用 mmCIF 格式

### 2. 生物学组装 vs 不对称单元
- **不对称单元（Asymmetric Unit）**：晶体学中的最小重复单元，可能不是生物活性形式
- **生物学组装（Biological Assembly）**：实际的生物活性状态（如二聚体、四聚体）
- 解决：分子对接和 MD 模拟通常使用生物学组装

### 3. 缺失残基/原子
- 柔性环区、N/C 末端可能因电子密度不清晰而缺失
- 氢原子通常不包含（X-ray 无法观察到氢）
- 解决：使用 MODELLER 等工具补全缺失残基；使用 reduce/pdb2pqr 添加氢原子

### 4. 配体与非标准残基
- **HETATM 记录**：配体、辅因子、溶剂（水分子）
- **修饰氨基酸**（如磷酸化 Ser）标记为 HETATM
- 解决：根据需求选择性保留/删除水分子和配体

### 5. 多个 Model（NMR 结构）
- NMR 结构包含多个构象（MODEL 1, MODEL 2, ...）
- 解决：通常选择 MODEL 1 或对所有构象做系综分析

### 6. API 限流
- RCSB 对 API 有速率限制（通常 ~20 请求/秒）
- 批量下载时添加延迟（`time.sleep(0.1)`）
- 解决：使用 rsync 镜像或 FTP 批量下载大量数据

### 7. 坐标系与链标识
- **链 ID（Chain ID）**：可能是单字母（A/B）或多字母（AA/AB）
- 复合物中不同分子有不同链
- 解决：明确指定需要的链（如只保留蛋白链A）

### 8. 旧版 PDB 格式问题
- 90 年代早期的结构可能格式不规范
- 解决：使用 PDBFixer 或 Biopython 修复

## 官方链接与引用

### 官方资源
- **主站**: https://www.rcsb.org/
- **文档**: https://www.rcsb.org/docs/
- **API 文档**: https://data.rcsb.org/
- **GraphQL 文档**: https://data.rcsb.org/graphql/index.html
- **FTP 镜像**: ftp://ftp.rcsb.org/pub/pdb/
- **帮助中心**: https://www.rcsb.org/pages/help

### 引用方式
主要引用：
```
H.M. Berman, J. Westbrook, Z. Feng, G. Gilliland, T.N. Bhat, H. Weissig, 
I.N. Shindyalov, P.E. Bourne. (2000) The Protein Data Bank. 
Nucleic Acids Research, 28: 235-242.
doi: 10.1093/nar/28.1.235
```

最新引用（2019）：
```
Burley SK, Bhikadiya C, Bi C, et al. (2019) 
RCSB Protein Data Bank: biological macromolecular structures 
enabling research and education in fundamental biology, biomedicine, 
biotechnology and energy. Nucleic Acids Research, 47:D464-D474.
doi: 10.1093/nar/gky1004
```

### 相关工具
- **PyMOL**: 结构可视化
- **Chimera/ChimeraX**: UCSF 开发的可视化工具
- **Biopython**: Python 解析 PDB 文件
- **MDAnalysis**: 轨迹分析
- **ProDy**: 蛋白动力学分析

### 数据统计（2024）
- 总结构数：>210,000
- 每周新增：~200 个
- X-ray：~85%
- EM：~10%
- NMR：~5%


