#!/bin/bash

# 创建剩余的 MD 文件（精简但符合要求）
cd /home/zyb/project/pingtai_test/frontend/app/content-md

# zinc.md
cat > zinc.md << 'EOF'
# ZINC - 虚拟筛选化合物库

## 简介
**ZINC** 是用于虚拟筛选的大型购买化合物数据库，由UCSF开发维护。包含数十亿个可购买的"类药"化合物，广泛用于基于结构和基于配体的药物设计。

**特点**：
- 可购买化合物（可直接订购）
- 按药物相似性分类
- 支持子结构/相似性搜索
- 提供多种格式下载
- 定期更新供应商信息

## 典型用例
1. 构建虚拟筛选化合物库
2. 寻找先导化合物
3. 相似性搜索已知活性化合物
4. 获取对接所需的3D构象

## 输入/输出
**输入**：SMILES、分子量范围、logP范围、供应商筛选  
**输出**：SMILES、SDF、MOL2格式，包含3D坐标

## 快速上手
1. 访问 https://zinc.docking.org/
2. 选择化合物子集（如"In Stock"、"Lead-like"）
3. 使用高级搜索或子结构搜索
4. 下载结果（批量或单个）

### 示例：下载特定化合物
```bash
# 通过ZINC ID下载
curl -o ZINC000001234567.sdf \
  "https://zinc.docking.org/substances/ZINC000001234567.sdf"

# 批量下载（需登录）
wget -r -np -nH --cut-dirs=2 \
  "https://zinc.docking.org/catalogs/in-stock/subsets/"
```

### Python API 示例
```python
import requests

def download_zinc_mol(zinc_id, format='sdf'):
    """下载ZINC化合物"""
    url = f"https://zinc.docking.org/substances/{zinc_id}.{format}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{zinc_id}.{format}", 'wb') as f:
            f.write(response.content)
        print(f"已下载: {zinc_id}.{format}")
    else:
        print(f"下载失败: {zinc_id}")

# 示例
download_zinc_mol("ZINC000001234567", "sdf")
```

## 注意事项与常见坑
1. **数据量大**：完整库有数十亿化合物，需选择合适子集
2. **3D构象**：部分格式包含预生成的3D构象，部分需自己生成
3. **供应商更新**：可购买性会变化，使用前确认
4. **License**：免费用于学术研究，商业使用需确认
5. **子集选择**：
   - `in-stock`: 立即可购买
   - `lead-like`: 符合类先导化合物规则
   - `fragment`: 片段库（MW<300）

## 官方链接与引用
- **主站**: https://zinc.docking.org/
- **文档**: https://zinc.docking.org/help/
- **FTP**: https://files.docking.org/

**引用**:
```
Irwin JJ, Shoichet BK. ZINC--a free database of commercially available 
compounds for virtual screening. J Chem Inf Model. 2005;45(1):177-182. 
doi: 10.1021/ci049714+
```
EOF

# pubchem.md
cat > pubchem.md << 'EOF'
# PubChem - NCBI公共化学数据库

## 简介
**PubChem** 是NCBI维护的开放化学数据库，包含1.1亿+化合物、274百万+生物活性数据。提供化合物结构、性质、生物活性、专利、文献等信息。

**核心组成**：
- **Compound**: 唯一化学结构（去同分异构体）
- **Substance**: 供应商提交的化合物记录  
- **BioAssay**: 生物活性测试数据

## 典型用例
1. 查询化合物性质（logP、TPSA等）
2. 下载化合物结构（SDF/MOL）
3. 子结构/相似性搜索
4. 获取生物活性数据
5. SMILES/InChI格式转换

## 输入/输出
**输入**：CID（Compound ID）、SMILES、InChI、化合物名称  
**输出**：JSON/XML/SDF，包含结构、性质、生物活性数据

## 快速上手

### PUG-REST API 示例

```bash
# SMILES → CID
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/CC(=O)OC1=CC=CC=C1C(=O)O/cids/JSON"

# CID → SDF下载
curl -o aspirin.sdf \
  "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/SDF"

# 获取分子量
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularWeight/JSON"

# 批量查询性质
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244,2245,2246/property/MolecularWeight,XLogP,TPSA/JSON"
```

### Python示例

```python
import requests

def smiles_to_cid(smiles):
    """SMILES转CID"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['IdentifierList']['CID'][0]
    return None

def get_compound_properties(cid):
    """获取化合物性质"""
    props = "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props}/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['PropertyTable']['Properties'][0]
    return None

# 示例
aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
cid = smiles_to_cid(aspirin_smiles)
print(f"阿司匹林 CID: {cid}")

props = get_compound_properties(cid)
print(f"分子量: {props['MolecularWeight']}")
print(f"logP: {props['XLogP']}")
print(f"TPSA: {props['TPSA']}")
```

## 注意事项与常见坑
1. **CID vs SID**: CID是唯一结构，SID是供应商记录（一个CID对应多个SID）
2. **API限流**: 每秒最多5个请求，超过会被限制
3. **大批量查询**: 使用POST而非GET，避免URL过长
4. **SMILES编码**: URL中的特殊字符需编码（如#→%23）
5. **异步查询**: 大批量查询使用ListKey机制

## 官方链接与引用
- **主站**: https://pubchem.ncbi.nlm.nih.gov/
- **PUG-REST文档**: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
- **Python SDK**: https://github.com/mcs07/PubChemPy

**引用**:
```
Kim S, et al. PubChem 2023 update. Nucleic Acids Res. 2023;51(D1):D1373-D1380.
doi: 10.1093/nar/gkac956
```
EOF

echo "剩余MD文件已创建"
