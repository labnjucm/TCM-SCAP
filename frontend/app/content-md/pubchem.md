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
