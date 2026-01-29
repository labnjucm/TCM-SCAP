# ChEMBL - 化合物生物活性数据库

## 简介
**ChEMBL** 是由欧洲生物信息研究所（EMBL-EBI）维护的开放化合物生物活性数据库，包含超过220万个化合物和1900万个生物活性数据点。数据来源于科学文献、专利和高通量筛选项目。

**核心内容**：小分子化合物、生物大分子靶标、生物活性数据、ADMET性质、药物代谢数据。

## 典型用例
1. 查找特定靶标的已知抑制剂
2. 获取化合物的生物活性谱
3. 药物重定位研究
4. 构建QSAR模型
5. 虚拟筛选化合物库构建

## 输入/输出
**输入**: ChEMBL ID (CHEMBL25, CHEMBL1234)、靶标名称、化合物名称、SMILES、InChI
**输出**: JSON/XML/SDF格式，包含化合物结构、活性数据、靶标信息、文献引用

## 快速上手

### Web 界面
1. 访问 https://www.ebi.ac.uk/chembl/
2. 搜索化合物/靶标/文献
3. 浏览生物活性数据
4. 下载数据（多种格式）

### Python API 示例
```python
from chembl_webresource_client.new_client import new_client

# 搜索靶标
targets = new_client.target
egfr = targets.filter(pref_name__iexact='EGFR')
print(f"找到 {len(egfr)} 个 EGFR 靶标")

# 获取活性数据
activities = new_client.activity
egfr_activities = activities.filter(target_chembl_id='CHEMBL203')
print(f"EGFR 活性数据: {len(egfr_activities)} 条")

# 搜索化合物
molecules = new_client.molecule
aspirin = molecules.get('CHEMBL25')
print(f"化合物名称: {aspirin['pref_name']}")
print(f"SMILES: {aspirin['molecule_structures']['canonical_smiles']}")
```

### REST API 示例
```bash
# 获取化合物信息
curl https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json

# 搜索活性数据
curl "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL203&limit=10"

# 子结构搜索
curl -X POST https://www.ebi.ac.uk/chembl/api/data/substructure/c1ccccc1.json
```

## 注意事项与常见坑
1. **License**: ChEMBL 数据采用 CC BY-SA 3.0 许可，商业使用需注意
2. **活性单位**: IC50/EC50/Ki 单位可能不同（nM/μM/M），需标准化
3. **数据质量**: 文献数据可能有误差，注意检查
4. **API 限流**: 大量请求时添加延迟
5. **版本更新**: 定期更新（每年3-4次），ChEMBL ID 可能变化

## 官方链接与引用
- **主站**: https://www.ebi.ac.uk/chembl/
- **文档**: https://chembl.gitbook.io/chembl-interface-documentation/
- **API**: https://www.ebi.ac.uk/chembl/api/data/docs
- **下载**: https://ftp.ebi.ac.uk/pub/databases/chembl/

**引用**: 
```
Gaulton A, et al. (2017) The ChEMBL database in 2017. 
Nucleic Acids Res. 45(D1):D945-D954. doi: 10.1093/nar/gkw1074
```
