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
