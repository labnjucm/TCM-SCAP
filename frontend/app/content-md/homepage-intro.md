# 🧪 欢迎使用 ChemHub 计算化学资源平台

ChemHub 是一个集成化的计算化学工具和资源导航平台，旨在为化学、生物、药物研发领域的研究人员提供便捷的工具访问和资源整合服务。

## 🎯 平台特色

### 📚 资源导航
- **数据库资源**：RCSB PDB、AlphaFold DB、ChEMBL、ZINC、PubChem 等权威数据源
- **工具链接**：快速访问常用计算化学工具的官方网站
- **详细说明**：每个工具都配有中文使用指南和注意事项

### 🚀 本地工具集成
- **分子对接**：内嵌 Gradio 界面，支持 AutoDock Vina 等对接工具
- **分子动力学**：GROMACS、OpenMM、NAMD 等模拟工具的快速访问
- **量子化学**：ORCA、Gaussian、Q-Chem 计算工具整合
- **ADMET 预测**：SwissADME、PreADMET 等药物性质预测工具

### 👤 用户系统
- **账号管理**：安全的用户注册和登录功能
- **数据持久化**：基于 MySQL 的用户数据存储
- **JWT 认证**：现代化的身份验证机制

## 📖 使用说明

### 快速开始

1. **浏览资源**：点击左侧侧边栏的分类，查看各类工具和数据库
2. **查看详情**：选择任一工具，主页面会显示简介和详细说明
3. **访问工具**：
   - 点击「打开官方链接」访问官方网站
   - 选择带有「Gradio」标识的工具可在页面内直接使用
4. **用户登录**：点击右下角 **⚙️ 齿轮图标** 注册或登录账号

### 功能板块

#### 🗂️ 获取数据
提供主流生物分子和化合物数据库的访问入口：
- **RCSB PDB**：蛋白质三维结构数据库
- **AlphaFold DB**：AI 预测的蛋白质结构
- **ChEMBL**：化合物生物活性数据
- **ZINC**：虚拟筛选化合物库
- **PubChem**：公共化学数据库

#### ⚗️ 分子对接
集成分子对接工具，用于预测配体-受体结合模式：
- **AutoDock Vina**：经典对接工具
- **Smina**：Vina 改进版
- **GNINA**：深度学习增强对接
- **本地 Gradio 界面**：可直接在浏览器中运行对接任务

#### 🔬 分子动力学
提供分子动力学模拟软件的快速访问：
- **GROMACS**：高性能 MD 软件
- **OpenMM**：GPU 加速 Python 库
- **NAMD**：并行 MD 引擎
- **本地 Gradio 界面**：简化的 MD 任务提交

#### 💊 ADMET 分析
药物性质预测和评估工具：
- **SwissADME**：ADME 性质预测
- **PreADMET**：全面的 ADMET 预测平台

#### 🧬 计算化学分析
量子化学计算工具集成：
- **ORCA**：现代量子化学程序
- **Gaussian**：经典量化软件
- **Q-Chem**：高效并行计算
- **本地 Gradio 界面**：ORCA 计算任务提交

## 💡 使用建议

### 工作流程示例

**药物虚拟筛选流程**：
1. 从 **RCSB PDB** 获取靶点蛋白结构
2. 从 **ZINC** 或 **PubChem** 获取化合物库
3. 使用 **AutoDock Vina**（本地 Gradio）进行分子对接
4. 用 **SwissADME** 评估候选化合物的药物性质
5. 对优选化合物进行 **GROMACS** 分子动力学验证

**蛋白质研究流程**：
1. 从 **AlphaFold DB** 获取预测结构
2. 使用 **GROMACS** 进行结构优化和 MD 模拟
3. 用 **ORCA** 计算关键残基的电子性质

## 🛠️ 技术栈

- **前端框架**：Next.js 14 + React + TypeScript
- **样式方案**：Tailwind CSS
- **数据库**：MySQL + Prisma ORM
- **认证方式**：JWT
- **工具集成**：Gradio（Python）
- **Markdown 渲染**：ReactMarkdown

## 📝 自定义配置

### 修改本介绍内容
编辑文件：`frontend/app/content-md/homepage-intro.md`

### 修改工具详细说明
编辑对应的 Markdown 文件：
- `frontend/app/content-md/autodock-vina.md`
- `frontend/app/content-md/gromacs.md`
- `frontend/app/content-md/orca.md`
- 等等...

### 配置 Gradio 地址
编辑文件：`frontend/.env.local`
```bash
NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:7861
NEXT_PUBLIC_MD_PATH=http://127.0.0.1:7862
NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:7863
```

## 🔐 安全提醒

- **密码安全**：请使用至少 8 位的强密码
- **生产环境**：修改 `.env.local` 中的 `JWT_SECRET` 为复杂随机字符串
- **数据库**：生产环境建议创建专用数据库用户，不使用 root

## 📚 更多文档

- **快速启动**：`QUICKSTART-SIMPLE.md`
- **运行命令**：`RUN_COMMANDS.md`
- **部署指南**：`DEPLOYMENT_WITH_AUTH.md`
- **IP 配置**：`HOW_TO_CHANGE_IP.md`
- **MySQL 问题**：`MYSQL_CONNECTION_FIX.md`

## 🤝 技术支持

如遇到问题：
1. 查看项目根目录下的文档文件
2. 检查控制台日志（按 F12）
3. 查看 `RUN_COMMANDS.md` 中的故障排查部分

---

**祝您使用愉快！开始探索计算化学的世界吧** 🚀

