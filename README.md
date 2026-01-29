# TCM-SCAP — 计算化学/对接/MD/ADMET 统一平台（Scaffold）

一个集成多种计算化学工具的 Web 平台，基于 Next.js 14 构建，支持内嵌 Gradio 应用和外部 ADMET 网站。

## ✨ 特性

- 🎨 **现代化 UI**：基于 Next.js 14 + TailwindCSS + lucide-react
- 🔌 **灵活集成**：通过 Nginx 反向代理内嵌 Gradio 应用和外部网站
- 📦 **模块化配置**：所有菜单和工具配置集中在 `config/catalog.ts`
- 🔄 **子路径部署**：支持在同域子路径下运行多个 Gradio 应用
- 📱 **响应式设计**：适配桌面和移动端

## 🗂️ 项目结构

```
TCM_SCAP/
├── frontend/                    # Next.js 前端应用
│   ├── package.json
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── postcss.config.mjs
│   ├── app/
│   │   ├── globals.css         # 全局样式
│   │   ├── layout.tsx          # 根布局
│   │   ├── page.tsx            # 主页面
│   │   ├── components/         # React 组件
│   │   │   ├── Header.tsx      # 顶部导航
│   │   │   ├── Sidebar.tsx     # 左侧菜单
│   │   │   ├── MainPane.tsx    # 主内容区
│   │   │   ├── IframePane.tsx  # iframe 容器
│   │   │   └── Markdown.tsx    # Markdown 渲染
│   │   ├── lib/
│   │   │   └── types.ts        # TypeScript 类型定义
│   │   └── config/
│   │       └── catalog.ts      # 工具目录配置（核心）
│   └── public/
│       └── logo.svg            # 网站 Logo
├── reverse-proxy/              # Nginx 反向代理配置
│   ├── nginx.conf              # Nginx 配置文件
│   └── README.md               # 反代配置说明
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 前端部署

```bash
cd frontend

# 创建环境变量文件（可选）
cat > .env.local << 'EOF'
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)
NEXT_PUBLIC_DOCKING_PATH=/apps/docking/
NEXT_PUBLIC_MD_PATH=/apps/md/
NEXT_PUBLIC_ORCA_PATH=/apps/orca/
NEXT_PUBLIC_SWISSADME_PATH=/embed/swissadme/
NEXT_PUBLIC_PREADMET_PATH=/embed/preadmet/
EOF

# 安装依赖
npm install

# 构建项目
npm run build

# 启动生产服务器（端口 5173）
npm run start
```

或者使用开发模式：

```bash
npm run dev
```

或者使用集成的脚本：

```bash
./yunxing.sh
```

### 2. 配置 Gradio 应用

**重要**：你的三个 Gradio 应用必须设置 `root_path` 参数才能在子路径下正常工作。同时需要配置好在instruments文件夹下的关键工具。

#### 示例：分子对接应用（端口 7861）

```python
import gradio as gr

def docking_fn(protein_file, ligand_file):
    # 你的对接逻辑
    return "对接完成"

demo = gr.Interface(
    fn=docking_fn,
    inputs=["file", "file"],
    outputs="text",
    title="分子对接工具"
)

# 关键：设置 root_path 与 nginx.conf 保持一致
demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    root_path="/apps/docking"
)
```

#### 示例：分子动力学应用（端口 7862）

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7862,
    root_path="/apps/md"
)
```

#### 示例：ORCA 应用（端口 7863）

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7863,
    root_path="/apps/orca"
)
```

### 3. 配置 Nginx 反向代理

```bash
# 复制配置文件
sudo cp reverse-proxy/nginx.conf /etc/nginx/nginx.conf

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

详细说明请参考 [reverse-proxy/README.md](reverse-proxy/README.md)

### 4. 访问平台

浏览器访问 `http://<你的服务器IP>/`

## 📋 功能模块

### 获取数据
- RCSB PDB - 蛋白质结构数据库
- AlphaFold DB - AI 预测的蛋白结构
- ChEMBL - 化合物生物活性数据库
- ZINC - 虚拟筛选化合物库
- PubChem - NCBI 公共化学数据库

### 分子对接
- AutoDock Vina - 经典对接工具
- Smina - Vina 改进版
- GNINA - 深度学习对接
- **你的 Gradio 对接界面**（内嵌）

### 分子动力学模拟
- GROMACS - 高性能 MD 软件
- OpenMM - GPU 加速 MD 库
- NAMD - 并行 MD 引擎
- **你的 Gradio MD 界面**（内嵌）

### ADMET 分析
- **SwissADME**（内嵌）- 药物性质预测
- **PreADMET**（内嵌）- ADMET 性质预测

### 计算化学分析
- ORCA - 量子化学程序
- Gaussian - 经典量化软件
- Q-Chem - 现代量化程序
- **你的 Gradio ORCA 界面**（内嵌）

## 🔧 自定义与扩展

### 添加新工具

编辑 `frontend/app/config/catalog.ts`：

```typescript
{
  title: "新分类",
  items: [
    {
      key: "new-tool",
      title: "新工具名称",
      intro: md(`工具简介支持 Markdown 格式`),
      link: "https://example.com/",       // 外部链接（可选）
      iframeSrc: "/apps/newtool/"        // 内嵌路径（可选）
    }
  ]
}
```

然后在 `nginx.conf` 中添加对应的反向代理配置。

### 修改样式

编辑 `frontend/app/globals.css` 和 `frontend/tailwind.config.ts`。

主题色在 `tailwind.config.ts` 中定义：

```typescript
colors: {
  brand: { DEFAULT: "#0ea5e9" }  // 修改为你喜欢的颜色
}
```

### 更改端口

1. 修改 `frontend/package.json` 中的启动端口
2. 修改 `nginx.conf` 中对应的 `proxy_pass` 配置
3. 重启服务

## 📝 环境变量

在 `frontend/.env.local` 中配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEXT_PUBLIC_APP_TITLE` | 网站标题 | ChemHub |
| `NEXT_PUBLIC_FOOTER_NOTE` | 页脚文本 | © 2025 ChemHub (internal) |
| `NEXT_PUBLIC_DOCKING_PATH` | 对接应用路径 | /apps/docking/ |
| `NEXT_PUBLIC_MD_PATH` | MD 应用路径 | /apps/md/ |
| `NEXT_PUBLIC_ORCA_PATH` | ORCA 应用路径 | /apps/orca/ |
| `NEXT_PUBLIC_SWISSADME_PATH` | SwissADME 路径 | /embed/swissadme/ |
| `NEXT_PUBLIC_PREADMET_PATH` | PreADMET 路径 | /embed/preadmet/ |

## 🐛 故障排查

### Gradio 界面无法加载

**问题**：点击"打开我的 Gradio"后页面空白或显示错误

**解决方案**：
1. 确认 Gradio 应用已启动并监听正确端口
2. 检查 Gradio 的 `root_path` 参数是否与 nginx.conf 一致
3. 查看浏览器控制台是否有跨域或 404 错误
4. 确认 Nginx 已正确配置并重启

### 样式丢失或资源 404

**问题**：Gradio 界面加载但样式错误

**原因**：静态资源路径错误，通常是 `root_path` 配置问题

**解决方案**：
```python
# 确保 root_path 不要遗漏尾部斜杠的一致性
demo.launch(root_path="/apps/docking")  # nginx 中是 location /apps/docking/
```

### ADMET 网站无法内嵌

**问题**：SwissADME 或 PreADMET 拒绝在 iframe 中显示

**原因**：某些网站使用 JavaScript 强制防止内嵌

**解决方案**：
1. 检查 Nginx 是否正确配置了 `proxy_hide_header`
2. 如果仍然失败，在页面中添加"在新窗口打开"按钮
3. 仅将反代用于内部研究/教学，遵守网站服务条款

### 端口冲突

**问题**：`Address already in use`

**解决方案**：
```bash
# 查找占用端口的进程
sudo lsof -i :5173  # 或其他端口

# 终止进程或更换端口
```

## ⚠️ 免责声明

- 本项目仅用于**内部研究与教学目的**
- 内嵌第三方网站（SwissADME、PreADMET）时，务必遵守其服务条款
- 不建议在生产环境中使用反向代理绕过外部网站的嵌入限制
- 请尊重原网站的访问政策和使用条款

## 📦 技术栈

- **前端框架**：Next.js 14 (App Router)
- **开发语言**：TypeScript
- **样式方案**：TailwindCSS
- **图标库**：lucide-react
- **Markdown 渲染**：react-markdown
- **反向代理**：Nginx
- **应用框架**：Gradio (Python)

## 📄 许可

本项目为内部工具脚手架，使用时请遵守相关开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**构建时间**：2025-11-04  
**版本**：v0.1.0  
**维护者**：njucm503(zhangshiyu654@gmail.com)

