# 🔧 如何修改 IP 地址配置

本文档说明如何将 IP 地址从 `127.0.0.1`（本地）修改为其他地址，以便从局域网或外网访问。

## 📋 需要修改的位置

### 1️⃣ Gradio 应用（3 个文件）

修改所有 Gradio 应用的 `server_name` 参数。

#### 修改 `examples/docking_app.py`

```python
demo.launch(
    server_name="0.0.0.0",  # 修改这里：0.0.0.0 允许所有 IP 访问
    server_port=7861,
    show_error=True,
    share=False
)
```

#### 修改 `examples/md_app.py`

```python
demo.launch(
    server_name="0.0.0.0",  # 修改这里
    server_port=7862,
    show_error=True,
    share=False
)
```

#### 修改 `examples/orca_app.py`

```python
demo.launch(
    server_name="0.0.0.0",  # 修改这里
    server_port=7863,
    show_error=True,
    share=False
)
```

### 2️⃣ 前端配置文件

修改 `frontend/.env.local`（如果文件不存在，创建它）：

```bash
# 可自定义页面标题与页脚
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)

# Gradio 应用访问地址
# 方式 1：使用服务器 IP 地址
NEXT_PUBLIC_DOCKING_PATH=http://192.168.1.100:7861
NEXT_PUBLIC_MD_PATH=http://192.168.1.100:7862
NEXT_PUBLIC_ORCA_PATH=http://192.168.1.100:7863

# 方式 2：使用域名（如果有）
# NEXT_PUBLIC_DOCKING_PATH=http://chemhub.example.com:7861
# NEXT_PUBLIC_MD_PATH=http://chemhub.example.com:7862
# NEXT_PUBLIC_ORCA_PATH=http://chemhub.example.com:7863

# 方式 3：保持本地访问（默认）
# NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:7861
# NEXT_PUBLIC_MD_PATH=http://127.0.0.1:7862
# NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:7863
```

> **注意**：将 `192.168.1.100` 替换为你的实际服务器 IP 地址。

### 3️⃣ 前端启动配置（可选）

如果需要前端也从外部访问，修改启动命令：

#### 开发模式

```bash
cd frontend
npm run dev -- -H 0.0.0.0
```

#### 生产模式

修改 `frontend/package.json`：

```json
{
  "scripts": {
    "dev": "next dev -p 5173 -H 0.0.0.0",
    "start": "next start -p 5173 -H 0.0.0.0"
  }
}
```

## 🔍 查找服务器 IP 地址

### Linux/Mac

```bash
# 方法 1：使用 ifconfig
ifconfig | grep "inet "

# 方法 2：使用 ip
ip addr show

# 方法 3：查看特定网卡
ip addr show eth0  # 或 wlan0
```

### Windows

```cmd
ipconfig
```

常见结果：
- **本地回环**：`127.0.0.1`（只能本机访问）
- **局域网**：`192.168.x.x` 或 `10.x.x.x`（局域网内访问）
- **公网 IP**：联系网络管理员获取

## 📝 完整修改步骤示例

假设你的服务器 IP 是 `192.168.1.100`：

### 步骤 1：修改 Gradio 应用

```bash
# 编辑 3 个文件
nano examples/docking_app.py
nano examples/md_app.py
nano examples/orca_app.py

# 将 server_name 改为 "0.0.0.0"
```

### 步骤 2：修改前端配置

```bash
# 创建或编辑配置文件
nano frontend/.env.local

# 写入以下内容（替换 IP 地址）：
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)
NEXT_PUBLIC_DOCKING_PATH=http://192.168.1.100:7861
NEXT_PUBLIC_MD_PATH=http://192.168.1.100:7862
NEXT_PUBLIC_ORCA_PATH=http://192.168.1.100:7863
```

### 步骤 3：修改前端启动配置（可选）

```bash
nano frontend/package.json

# 修改 scripts 部分：
"dev": "next dev -p 5173 -H 0.0.0.0",
"start": "next start -p 5173 -H 0.0.0.0"
```

### 步骤 4：重启服务

```bash
# 停止现有服务
./stop-all.sh

# 重新启动
./start-all.sh
```

### 步骤 5：访问测试

从其他设备访问：

```
http://192.168.1.100:5173
```

## 🔐 安全注意事项

### 1. 防火墙配置

确保防火墙允许相应端口：

```bash
# Ubuntu/Debian
sudo ufw allow 5173/tcp
sudo ufw allow 7861/tcp
sudo ufw allow 7862/tcp
sudo ufw allow 7863/tcp
sudo ufw reload

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=5173/tcp
sudo firewall-cmd --permanent --add-port=7861/tcp
sudo firewall-cmd --permanent --add-port=7862/tcp
sudo firewall-cmd --permanent --add-port=7863/tcp
sudo firewall-cmd --reload
```

### 2. 访问控制

如果需要限制访问，考虑：

- 使用 VPN 连接
- 配置 IP 白名单
- 添加用户认证（需自行实现）
- 使用 HTTPS（生产环境推荐）

### 3. 公网暴露风险

⚠️ **不建议直接将服务暴露到公网**，因为：
- 示例应用没有认证机制
- 可能被恶意访问
- 建议通过 VPN 或反向代理（如 Nginx + HTTPS）

## 🧪 测试连接

### 从本机测试

```bash
# 测试前端
curl http://localhost:5173

# 测试 Gradio 应用
curl http://127.0.0.1:7861
curl http://127.0.0.1:7862
curl http://127.0.0.1:7863
```

### 从其他设备测试

```bash
# 替换 IP 地址
curl http://192.168.1.100:5173
curl http://192.168.1.100:7861
```

或在浏览器中直接访问。

## 🐛 故障排查

### 无法从外部访问

**检查清单**：
1. ✅ Gradio `server_name` 改为 `0.0.0.0`？
2. ✅ 前端 `.env.local` 配置正确的 IP？
3. ✅ 防火墙端口已开放？
4. ✅ 服务已重启？
5. ✅ IP 地址正确？

**测试命令**：

```bash
# 在服务器上测试
netstat -tuln | grep -E '5173|7861|7862|7863'

# 应该显示 0.0.0.0:端口 或 :::端口
```

### iframe 加载失败

如果 iframe 无法加载 Gradio：

1. 检查浏览器控制台错误（F12）
2. 确认 Gradio 地址可以直接访问
3. 检查是否有混合内容警告（HTTP vs HTTPS）

### 混合内容问题

如果前端使用 HTTPS，Gradio 必须也使用 HTTPS：

```bash
# 方式 1：前端也用 HTTP
# 方式 2：为 Gradio 配置 HTTPS（需要证书）
# 方式 3：使用 Nginx 统一处理 HTTPS
```

## 📚 相关文档

- 基础启动：`QUICKSTART-SIMPLE.md`
- 完整文档：`README.md`
- Nginx 配置：`reverse-proxy/README.md`（高级用法）
- **MySQL 连接问题**：`MYSQL_CONNECTION_FIX.md`（用户登录/注册错误修复）

## 💡 常见场景

### 场景 1：局域网内其他电脑访问

```bash
# Gradio: server_name="0.0.0.0"
# 前端 .env.local: http://192.168.1.100:786x
# 访问: http://192.168.1.100:5173
```

### 场景 2：只在本机使用

```bash
# Gradio: server_name="127.0.0.1"
# 前端 .env.local: http://127.0.0.1:786x
# 访问: http://localhost:5173
```

### 场景 3：使用域名

```bash
# 前提：域名已解析到服务器 IP
# Gradio: server_name="0.0.0.0"
# 前端 .env.local: http://example.com:786x
# 访问: http://example.com:5173
```

---

**修改完成后，记得重启所有服务！** 🚀


