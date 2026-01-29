# Nginx 反代说明

## 配置步骤

### 1. 启动前端

```bash
cd ../frontend
npm install
npm run build
npm run start  # 默认监听 5173 端口
```

### 2. 启动你的 3 个 Gradio 应用

**重要：必须设置 `root_path` 参数以确保子路径下正常工作**

#### Gradio Docking 应用（监听 7861）

```python
import gradio as gr

# 你的 docking 界面
demo = gr.Interface(...)

demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    root_path="/apps/docking"  # 关键：与 nginx.conf 保持一致
)
```

#### Gradio MD 应用（监听 7862）

```python
import gradio as gr

# 你的分子动力学界面
demo = gr.Interface(...)

demo.launch(
    server_name="0.0.0.0",
    server_port=7862,
    root_path="/apps/md"  # 关键：与 nginx.conf 保持一致
)
```

#### Gradio ORCA 应用（监听 7863）

```python
import gradio as gr

# 你的 ORCA 界面
demo = gr.Interface(...)

demo.launch(
    server_name="0.0.0.0",
    server_port=7863,
    root_path="/apps/orca"  # 关键：与 nginx.conf 保持一致
)
```

> **说明**：`root_path` 能确保 Gradio 的静态资源（JS/CSS）和 WebSocket 路径在子路径下仍可被正确加载。

### 3. 应用 Nginx 配置

#### 方法 1：替换默认配置（需 root 权限）

```bash
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo nginx -t  # 测试配置
sudo systemctl restart nginx
```

#### 方法 2：使用 sites-available（推荐）

```bash
sudo cp nginx.conf /etc/nginx/sites-available/chemhub
sudo ln -s /etc/nginx/sites-available/chemhub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 方法 3：Docker 运行 Nginx

```bash
docker run -d \
  --name chemhub-nginx \
  -p 80:80 \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  --network host \
  nginx:latest
```

### 4. 访问平台

浏览器访问 `http://<你的服务器IP>/`

- 左侧选择分类与工具
- 点击"打开我的 Gradio ..."即可在中间区内嵌打开
- ADMET 工具通过反代路径内嵌

## 注意事项

### 关于外部网站内嵌

- SwissADME 和 PreADMET 通过反向代理隐藏了 `X-Frame-Options` 和 `Content-Security-Policy` 响应头
- **仅用于内部研究/教学目的**，务必遵守其服务条款
- 若站点通过其他方式禁止嵌入（如 JavaScript 检测），请改用"在新窗口打开"

### 端口冲突

如果 5173 或 786x 端口被占用，请修改：
1. `frontend/.env.local` 中的路径配置
2. `nginx.conf` 中的 `proxy_pass` 端口
3. Gradio 应用的 `server_port` 参数

### 防火墙

确保服务器防火墙开放了 80 端口（或你配置的其他端口）：

```bash
sudo ufw allow 80/tcp
sudo ufw reload
```

## 故障排查

### Gradio 界面无法加载或样式错误

- 检查 Gradio 的 `root_path` 是否与 `nginx.conf` 的 `location` 一致
- 查看浏览器控制台是否有 404 错误（静态资源路径问题）
- 确认 Gradio 应用已启动且监听正确端口

### WebSocket 连接失败

- 确保 `nginx.conf` 中配置了 `Upgrade` 和 `Connection` 头
- 检查 Gradio 版本是否支持子路径部署

### ADMET 网站无法内嵌

- 某些网站的 CSP 策略可能通过 JavaScript 强制执行
- 如果内嵌失败，在页面中提供"在新窗口打开"链接作为备选方案

## 扩展功能

### 启用 HTTPS

1. 安装 Let's Encrypt 证书：

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

2. Certbot 会自动修改 `nginx.conf` 添加 SSL 配置

### 添加访问控制

在 `nginx.conf` 的 `server` 块中添加：

```nginx
auth_basic "Restricted Access";
auth_basic_user_file /etc/nginx/.htpasswd;
```

生成密码文件：

```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd username
```

