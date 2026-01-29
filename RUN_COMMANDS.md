# 🚀 ChemHub 完整运行命令指南

## ✅ 数据库已配置完成

数据库连接和权限已经全部配置好：

- ✅ MySQL root 用户权限已授予（所有访问方式）
- ✅ 数据库 `chemhub` 已创建
- ✅ 数据表 `users` 已创建
- ✅ Prisma Client 已生成
- ✅ 连接地址：`mysql://root:pass@127.0.0.1:3306/chemhub`

## 🎯 立即启动（快速版）

### 方式 1：使用启动脚本（推荐）

```bash
cd /home/zyb/project/pingtai_test
./start-with-mysql.sh
```

### 方式 2：手动启动前端

```bash
cd /home/zyb/project/pingtai_test/frontend
npm run dev
```

## 📖 详细启动步骤

### 1. 启动 MySQL（如果未运行）

```bash
# 检查 MySQL 状态
sudo systemctl status mysql

# 如果未运行，启动它
sudo systemctl start mysql

# 设置开机自启（可选）
sudo systemctl enable mysql
```

### 2. 验证数据库连接

```bash
# 测试 MySQL 连接
mysql -u root -ppass -e "USE chemhub; SHOW TABLES;"

# 应该显示：
# - _prisma_migrations
# - users
```

### 3. 启动前端服务

```bash
cd /home/zyb/project/pingtai_test/frontend

# 确保依赖已安装
npm install

# 启动开发服务器
npm run dev
```

### 4. 启动 Gradio 应用（可选）

如果需要使用分子对接、分子动力学等功能：

```bash
# 在新终端窗口 1
cd /home/zyb/project/pingtai_test/examples
python docking_app.py

# 在新终端窗口 2
python md_app.py

# 在新终端窗口 3
python orca_app.py
```

## 🌐 访问地址

启动后，在浏览器中访问：

- **主界面**: http://localhost:5173
- **分子对接**: http://localhost:7861（如果启动了）
- **分子动力学**: http://localhost:7862（如果启动了）
- **量子化学**: http://localhost:7863（如果启动了）

## 🧪 测试用户认证

### 测试注册

1. 打开 http://localhost:5173
2. 点击右下角的 **⚙️ 齿轮图标**
3. 切换到 **注册** 标签
4. 输入邮箱和密码（至少 8 位）
5. 点击 **注册** 按钮

### 测试登录

1. 使用注册的账号登录
2. 成功后会显示用户邮箱

### 验证数据库

```bash
# 查看注册的用户
mysql -u root -ppass -e "USE chemhub; SELECT id, email, createdAt FROM users;"
```

## 🛑 停止服务

```bash
# 方式 1：在运行的终端按 Ctrl+C

# 方式 2：使用停止脚本
cd /home/zyb/project/pingtai_test
./stop-all.sh

# 方式 3：手动查找并关闭进程
ps aux | grep "npm run dev"
kill <进程ID>
```

## 🔧 故障排查

### 问题 1：端口被占用

```bash
# 查看端口占用
sudo lsof -i :5173
sudo lsof -i :7861
sudo lsof -i :7862
sudo lsof -i :7863

# 杀死占用端口的进程
sudo kill -9 <进程ID>
```

### 问题 2：MySQL 连接失败

```bash
# 1. 检查 MySQL 是否运行
sudo systemctl status mysql

# 2. 测试连接
mysql -u root -ppass -e "SELECT 1;"

# 3. 检查权限
mysql -u root -ppass -e "SELECT host, user FROM mysql.user WHERE user='root';"

# 4. 如果密码不对，重置密码
sudo mysql
ALTER USER 'root'@'localhost' IDENTIFIED BY 'pass';
FLUSH PRIVILEGES;
EXIT;
```

### 问题 3：数据库表不存在

```bash
cd /home/zyb/project/pingtai_test/frontend
export DATABASE_URL="mysql://root:pass@127.0.0.1:3306/chemhub"
npx prisma db push
```

### 问题 4：Prisma Client 报错

```bash
cd /home/zyb/project/pingtai_test/frontend
npx prisma generate
```

### 问题 5：npm 依赖问题

```bash
cd /home/zyb/project/pingtai_test/frontend
rm -rf node_modules package-lock.json
npm install
```

## 📝 环境变量配置

当前配置文件 `frontend/.env.local`：

```bash
# 页面配置
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)

# Gradio 应用地址
NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:7861
NEXT_PUBLIC_MD_PATH=http://127.0.0.1:7862
NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:7863

# 数据库配置
DATABASE_URL="mysql://root:pass@127.0.0.1:3306/chemhub"

# JWT 密钥
JWT_SECRET="please_change_me_to_a_strong_random_secret"
```

## 🔍 查看日志

### 前端日志

启动 `npm run dev` 时会在终端显示日志，包括：
- 编译信息
- 请求日志
- 错误信息

### 数据库查询日志

Prisma Client 已配置为在开发模式显示查询日志。

### MySQL 日志

```bash
# 查看 MySQL 错误日志
sudo tail -f /var/log/mysql/error.log

# 查看通用查询日志（如果启用）
sudo tail -f /var/log/mysql/mysql.log
```

## 🛠️ 开发工具

### Prisma Studio（数据库管理界面）

```bash
cd /home/zyb/project/pingtai_test/frontend
export DATABASE_URL="mysql://root:pass@127.0.0.1:3306/chemhub"
npx prisma studio
```

然后访问 http://localhost:5555

### 重置数据库

```bash
# 清空所有数据（保留表结构）
mysql -u root -ppass -e "USE chemhub; TRUNCATE TABLE users;"

# 完全重建数据库
mysql -u root -ppass << 'EOF'
DROP DATABASE IF EXISTS chemhub;
CREATE DATABASE chemhub CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EOF

# 重新推送 schema
cd /home/zyb/project/pingtai_test/frontend
export DATABASE_URL="mysql://root:pass@127.0.0.1:3306/chemhub"
npx prisma db push
```

## 🚀 生产部署

如需部署到生产环境，请参考：

- `DEPLOYMENT_WITH_AUTH.md` - 完整部署指南
- `DEPLOYMENT_CHECKLIST.md` - 部署检查清单
- `reverse-proxy/README.md` - Nginx 反向代理配置

## 📚 其他文档

- `README.md` - 项目总览
- `QUICKSTART-SIMPLE.md` - 快速开始
- `HOW_TO_CHANGE_IP.md` - 修改 IP 配置
- `MYSQL_CONNECTION_FIX.md` - MySQL 连接问题修复
- `PROJECT_STRUCTURE.md` - 项目结构说明

## ✨ 快速命令参考

```bash
# === 启动服务 ===
cd /home/zyb/project/pingtai_test
./start-with-mysql.sh                    # 一键启动

# === 只启动前端 ===
cd /home/zyb/project/pingtai_test/frontend
npm run dev                               # 开发模式

# === 数据库管理 ===
mysql -u root -ppass                      # 登录 MySQL
mysql -u root -ppass chemhub              # 直接进入 chemhub 数据库

# === Prisma 操作 ===
cd /home/zyb/project/pingtai_test/frontend
export DATABASE_URL="mysql://root:pass@127.0.0.1:3306/chemhub"
npx prisma studio                         # 数据库管理界面
npx prisma db push                        # 同步 schema
npx prisma generate                       # 生成 Client

# === 查看状态 ===
sudo systemctl status mysql               # MySQL 状态
ps aux | grep node                        # Node 进程
ps aux | grep python                      # Python 进程
sudo lsof -i :5173                       # 端口占用

# === 停止服务 ===
./stop-all.sh                            # 一键停止
# 或按 Ctrl+C
```

---

## 🎉 现在可以开始使用了！

执行以下命令立即启动：

```bash
cd /home/zyb/project/pingtai_test
./start-with-mysql.sh
```

然后访问 http://localhost:5173 开始使用 ChemHub！

