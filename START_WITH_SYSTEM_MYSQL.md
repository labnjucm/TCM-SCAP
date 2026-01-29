# 🚀 使用系统 MySQL 启动 ChemHub 完整指南

## 📋 前提条件

- ✅ MySQL 已安装在系统中
- ✅ MySQL root 密码：`pass`
- ✅ MySQL 正在运行

---

## 🗄️ 步骤 1：创建数据库

```bash
# 连接到 MySQL
mysql -u root -ppass

# 或者如果上面的命令不行，使用：
mysql -u root -p
# 然后输入密码：pass
```

在 MySQL 命令行中执行：

```sql
-- 创建数据库
CREATE DATABASE chemhub CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 验证数据库已创建
SHOW DATABASES;

-- 退出 MySQL
EXIT;
```

应该看到 `chemhub` 数据库在列表中。

---

## ⚙️ 步骤 2：配置环境变量

```bash
cd /home/zyb/project/pingtai_test/frontend

# 创建 .env.local 文件
cat > .env.local << 'EOF'
# 页面配置
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)

# Gradio 应用地址
NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:7861
NEXT_PUBLIC_MD_PATH=http://127.0.0.1:7862
NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:7863

# 数据库配置（系统 MySQL）
DATABASE_URL="mysql://root:pass@localhost:3306/chemhub"

# JWT 密钥（生产环境请更换）
JWT_SECRET="chemhub_jwt_secret_change_in_production_2024"
EOF

# 验证文件已创建
cat .env.local
```

---

## 📦 步骤 3：安装依赖

```bash
# 确保在 frontend 目录
cd /home/zyb/project/pingtai_test/frontend

# 安装所有 npm 依赖
npm install

# 等待安装完成（可能需要 1-3 分钟）
```

如果 bcrypt 编译失败，执行：

```bash
# Ubuntu/Debian
sudo apt-get install python3 make g++

# 然后重新安装
npm install
```

---

## 🔧 步骤 4：初始化数据库表

```bash
# 1. 生成 Prisma Client
npx prisma generate

# 2. 创建数据库表（users 表）
npx prisma migrate dev --name init

# 如果提示确认，输入：yes
```

**预期输出**：
```
✔ Prisma Migrate applied the following migration(s):
  └─ 20241104000000_init/
    └─ migration.sql

✔ Generated Prisma Client (...)
```

---

## ▶️ 步骤 5：启动前端

### 方式 1：开发模式（推荐）

```bash
npm run dev
```

**预期输出**：
```
▲ Next.js 14.2.5
- Local:        http://localhost:5173
- Ready in 2.3s
```

### 方式 2：生产模式

```bash
# 构建
npm run build

# 启动
npm run start
```

---

## 🧪 步骤 6：测试新功能

### 1. 访问主界面

```bash
# 在浏览器打开
http://localhost:5173
```

### 2. 测试用户注册

1. 点击右下角 **齿轮按钮** ⚙️
2. 切换到 **"注册"** 标签
3. 输入：
   - 邮箱：`test@example.com`
   - 密码：`12345678`
4. 点击 **"注册"**
5. ✅ 应显示："注册成功！请登录"

### 3. 测试登录

1. 切换到 **"登录"** 标签
2. 输入相同的邮箱和密码
3. 点击 **"登录"**
4. ✅ Header 右上角应显示：`test@example.com`

### 4. 测试详细说明

1. 左侧菜单点击：**"获取数据"** → **"RCSB PDB"**
2. 点击绿色按钮：**"查看详细说明"**
3. ✅ 应弹出详细说明对话框

---

## 🐍 步骤 7：启动 Gradio 应用（可选）

在新的终端窗口中：

```bash
# 终端 1：分子对接
cd /home/zyb/project/pingtai_test
python3 examples/docking_app.py

# 终端 2：分子动力学
python3 examples/md_app.py

# 终端 3：ORCA 计算
python3 examples/orca_app.py
```

然后在 ChemHub 中点击"打开我的 Gradio"即可内嵌使用。

---

## 🛠️ 常用管理命令

### 查看数据库内容

```bash
# 方式 1：Prisma Studio（可视化）
cd /home/zyb/project/pingtai_test/frontend
npx prisma studio

# 访问：http://localhost:5555
```

```bash
# 方式 2：MySQL 命令行
mysql -u root -ppass chemhub

# 查看用户表
SELECT * FROM users;

# 查看表结构
DESCRIBE users;

# 退出
EXIT;
```

### 重置数据库

```bash
cd /home/zyb/project/pingtai_test/frontend

# 警告：这会删除所有数据！
npx prisma migrate reset

# 确认后会重新创建所有表
```

### 停止服务

```bash
# 前端：在终端按 Ctrl+C

# Gradio 应用：在各自终端按 Ctrl+C
```

---

## 📝 一键启动脚本

创建启动脚本以便快速启动：

```bash
cd /home/zyb/project/pingtai_test

# 创建启动脚本
cat > start-with-mysql.sh << 'SCRIPT'
#!/bin/bash

echo "🚀 ChemHub 启动脚本（系统 MySQL）"
echo "=================================="
echo ""

# 检查 MySQL 是否运行
if ! mysqladmin ping -h localhost -u root -ppass &> /dev/null; then
    echo "❌ MySQL 未运行或密码错误"
    echo "请启动 MySQL 服务：sudo systemctl start mysql"
    exit 1
fi

echo "✅ MySQL 正在运行"

# 检查数据库是否存在
if ! mysql -u root -ppass -e "USE chemhub" &> /dev/null; then
    echo "❌ 数据库 chemhub 不存在"
    echo "正在创建数据库..."
    mysql -u root -ppass -e "CREATE DATABASE chemhub CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    echo "✅ 数据库已创建"
fi

cd frontend

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
fi

# 初始化数据库（如果需要）
if [ ! -d "node_modules/.prisma" ]; then
    echo "🔧 初始化 Prisma..."
    npx prisma generate
fi

# 运行迁移
echo "🗄️  运行数据库迁移..."
npx prisma migrate deploy

# 启动前端
echo "▶️  启动前端..."
echo ""
echo "访问：http://localhost:5173"
echo ""
npm run dev
SCRIPT

# 添加执行权限
chmod +x start-with-mysql.sh

echo "✅ 启动脚本已创建：start-with-mysql.sh"
```

以后只需运行：

```bash
cd /home/zyb/project/pingtai_test
./start-with-mysql.sh
```

---

## 🐛 常见问题排查

### 问题 1：MySQL 连接失败

**错误**：`Can't reach database server at localhost:3306`

**解决方案**：

```bash
# 检查 MySQL 是否运行
sudo systemctl status mysql

# 如果未运行，启动它
sudo systemctl start mysql

# 测试连接
mysql -u root -ppass -e "SELECT 1;"
```

### 问题 2：数据库不存在

**错误**：`Unknown database 'chemhub'`

**解决方案**：

```bash
mysql -u root -ppass -e "CREATE DATABASE chemhub CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

### 问题 3：密码错误

**错误**：`Access denied for user 'root'@'localhost'`

**解决方案**：

```bash
# 如果密码不是 pass，更新 .env.local
nano frontend/.env.local

# 修改 DATABASE_URL：
DATABASE_URL="mysql://root:你的实际密码@localhost:3306/chemhub"
```

### 问题 4：端口 3306 被占用

**错误**：MySQL 无法启动

**解决方案**：

```bash
# 查找占用端口的进程
sudo lsof -i :3306

# 或查看 MySQL 配置
sudo nano /etc/mysql/my.cnf
```

### 问题 5：Prisma 迁移失败

**错误**：Migration failed

**解决方案**：

```bash
# 删除迁移历史
rm -rf frontend/prisma/migrations

# 重新生成迁移
cd frontend
npx prisma migrate dev --name init
```

---

## 📊 验证安装

运行以下命令验证一切正常：

```bash
# 1. 验证 MySQL
mysql -u root -ppass -e "SELECT VERSION();"

# 2. 验证数据库
mysql -u root -ppass -e "SHOW DATABASES LIKE 'chemhub';"

# 3. 验证表
mysql -u root -ppass chemhub -e "SHOW TABLES;"

# 4. 验证前端
cd /home/zyb/project/pingtai_test/frontend
npm run dev -- --help
```

---

## 🎉 完整启动流程总结

```bash
# === 一次性设置（只需运行一次）===

# 1. 创建数据库
mysql -u root -ppass -e "CREATE DATABASE chemhub CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 2. 进入前端目录
cd /home/zyb/project/pingtai_test/frontend

# 3. 创建配置文件
cat > .env.local << 'EOF'
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)
NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:7861
NEXT_PUBLIC_MD_PATH=http://127.0.0.1:7862
NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:7863
DATABASE_URL="mysql://root:pass@localhost:3306/chemhub"
JWT_SECRET="chemhub_jwt_secret_change_in_production_2024"
EOF

# 4. 安装依赖
npm install

# 5. 初始化数据库
npx prisma generate
npx prisma migrate dev --name init

# === 日常启动（每次都需要）===

# 6. 启动前端
npm run dev

# 访问 http://localhost:5173
```

---

## 🔐 生产环境建议

如果要部署到生产环境：

1. **更换 JWT 密钥**：
   ```bash
   # 生成强随机密钥
   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
   
   # 更新到 .env.local
   JWT_SECRET="生成的密钥"
   ```

2. **创建专用数据库用户**：
   ```sql
   CREATE USER 'chemhub'@'localhost' IDENTIFIED BY '强密码';
   GRANT ALL PRIVILEGES ON chemhub.* TO 'chemhub'@'localhost';
   FLUSH PRIVILEGES;
   
   -- 更新 DATABASE_URL
   DATABASE_URL="mysql://chemhub:强密码@localhost:3306/chemhub"
   ```

3. **配置 SSL**：
   ```bash
   # 使用 Let's Encrypt 等工具配置 HTTPS
   ```

---

**现在可以开始使用了！** 🎊

有问题请参考：
- `QUICK_TEST_GUIDE.md` - 快速测试指南
- `DEPLOYMENT_WITH_AUTH.md` - 详细部署文档
- `FEATURE_COMPLETE_SUMMARY.md` - 功能完整说明

