#!/usr/bin/env bash
set -Eeuo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║        🚀 ChemHub 启动脚本（系统 MySQL）                    ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# MySQL 配置
MYSQL_USER="root"
MYSQL_PASS="pass"
DB_NAME="chemhub"

# 让非交互 shell 也能用 conda
if ! command -v conda >/dev/null 2>&1; then
  # 视你安装位置调整 anaconda3/miniconda3
  . "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi

# 检查 MySQL 是否运行
echo "🔍 检查 MySQL 服务..."
if ! mysqladmin ping -h localhost -u "$MYSQL_USER" -p"$MYSQL_PASS" &> /dev/null; then
    echo -e "${RED}❌ MySQL 未运行或密码错误${NC}"
    echo ""
    echo "请执行以下操作之一："
    echo "1. 启动 MySQL：sudo systemctl start mysql"
    echo "2. 检查密码是否为 'pass'"
    echo "3. 如果密码不同，编辑本脚本的 MYSQL_PASS"
    exit 1
fi
echo -e "${GREEN}✅ MySQL 正在运行${NC}"

# 检查数据库是否存在
echo "🗄️  检查数据库..."
if ! mysql -u "$MYSQL_USER" -p"$MYSQL_PASS" -e "USE \`$DB_NAME\`" &> /dev/null; then
    echo -e "${YELLOW}⚠️  数据库 $DB_NAME 不存在${NC}"
    echo "正在创建数据库..."
    if mysql -u "$MYSQL_USER" -p"$MYSQL_PASS" -e "CREATE DATABASE \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"; then
        echo -e "${GREEN}✅ 数据库已创建${NC}"
    else
        echo -e "${RED}❌ 数据库创建失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ 数据库已存在${NC}"
fi

# ------- 启动后端三个服务（后台）并等待就绪 -------
ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# 服务端口（与你 .env.local 中保持一致）
PORT_DOCKING=7861
PORT_MD=7862
PORT_ORCA=7863

start_service() {
  local rel_dir="$1" env="$2" entry="$3" name="$4"
  (
    cd "$ROOT/$rel_dir"
    # 若你的 python 服务支持从环境变量读取端口，可在此处 export 对应变量
    # export PORT=xxxx  或  export GRADIO_SERVER_PORT=xxxx  等
    nohup conda run -n "$env" --no-capture-output python "$entry" \
      >"$LOG_DIR/$name.out" 2>&1 < /dev/null &
    echo $! > "$PID_DIR/$name.pid"
  )
  echo "[start] $name (pid=$(cat "$PID_DIR/$name.pid")) -> $LOG_DIR/$name.out"
}

wait_port() { # host port timeout name
  local host="$1" port="$2" timeout="${3:-90}" name="$4"
  echo "[wait] $name: tcp://${host}:${port} (<= ${timeout}s)"
  local t=0
  while ! (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; do
    ((t++)); if ((t>=timeout)); then
      echo "[fail] $name: 端口 ${port} 超时未就绪"
      tail -n 120 "$LOG_DIR/$name.out" || true
      exit 1
    fi
    sleep 1
  done
  echo "[ok]   $name: 端口已就绪"
}

# 启动
start_service "components/Comchemistry" kaifa      "orca_gradio_app.py" "orca"
start_service "components/Modynamics"   kaifa      "gmx_gui_runner.py"  "md"
start_service "components/docking"      test1_bat1 "app/main.py"        "docking"

# 等待对应端口（按你的实际端口修改）
wait_port 127.0.0.1 "$PORT_ORCA"   90 "orca"
wait_port 127.0.0.1 "$PORT_MD"     90 "md"
wait_port 127.0.0.1 "$PORT_DOCKING" 90 "docking"

echo "[ready] 三个后端服务均已就绪"

# ------- 前端与数据库迁移 -------
cd "$ROOT/frontend" || exit 1

# .env.local（若不存在则创建）
if [ ! -f ".env.local" ]; then
    echo "📝 创建配置文件..."
    cat > .env.local << EOF
NEXT_PUBLIC_APP_TITLE=ChemHub
NEXT_PUBLIC_FOOTER_NOTE=© 2025 ChemHub (internal)
NEXT_PUBLIC_DOCKING_PATH=http://127.0.0.1:${PORT_DOCKING}
NEXT_PUBLIC_MD_PATH=http://127.0.0.1:${PORT_MD}
NEXT_PUBLIC_ORCA_PATH=http://127.0.0.1:${PORT_ORCA}
DATABASE_URL="mysql://${MYSQL_USER}:${MYSQL_PASS}@localhost:3306/${DB_NAME}"
JWT_SECRET="chemhub_jwt_secret_change_in_production_2024"
EOF
    echo -e "${GREEN}✅ 配置文件已创建${NC}"
fi

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖（首次运行需要 1-3 分钟）..."
    npm install
    echo -e "${GREEN}✅ 依赖安装完成${NC}"
else
    echo -e "${GREEN}✅ 依赖已安装${NC}"
fi

# Prisma
if [ ! -d "node_modules/.prisma" ]; then
    echo "🔧 生成 Prisma Client..."
    npx prisma generate
    echo -e "${GREEN}✅ Prisma Client 已生成${NC}"
fi

# 检查是否需要迁移
echo "🔄 检查数据库迁移..."
TABLE_COUNT=$(mysql -u "$MYSQL_USER" -p"$MYSQL_PASS" "$DB_NAME" -se "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '$DB_NAME';")
if [ "${TABLE_COUNT:-0}" -eq 0 ]; then
    echo "🗄️  运行数据库迁移（创建 users 表）..."
    npx prisma migrate dev --name init
    echo -e "${GREEN}✅ 数据库表已创建${NC}"
else
    echo -e "${GREEN}✅ 数据库表已存在${NC}"
    npx prisma migrate deploy &> /dev/null || true
fi

npm run prisma:seed

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ 所有准备工作完成！${NC}"
echo ""
echo "▶️  启动前端服务..."
echo ""
echo "访问地址："
echo "  • 主界面：http://localhost:5173"
echo "  • 数据库管理：npx prisma studio"
echo ""
echo "测试新功能："
echo "  1. 点击右下角齿轮按钮 ⚙️  注册/登录"
echo "  2. 点击左侧资源 → 查看详细说明"
echo ""
echo "停止服务：按 Ctrl+C"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# ------- Ctrl+C / 退出 时联动终止三个后端 -------
cleanup_services() {
  echo
  echo "[cleanup] 停止后台服务..."
  for name in orca md docking; do
    pid_file="$PID_DIR/$name.pid"
    if [[ -f "$pid_file" ]]; then
      pid=$(cat "$pid_file" 2>/dev/null || echo "")
      if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
        # 尝试优雅终止主进程与其进程组
        pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ' || echo "")
        kill -TERM "$pid" 2>/dev/null || true
        [[ -n "$pgid" ]] && kill -TERM "-$pgid" 2>/dev/null || true
      fi
    fi
  done
  sleep 2
  for name in orca md docking; do
    pid_file="$PID_DIR/$name.pid"
    if [[ -f "$pid_file" ]]; then
      pid=$(cat "$pid_file" 2>/dev/null || echo "")
      if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  done
  echo "[cleanup] 已处理。"
}

trap 'cleanup_services' INT TERM EXIT

# 前端前台运行；Ctrl+C 会触发上面的 trap，从而连带关停三个后台
npm run dev
