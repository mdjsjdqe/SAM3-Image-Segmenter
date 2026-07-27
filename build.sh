#!/bin/bash
# ============================================================
# SAM3 Image Segmenter - macOS .app 打包脚本
# 使用 PyInstaller 将 Python 应用打包为原生 macOS 应用
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="SAM3 Image Segmenter"
SPEC_FILE="SAM3ImageSegmenter.spec"
DIST_DIR="dist"
BUILD_DIR="build"

echo "========================================="
echo "  SAM3 Image Segmenter - macOS 打包"
echo "========================================="
echo ""

# ── 1. 检查虚拟环境 ──
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "🔄 激活虚拟环境: .venv"
        source .venv/bin/activate
    else
        echo "❌ 未检测到虚拟环境"
        echo "   请先运行 run.sh 创建虚拟环境并安装依赖"
        echo "   或手动激活: source .venv/bin/activate"
        exit 1
    fi
fi

echo "🐍 Python: $(python3 --version)"
echo "📍 路径: $(which python3)"
echo ""

# ── 2. 安装 PyInstaller 和 Pillow ──
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "📦 安装 PyInstaller..."
    pip install pyinstaller
fi
if ! python3 -c "from PIL import Image" 2>/dev/null; then
    echo "📦 安装 Pillow（图标生成需要）..."
    pip install Pillow
fi

echo "✅ PyInstaller 已就绪"
echo ""

# ── 3. 生成应用图标 ──
if [ ! -f "icon.icns" ]; then
    echo "🎨 生成应用图标..."
    python3 generate_icon.py
else
    echo "✅ 图标已存在: icon.icns"
fi
echo ""

# ── 3. 检查必要依赖 ──
echo "🔍 检查依赖..."
REQUIRED_PACKAGES=("torch" "ultralytics" "PIL" "numpy")
MISSING=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "⚠️  缺少依赖: ${MISSING[*]}"
    echo "   正在安装..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -U ultralytics
    pip install Pillow numpy
fi

echo "✅ 所有依赖已就绪"
echo ""

# ── 4. 彻底清理旧的构建（含 PyInstaller 全局缓存）──
echo "🧹 清理旧的构建文件..."
rm -rf "$BUILD_DIR" "$DIST_DIR"
# 清理 PyInstaller 全局缓存（避免旧缓存残留导致 importlib 丢失）
PYI_CACHE="$HOME/Library/Application Support/pyinstaller"
if [ -d "$PYI_CACHE" ]; then
    echo "🧹 清理 PyInstaller 全局缓存: $PYI_CACHE"
    rm -rf "$PYI_CACHE"
fi

# ── 5. 执行打包 ──
echo ""
echo "🚀 开始打包 (这可能需要几分钟)..."
echo ""

pyinstaller "$SPEC_FILE" \
    --clean \
    --noconfirm \
    --workpath="$BUILD_DIR" \
    --distpath="$DIST_DIR"

# ── 6. 检查打包结果 ──
APP_PATH="$DIST_DIR/$APP_NAME.app"

if [ -d "$APP_PATH" ]; then
    APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
    echo ""
    echo "========================================="
    echo "  ✅ 打包成功！"
    echo "========================================="
    echo ""
    echo "  📱 应用: $APP_PATH"
    echo "  📦 大小: $APP_SIZE"
    echo ""
    echo "  📋 使用方法:"
    echo "     1. 双击 $APP_PATH 启动"
    echo "     2. 或终端执行: open \"$APP_PATH\""
    echo "     3. 或拖拽到 /Applications 文件夹"
    echo ""
    echo "  ⚠️  首次打开可能被 macOS Gatekeeper 拦截："
    echo "     右键点击 → 打开 → 仍要打开"
    echo "     或在 系统设置 → 隐私与安全性 中允许"
    echo ""
    echo "  💡 模型文件 (sam3.pt) 不会打包进 app，"
    echo "     首次使用时需手动加载模型文件"
    echo ""

    # ── 自动打开 app ──
    echo "🚀 正在启动应用..."
    open "dist/$APP_NAME.app"
else
    echo ""
    echo "❌ 打包失败，请检查上方错误信息"
    exit 1
fi
