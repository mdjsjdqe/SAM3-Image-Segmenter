# SAM3 Image Segmenter v1.2.0

基于 Meta SAM3 (Segment Anything Model 3) 的交互式图像分割桌面应用。

## ✨ 功能

- **📝 文字提示分割**：输入文字描述（如 "cat"、"defect"），自动检测并分割所有匹配对象
- **🖱️ 点击提示分割**：左键添加正提示点（前景），右键添加负提示点（背景）
- **📦 框选提示分割**：鼠标拖拽画框，框选目标区域
- **🔗 多次分割累积**：分割结果累积保留，可反复操作直到满意
- **🔗 合并 Mask**：一键合并所有可见 mask 为 combined mask
- **💾 多格式导出**：叠加图、彩色/二值/实例 mask、标签映射文件
- **💾 会话持久化**：自动保存/恢复图片路径、模型路径、分割结果，重启不丢失
- **🍎 macOS 原生 .app**：PyInstaller 打包，支持 Retina，含应用图标
- **🛡️ 防闪退**：全局异常捕获 + 日志记录，出错不闪退

## 🚀 快速开始

### 方式一：直接运行（开发模式）

```bash
# 1. 克隆/下载项目
cd SAM3Segmenter

# 2. 一键启动（自动创建虚拟环境、安装依赖、生成图标）
bash run.sh

# 3. 在应用中加载模型
#    点击「🧠 加载模型」→ 选择 sam3.pt 文件
```

### 方式二：打包为 macOS .app

```bash
# 1. 确保依赖已安装
source .venv/bin/activate

# 2. 一键打包（自动生成图标 + 打包 .app）
bash build.sh

# 3. 打包完成后
open "dist/SAM3 Image Segmenter.app"
# 或拖到 /Applications
```

## 📋 依赖

| 依赖        | 版本     | 说明                          |
| ----------- | -------- | ----------------------------- |
| Python      | 3.9+     | 推荐 3.11                     |
| PyTorch     | 2.0+     | macOS CPU 版（支持 MPS 加速） |
| Ultralytics | 8.3.237+ | SAM3 核心依赖                 |
| Pillow      | 10.0+    | 图像处理 + 图标生成           |
| NumPy       | 1.24+    | 数值计算                      |
| PyInstaller | 6.0+     | 仅打包时需要                  |

## 🎯 使用流程

```
打开图片 → 加载模型 → 多次分割（文字/点击/框选）
    ↓
结果累积在列表中 → 可隐藏/删除不需要的
    ↓
点击「🔗 合并所有 Mask」预览
    ↓
点击「💾 保存结果」→ 导出 combined mask + 单个 mask + 标签映射
```

## 💾 会话持久化

应用会自动保存以下状态，重启后自动恢复：

| 保存内容 | 说明                                       |
| -------- | ------------------------------------------ |
| 模型路径 | 重启后自动加载上次使用的模型               |
| 图片路径 | 重启后自动打开上次的图片                   |
| 分割结果 | 每个 mask 保存为 .npy 文件，重启后自动恢复 |
| 窗口大小 | 记住上次关闭时的窗口尺寸                   |

文件存储位置：

- `session.json` — 会话基本状态
- `saved_masks/{图片名}/` — 分割结果（mask .npy + 元数据 JSON）
- `sam3_segmenter.log` — 运行日志

## 🛡️ 防闪退机制

- **全局异常捕获**：`sys.excepthook` + Tkinter 回调异常处理
- **日志记录**：所有错误写入 `sam3_segmenter.log`
- **友好提示**：出错时弹出对话框，显示错误信息和日志路径

## 📦 导出文件说明

| 文件                            | 说明                                     |
| ------------------------------- | ---------------------------------------- |
| `{name}_overlay.png`            | 原图 + mask 叠加图                       |
| `{name}_combined_color.png`     | 彩色 combined mask（每个实例不同颜色）   |
| `{name}_combined_binary.png`    | 二值 combined mask（所有实例合并为白色） |
| `{name}_combined_instances.png` | 实例编号灰度图（1, 2, 3...）             |
| `{name}_masks/`                 | 每个 mask 单独保存                       |
| `{name}_label_map.txt`          | 标签映射文件                             |

## ⚠️ 模型下载

SAM3 模型权重不会自动下载，需手动获取：

1. 访问 https://huggingface.co/facebook/sam3
2. 登录 HuggingFace 并申请访问权限（免费，通常几小时内批准）
3. 下载 `sam3.pt`（约 3.4 GB）
4. 在应用中用「加载模型」选择该文件

## 🍎 macOS .app 注意事项

首次打开 .app 可能被 macOS Gatekeeper 拦截：

```bash
# 方式一：右键点击 .app → 打开 → 仍要打开
# 方式二：终端执行
xattr -cr "dist/SAM3 Image Segmenter.app"
```
