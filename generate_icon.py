"""
SAM3 Image Segmenter - 图标生成脚本
使用 Pillow 生成 macOS .icns 图标文件

macOS .icns 格式需要以下尺寸的图标：
16x16, 32x32, 64x64, 128x128, 256x256, 512x512, 1024x1024

使用方法:
    python generate_icon.py
    → 输出: icon.icns (macOS 图标)
    → 输出: icon_preview.png (预览图)
"""

import os
import struct
from PIL import Image, ImageDraw, ImageFont


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """
    查找可用的 TrueType 字体
    按优先级尝试多个路径，确保在所有 macOS 版本上都能找到有效字体
    """
    # macOS 系统字体候选列表（按优先级排序）
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/SFCompact.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                # 验证字体可用：尝试获取 bbox
                font.getbbox("S3")
                return font
            except (OSError, IOError, Exception):
                continue

    # 所有 TrueType 字体都失败，使用 load_default 但设置 size
    # Pillow >= 10.1 支持 load_default(size=)
    try:
        font = ImageFont.load_default(size=size)
        font.getbbox("S3")
        return font
    except (TypeError, OSError, AttributeError, Exception):
        pass

    # 最终回退：load_default() 无参数
    try:
        font = ImageFont.load_default()
        # 默认字体可能很小，手动缩放
        return font
    except Exception:
        pass

    # 绝对不可能到这里，但以防万一
    raise RuntimeError("无法加载任何字体，请检查 Pillow 安装")


def create_icon_image(size: int) -> Image.Image:
    """
    生成 SAM3 图标图像
    设计：圆角矩形背景 + 分割区域图案 + "S3" 文字
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 圆角矩形背景 - 渐变蓝紫色
    radius = size // 5
    border = max(1, size // 64)
    outer_rect = [border, border, size - border - 1, size - border - 1]
    if outer_rect[2] > outer_rect[0] and outer_rect[3] > outer_rect[1]:
        draw.rounded_rectangle(
            outer_rect,
            radius=radius,
            fill=(45, 55, 120, 255),  # 深蓝紫
        )
    # 内层圆角矩形（渐变效果 - 用两层模拟）
    inset = max(2, size // 32)
    inner_rect = [inset, inset, size - inset - 1, size - inset - 1]
    if inner_rect[2] > inner_rect[0] and inner_rect[3] > inner_rect[1]:
        draw.rounded_rectangle(
            inner_rect,
            radius=max(1, radius - inset // 2),
            fill=(65, 85, 180, 255),  # 蓝紫
        )
    # 最内层（高光）
    inner = max(4, size // 16)
    highlight_rect = [inner, inner, size - inner - 1, size * 2 // 3]
    if highlight_rect[2] > highlight_rect[0] and highlight_rect[3] > highlight_rect[1]:
        draw.rounded_rectangle(
            highlight_rect,
            radius=max(1, radius - inner),
            fill=(85, 110, 210, 80),  # 半透明高光
        )

    # 分割线图案（对角虚线 + 多边形片段）
    line_width = max(2, size // 32)
    # 主对角线
    draw.line(
        [(size * 2 // 10, size * 8 // 10), (size * 8 // 10, size * 2 // 10)],
        fill=(255, 255, 255, 200),
        width=line_width,
    )
    # 上方三角片段（被分割出的部分）
    tri_margin = size // 6
    draw.polygon(
        [
            (tri_margin, tri_margin),
            (size - tri_margin, tri_margin),
            (size - tri_margin, size // 3),
        ],
        fill=(0, 220, 180, 160),  # 青绿色 - 分割区域
    )
    # 下方三角片段
    draw.polygon(
        [
            (tri_margin, size - tri_margin),
            (tri_margin, size * 2 // 3),
            (size - tri_margin, size - tri_margin),
        ],
        fill=(255, 180, 50, 160),  # 橙色 - 另一个分割区域
    )

    # "S3" 文字
    font_size = size // 3
    font = _find_font(font_size)

    text = "S3"
    # 安全获取文字尺寸
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except (OSError, Exception):
        # textbbox 失败时使用 font.getbbox
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except (OSError, Exception):
            # 最终回退：估算尺寸
            text_w = font_size * len(text) * 2 // 3
            text_h = font_size

    text_x = (size - text_w) // 2
    text_y = (size - text_h) // 2

    # 文字阴影
    shadow_offset = max(1, size // 64)
    try:
        draw.text(
            (text_x + shadow_offset, text_y + shadow_offset),
            text, fill=(0, 0, 0, 120), font=font,
        )
        # 文字主体
        draw.text(
            (text_x, text_y),
            text, fill=(255, 255, 255, 240), font=font,
        )
    except (OSError, Exception):
        # 字体渲染失败时跳过文字，只保留图形
        pass

    return img


def create_icns(sizes_dict: dict, output_path: str):
    """
    将多尺寸 PNG 图标打包为 macOS .icns 文件

    .icns 文件格式:
    - 文件头: 'icns' + 4字节文件总大小
    - 图标数据: 类型标识 + 4字节块大小 + 像素数据
    """
    # macOS .icns 图标类型映射
    ICNS_TYPES = {
        16: "icp4",    # 16x16
        32: "ic11",    # 16x16@2x / 32x32
        64: "ic12",    # 32x32@2x / 64x64
        128: "icp6",   # 128x128
        256: "ic08",   # 256x256
        512: "ic07",   # 512x512
        1024: "ic09",  # 512x512@2x / 1024x1024
    }

    icns_data = b""

    for size, icon_type in sorted(ICNS_TYPES.items()):
        if size in sizes_dict:
            png_data = sizes_dict[size]
            type_bytes = icon_type.encode("ascii")
            block_size = len(png_data) + 8
            icns_data += type_bytes + struct.pack(">I", block_size) + png_data

    file_size = len(icns_data) + 8
    with open(output_path, "wb") as f:
        f.write(b"icns")
        f.write(struct.pack(">I", file_size))
        f.write(icns_data)

    print(f"✅ 图标已保存: {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir

    # 生成各尺寸图标
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    sizes_dict = {}

    print("🎨 生成 SAM3 Image Segmenter 图标...")
    for size in sizes:
        img = create_icon_image(size)
        png_path = os.path.join(output_dir, f"icon_{size}x{size}.png")
        img.save(png_path, "PNG")
        with open(png_path, "rb") as f:
            sizes_dict[size] = f.read()
        print(f"  ✅ {size}x{size}")

    # 生成 .icns
    icns_path = os.path.join(output_dir, "icon.icns")
    create_icns(sizes_dict, icns_path)

    # 生成预览图（512x512）
    preview = create_icon_image(512)
    preview_path = os.path.join(output_dir, "icon_preview.png")
    preview.save(preview_path, "PNG")
    print(f"✅ 预览图已保存: {preview_path}")

    # 清理临时 PNG 文件
    for size in sizes:
        png_path = os.path.join(output_dir, f"icon_{size}x{size}.png")
        if os.path.exists(png_path):
            os.remove(png_path)

    print(f"\n🎉 图标生成完成！")
    print(f"   📱 macOS 图标: {icns_path}")
    print(f"   🖼️  预览图: {preview_path}")


if __name__ == "__main__":
    main()
