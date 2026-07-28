#!/usr/bin/env python3
"""
SAM3 Image Segmenter — 应用图标生成器
生成 icon.ico（Windows）和 icon.icns（macOS）用于 PyInstaller 打包

图标设计：蓝色圆形背景 + 白色 SAM（Segment Anything Model）剪影
"""

import os
import sys
import struct

def generate_icon_ico(output_path="icon.ico", size=256):
    """生成 Windows .ico 格式图标（纯代码绘制，无需 Pillow）"""
    
    # ── 用 BMP 数据生成图标 ──
    # ICO 文件格式：ICONDIR + ICONDIRENTRY + BMP 数据
    
    width = size
    height = size
    
    # 生成像素数据（BGRA 格式，从下到上）
    pixels = []
    cx, cy = width // 2, height // 2
    r_outer = width // 2 - 4  # 外圆半径（留边距）
    r_inner = width // 4      # 内圆（SAM 中心标记）
    
    for y in range(height):
        row = []
        for x in range(width):
            # 距离中心
            dx = x - cx
            dy = y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            
            # 背景透明
            if dist > r_outer:
                row.append((0, 0, 0, 0))  # BGRA: 透明
            # 外圆：蓝色渐变
            elif dist > r_inner + 8:
                # 蓝色渐变（从深蓝到浅蓝）
                t = (dist - r_inner - 8) / (r_outer - r_inner - 8)
                b = int(40 + t * 80)    # Blue: 40→120
                g = int(80 + t * 40)    # Green: 80→120  
                r = int(20 + t * 30)    # Red: 20→50
                a = 255
                row.append((b, g, r, a))  # BGRA
            # 内圆：白色
            elif dist > r_inner - 2:
                row.append((255, 255, 255, 255))  # BGRA: 白色
            # 中心：深蓝色
            else:
                row.append((120, 100, 40, 255))  # BGRA: 深蓝
            
        pixels.append(row)
    
    # BMP 数据（从下到上存储）
    bmp_data = b''
    for row in reversed(pixels):
        for b, g, r, a in row:
            bmp_data += struct.pack('BBBB', b, g, r, a)
    
    # AND mask（全0 = 不透明）
    and_mask = b'\x00' * ((width + 31) // 32 * 4 * height)
    
    # ── ICO 文件结构 ──
    # ICONDIR (6 bytes)
    icondir = struct.pack('<HHH', 0, 1, 1)  # Reserved, Type=ICO, Count=1
    
    # ICONDIRENTRY (16 bytes)
    image_data = bmp_data + and_mask
    bmp_header = struct.pack('<IiiHHIIiiII',
        40,  # biSize (BITMAPINFOHEADER)
        width,  # biWidth
        height * 2,  # biHeight (2x because AND mask is included)
        1,  # biPlanes
        32,  # biBitCount (BGRA)
        0,  # biCompression (BI_RGB)
        len(bmp_data),  # biSizeImage
        0,  # biXPelsPerMeter
        0,  # biYPelsPerMeter
        0,  # biClrUsed
        0,  # biClrImportant
    )
    
    entry_data = bmp_header + image_data
    entry_offset = 6 + 16  # ICONDIR + ICONDIRENTRY
    
    icondirentry = struct.pack('<BBBBHHII',
        0 if width >= 256 else width,   # Width (0 = 256+)
        0 if height >= 256 else height,  # Height (0 = 256+)
        0,  # ColorCount
        0,  # Reserved
        1,  # Planes
        32, # BitCount
        len(entry_data),  # SizeInBytes
        entry_offset,     # FileOffset
    )
    
    ico_data = icondir + icondirentry + entry_data
    
    with open(output_path, 'wb') as f:
        f.write(ico_data)
    
    print(f"[OK] Windows icon generated: {output_path} ({len(ico_data)} bytes)")


def generate_icon_png(output_path="icon.png", size=256):
    """生成 PNG 格式图标（用于 Linux 和 macOS icns 转换）"""
    try:
        from PIL import Image, ImageDraw
        
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        cx, cy = size // 2, size // 2
        r_outer = size // 2 - 4
        r_inner = size // 4
        
        # 外圆：蓝色渐变
        for r in range(r_outer, r_inner + 8, -1):
            t = (r - r_inner - 8) / (r_outer - r_inner - 8)
            color = (int(20 + t * 30), int(80 + t * 40), int(40 + t * 80), 255)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        
        # 内圆：白色
        draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(255, 255, 255, 255))
        
        # 中心：深蓝
        r_center = r_inner - 2
        draw.ellipse([cx - r_center, cy - r_center, cx + r_center, cy + r_center], fill=(40, 100, 120, 255))
        
        # SAM 文字（如果字体可用）
        try:
            draw.text((cx - 30, cy - 15), "SAM", fill=(255, 255, 255, 255), font=None)
        except Exception:
            pass
        
        img.save(output_path, 'PNG')
        print(f"[OK] PNG icon generated: {output_path}")
        
    except ImportError:
        print("[WARN] Pillow not installed, PNG icon generation skipped")


def generate_icon_icns(output_path="icon.icns"):
    """生成 macOS .icns 格式图标（依赖 Pillow + png2icns 或 sips）"""
    # macOS 上可用 sips 命令从 PNG 生成 icns
    if sys.platform == 'darwin':
        png_path = "icon.png"
        generate_icon_png(png_path)
        
        iconset_dir = "icon.iconset"
        os.makedirs(iconset_dir, exist_ok=True)
        
        sizes = {
            'icon_16x16.png': 16,
            'icon_16x16@2x.png': 32,
            'icon_32x32.png': 32,
            'icon_32x32@2x.png': 64,
            'icon_128x128.png': 128,
            'icon_128x128@2x.png': 256,
            'icon_256x256.png': 256,
            'icon_256x256@2x.png': 512,
            'icon_512x512.png': 512,
            'icon_512x512@2x.png': 1024,
        }
        
        try:
            from PIL import Image
            base_img = Image.open(png_path)
            for name, sz in sizes.items():
                resized = base_img.resize((sz, sz), Image.LANCZOS)
                resized.save(os.path.join(iconset_dir, name), 'PNG')
            
            # 使用 sips 或 iconutil 生成 icns
            os.system(f'iconutil -c icns {iconset_dir} -o {output_path}')
            
            # 清理临时文件
            import shutil
            shutil.rmtree(iconset_dir, ignore_errors=True)
            
            if os.path.exists(output_path):
                print(f"[OK] macOS icon generated: {output_path}")
            else:
                print("[WARN] icns generation failed, will use PNG as icon")
        except ImportError:
            print("[WARN] Pillow not installed, skipping icns icon generation")
    else:
        print("[WARN] Non-macOS system, skipping icns icon generation")


if __name__ == '__main__':
    import sys
    import io
    # Fix Windows cp1252 encoding issue in GitHub Actions
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=== SAM3 Image Segmenter Icon Generator ===")
    
    # Generate all icon formats
    generate_icon_ico("icon.ico")     # Windows
    generate_icon_png("icon.png")     # Linux / General
    generate_icon_icns("icon.icns")   # macOS
    
    print("")
    print("Icon files generated, ready for PyInstaller packaging")
