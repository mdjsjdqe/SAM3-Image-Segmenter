"""
SAM3 Image Segmenter - 主应用 GUI
基于 Tkinter 构建，支持点提示、框提示的交互式图像分割
支持多次分割结果累积、合并 mask 导出
支持会话持久化（自动保存/恢复图片路径、模型路径、分割结果）
"""

import os
import sys
import json
import math
import logging
import traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np

from sam3_engine import SAM3Model

# ── tkinterdnd2 拖拽文件支持（可选）──
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False


# ── 日志配置 ──
def setup_logging():
    """配置日志，输出到文件和控制台"""
    log_dir = app_base_dir()
    log_path = os.path.join(log_dir, "sam3_segmenter.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("SAM3App")


# ── PyInstaller 打包适配 ──
def resource_path(relative_path: str) -> str:
    """
    获取资源文件的绝对路径（兼容 PyInstaller 打包和开发环境）
    PyInstaller 打包后，资源文件在 sys._MEIPASS 目录下
    开发环境中，资源文件在脚本所在目录
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)


def app_base_dir() -> str:
    """
    获取应用基础目录（用于存放模型、配置等用户数据）
    PyInstaller 打包后，.app 内部是只读的，用户数据应放在 .app 同级目录
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包模式：.app 所在目录
        # sys.executable = .../SAM3 Image Segmenter.app/Contents/MacOS/SAM3 Image Segmenter
        # 需要往上3层到 .app 所在目录
        return os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    # 开发模式：脚本所在目录
    return os.path.abspath(os.path.dirname(__file__))


# ── 会话持久化 ──
class SessionManager:
    """管理会话状态，自动保存和恢复应用状态"""

    SESSION_FILE = "session.json"

    def __init__(self):
        self.base_dir = app_base_dir()
        self.session_path = os.path.join(self.base_dir, self.SESSION_FILE)
        self.data = {
            "image_path": None,
            "model_path": None,
            "window_geometry": "1280x800",
            "masks_dir": None,  # 保存分割结果的目录
            "show_help_on_start": True,  # 启动时是否自动显示 Help
        }

    def load(self) -> dict:
        """加载上次的会话状态"""
        try:
            if os.path.exists(self.session_path):
                with open(self.session_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.data.update(saved)
                logging.getLogger("SAM3App").info(f"会话已恢复: {self.session_path}")
            return self.data.copy()
        except Exception as e:
            logging.getLogger("SAM3App").warning(f"加载会话失败: {e}")
            return self.data.copy()

    def save(self, **kwargs):
        """保存当前会话状态"""
        self.data.update(kwargs)
        try:
            with open(self.session_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.getLogger("SAM3App").warning(f"保存会话失败: {e}")

    def save_masks_session(self, image_path: str, masks: list, mask_visible: list):
        """
        保存分割结果到独立的 JSON 文件
        mask 本身保存为 numpy .npy 文件，元数据保存为 JSON
        """
        if not image_path or not masks:
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        masks_session_dir = os.path.join(self.base_dir, "saved_masks", base_name)
        os.makedirs(masks_session_dir, exist_ok=True)

        # 保存每个 mask 为 .npy 文件
        masks_meta = []
        for i, r in enumerate(masks):
            mask_path = os.path.join(masks_session_dir, f"mask_{i:03d}.npy")
            np.save(mask_path, r["mask"])
            masks_meta.append({
                "mask_file": f"mask_{i:03d}.npy",
                "bbox": r["bbox"],
                "score": r["score"],
                "label": r["label"],
                "name": r.get("name", f"Mask_{i+1}"),
                "area": r["area"],
                "prompt_type": r.get("prompt_type", "point"),
                "visible": mask_visible[i] if i < len(mask_visible) else True,
            })

        # 保存元数据
        meta_path = os.path.join(masks_session_dir, "masks_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "image_path": image_path,
                "masks": masks_meta,
            }, f, ensure_ascii=False, indent=2)

        self.save(masks_dir=masks_session_dir)
        logging.getLogger("SAM3App").info(f"分割结果已保存: {masks_session_dir} ({len(masks)} 个 mask)")

    def load_masks_session(self, image_path: str) -> tuple:
        """
        加载之前保存的分割结果
        Returns: (masks_list, mask_visible_list) 或 ([], [])
        """
        if not image_path:
            return [], []

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        masks_session_dir = os.path.join(self.base_dir, "saved_masks", base_name)
        meta_path = os.path.join(masks_session_dir, "masks_meta.json")

        if not os.path.exists(meta_path):
            return [], []

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 验证图片路径匹配
            saved_image = meta.get("image_path", "")
            if saved_image != image_path and os.path.basename(saved_image) != os.path.basename(image_path):
                logging.getLogger("SAM3App").info("图片路径不匹配，跳过恢复分割结果")
                return [], []

            masks = []
            visible = []
            for m in meta.get("masks", []):
                mask_path = os.path.join(masks_session_dir, m["mask_file"])
                if os.path.exists(mask_path):
                    mask = np.load(mask_path)
                    masks.append({
                        "mask": mask,
                        "bbox": m["bbox"],
                        "score": m["score"],
                        "label": m["label"],
                        "name": m.get("name", f"Mask_{i+1}"),
                        "area": m["area"],
                        "prompt_type": m["prompt_type"],
                    })
                    visible.append(m.get("visible", True))

            logging.getLogger("SAM3App").info(f"已恢复 {len(masks)} 个分割结果")
            return masks, visible

        except Exception as e:
            logging.getLogger("SAM3App").warning(f"加载分割结果失败: {e}")
            return [], []



class SAM3App:
    """SAM3 图像分割应用主窗口"""

    # 应用标题和版本
    APP_TITLE = "SAM3 Image Segmenter"
    APP_VERSION = "1.5.0"

    # 画布最大尺寸
    CANVAS_MAX_W = 900
    CANVAS_MAX_H = 700

    # 颜色调色板
    COLORS = [
        "#FF0000", "#00CC00", "#0066FF", "#FFCC00", "#FF00FF",
        "#00CCCC", "#FF8800", "#8800FF", "#0088FF", "#FF8888",
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{self.APP_TITLE} v{self.APP_VERSION}")

        # 会话管理器
        self.session = SessionManager()
        session_data = self.session.load()

        # 恢复窗口大小
        geometry = session_data.get("window_geometry", "1280x800")
        self.root.geometry(geometry)
        self.root.minsize(1024, 640)

        # 窗口关闭时保存会话
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # 核心模型
        self.engine = SAM3Model()

        # 状态变量
        self.image_path = None          # 当前图片路径
        self.image_pil = None           # PIL Image
        self.image_np = None            # numpy array (H, W, 3) RGB
        self.display_scale = 1.0        # 显示缩放比例
        self.segmentation_results = []  # 分割结果（累积，不替换）
        self.overlay_image = None       # 叠加后的图片
        self.mask_visible = []          # 每个 mask 是否可见

        # 交互状态
        self.prompt_mode = tk.StringVar(value="point")  # point / box
        self.click_point_list = []    # 点提示列表 [{id, name, coords, label, visible}, ...]
        self.click_point_counter = 0  # 点计数器
        self.box_start = None           # 框选起点
        self.box_end = None             # 框选终点
        self.is_drawing_box = False     # 是否正在画框
        self.box_rect_id = None         # 画布上的框矩形 ID
        self.box_start_img = None       # 框选起点（原图坐标）
        self.box_end_img = None         # 框选终点（原图坐标）

        # 框选形状模式
        self.box_shape_mode = tk.StringVar(value="rectangle")  # rect / ellipse / polygon
        self.shift_pressed = False      # Shift 键状态
        self.polygon_points = []        # 多边形顶点（画布坐标）
        self.polygon_img_points = []    # 多边形顶点（原图坐标）
        self.polygon_line_ids = []      # 多边形线段画布 ID
        self.polygon_point_ids = []     # 多边形顶点画布 ID

        # ROI 列表管理
        self.roi_list = []              # [{name, shape, bbox, canvas_ids, img_points, mask}, ...]
        self.roi_counter = 0            # ROI 编号计数器
        self.roi_visible = []           # ROI 可见性列表
        self.selected_roi_idx = -1      # 当前选中的 ROI 索引
        self.current_roi_canvas_ids = []  # 正在绘制的 ROI 画布 ID（未完成时）

        # 校准数据（直线拖拽标定）
        self.calibration_data = None    # {"point1": (x,y), "point2": (x,y), "pixel_dist": float, "real_dist": float, "unit": str, "scale": float}
        self.calibration_mode = False   # 是否处于校准模式
        self.calib_start_img = None     # 校准直线起点（原图坐标）
        self.calib_end_img = None       # 校准直线终点（原图坐标）
        self.calib_start_canvas = None  # 校准直线起点（画布坐标）
        self.calib_is_dragging = False  # 是否正在拖拽画校准直线
        self.calib_line_id = None       # 正在拖拽的校准线画布 ID
        self.calib_shift_end_canvas = None  # Shift 约束后的终点画布坐标
        self.calib_canvas_ids = []      # 校准线的画布 ID（完成后保留的）
        self.display_unit = tk.StringVar(value="mm")  # 面积显示单位

        # 缩放相关变量
        self.zoom_level = 1.0           # 当前缩放倍数（1.0 = fit 窗口）
        self.zoom_center_img = None     # 缩放中心（原图坐标）
        self.is_panning = False         # 是否正在平移拖拽
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.image_locked = True        # 默认锁定（不允许缩放平移）
        self.zoom_rect_start = None     # 局部放大矩形起点（画布坐标）
        self.zoom_rect_id = None        # 局部放大矩形画布 ID
        self.is_zoom_rect = False       # 是否正在画局部放大矩形

        # 模型路径变量（在 __init__ 中提前创建，避免会话恢复时 AttributeError）
        self._model_path_var = tk.StringVar()

        # 结果计数变量（在 __init__ 中提前创建，避免 _build_ui 时 AttributeError）
        self.result_count_var = tk.StringVar(value="累计分割: 0 个区域")

        # 构建 UI
        self._build_ui()

        # ── 恢复上次会话 ──
        self._restore_session(session_data)

        # ── 启动时自动显示 Help（仅当用户未关闭时） ──
        if session_data.get("show_help_on_start", True):
            self.root.after(300, lambda: self._show_help(on_start=True))

    def _restore_session(self, session_data: dict):
        """恢复上次会话的状态 — 记住路径，自动加载模型"""
        logger = logging.getLogger("SAM3App")

        # 1. 自动加载模型（如有上次路径）
        model_path = session_data.get("model_path")
        if model_path and os.path.isfile(model_path):
            logger.info(f"自动加载上次模型: {model_path}")
            self._model_path_var.set(model_path)
            self._auto_load_model(model_path)
        else:
            self._set_status("就绪 — 请先加载模型和图片")

        # 2. 记住图片路径（不自动加载，避免每次打开都是上次的图）
        image_path = session_data.get("image_path")
        if image_path and os.path.isfile(image_path):
            logger.info(f"记住上次图片路径: {image_path}")
            if not model_path:
                self._set_status(f"上次图片: {os.path.basename(image_path)} — 点击「🖼️ 打开图片」重新加载")

    # ================================================================
    #  UI 构建
    # ================================================================

    def _build_ui(self):
        """构建完整 UI 布局"""
        # ---- 顶部工具栏 ----
        self._build_toolbar()

        # ---- grid 配置 ----
        self.root.rowconfigure(0, weight=0)   # toolbar — 固定高度
        self.root.rowconfigure(1, weight=1)   # _vpaned — 占满剩余空间
        self.root.rowconfigure(3, weight=0)   # status_frame — 固定高度
        self.root.columnconfigure(0, weight=1)

        # ---- 用垂直 PanedWindow 管理 main_frame 和 log_panel 的比例 ----
        self._vpaned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self._vpaned.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # ---- 主体区域 ----
        main_frame = ttk.PanedWindow(self._vpaned, orient=tk.HORIZONTAL)
        self.main_frame = main_frame
        self._vpaned.add(main_frame, weight=3)

        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=4)
        self._build_canvas(left_frame)

        # 右侧：控制面板
        right_frame = ttk.Frame(main_frame, width=320)
        main_frame.add(right_frame, weight=1)
        self._build_control_panel(right_frame)

        # ---- 底部状态栏 ----
        self._build_statusbar()

    def _build_toolbar(self):
        """顶部工具栏（单行布局）"""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))

        btn_open_img = ttk.Button(toolbar, text="🖼️ 打开图片", command=self._open_image)
        btn_open_img.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(btn_open_img, "打开图片文件\n支持 PNG/JPG/BMP/TIFF")

        btn_open_proj = ttk.Button(toolbar, text="🧊 打开项目", command=self._open_project)
        btn_open_proj.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(btn_open_proj, "打开 .sam3proj 项目文件\n恢复所有 mask、ROI 和校准数据")

        self.model_btn = ttk.Button(toolbar, text="🧠 加载模型", command=self._load_model)
        self.model_btn.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self.model_btn, "加载 SAM3 模型文件 (.pt)\n支持 sam2.1_l / sam3 模型")

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(toolbar, text="ROI生成模式:").pack(side=tk.LEFT, padx=(4, 2))
        for text, mode in [("🖱️ 点击ROI", "point"), ("✏️ 框选ROI", "box")]:
            rb = ttk.Radiobutton(
                toolbar, text=text, variable=self.prompt_mode, value=mode,
                command=self._on_prompt_mode_change
            )
            rb.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        btn_clear = ttk.Button(toolbar, text="🧹 清除所有ROIs", command=self._clear_prompts_with_confirm)
        btn_clear.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(btn_clear, "清除所有提示点和 ROI")

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        btn_save = ttk.Button(toolbar, text="💾 保存项目", command=self._save_project)
        btn_save.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(btn_save, "保存项目为 .sam3proj 文件\n包含原图、mask、ROI 和校准数据")

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        self.calib_btn = ttk.Button(toolbar, text="📏 Scale Bar校准", command=self._start_calibration)
        self.calib_btn.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self.calib_btn, "Scale Bar 校准：拖拽画直线对应 scale bar\n按住 Shift 约束为水平/垂直/45°")

        self.manual_map_btn = ttk.Button(toolbar, text="🔢 手动 Mapping", command=self._manual_mapping)
        self.manual_map_btn.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self.manual_map_btn, "手动 Mapping（单位/px）\n如 0.32 μm/px 或 0.001 mm/px")

        self.calib_info_var = tk.StringVar(value="⚠️ 未校准")
        self.calib_info_label = ttk.Label(toolbar, textvariable=self.calib_info_var, foreground="blue")
        self.calib_info_label.pack(side=tk.LEFT, padx=2)
        self._add_dynamic_tooltip(self.calib_info_label, self._get_calib_tooltip_text)



    def _build_canvas(self, parent):
        """图片显示画布（含顶部信息栏）"""
        # ── 顶部信息栏：图片名 + Fit + Lock/Unlock ──
        self.canvas_top_bar = tk.Frame(parent, bg="#3c3c3c", height=28)
        self.canvas_top_bar.pack(fill=tk.X, side=tk.TOP)
        self.canvas_top_bar.pack_propagate(False)  # 固定高度

        # 图片名称（左侧）
        self.canvas_img_name_var = tk.StringVar(value="")
        self.img_name_label = tk.Label(
            self.canvas_top_bar, textvariable=self.canvas_img_name_var,
            bg="#3c3c3c", fg="#e0e0e0", font=("", 10), anchor=tk.W
        )
        self.img_name_label.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # ── 右侧按钮组（从右到左 pack）──

        # 👤作者 — 最右边，浅蓝文字
        self.author_btn = tk.Button(
            self.canvas_top_bar, text="👤作者", font=("", 10),
            command=self._show_author, bg="#3c3c3c", fg="#90CAF9",
            activebackground="#3c3c3c", activeforeground="#B3D9FF",
            relief=tk.FLAT, padx=6, pady=2, cursor="hand2", bd=0,
            highlightthickness=0, highlightbackground="#3c3c3c"
        )
        self.author_btn.pack(side=tk.RIGHT, padx=(0, 4))
        self._add_tooltip(self.author_btn, "作者信息")

        # 分割条 — 作者前面
        sep_author = tk.Label(
            self.canvas_top_bar, text="│", bg="#3c3c3c", fg="#666666",
            font=("", 10), padx=2
        )
        sep_author.pack(side=tk.RIGHT)

        # ❓Help — 金色文字
        self.help_btn = tk.Button(
            self.canvas_top_bar, text="❓Help", font=("", 10),
            command=self._show_help, bg="#3c3c3c", fg="#FFD700",
            activebackground="#3c3c3c", activeforeground="#FFE44D",
            relief=tk.FLAT, padx=6, pady=2, cursor="hand2", bd=0,
            highlightthickness=0, highlightbackground="#3c3c3c"
        )
        self.help_btn.pack(side=tk.RIGHT, padx=(2, 2))
        self._add_tooltip(self.help_btn, "使用帮助 — 数据分析步骤")

        # 分割条 — Help 和 Lock 之间
        sep_help_lock = tk.Label(
            self.canvas_top_bar, text="│", bg="#3c3c3c", fg="#666666",
            font=("", 10), padx=2
        )
        sep_help_lock.pack(side=tk.RIGHT)

        # 🔒 Lock/Unlock — 绿色/橙色文字
        self.lock_btn = tk.Button(
            self.canvas_top_bar, text="🔒 Lock", font=("", 10),
            command=self._toggle_lock, bg="#3c3c3c", fg="#4CAF50",
            activebackground="#3c3c3c", activeforeground="#66BB6A",
            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", bd=0,
            highlightthickness=0, highlightbackground="#3c3c3c"
        )
        self.lock_btn.pack(side=tk.RIGHT, padx=(0, 2))
        self._add_tooltip(self.lock_btn, "切换 Lock/Unlock\n🔒 Lock: ROI 操作\n🔓 Unlock: 缩放/平移/局部放大")

        # 分割条 — Lock 前面
        sep_lock = tk.Label(
            self.canvas_top_bar, text="│", bg="#3c3c3c", fg="#666666",
            font=("", 10), padx=2
        )
        sep_lock.pack(side=tk.RIGHT)

        # 🔍Fit — emoji 风格匹配
        self.zoom_fit_btn = tk.Button(
            self.canvas_top_bar, text="🔲 Fit Window", font=("", 10),
            command=self._zoom_fit, bg="#3c3c3c", fg="#000000",
            activebackground="#3c3c3c", activeforeground="#ffffff",
            relief=tk.FLAT, padx=6, pady=2, cursor="hand2", bd=0,
            highlightthickness=0, highlightbackground="#3c3c3c"
        )
        self.zoom_fit_btn.pack(side=tk.RIGHT, padx=(0, 2))
        self._add_tooltip(self.zoom_fit_btn, "重置为 Fit 窗口显示\n快捷键: 双击画布")

        # ── 画布区域 ──
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 滚动条
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#2b2b2b",
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
        )

        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── 拖拽文件支持 ──
        if HAS_DND:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self._on_file_drop)

        # 图片偏移量（居中用）
        self.image_offset_x = 0
        self.image_offset_y = 0

        # 窗口 resize 时重新 fit 图片
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Button-2>", self._on_canvas_right_click)  # macOS 右键=Button-2, 负提示
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)  # Linux/Windows 右键=Button-3
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)

        # 缩放和平移事件（平移改为 Ctrl+左键拖拽，兼容 macOS 无中键的情况）
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)        # macOS/Windows 滚轮缩放
        self.canvas.bind("<Button-4>", self._on_mouse_wheel_linux_up)  # Linux 滚轮上
        self.canvas.bind("<Button-5>", self._on_mouse_wheel_linux_down) # Linux 滚轮下
        self.canvas.bind("<Control-Button-1>", self._on_pan_start)     # Ctrl+左键开始平移
        self.canvas.bind("<Control-B1-Motion>", self._on_pan_motion)   # Ctrl+左键拖拽平移
        self.canvas.bind("<Control-ButtonRelease-1>", self._on_pan_end) # Ctrl+左键结束平移

        # Shift 键监听
        self.root.bind("<KeyPress-Shift_L>", self._on_shift_press)
        self.root.bind("<KeyRelease-Shift_L>", self._on_shift_release)
        self.root.bind("<KeyPress-Shift_R>", self._on_shift_press)
        self.root.bind("<KeyRelease-Shift_R>", self._on_shift_release)

        # 全局键盘快捷键
        self.root.bind("<Escape>", self._on_key_press)
        self.root.bind("<Key-plus>", self._on_key_press)
        self.root.bind("<Key-equal>", self._on_key_press)
        self.root.bind("<Key-minus>", self._on_key_press)


    def _on_key_press(self, event):
        """全局键盘快捷键处理"""
        key = event.keysym
        if key == "Escape":
            # ESC: 取消当前操作
            if self.calibration_mode:
                self.calibration_mode = False
                self.calib_is_dragging = False
                self.calib_start_img = None
                self.calib_end_img = None
                if self.calib_line_id:
                    self.canvas.delete(self.calib_line_id)
                    self.calib_line_id = None
                self.calib_btn.config(text="📏 Scale Bar校准")
                self._set_status("📏 校准已取消")
                self._update_cursor()
            elif self.is_drawing_box:
                self.is_drawing_box = False
                if self.box_rect_id:
                    self.canvas.delete(self.box_rect_id)
                    self.box_rect_id = None
                self._set_status("框选已取消")
            elif self.polygon_points:
                self._clear_polygon()
                self._set_status("多边形绘制已取消")
        elif key in ("plus", "equal"):
            # +/=: 放大（Unlock 模式）
            if not self.image_locked and self.image_np is not None:
                cw = self.canvas.winfo_width()
                ch = self.canvas.winfo_height()
                self._apply_zoom(cw // 2, ch // 2, 1.2)
        elif key == "minus":
            # -: 缩小（Unlock 模式）
            if not self.image_locked and self.image_np is not None:
                cw = self.canvas.winfo_width()
                ch = self.canvas.winfo_height()
                self._apply_zoom(cw // 2, ch // 2, 1 / 1.2)

    def _build_control_panel(self, parent):
        """右侧控制面板"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        # 绑定 Tab 切换事件，同步更新ROI生成模式
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        # ---- Tab 1: 点击ROI模式 ----
        point_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(point_frame, text="🖱️ 点击ROI模式")

        ttk.Label(point_frame, text="操作说明:").pack(anchor=tk.W)
        ttk.Label(point_frame, text="• 左键点击 = 正提示（前景）", foreground="green").pack(anchor=tk.W)
        ttk.Label(point_frame, text="• 右键点击 = 负提示（背景）", foreground="red").pack(anchor=tk.W)
        ttk.Label(point_frame, text="• 可添加多个点后一起分割", foreground="gray").pack(anchor=tk.W)

        ttk.Separator(point_frame).pack(fill=tk.X, pady=8)

        ttk.Label(point_frame, text="已添加的提示点:").pack(anchor=tk.W)
        # Treeview 显示点列表
        point_cols = ("id", "name", "type", "visible")
        self.point_tree = ttk.Treeview(point_frame, columns=point_cols, show="headings", height=6, selectmode="extended")
        self.point_tree.heading("id", text="ID")
        self.point_tree.heading("name", text="名字")
        self.point_tree.heading("type", text="类型")
        self.point_tree.heading("visible", text="可见")
        self.point_tree.column("id", width=40, anchor=tk.CENTER)
        self.point_tree.column("name", width=100, anchor=tk.CENTER)
        self.point_tree.column("type", width=70, anchor=tk.CENTER)
        self.point_tree.column("visible", width=40, anchor=tk.CENTER)
        self.point_tree.pack(fill=tk.X, pady=4)
        self.point_tree.bind("<Double-1>", self._on_point_tree_double_click)
        self.point_tree.bind("<ButtonRelease-1>", self._on_point_tree_click)

        btn_frame = ttk.Frame(point_frame)
        btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="👁️ 切换可见性", command=self._toggle_selected_point_visibility).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="❌ 删除选中", command=self._delete_selected_point).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🧹 清空所有", command=self._clear_point_prompts).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🙈 隐藏所有", command=self._hide_point_prompts).pack(
            side=tk.LEFT, padx=2)

        ttk.Button(point_frame, text="🔍 执行分割", command=self._segment_points).pack(
            fill=tk.X, pady=(12, 4))

        # ---- Tab 2: 框选ROI模式 (ROI 编辑模式) ----
        box_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(box_frame, text="✏️ 框选ROI模式")

        # 形状选择
        ttk.Label(box_frame, text="形状工具:", font=("", 10, "bold")).pack(anchor=tk.W)
        shape_frame = ttk.Frame(box_frame)
        shape_frame.pack(fill=tk.X, pady=4)
        for text, mode in [("矩形", "rectangle"), ("椭圆", "ellipse"), ("多边形", "polygon")]:
            ttk.Radiobutton(
                shape_frame, text=text, variable=self.box_shape_mode, value=mode
            ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(box_frame).pack(fill=tk.X, pady=8)

        # 操作说明（动态）
        ttk.Label(box_frame, text="操作说明:", font=("", 9, "bold")).pack(anchor=tk.W)
        self.box_help_label = ttk.Label(box_frame, text="", foreground="blue", wraplength=250, justify=tk.LEFT)
        self.box_help_label.pack(anchor=tk.W, pady=2)
        self.box_shape_mode.trace_add("write", lambda *_: self._update_box_help())
        self._update_box_help()

        ttk.Separator(box_frame).pack(fill=tk.X, pady=8)

        # ROI 列表（Treeview，复刻结果Tab管理）
        self.roi_count_var = tk.StringVar(value="累计选区: 0 个")
        ttk.Label(box_frame, textvariable=self.roi_count_var, font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(box_frame, text="选区列表（点击可见列切换 / 双击名字编辑）:").pack(anchor=tk.W)
        self.roi_tree = ttk.Treeview(
            box_frame,
            columns=("label_id", "name", "area", "vis"),
            show="headings",
            height=8,
        )
        self.roi_tree.heading("label_id", text="ID")
        self.roi_tree.heading("name", text="名字")
        self.roi_tree.heading("area", text="面积")
        self.roi_tree.heading("vis", text="可见")
        self.roi_tree.column("label_id", width=100, anchor=tk.CENTER)
        self.roi_tree.column("name", width=120, anchor=tk.CENTER)
        self.roi_tree.column("area", width=130, anchor=tk.CENTER)
        self.roi_tree.column("vis", width=35, anchor=tk.CENTER)
        self.roi_tree.pack(fill=tk.BOTH, expand=True, pady=4)

        self.roi_tree.bind("<<TreeviewSelect>>", self._on_roi_select)
        self.roi_tree.bind("<Double-1>", self._on_roi_double_click)
        self.roi_tree.bind("<ButtonRelease-1>", self._on_roi_click)

        # ROI 操作按钮
        roi_btn_frame = ttk.Frame(box_frame)
        roi_btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(roi_btn_frame, text="👁️ 切换选中可见性", command=self._toggle_selected_roi_visibility).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(roi_btn_frame, text="❌ 删除选中", command=self._delete_selected_roi).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(roi_btn_frame, text="🧹 清空所有", command=lambda: self._clear_rois_by_type("box")).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(roi_btn_frame, text="🙈 隐藏所有", command=lambda: self._hide_rois_by_type("box")).pack(
            side=tk.LEFT, padx=2)

        # ── ROI 布尔运算区 ──
        ttk.Separator(box_frame).pack(fill=tk.X, pady=8)
        ttk.Label(box_frame, text="ROI 运算:", font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(box_frame, text="对两个 ROI 执行 Union/Intersection/Minus", foreground="gray").pack(anchor=tk.W)

        # ROI A 选择
        roi_op_frame_a = ttk.Frame(box_frame)
        roi_op_frame_a.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(roi_op_frame_a, text="ROI A:").pack(side=tk.LEFT)
        self.roi_a_var = tk.StringVar(value="0")
        self.roi_a_combo = ttk.Combobox(roi_op_frame_a, textvariable=self.roi_a_var, width=20, state="readonly", height=10)
        self.roi_a_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.roi_a_combo.bind("<<ComboboxSelected>>", lambda e: self._update_roi_op_label())

        # 运算类型选择
        roi_op_frame_op = ttk.Frame(box_frame)
        roi_op_frame_op.pack(fill=tk.X, pady=2)
        ttk.Label(roi_op_frame_op, text="运算:").pack(side=tk.LEFT)
        self.roi_op_var = tk.StringVar(value="Minus")
        for op_text, op_val in [("Union 并集", "Union"), ("Intersection 交集", "Intersection"), ("Minus 差集", "Minus")]:
            ttk.Radiobutton(roi_op_frame_op, text=op_text, variable=self.roi_op_var, value=op_val, command=self._update_roi_op_label).pack(side=tk.LEFT, padx=2)

        # ROI B 选择
        roi_op_frame_b = ttk.Frame(box_frame)
        roi_op_frame_b.pack(fill=tk.X, pady=2)
        ttk.Label(roi_op_frame_b, text="ROI B:").pack(side=tk.LEFT)
        self.roi_b_var = tk.StringVar(value="1")
        self.roi_b_combo = ttk.Combobox(roi_op_frame_b, textvariable=self.roi_b_var, width=20, state="readonly", height=10)
        self.roi_b_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.roi_b_combo.bind("<<ComboboxSelected>>", lambda e: self._update_roi_op_label())

        # 运算结果标签名
        roi_op_frame_name = ttk.Frame(box_frame)
        roi_op_frame_name.pack(fill=tk.X, pady=2)
        ttk.Label(roi_op_frame_name, text="结果名:").pack(side=tk.LEFT)
        self.roi_op_label_var = tk.StringVar(value="选择 ROI 和运算后自动生成")
        self.roi_op_label_entry = ttk.Entry(roi_op_frame_name, textvariable=self.roi_op_label_var, width=20)
        self.roi_op_label_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # 运算按钮
        roi_op_btn_frame = ttk.Frame(box_frame)
        roi_op_btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(roi_op_btn_frame, text="⚡ 执行运算", command=self._execute_roi_operation).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(roi_op_btn_frame, text="👁️ 仅预览", command=self._preview_roi_operation).pack(
            side=tk.LEFT, padx=2)

        ttk.Separator(box_frame).pack(fill=tk.X, pady=8)

        ttk.Button(box_frame, text="🔍 执行分割（选中 ROI）", command=self._segment_box).pack(
            fill=tk.X, pady=(12, 4))

        # ---- Tab 4: Mask结果 ----
        result_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(result_frame, text="📊 Mask结果")

        # 结果统计 + 单位选择（同行）
        result_header = ttk.Frame(result_frame)
        result_header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(result_header, textvariable=self.result_count_var,
                  font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(result_header, text="单位:").pack(side=tk.RIGHT, padx=(4, 0))
        unit_combo = ttk.Combobox(result_header, textvariable=self.display_unit,
                                   values=["μm", "mm", "cm", "in"], width=5, state="readonly")
        unit_combo.pack(side=tk.RIGHT, padx=2)
        self.display_unit.trace_add("write", lambda *_: self._on_unit_change())

        # ── Mask 布尔运算区（放在顶部，下拉列表有充足空间） ──
        ttk.Separator(result_frame).pack(fill=tk.X, pady=8)
        ttk.Label(result_frame, text="Mask 运算:", font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(result_frame, text="对两个 mask 执行 Union/Intersection/Minus", foreground="gray").pack(anchor=tk.W)

        # Mask A 选择
        mask_op_frame_a = ttk.Frame(result_frame)
        mask_op_frame_a.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(mask_op_frame_a, text="Mask A:").pack(side=tk.LEFT)
        self.mask_a_var = tk.StringVar(value="0")
        self.mask_a_combo = ttk.Combobox(mask_op_frame_a, textvariable=self.mask_a_var, width=20, state="readonly", height=10)
        self.mask_a_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.mask_a_combo.bind("<<ComboboxSelected>>", lambda e: self._update_mask_op_label())

        # 运算类型选择
        mask_op_frame_op = ttk.Frame(result_frame)
        mask_op_frame_op.pack(fill=tk.X, pady=2)
        ttk.Label(mask_op_frame_op, text="运算:").pack(side=tk.LEFT)
        self.mask_op_var = tk.StringVar(value="Minus")
        for op_text, op_val in [("Union 并集", "Union"), ("Intersection 交集", "Intersection"), ("Minus 差集", "Minus")]:
            ttk.Radiobutton(mask_op_frame_op, text=op_text, variable=self.mask_op_var, value=op_val, command=self._update_mask_op_label).pack(side=tk.LEFT, padx=2)

        # Mask B 选择
        mask_op_frame_b = ttk.Frame(result_frame)
        mask_op_frame_b.pack(fill=tk.X, pady=2)
        ttk.Label(mask_op_frame_b, text="Mask B:").pack(side=tk.LEFT)
        self.mask_b_var = tk.StringVar(value="1")
        self.mask_b_combo = ttk.Combobox(mask_op_frame_b, textvariable=self.mask_b_var, width=20, state="readonly", height=10)
        self.mask_b_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.mask_b_combo.bind("<<ComboboxSelected>>", lambda e: self._update_mask_op_label())

        # 运算结果标签名
        mask_op_frame_name = ttk.Frame(result_frame)
        mask_op_frame_name.pack(fill=tk.X, pady=2)
        ttk.Label(mask_op_frame_name, text="结果名:").pack(side=tk.LEFT)
        self.mask_op_label_var = tk.StringVar(value="选择 Mask 和运算后自动生成")
        self.mask_op_label_entry = ttk.Entry(mask_op_frame_name, textvariable=self.mask_op_label_var, width=20)
        self.mask_op_label_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # 运算按钮
        mask_op_btn_frame = ttk.Frame(result_frame)
        mask_op_btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(mask_op_btn_frame, text="⚡ 执行运算", command=self._execute_mask_operation).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(mask_op_btn_frame, text="👁️ 仅预览", command=self._preview_mask_operation).pack(
            side=tk.LEFT, padx=2)

        ttk.Separator(result_frame).pack(fill=tk.X, pady=8)

        ttk.Label(result_frame, text="分割结果列表（点击查看/隐藏）:").pack(anchor=tk.W)

        # Treeview + Scrollbar
        tree_container = ttk.Frame(result_frame)
        tree_container.pack(fill=tk.BOTH, expand=True, pady=4)

        result_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_tree = ttk.Treeview(
            tree_container,
            columns=("label_id", "name", "score", "area", "vis"),
            show="headings",
            height=10,
            yscrollcommand=result_scrollbar.set,
        )
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.config(command=self.result_tree.yview)

        self.result_tree.heading("label_id", text="ID")
        self.result_tree.heading("name", text="名字")
        self.result_tree.heading("score", text="置信度")
        self.result_tree.heading("area", text="面积")
        self.result_tree.heading("vis", text="可见")
        self.result_tree.column("label_id", width=100, anchor=tk.CENTER)
        self.result_tree.column("name", width=90, anchor=tk.CENTER)
        self.result_tree.column("score", width=55, anchor=tk.CENTER)
        self.result_tree.column("area", width=130, anchor=tk.CENTER)
        self.result_tree.column("vis", width=35, anchor=tk.CENTER)

        self.result_tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self.result_tree.bind("<Double-1>", self._on_result_double_click)
        self.result_tree.bind("<ButtonRelease-1>", self._on_result_click)

        # 结果操作按钮
        result_btn_frame = ttk.Frame(result_frame)
        result_btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(result_btn_frame, text="👁️ 切换选中可见性", command=self._toggle_selected_visibility).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(result_btn_frame, text="❌ 删除选中", command=self._delete_selected_result).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(result_btn_frame, text="💾 保存选中Mask", command=self._save_selected_result).pack(
            side=tk.LEFT, padx=2)

        # 合并 Mask 按钮
        ttk.Separator(result_frame).pack(fill=tk.X, pady=8)
        ttk.Button(result_frame, text="🔗 合并所有 Mask 并预览", command=self._combine_and_preview).pack(
            fill=tk.X, pady=4)

        # 透明度调节
        alpha_frame = ttk.Frame(result_frame)
        alpha_frame.pack(fill=tk.X, pady=4)
        ttk.Label(alpha_frame, text="叠加透明度:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.5)
        ttk.Scale(alpha_frame, from_=0.1, to=0.9, variable=self.alpha_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._update_overlay()).pack(
            side=tk.LEFT, fill=tk.X, expand=True)



    def _build_statusbar(self):
        """底部状态栏 + 可展开的 log 面板"""
        # ── 可展开的 log 面板（作为 _vpaned 的第二个 pane）──
        self._log_panel = ttk.Frame(self._vpaned, relief=tk.GROOVE, borderwidth=2)
        # 默认不添加到 _vpaned（收起状态）

        # ✕ 关闭按钮已移除 — 用状态栏 ▶/▼ 按钮切换展开/收起

        # log 文本区域和滚动条
        self._log_text = tk.Text(self._log_panel, height=6, wrap=tk.WORD,
                                  font=("Menlo", 13), state=tk.DISABLED,
                                  bg="#2d2d2d", fg="#e0e0e0",
                                  insertbackground="#e0e0e0",
                                  selectbackground="#264f78",
                                  relief=tk.SUNKEN, bd=1)
        log_scrollbar = ttk.Scrollbar(self._log_panel, orient=tk.VERTICAL,
                                       command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scrollbar.set)

        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # log 文本颜色标签
        self._log_text.tag_configure("info", foreground="#e0e0e0")
        self._log_text.tag_configure("success", foreground="#6a9955")
        self._log_text.tag_configure("error", foreground="#f44747")
        self._log_text.tag_configure("progress", foreground="#569cd6")
        self._log_text.tag_configure("time", foreground="#858585")

        # ── 状态栏 ──
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 0))

        self.status_var = tk.StringVar(value="就绪 — 请先加载模型和图片")
        statusbar = ttk.Label(self.status_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        statusbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # log 面板切换按钮
        self._log_expanded = False
        self._log_toggle_btn = ttk.Button(self.status_frame, text="▶",
                                           width=3, command=self._toggle_log_panel)
        self._log_toggle_btn.pack(side=tk.RIGHT, padx=(4, 0))

    def _toggle_log_panel(self):
        """切换 log 面板的展开/收起"""
        if self._log_expanded:
            # 收起 — 从 _vpaned 移除 log_panel
            self._vpaned.remove(self._log_panel)
            self._log_toggle_btn.config(text="▶")
            self._log_expanded = False
        else:
            # 展开 — 添加 log_panel 为 _vpaned 的第二个 pane，weight=1（main_frame weight=3）
            self._vpaned.add(self._log_panel, weight=1)
            self._log_toggle_btn.config(text="▼")
            self._log_expanded = True
            # 滚动到底部
            self._log_text.see(tk.END)

    # ================================================================
    #  会话保存与关闭
    # ================================================================

    def _on_closing(self):
        """窗口关闭时保存会话"""
        logger = logging.getLogger("SAM3App")
        try:
            # 保存窗口大小
            geometry = self.root.geometry()

            # 保存基本会话信息
            self.session.save(
                image_path=self.image_path,
                model_path=self.engine.model_path if self.engine.is_loaded() else None,
                window_geometry=geometry,
            )

            # 保存分割结果
            if self.image_path and self.segmentation_results:
                self.session.save_masks_session(
                    self.image_path,
                    self.segmentation_results,
                    self.mask_visible,
                )

            logger.info("会话已保存")
        except Exception as e:
            logger.warning(f"保存会话时出错: {e}")

        self.root.destroy()

    # ================================================================
    #  图片加载与显示
    # ================================================================

    def _open_image(self):
        """打开图片文件"""
        filetypes = [
            ("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
            ("所有文件", "*.*"),
        ]
        path = filedialog.askopenfilename(title="选择图片", filetypes=filetypes)
        if not path:
            return
        self._load_image_from_path(path)

    def _load_image_from_path(self, path: str):
        """从路径加载图片（支持恢复会话时调用）"""
        self.image_path = path
        self.image_pil = Image.open(path).convert("RGB")
        self.image_np = np.array(self.image_pil)

        # 清除之前的分割结果（恢复会话时会单独恢复）
        self.segmentation_results = []
        self.mask_visible = []
        self.overlay_image = None
        self._update_result_tree()
        self._clear_prompts()
        # 清除校准数据
        self._clear_calibration()
        # 重置缩放
        self.zoom_level = 1.0
        self.zoom_center_img = None

        # 显示图片
        self._display_image()
        self._set_status(f"已加载图片: {os.path.basename(path)} ({self.image_np.shape[1]}x{self.image_np.shape[0]})")

        # 保存会话
        self.session.save(image_path=path)

    def _display_image(self, overlay_np=None):
        """在画布上显示图片（居中 + fit 窗口 + 保持比例，支持缩放）"""
        if overlay_np is not None:
            display_img = Image.fromarray(overlay_np)
        elif self.image_pil is not None:
            display_img = self.image_pil.copy()
        else:
            return

        # 获取画布实际尺寸
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            # 窗口还没渲染完，用默认尺寸
            canvas_w = 800
            canvas_h = 600

        # 计算基础缩放比例（fit 窗口）
        img_w, img_h = display_img.size
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        base_scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小

        # 应用用户缩放倍数
        self.display_scale = base_scale * self.zoom_level

        new_w = int(img_w * self.display_scale)
        new_h = int(img_h * self.display_scale)

        # 居中偏移（考虑缩放中心）
        if self.zoom_center_img is not None and self.zoom_level > 1.0:
            # 缩放时以用户指定的中心点为基准
            center_cx = self.zoom_center_img[0] * self.display_scale
            center_cy = self.zoom_center_img[1] * self.display_scale
            self.image_offset_x = canvas_w / 2 - center_cx
            self.image_offset_y = canvas_h / 2 - center_cy
        else:
            # fit 窗口时居中
            self.image_offset_x = (canvas_w - new_w) / 2
            self.image_offset_y = (canvas_h - new_h) / 2

        display_img = display_img.resize((new_w, new_h), Image.LANCZOS)

        self._tk_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, canvas_w, canvas_h))
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, anchor=tk.NW, image=self._tk_image)

        # 在左上角显示图片信息
        self._draw_image_info()

        # 缩放倍数已合并到 _draw_image_info 中

        # 重绘提示点
        self._redraw_points()

        # 重绘校准线
        if self.calibration_data and self.calib_start_img and self.calib_end_img:
            self._redraw_calibration_line()

        # 重绘所有 ROI（_display_image 会 delete("all")，需要重新绘制）
        if hasattr(self, 'roi_list') and self.roi_list:
            self._redraw_all_rois()

    def _draw_image_info(self):
        """更新顶部栏的图片文件名、大小和缩放倍数信息"""
        if not self.image_path:
            self.canvas_img_name_var.set("")
            return

        import os
        filename = os.path.basename(self.image_path)
        h, w = self.image_np.shape[:2]

        # 获取文件大小
        try:
            file_size_bytes = os.path.getsize(self.image_path)
            if file_size_bytes >= 1024 * 1024:
                file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
            else:
                file_size = f"{file_size_bytes / 1024:.1f} KB"
        except OSError:
            file_size = ""

        info_text = f"{filename}  {w}×{h}  {file_size}"

        # 缩放倍数（与工具栏指示器同步，使用 display_scale）
        actual_scale = self.display_scale if hasattr(self, 'display_scale') and self.display_scale else 1.0
        if actual_scale >= 1.0:
            zoom_text = f"🔍 {actual_scale:.1f}x"
        elif actual_scale > 0:
            zoom_text = f"🔍 {actual_scale * 100:.0f}%"
        else:
            zoom_text = ""

        if zoom_text:
            info_text = info_text + "  " + zoom_text

        self.canvas_img_name_var.set(info_text)

    def _toggle_lock(self):
        """切换 Lock/Unlock 状态"""
        self.image_locked = not self.image_locked
        if self.image_locked:
            self.lock_btn.config(text="🔒 Lock", fg="#4CAF50")  # 绿色 = 已锁定
            # 不自动 fit，保持当前缩放级别（用户可能需要在放大模式下校准）
            self._set_status("🔒 已锁定 — ROI 操作已恢复，缩放/平移已禁用")
        else:
            self.lock_btn.config(text="🔓 Unlock", fg="#FF9800")  # 橙色 = 已解锁
            self._set_status("🔓 已解锁 — 滚轮缩放 + Ctrl+拖拽平移 + 拖拽局部放大已启用，ROI 操作已临时禁止")
        self._update_cursor()

    # ================================================================
    #  缩放与平移
    # ================================================================

    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放（macOS/Windows）"""
        if self.image_locked or self.image_np is None:
            return
        # macOS: event.delta 正/负; Windows: event.delta 120/-120
        if event.delta > 0:
            factor = 1.2
        else:
            factor = 1 / 1.2
        self._apply_zoom(event.x, event.y, factor)

    def _on_mouse_wheel_linux_up(self, event):
        """Linux 滚轮上滚"""
        if self.image_locked or self.image_np is None:
            return
        self._apply_zoom(event.x, event.y, 1.2)

    def _on_mouse_wheel_linux_down(self, event):
        """Linux 滚轮下滚"""
        if self.image_locked or self.image_np is None:
            return
        self._apply_zoom(event.x, event.y, 1 / 1.2)

    def _apply_zoom(self, cx, cy, factor):
        """以画布坐标 (cx, cy) 为中心缩放"""
        # 记录缩放中心的原图坐标
        img_coords = self._canvas_to_image_coords(cx, cy)
        self.zoom_center_img = img_coords

        new_zoom = self.zoom_level * factor
        # 限制缩放范围：0.5x ~ 20x
        new_zoom = max(0.5, min(20.0, new_zoom))
        self.zoom_level = new_zoom

        # 重绘图片
        self._display_image(self.overlay_image if self.overlay_image is not None else None)
        self._draw_image_info()

    def _on_pan_start(self, event):
        """中键按下开始平移"""
        if self.image_locked:
            return
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def _on_pan_motion(self, event):
        """中键拖拽平移"""
        if self.image_locked or not self.is_panning or self.image_np is None:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        # 更新偏移量
        self.image_offset_x += dx
        self.image_offset_y += dy

        # 更新缩放中心（原图坐标）
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        center_cx = canvas_w / 2 - self.image_offset_x
        center_cy = canvas_h / 2 - self.image_offset_y
        if self.display_scale > 0:
            self.zoom_center_img = [center_cx / self.display_scale, center_cy / self.display_scale]

        self.pan_start_x = event.x
        self.pan_start_y = event.y

        # 重绘
        self._display_image(self.overlay_image if self.overlay_image is not None else None)

    def _on_pan_end(self, event):
        """中键释放结束平移"""
        self.is_panning = False

    def _update_cursor(self):
        """根据当前状态更新光标样式"""
        if self.calibration_mode:
            self.canvas.config(cursor="crosshair")
        elif not self.image_locked:
            self.canvas.config(cursor="sizing")  # 放大镜/缩放光标
        else:
            # Lock 模式：根据 ROI 生成模式设置光标
            mode = self.prompt_mode.get()
            if mode == "point":
                self.canvas.config(cursor="crosshair")
            elif mode == "box":
                self.canvas.config(cursor="cross")

    def _zoom_fit(self):
        """重置显示为 fit 窗口模式"""
        if self.image_pil is None:
            return
        self.zoom_level = 1.0
        self.zoom_center_img = None
        self.image_offset_x = 0
        self.image_offset_y = 0
        self._display_image(self.overlay_image if self.overlay_image is not None else None)
        self._draw_image_info()
        self._update_cursor()

    def _on_file_drop(self, event):
        """拖拽文件到窗口打开"""
        self._process_dropped_file(event.data)

    def _process_dropped_file(self, raw):
        """解析拖拽文件路径并打开"""
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ''
        if not isinstance(raw, str):
            raw = str(raw)
        # macOS: tkinterdnd2 返回 file:///Users/... URI 格式
        if raw.startswith('file://'):
            import urllib.parse
            raw = urllib.parse.unquote(raw[7:])
        # 花括号包裹的路径（含空格的文件名）
        if raw.startswith('{') and raw.endswith('}'):
            raw = raw[1:-1]
        # 可能返回多个文件路径
        if ' ' in raw and not os.path.exists(raw):
            parts = raw.split()
            for p in parts:
                if os.path.exists(p):
                    raw = p
                    break
        raw = raw.strip()

        if not os.path.exists(raw):
            self._set_status(f"文件不存在: {raw}")
            return

        if raw.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')):
            self._load_image_from_path(raw)
        elif raw.endswith('.sam3proj'):
            self._open_project_from_path(raw)
        else:
            self._set_status(f"不支持的文件格式: {raw}")

    def _show_author(self):
        """显示作者信息弹窗"""
        dialog = tk.Toplevel(self.root)
        dialog.title("作者信息")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # 居中显示
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width() // 2 - 180,
            self.root.winfo_rooty() + self.root.winfo_height() // 2 - 100
        ))

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="SAM3 Image Segmenter", font=("Helvetica", 14, "bold")).pack(pady=(0, 8))
        ttk.Label(frame, text=f"Version {self.APP_VERSION}", font=("Helvetica", 10)).pack()
        ttk.Separator(frame).pack(fill=tk.X, pady=10)
        ttk.Label(frame, text="黄华添 (Huatian Huang)", font=("Helvetica", 12, "bold")).pack()
        ttk.Label(frame, text="Sunny FACA Team", font=("Helvetica", 12, "bold")).pack()
        ttk.Label(frame, text="舞宇集团 (Sunny Group)", font=("Helvetica", 12, "bold")).pack()
        ttk.Label(frame, text=f"v{self.APP_VERSION} (Based on SAM)", font=("Helvetica", 9), foreground="gray").pack(pady=(6, 0))

        ttk.Button(frame, text="关闭", command=dialog.destroy).pack(pady=(12, 0))

    def _show_help(self, on_start=False):
        """显示使用帮助弹窗 — 数据分析步骤
        
        Args:
            on_start: 是否为启动时自动弹出（显示 Don't show next time 按钮）
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("使用帮助")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # 居中显示
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width() // 2 - 250,
            self.root.winfo_rooty() + self.root.winfo_height() // 2 - 180
        ))

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="📊 数据分析步骤", font=("Helvetica", 14, "bold")).pack(pady=(0, 10))

        steps = [
            ("1️⃣  导入图片", "点击 🖼️ 打开图片 或拖拽文件到窗口\n支持 PNG / JPG / TIFF 等格式"),
            ("2️⃣  导入模型", "点击 🧠 加载模型 → 选择 .pt 模型文件\n首次使用需从 HuggingFace 下载 SAM 模型"),
            ("3️⃣  Scale Bar 校准", "① 点击 🔓 Unlock → 切换到缩放/平移模式\n② 在图片上画矩形框放大定位 scale bar 区域\n③ 点击 📏 Scale Bar校准 → 自动切换为 Lock 模式，在图片上拖拽画直线对应 scale bar\n   按住 Shift 约束为水平/垂直/45° 方向\n④ 输入实际距离和单位（μm / mm / cm / in）"),
            ("4️⃣  分割（Lock 模式）", "点击 🔒 Lock → 选择 ROI 生成模式\n• 点击 ROI: 左键=前景，右键=背景\n• 框选 ROI: 矩形/椭圆/多边形\n点击 🔍 执行分割"),
            ("5️⃣  合并 Mask", "在 📊 Mask结果 Tab 中选中多个 Mask\n点击 🔀 合并选中 → 合并为 combined mask\n或使用 ROI/Mask 布尔运算（Union / Intersection / Minus）"),
            ("6️⃣  导出结果", "点击 💾 保存选中Mask → 输出 overlay + binary\n校准后自动添加面积水印（左上角黑底白字）\n点击 💾 保存项目 → 保存为 .sam3proj 可下次打开"),
        ]

        for title, desc in steps:
            step_frame = ttk.Frame(frame)
            step_frame.pack(fill=tk.X, pady=4)
            ttk.Label(step_frame, text=title, font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(step_frame, text=desc, font=("Helvetica", 9), foreground="#555555",
                      wraplength=420, justify=tk.LEFT).pack(anchor=tk.W, padx=(16, 0))

        ttk.Separator(frame).pack(fill=tk.X, pady=8)

        shortcuts_frame = ttk.Frame(frame)
        shortcuts_frame.pack(fill=tk.X)
        ttk.Label(shortcuts_frame, text="⌨️ 快捷键", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        shortcut_text = "ESC = 取消操作  |  +/- = 缩放（Unlock）  |  双击画布 = Fit Window"
        ttk.Label(shortcuts_frame, text=shortcut_text, font=("Helvetica", 9), foreground="gray").pack(anchor=tk.W, padx=(16, 0))

        # 按钮区域
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=(10, 0))

        if on_start:
            # 启动时自动弹出：显示 Don't show next time 按钮
            def close_and_dont_show():
                self.session.data["show_help_on_start"] = False
                self.session.save(show_help_on_start=False)
                dialog.destroy()

            ttk.Button(btn_frame, text="Don't show next time", command=close_and_dont_show).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(side=tk.LEFT, padx=4)
        else:
            # 手动点击 Help 时：只显示关闭按钮
            ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(padx=4)

    def _add_tooltip(self, widget, text):
        """为按钮/标签添加 hover 提示（Tooltip）"""
        tooltip_window = None

        def show_tooltip(event):
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            nonlocal tooltip_window
            tooltip_window = tk.Toplevel(widget)
            tooltip_window.wm_overrideredirect(True)
            tooltip_window.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip_window, text=text, justify=tk.LEFT,
                background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                font=("", 10), padx=6, pady=4
            )
            label.pack()

        def hide_tooltip(event):
            nonlocal tooltip_window
            if tooltip_window:
                tooltip_window.destroy()
                tooltip_window = None

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def _add_dynamic_tooltip(self, widget, text_func):
        """为按钮/标签添加动态 hover 提示（每次显示时调用 text_func 获取最新文本）"""
        tooltip_window = None

        def show_tooltip(event):
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            nonlocal tooltip_window
            tooltip_window = tk.Toplevel(widget)
            tooltip_window.wm_overrideredirect(True)
            tooltip_window.wm_geometry(f"+{x}+{y}")
            current_text = text_func()
            label = tk.Label(
                tooltip_window, text=current_text, justify=tk.LEFT,
                background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                font=("", 10), padx=6, pady=4
            )
            label.pack()

        def hide_tooltip(event):
            nonlocal tooltip_window
            if tooltip_window:
                tooltip_window.destroy()
                tooltip_window = None

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)


    def _redraw_calibration_line(self):
        """重绘校准线（在 _display_image 清除画布后需要重绘）"""
        if not self.calibration_data or not self.calib_start_img or not self.calib_end_img:
            return
        self.canvas.delete("calibration")
        self.calib_canvas_ids = []
        p1_cx, p1_cy = self._image_to_canvas_coords(*self.calib_start_img)
        p2_cx, p2_cy = self._image_to_canvas_coords(*self.calib_end_img)
        r = 4
        p1_id = self.canvas.create_oval(
            p1_cx - r, p1_cy - r, p1_cx + r, p1_cy + r,
            fill="#00FF00", outline="white", width=1, tags="calibration"
        )
        p2_id = self.canvas.create_oval(
            p2_cx - r, p2_cy - r, p2_cx + r, p2_cy + r,
            fill="#FF0000", outline="white", width=1, tags="calibration"
        )
        line_id = self.canvas.create_line(
            p1_cx, p1_cy, p2_cx, p2_cy,
            fill="#FFFF00", width=2, dash=(8, 4), tags="calibration"
        )
        # 像素距离标注（中点上方）
        mid_x, mid_y = (p1_cx + p2_cx) / 2, (p1_cy + p2_cy) / 2
        pixel_dist = self.calibration_data["pixel_dist"]
        dist_label_id = self.canvas.create_text(
            mid_x, mid_y - 12,
            text=f"{pixel_dist:.1f} px",
            fill="#FFFF00", font=("", 10, "bold"), tags="calibration"
        )
        self.calib_canvas_ids = [p1_id, p2_id, line_id, dist_label_id]

    def _redraw_points(self):
        """重绘所有可见的提示点到画布上"""
        self.canvas.delete("prompt_point")
        for pt_data in self.click_point_list:
            if not pt_data["visible"]:
                continue
            pt = pt_data["coords"]
            label = pt_data["label"]
            x, y = self._image_to_canvas_coords(pt[0], pt[1])
            color = "#00FF00" if label == 1 else "#FF0000"
            r = 6
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="white", width=2,
                tags="prompt_point",
            )

    # ================================================================
    #  模型加载
    # ================================================================

    def _auto_load_model(self, model_path: str):
        """启动时自动加载上次模型（后台线程，无弹窗）"""
        model_name = os.path.basename(model_path)
        self.model_btn.config(text=f"⏳ 加载中...", state=tk.DISABLED)
        self._set_status(f"正在自动加载模型 {model_name}...")
        self.root.config(cursor="watch")

        def auto_load_thread():
            try:
                self.engine.load_model(model_path)
                self.session.save(model_path=model_path)
                self.root.after(0, lambda: self.model_btn.config(
                    text=f"✅ {model_name}", state=tk.NORMAL))
                self.root.after(0, lambda: self._set_status(
                    f"✅ 模型自动加载完成: {model_name} (设备: {self.engine.device})"))
                self.root.after(0, lambda: self._log(f"✅ {model_name} auto-loaded"))
            except Exception as e:
                logger = logging.getLogger("SAM3App")
                logger.warning(f"自动加载模型失败: {e}")
                self.root.after(0, lambda: self.model_btn.config(
                    text="🧠 加载模型", state=tk.NORMAL))
                self.root.after(0, lambda: self._set_status(
                    f"自动加载失败 — 请手动点击「🧠 加载模型」"))
                self.root.after(0, lambda: self._log(f"⚠️ 自动加载失败: {e}", "error"))
            finally:
                self.root.after(0, lambda: self.root.config(cursor=""))

        threading.Thread(target=auto_load_thread, daemon=True).start()

    def _load_model(self):
        """加载 SAM3 模型"""
        if self.engine.is_loaded():
            if not messagebox.askyesno("提示", "模型已加载，是否重新加载？"):
                return

        # 弹出选择对话框：选择已有模型文件 或 查看下载指引
        choice_dialog = tk.Toplevel(self.root)
        choice_dialog.title("加载 SAM3 模型")
        choice_dialog.geometry("540x400")
        choice_dialog.resizable(False, False)
        choice_dialog.transient(self.root)
        choice_dialog.grab_set()

        ttk.Label(choice_dialog, text="加载 SAM3 模型", font=("", 14, "bold")).pack(pady=(16, 8))

        # 选项1：选择本地模型文件
        local_frame = ttk.LabelFrame(choice_dialog, text="方式一：选择本地模型文件", padding=10)
        local_frame.pack(fill=tk.X, padx=16, pady=4)

        ttk.Label(local_frame, text="选择已下载的 .pt 模型文件（如 sam3.pt）:").pack(anchor=tk.W)
        path_frame = ttk.Frame(local_frame)
        path_frame.pack(fill=tk.X, pady=4)
        ttk.Entry(path_frame, textvariable=self._model_path_var, width=42).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(path_frame, text="浏览...", command=lambda: self._browse_model_file()).pack(side=tk.RIGHT)

        # 选项2：下载指引
        guide_frame = ttk.LabelFrame(choice_dialog, text="方式二：下载模型（首次使用必看）", padding=10)
        guide_frame.pack(fill=tk.X, padx=16, pady=4)

        guide_text = (
            "SAM3 模型权重不会自动下载，需手动获取：\n"
            "1. 访问 https://huggingface.co/facebook/sam3\n"
            "2. 登录 HuggingFace 并申请访问权限（免费，通常几小时内批准）\n"
            "3. 下载 sam3.pt（约 3.4 GB）\n"
            "4. 放到项目目录或任意位置，用「方式一」加载"
        )
        ttk.Label(guide_frame, text=guide_text, justify=tk.LEFT, foreground="#555").pack(anchor=tk.W)

        # 按钮
        btn_frame = ttk.Frame(choice_dialog)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="✅ 加载模型", command=lambda: self._do_load_model(choice_dialog)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="❌ 取消", command=choice_dialog.destroy).pack(side=tk.LEFT, padx=4)

    def _browse_model_file(self):
        """浏览选择模型文件"""
        path = filedialog.askopenfilename(
            title="选择 SAM3 模型文件",
            filetypes=[("模型文件", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            self._model_path_var.set(path)

    def _do_load_model(self, dialog):
        """执行模型加载"""
        path = self._model_path_var.get().strip()
        if not path:
            messagebox.showwarning("提示", "请先选择模型文件路径", parent=dialog)
            return
        if not os.path.isfile(path):
            messagebox.showerror("错误", f"文件不存在: {path}", parent=dialog)
            return

        dialog.destroy()

        # 创建进度条窗口
        progress_win = tk.Toplevel(self.root)
        progress_win.title("加载模型")
        progress_win.geometry("420x120")
        progress_win.resizable(False, False)
        progress_win.transient(self.root)
        progress_win.grab_set()
        progress_win.protocol("WM_DELETE_WINDOW", lambda: None)  # 禁止关闭

        ttk.Label(progress_win, text="正在加载 SAM3 模型，请稍候...",
                  font=("", 11)).pack(pady=(16, 8))
        self._progress_bar = ttk.Progressbar(progress_win, mode='indeterminate', length=380)
        self._progress_bar.pack(padx=20)
        self._progress_bar.start(15)  # 开始动画
        self._progress_label = ttk.Label(progress_win, text="初始化中...", foreground="#666")
        self._progress_label.pack(pady=(4, 0))

        self._set_status("正在加载模型，请稍候...")
        self.root.config(cursor="watch")
        self.model_btn.config(text="⏳ 加载中...", state=tk.DISABLED)

        def update_progress(step_or_msg, total=None, message=None):
            """从后台线程更新进度信息（兼容两种回调签名）"""
            if total is not None and message is not None:
                # sam3_engine 回调格式: (step, total, message)
                display_msg = f"[{step_or_msg}/{total}] {message}"
            else:
                # 简单字符串格式
                display_msg = str(step_or_msg)
            self.root.after(0, lambda: self._progress_label.config(text=display_msg))
            self.root.after(0, lambda: self._log(display_msg))

        def load_thread():
            try:
                update_progress("导入 PyTorch...")
                self.engine.load_model(path, progress_callback=update_progress)
                # 保存模型路径到会话
                self.session.save(model_path=path)
                model_name = os.path.basename(path)
                self.root.after(0, lambda: self.model_btn.config(
                    text=f"✅ {model_name}", state=tk.NORMAL))
                self.root.after(0, lambda: self._set_status(
                    f"✅ 模型加载完成: {model_name} (设备: {self.engine.device})"))
                self.root.after(0, lambda: self._log(f"✅ {model_name} loaded"))
                # 关闭进度窗口
                self.root.after(500, lambda: progress_win.destroy())
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"模型加载异常:\n{tb}", "error")
                self.root.after(0, lambda: progress_win.destroy())
                self.root.after(0, lambda: self.model_btn.config(
                    text="🧠 加载模型", state=tk.NORMAL))
                err_text = f"{e}\n\n堆栈:\n{tb[:500]}"
                self.root.after(0, lambda: messagebox.showerror("加载失败", err_text))
                self.root.after(0, lambda: self._set_status("❌ 模型加载失败"))
                self.root.after(0, lambda: self._log(f"❌ 加载失败: {e}", "error"))
            finally:
                self.root.after(0, lambda: self.root.config(cursor=""))

        threading.Thread(target=load_thread, daemon=True).start()

    # ================================================================
    #  交互事件处理
    # ================================================================

    def _on_prompt_mode_change(self):
        """ROI生成模式切换时清除交互状态，并自动切换到对应 Tab"""
        self.is_drawing_box = False
        self.box_start = None
        self.box_end = None
        if self.box_rect_id:
            self.canvas.delete(self.box_rect_id)
            self.box_rect_id = None
        self._clear_polygon()

        # 自动切换到对应的 Tab
        mode = self.prompt_mode.get()
        if mode == "point":
            self.notebook.select(0)  # Tab 0: 🖱️ 点击ROI模式
        elif mode == "box":
            self.notebook.select(1)  # Tab 1: ✏️ 框选ROI模式

    def _on_notebook_tab_changed(self, event=None):
        """当用户点击 Notebook Tab 时，同步更新工具栏的ROI生成模式"""
        current_tab = self.notebook.index(self.notebook.select())
        target_mode = "point" if current_tab == 0 else "box"
        if self.prompt_mode.get() != target_mode:
            # 同步ROI生成模式（设置 prompt_mode 会触发 radiobutton 更新，
            # 但不会触发 _on_prompt_mode_change，需手动调用）
            self.prompt_mode.set(target_mode)
            # 清除交互状态（不再调用 _on_prompt_mode_change 避免递归 select）
            self.is_drawing_box = False
            self.box_start = None
            self.box_end = None
            if self.box_rect_id:
                self.canvas.delete(self.box_rect_id)
                self.box_rect_id = None
            self._clear_polygon()

    def _on_shift_press(self, event):
        """Shift 键按下"""
        self.shift_pressed = True

    def _on_shift_release(self, event):
        """Shift 键释放"""
        self.shift_pressed = False

    def _update_box_help(self):
        """根据形状模式更新操作说明"""
        shape = self.box_shape_mode.get()
        if shape == "rectangle":
            self.box_help_label.config(text="• 按住左键拖拽画矩形框\n• 松开鼠标完成框选")
        elif shape == "ellipse":
            self.box_help_label.config(text="• 按住左键拖拽画椭圆\n• 按住 Shift 画正圆\n• 松开鼠标完成框选")
        elif shape == "polygon":
            self.box_help_label.config(text="• 左键点击添加多边形顶点\n• 右键🖱️闭合多边形\n• 至少需要 3 个顶点")

    def _clear_polygon(self):
        """清除多边形绘制状态"""
        for lid in self.polygon_line_ids:
            self.canvas.delete(lid)
        for pid in self.polygon_point_ids:
            self.canvas.delete(pid)
        self.polygon_points = []
        self.polygon_img_points = []
        self.polygon_line_ids = []
        self.polygon_point_ids = []

    def _on_canvas_resize(self, event=None):
        """画布大小变化时重新 fit 图片 + 重绘 ROI"""
        if self.image_pil is not None:
            self._display_image(self.overlay_image if self.overlay_image is not None else None)
            self._redraw_all_rois()

    def _canvas_to_image_coords(self, cx, cy):
        """画布坐标转原图坐标（考虑偏移）"""
        ix = (cx - self.image_offset_x) / self.display_scale
        iy = (cy - self.image_offset_y) / self.display_scale
        return [ix, iy]

    def _image_to_canvas_coords(self, ix, iy):
        """原图坐标转画布坐标（考虑偏移）"""
        cx = ix * self.display_scale + self.image_offset_x
        cy = iy * self.display_scale + self.image_offset_y
        return cx, cy

    def _on_canvas_click(self, event):
        """左键点击（Ctrl+左键用于平移，不触发点击/框选）"""
        if self.image_np is None:
            return

        # Ctrl+左键 = 平移，不触发点击逻辑
        if event.state & 0x4:  # Ctrl 键修饰符
            return

        # ── Unlock 模式：禁止点击ROI和框选ROI，允许画局部放大矩形 ──
        if not self.image_locked:
            # 开始画局部放大矩形
            self.zoom_rect_start = (event.x, event.y)
            self.is_zoom_rect = True
            return

        # ── 校准模式：按下鼠标开始画校准直线 ──
        if self.calibration_mode:
            self.calib_start_img = self._canvas_to_image_coords(event.x, event.y)
            self.calib_start_canvas = (event.x, event.y)
            self.calib_is_dragging = True
            self._set_status("📏 校准: 拖拽画直线（按住 Shift 约束为水平/垂直/45°），释放鼠标完成")
            return

        mode = self.prompt_mode.get()

        if mode == "point":
            img_coords = self._canvas_to_image_coords(event.x, event.y)
            self._add_click_point(img_coords, label=1)  # 正提示

        elif mode == "box":
            shape = self.box_shape_mode.get()
            if shape == "polygon":
                # 多边形模式：点击添加顶点
                self.polygon_points.append((event.x, event.y))
                img_pt = self._canvas_to_image_coords(event.x, event.y)
                self.polygon_img_points.append(img_pt)
                # 画顶点
                r = 4
                pid = self.canvas.create_oval(
                    event.x - r, event.y - r, event.x + r, event.y + r,
                    fill="#FF6600", outline="white", width=1
                )
                self.polygon_point_ids.append(pid)
                # 画线段（连接到上一个点）
                if len(self.polygon_points) >= 2:
                    x0, y0 = self.polygon_points[-2]
                    x1, y1 = self.polygon_points[-1]
                    lid = self.canvas.create_line(
                        x0, y0, x1, y1, fill="#FF6600", width=2, dash=(6, 3)
                    )
                    self.polygon_line_ids.append(lid)
                # 多边形顶点添加中（状态栏提示）
                self._set_status(f"多边形绘制中: {len(self.polygon_points)} 个顶点，右键🖱️闭合")
            else:
                # 矩形/椭圆模式：开始拖拽
                self.box_start = (event.x, event.y)
                self.is_drawing_box = True

    def _on_canvas_right_click(self, event):
        """右键点击：polygon 模式闭合多边形 / point 模式添加负提示"""
        if self.image_np is None:
            return

        # ── Unlock 模式：禁止添加负提示和闭合多边形 ──
        if not self.image_locked:
            return

        mode = self.prompt_mode.get()

        if mode == "box" and self.box_shape_mode.get() == "polygon":
            # 右键闭合多边形
            self._close_polygon()
            return

        if mode == "point":
            img_coords = self._canvas_to_image_coords(event.x, event.y)
            self._add_click_point(img_coords, label=0)  # 负提示

    def _on_canvas_drag(self, event):
        """鼠标拖拽（画矩形/椭圆 或 校准直线）"""
        # ── Unlock 模式：禁止框选ROI，允许画局部放大矩形 ──
        if not self.image_locked and not self.calib_is_dragging:
            if self.is_zoom_rect:
                if self.zoom_rect_id:
                    self.canvas.delete(self.zoom_rect_id)
                sx, sy = self.zoom_rect_start
                self.zoom_rect_id = self.canvas.create_rectangle(
                    sx, sy, event.x, event.y,
                    outline="#00BFFF", width=2, dash=(6, 3)
                )
            return

        # ── 校准模式：实时画校准直线 ──
        if self.calib_is_dragging:
            if self.calib_line_id:
                self.canvas.delete(self.calib_line_id)
            sx, sy = self.calib_start_canvas
            ex, ey = event.x, event.y
            # Shift 约束：水平/垂直/45°
            if self.shift_pressed:
                dx = ex - sx
                dy = ey - sy
                angle = math.atan2(dy, dx)
                # 将角度吸附到 0°/90°/180°/270°/45°/135°/225°/315°
                snap_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4,
                               math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
                # 找最近的吸附角度
                min_diff = min(abs(angle - sa) for sa in snap_angles)
                best_angle = min(snap_angles, key=lambda sa: abs(angle - sa))
                # 用原始距离作为约束后的直线长度
                dist = math.sqrt(dx * dx + dy * dy)
                ex = sx + dist * math.cos(best_angle)
                ey = sy + dist * math.sin(best_angle)
            self.calib_line_id = self.canvas.create_line(
                sx, sy, ex, ey,
                fill="#FFFF00", width=2, dash=(8, 4)
            )
            # Shift 约束时更新终点为约束后的画布坐标
            self.calib_shift_end_canvas = (ex, ey) if self.shift_pressed else None
            return

        if not self.is_drawing_box or self.prompt_mode.get() != "box":
            return

        shape = self.box_shape_mode.get()
        if shape == "polygon":
            return  # 多边形不用拖拽

        if self.box_rect_id:
            self.canvas.delete(self.box_rect_id)

        x0, y0 = self.box_start
        x1, y1 = event.x, event.y

        if shape == "rectangle":
            self.box_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="#00FF00", width=2, dash=(6, 3),
            )
        elif shape == "ellipse":
            # Shift = 正圆
            if self.shift_pressed:
                dx = x1 - x0
                dy = y1 - y0
                size = max(abs(dx), abs(dy))
                x1 = x0 + size * (1 if dx >= 0 else -1)
                y1 = y0 + size * (1 if dy >= 0 else -1)
            self.box_rect_id = self.canvas.create_oval(
                x0, y0, x1, y1,
                outline="#00FF00", width=2, dash=(6, 3),
            )

    def _on_canvas_release(self, event):
        """鼠标释放（完成矩形/椭圆框选 → 自动添加 ROI，或完成校准直线）"""
        # ── 校准模式：完成校准直线 ──
        if self.calib_is_dragging:
            self.calib_is_dragging = False
            # 清除拖拽时的临时线
            if self.calib_line_id:
                self.canvas.delete(self.calib_line_id)
                self.calib_line_id = None

            # 如果 Shift 约束了终点，使用约束后的画布坐标计算原图坐标
            if self.calib_shift_end_canvas:
                end_canvas_x, end_canvas_y = self.calib_shift_end_canvas
                self.calib_end_img = self._canvas_to_image_coords(end_canvas_x, end_canvas_y)
                self.calib_shift_end_canvas = None
            else:
                # 记录终点
                self.calib_end_img = self._canvas_to_image_coords(event.x, event.y)

            # 计算像素距离
            import math
            dx = self.calib_end_img[0] - self.calib_start_img[0]
            dy = self.calib_end_img[1] - self.calib_start_img[1]
            pixel_dist = math.sqrt(dx * dx + dy * dy)

            if pixel_dist < 5:
                # 太短，取消校准
                self.calib_start_img = None
                self.calib_end_img = None
                self._set_status("📏 校准线太短（<5px），请重新拖拽画更长的直线")
                return

            # 在画布上绘制正式的校准参考线（保留）
            sx, sy = self._image_to_canvas_coords(*self.calib_start_img)
            ex, ey = self._image_to_canvas_coords(*self.calib_end_img)

            # 两端端点标记
            r = 4
            p1_id = self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill="#00FF00", outline="white", width=1, tags="calibration"
            )
            p2_id = self.canvas.create_oval(
                ex - r, ey - r, ex + r, ey + r,
                fill="#FF0000", outline="white", width=1, tags="calibration"
            )
            # 校准直线
            line_id = self.canvas.create_line(
                sx, sy, ex, ey,
                fill="#FFFF00", width=2, dash=(8, 4), tags="calibration"
            )
            # 像素距离标注（中点上方）
            mid_x, mid_y = (sx + ex) / 2, (sy + ey) / 2
            dist_label_id = self.canvas.create_text(
                mid_x, mid_y - 12,
                text=f"{pixel_dist:.1f} px",
                fill="#FFFF00", font=("", 10, "bold"), tags="calibration"
            )
            self.calib_canvas_ids = [p1_id, p2_id, line_id, dist_label_id]

            self._set_status(f"📏 校准: 像素距离 = {pixel_dist:.1f} px，请输入实际距离")
            # 退出校准模式，弹出输入对话框
            self.calibration_mode = False
            self.calib_btn.config(text="📏 Scale Bar校准")
            self._finish_calibration(pixel_dist)
            return

        # ── Unlock 模式 或 校准模式：完成局部放大矩形 ──
        if not self.image_locked or self.calibration_mode:
            self.is_zoom_rect = False
            if self.zoom_rect_id:
                self.canvas.delete(self.zoom_rect_id)
                self.zoom_rect_id = None
            if self.zoom_rect_start is None or self.image_np is None:
                return
            sx, sy = self.zoom_rect_start
            ex, ey = event.x, event.y
            self.zoom_rect_start = None
            # 计算矩形区域的原图坐标
            img_x1, img_y1 = self._canvas_to_image_coords(min(sx, ex), min(sy, ey))
            img_x2, img_y2 = self._canvas_to_image_coords(max(sx, ex), max(sy, ey))
            rect_w = abs(img_x2 - img_x1)
            rect_h = abs(img_y2 - img_y1)
            if rect_w < 5 or rect_h < 5:
                # 太小，忽略
                return
            # 计算缩放倍数：让该区域 fit 铺满窗口
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            # 计算基础 scale（fit 窗口时的 scale）
            img_full_w, img_full_h = self.image_pil.size
            base_scale_w = canvas_w / img_full_w
            base_scale_h = canvas_h / img_full_h
            base_scale = min(base_scale_w, base_scale_h, 1.0)
            # 目标：让 rect_w * base_scale * zoom_level = canvas_w 或 rect_h * base_scale * zoom_level = canvas_h
            zoom_for_w = canvas_w / (rect_w * base_scale)
            zoom_for_h = canvas_h / (rect_h * base_scale)
            # 取较小的缩放倍数，保证整个矩形区域可见
            self.zoom_level = min(zoom_for_w, zoom_for_h)
            # 缩放中心 = 矩形区域的中心（原图坐标）
            center_x = (img_x1 + img_x2) / 2
            center_y = (img_y1 + img_y2) / 2
            self.zoom_center_img = (center_x, center_y)
            # 重绘
            self._display_image(self.overlay_image if self.overlay_image is not None else None)
            self._draw_image_info()
            self._set_status(f"🔍 局部放大 {self.display_scale:.1f}x — 画矩形框选择区域")
            return

        if not self.is_drawing_box or self.prompt_mode.get() != "box":
            return

        shape = self.box_shape_mode.get()
        if shape == "polygon":
            return  # 多边形不用释放

        self.is_drawing_box = False
        self.box_end = (event.x, event.y)

        x0, y0 = self.box_start
        x1, y1 = event.x, event.y

        # 椭圆模式 Shift=正圆
        if shape == "ellipse" and self.shift_pressed:
            dx = x1 - x0
            dy = y1 - y0
            size = max(abs(dx), abs(dy))
            x1 = x0 + size * (1 if dx >= 0 else -1)
            y1 = y0 + size * (1 if dy >= 0 else -1)

        # 转换为原图坐标
        ix1, iy1 = self._canvas_to_image_coords(x0, y0)
        ix2, iy2 = self._canvas_to_image_coords(x1, y1)

        # 确保坐标顺序正确
        bbox = [min(ix1, ix2), min(iy1, iy2), max(ix1, ix2), max(iy1, iy2)]

        # 将临时绘制转为正式 ROI（重绘为实线+半透明填充）
        if self.box_rect_id:
            self.canvas.delete(self.box_rect_id)
            self.box_rect_id = None

        # 自动添加 ROI
        self._add_roi(shape, bbox)

    def _on_canvas_double_click(self, event):
        """双击事件 — polygon 模式闭合多边形，其他模式 Fit 恢复窗口"""
        if self.image_np is None:
            return
        mode = self.prompt_mode.get()
        if mode == "box" and self.box_shape_mode.get() == "polygon":
            # 双击闭合多边形（与右键闭合等效）
            # 先移除双击产生的重复顶点（双击会触发2次click + 1次double-click）
            if self.polygon_points:
                # 移除最后一个顶点（双击的第二次click产生的）
                self.polygon_points.pop()
                if self.polygon_img_points:
                    self.polygon_img_points.pop()
                if self.polygon_point_ids:
                    pid = self.polygon_point_ids.pop()
                    self.canvas.delete(pid)
                # 双击的第一次和第二次click位置很近，检查并移除重复
                if self.polygon_points and len(self.polygon_points) >= 2:
                    x0, y0 = self.polygon_points[-1]
                    x1, y1 = self.polygon_points[-2]
                    if abs(x0 - x1) < 5 and abs(y0 - y1) < 5:
                        self.polygon_points.pop()
                        if self.polygon_img_points:
                            self.polygon_img_points.pop()
                        if self.polygon_point_ids:
                            pid = self.polygon_point_ids.pop()
                            self.canvas.delete(pid)
                        if self.polygon_line_ids:
                            lid = self.polygon_line_ids.pop()
                            self.canvas.delete(lid)
            # 执行闭合
            self._close_polygon()
        else:
            # 其他模式：双击 Fit 恢复窗口
            self._zoom_fit()

    def _close_polygon(self):
        """闭合多边形并添加为 ROI（右键和双击共用）"""
        if len(self.polygon_points) < 3:
            messagebox.showinfo("提示", "多边形至少需要 3 个顶点")
            return

        # 闭合多边形：画最后一条线
        x0, y0 = self.polygon_points[-1]
        x1, y1 = self.polygon_points[0]
        lid = self.canvas.create_line(
            x0, y0, x1, y1, fill="#FF6600", width=2, dash=(6, 3)
        )
        self.polygon_line_ids.append(lid)

        # 计算多边形的外接矩形作为 bbox
        xs = [p[0] for p in self.polygon_img_points]
        ys = [p[1] for p in self.polygon_img_points]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        # 收集临时画布 ID 转为 ROI
        temp_ids = self.polygon_line_ids + self.polygon_point_ids

        # 清除临时绘制，_add_roi 会重新画
        for cid in temp_ids:
            self.canvas.delete(cid)

        # 自动添加 ROI
        self._add_roi("polygon", bbox, list(self.polygon_img_points), prompt_type="box")

        # 清除多边形临时状态，用户可以继续创建新多边形
        self.polygon_points = []
        self.polygon_img_points = []
        self.polygon_line_ids = []
        self.polygon_point_ids = []

        self._set_status("多边形已闭合，可继续点击创建新多边形")

    # ================================================================
    #  ROI 管理
    # ================================================================

    def _add_roi(self, shape, bbox, img_points=None, prompt_type="box"):
        """添加一个 ROI 到列表，并在画布上绘制
        
        Args:
            prompt_type: "point" 或 "box" 或 "op_xxx"（运算结果）
        """
        self.roi_counter += 1
        shape_cn = {"rectangle": "矩形", "ellipse": "椭圆", "polygon": "多边形"}.get(shape, shape)
        roi_name = f"ROI_{self.roi_counter} ({shape_cn})"
        roi_color = self._get_roi_color(self.roi_counter - 1)

        # 在画布上绘制 ROI（实线+半透明填充）
        canvas_ids = self._draw_roi_on_canvas(shape, bbox, roi_color, img_points)

        # 生成 ROI 的二值 mask（用于布尔运算）
        roi_mask = self._generate_roi_mask(shape, bbox, img_points)

        roi = {
            "name": roi_name,
            "shape": shape,
            "bbox": bbox,          # [x1, y1, x2, y2] 原图坐标
            "img_points": img_points,  # 多边形顶点（原图坐标）
            "canvas_ids": canvas_ids,
            "color": roi_color,
            "mask": roi_mask,      # 二值 mask (H, W) bool
            "prompt_type": prompt_type,  # "point" / "box" / "op_xxx"
        }
        self.roi_list.append(roi)

        # 更新 ROI 可见性列表
        self.roi_visible.append(True)

        # 更新 ROI 列表 UI
        self._update_roi_tree()
        # 自动选中新添加的 ROI
        new_idx = len(self.roi_list) - 1
        self.roi_tree.selection_set(str(new_idx))
        self.roi_tree.see(str(new_idx))

        # 更新 ROI 运算下拉框
        self._update_roi_combos()

        self._set_status(f"已添加 {roi_name} [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    def _get_roi_color(self, idx):
        """根据索引返回 ROI 颜色"""
        colors = ["#00FF00", "#FF6600", "#00BFFF", "#FF00FF", "#FFD700", "#00FFFF"]
        return colors[idx % len(colors)]

    def _draw_roi_on_canvas(self, shape, bbox, color, img_points=None, width=2):
        """在画布上绘制 ROI（虚线边框，无填充），返回画布 ID 列表"""
        canvas_ids = []
        x1, y1 = self._image_to_canvas_coords(bbox[0], bbox[1])
        x2, y2 = self._image_to_canvas_coords(bbox[2], bbox[3])
        dash = (8, 4)  # 虚线样式

        if shape == "rectangle":
            rid = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=width, dash=dash,
                tags="roi_shape",
            )
            canvas_ids.append(rid)

        elif shape == "ellipse":
            oid = self.canvas.create_oval(
                x1, y1, x2, y2,
                outline=color, width=width, dash=dash,
                tags="roi_shape",
            )
            canvas_ids.append(oid)

        elif shape == "polygon" and img_points:
            canvas_pts = []
            for pt in img_points:
                cx, cy = self._image_to_canvas_coords(pt[0], pt[1])
                canvas_pts.extend([cx, cy])
            if len(canvas_pts) >= 6:
                pid = self.canvas.create_polygon(
                    canvas_pts,
                    outline=color, width=width, fill="", dash=dash,
                    tags="roi_shape",
                )
                canvas_ids.append(pid)

        return canvas_ids

    def _redraw_all_rois(self):
        """重绘所有 ROI（窗口 resize 后调用，尊重可见性）"""
        # 清除旧的 ROI 画布元素
        self.canvas.delete("roi_shape")
        # 重新绘制（只绘制可见的）
        for i, roi in enumerate(self.roi_list):
            if i < len(self.roi_visible) and self.roi_visible[i]:
                if roi["shape"] == "mask" or (roi["shape"] == "polygon" and roi.get("img_points") is None and roi.get("mask") is not None):
                    # mask 类型 ROI
                    roi["canvas_ids"] = self._draw_roi_mask_on_canvas(roi["mask"], roi["color"])
                else:
                    roi["canvas_ids"] = self._draw_roi_on_canvas(
                        roi["shape"], roi["bbox"], roi["color"], roi.get("img_points")
                    )
            else:
                roi["canvas_ids"] = []
        # 重新高亮选中的 ROI
        if 0 <= self.selected_roi_idx < len(self.roi_list):
            self._highlight_roi(self.selected_roi_idx)

    def _on_roi_select(self, event):
        """ROI 列表选中事件 → 高亮选中的 ROI（支持多选）"""
        selected = self.roi_tree.selection()
        if not selected:
            self.selected_roi_idx = -1
            self._highlight_rois([])
            return

        self.selected_roi_idx = int(selected[-1])  # 最后选中的作为主选
        self._highlight_rois([int(s) for s in selected])

    def _highlight_roi(self, idx):
        """高亮单个 ROI（兼容旧调用）"""
        self._highlight_rois([idx])

    def _highlight_rois(self, indices):
        """高亮选中的 ROI（加粗虚线边框，支持多选）"""
        # 先重绘所有 ROI
        self.canvas.delete("roi_shape")
        for i, roi in enumerate(self.roi_list):
            # 不可见的 ROI 不绘制
            if i < len(self.roi_visible) and not self.roi_visible[i]:
                roi["canvas_ids"] = []
                continue
            color = roi["color"]
            is_selected = i in indices
            width = 4 if is_selected else 2
            dash = (12, 4) if is_selected else (8, 4)  # 选中的虚线更长更醒目
            bbox = roi["bbox"]
            x1, y1 = self._image_to_canvas_coords(bbox[0], bbox[1])
            x2, y2 = self._image_to_canvas_coords(bbox[2], bbox[3])
            canvas_ids = []

            if roi["shape"] == "rectangle":
                rid = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=color, width=width, dash=dash,
                    tags="roi_shape",
                )
                canvas_ids.append(rid)

            elif roi["shape"] == "ellipse":
                oid = self.canvas.create_oval(
                    x1, y1, x2, y2,
                    outline=color, width=width, dash=dash,
                    tags="roi_shape",
                )
                canvas_ids.append(oid)

            elif roi["shape"] == "polygon" and roi.get("img_points"):
                canvas_pts = []
                for pt in roi["img_points"]:
                    cx, cy = self._image_to_canvas_coords(pt[0], pt[1])
                    canvas_pts.extend([cx, cy])
                if len(canvas_pts) >= 6:
                    pid = self.canvas.create_polygon(
                        canvas_pts,
                        outline=color, width=width, fill="", dash=dash,
                        tags="roi_shape",
                    )
                    canvas_ids.append(pid)

            # mask 类型 ROI（运算结果）
            elif roi["shape"] == "mask" or (roi["shape"] == "polygon" and roi.get("img_points") is None and roi.get("mask") is not None):
                canvas_ids = self._draw_roi_mask_on_canvas(roi["mask"], color, width=width, dash=dash)

            roi["canvas_ids"] = canvas_ids

    def _delete_selected_roi(self):
        """删除选中的 ROI（支持多选）"""
        selected = self.roi_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在选区列表中选择要删除的项")
            return

        # 从大到小排序避免索引偏移
        indices = sorted([int(s) for s in selected], reverse=True)
        labels = [f"#{idx+1} {self.roi_list[idx]['name']}" for idx in indices]

        if not messagebox.askyesno("确认删除", f"删除以下 {len(indices)} 个选区？\n\n" + "\n".join(labels)):
            return

        for idx in indices:
            roi = self.roi_list[idx]
            for cid in roi["canvas_ids"]:
                self.canvas.delete(cid)
            self.roi_list.pop(idx)
            self.roi_visible.pop(idx)

        self.selected_roi_idx = -1
        self._update_roi_tree()
        self._update_roi_combos()

    def _clear_point_prompts(self):
        """清空点击ROI模式的全部提示点"""
        if not self.click_point_list:
            messagebox.showinfo("提示", "没有提示点需要清空")
            return
        if not messagebox.askyesno("确认清空", f"清空全部 {len(self.click_point_list)} 个提示点？"):
            return
        self.click_point_list = []
        self.click_point_counter = 0
        self._update_point_tree()
        self._redraw_points()
        self._set_status("已清空所有提示点")

    def _hide_point_prompts(self):
        """隐藏点击ROI模式的全部提示点"""
        if not self.click_point_list:
            messagebox.showinfo("提示", "没有提示点需要隐藏")
            return
        # 设置所有点为不可见
        for pt_data in self.click_point_list:
            pt_data["visible"] = False
        # 隐藏画布上的提示点
        self.canvas.delete("prompt_point")
        self._update_point_tree()
        self._set_status(f"已隐藏 {len(self.click_point_list)} 个提示点")

    def _clear_rois_by_type(self, prompt_type):
        """只清空指定类型的 ROI（point / box）
        
        point 类型：点击ROI模式创建的 ROI
        box 类型：框选ROI模式创建的 ROI（含 op_xxx 运算结果）
        """
        if prompt_type == "point":
            # 点击模式：只清空 prompt_type=="point" 的 ROI
            indices = [i for i, roi in enumerate(self.roi_list) if roi.get("prompt_type") == "point"]
            type_cn = "点击ROI"
        else:
            # 框选模式：清空 prompt_type=="box" 或 "op_xxx" 的 ROI
            indices = [i for i, roi in enumerate(self.roi_list)
                       if roi.get("prompt_type") == "box" or (roi.get("prompt_type") or "").startswith("op_")]
            type_cn = "框选ROI"

        if not indices:
            messagebox.showinfo("提示", f"没有{type_cn}类型的选区需要清空")
            return

        if not messagebox.askyesno("确认清空", f"清空 {len(indices)} 个{type_cn}选区？"):
            return

        # 从大到小删除，避免索引偏移
        for idx in sorted(indices, reverse=True):
            for cid in self.roi_list[idx]["canvas_ids"]:
                self.canvas.delete(cid)
            self.roi_list.pop(idx)
            self.roi_visible.pop(idx)

        self.selected_roi_idx = -1
        self._update_roi_tree()
        self._update_roi_combos()
        self._redraw_all_rois()

    def _hide_rois_by_type(self, prompt_type):
        """只隐藏指定类型的 ROI（point / box）"""
        if prompt_type == "point":
            indices = [i for i, roi in enumerate(self.roi_list) if roi.get("prompt_type") == "point"]
            type_cn = "点击ROI"
        else:
            indices = [i for i, roi in enumerate(self.roi_list)
                       if roi.get("prompt_type") == "box" or (roi.get("prompt_type") or "").startswith("op_")]
            type_cn = "框选ROI"

        if not indices:
            messagebox.showinfo("提示", f"没有{type_cn}类型的选区需要隐藏")
            return

        hidden_count = 0
        for idx in indices:
            if idx < len(self.roi_visible) and self.roi_visible[idx]:
                self.roi_visible[idx] = False
                # 隐藏画布元素
                for cid in self.roi_list[idx].get("canvas_ids", []):
                    try:
                        self.canvas.itemconfigure(cid, state="hidden")
                    except Exception:
                        pass
                hidden_count += 1

        if hidden_count == 0:
            messagebox.showinfo("提示", f"所有{type_cn}选区已经是隐藏状态")
            return

        self._update_roi_tree()
        self._set_status(f"已隐藏 {hidden_count} 个{type_cn}选区")

    def _clear_all_rois(self):
        """清空所有 ROI"""
        if not self.roi_list:
            return
        if not messagebox.askyesno("确认清空", f"清空全部 {len(self.roi_list)} 个选区？"):
            return
        for roi in self.roi_list:
            for cid in roi["canvas_ids"]:
                self.canvas.delete(cid)
        self.roi_list = []
        self.roi_visible = []
        self.roi_counter = 0
        self.selected_roi_idx = -1
        self._update_roi_tree()
        self._update_roi_combos()

    # ================================================================
    #  ROI 布尔运算
    # ================================================================

    def _update_roi_tree(self):
        """更新 ROI Treeview 列表"""
        self.roi_tree.delete(*self.roi_tree.get_children())
        for i, roi in enumerate(self.roi_list):
            area = int(roi["mask"].sum()) if roi.get("mask") is not None else 0
            area_str = self._format_calibrated_area(area)
            shape_cn = {"rectangle": "矩形", "ellipse": "椭圆", "polygon": "多边形"}.get(roi["shape"], roi["shape"])
            label_id = f"#{i+1}"
            vis = "✅" if (i < len(self.roi_visible) and self.roi_visible[i]) else "❌"
            self.roi_tree.insert("", tk.END, iid=str(i), values=(label_id, roi["name"], area_str, vis))

        self.roi_count_var.set(f"累计选区: {len(self.roi_list)} 个")
        # 更新 ROI 运算下拉框
        self._update_roi_combos()

    def _on_roi_click(self, event):
        """点击 ROI 列表 — 点击可见列切换可见性"""
        region = self.roi_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.roi_tree.identify_column(event.x)
        # column "#4" = vis column (1-based: #1=label_id, #2=name, #3=area, #4=vis)
        if column != "#4":
            return
        selected = self.roi_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        if idx < len(self.roi_visible):
            self.roi_visible[idx] = not self.roi_visible[idx]
            self._update_roi_tree()
            # 重绘 ROI（隐藏/显示）
            self._highlight_roi(self.selected_roi_idx)
            # 重新选中该行
            self.roi_tree.selection_set(str(idx))

    def _on_roi_double_click(self, event):
        """双击 ROI 列表 — 编辑名字"""
        region = self.roi_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.roi_tree.identify_column(event.x)
        # column "#2" = name column
        if column != "#2":
            return
        selected = self.roi_tree.selection()
        if not selected:
            return
        item_id = selected[0]
        idx = int(item_id)
        current_values = self.roi_tree.item(item_id, "values")
        current_name = current_values[1] if len(current_values) > 1 else self.roi_list[idx]["name"]
        new_name = simpledialog.askstring(
            "修改名字", f"输入新名字（当前: {current_name}）:",
            initialvalue=current_name,
            parent=self.root,
        )
        if new_name is not None and new_name.strip():
            new_name = new_name.strip()
            self.roi_list[idx]["name"] = new_name
            self._update_roi_tree()
            self.roi_tree.selection_set(str(idx))
            self._set_status(f"已将 #{idx+1} 名字改为: {new_name}")

    def _toggle_selected_roi_visibility(self):
        """切换选中 ROI 的可见性（支持多选，每个 ROI 反转当前状态）"""
        selected = self.roi_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在选区列表中选择一个或多个区域")
            return

        for item_id in selected:
            idx = int(item_id)
            if idx < len(self.roi_visible):
                self.roi_visible[idx] = not self.roi_visible[idx]

        self._update_roi_tree()
        self._highlight_roi(self.selected_roi_idx)

        # 重新选中之前的行
        for item_id in selected:
            if self.roi_tree.exists(item_id):
                self.roi_tree.selection_add(item_id)

    def _generate_roi_mask(self, shape, bbox, img_points=None):
        """根据 ROI 形状生成二值 mask (H, W) bool"""
        if self.image_np is None:
            return np.zeros((100, 100), dtype=bool)

        h, w = self.image_np.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if shape == "rectangle":
            mask[y1:y2, x1:x2] = True

        elif shape == "ellipse":
            # 生成椭圆 mask
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
            if rx > 0 and ry > 0:
                yy, xx = np.ogrid[:h, :w]
                ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
                mask = ellipse

        elif shape == "polygon" and img_points and len(img_points) >= 3:
            # 用多边形顶点生成 mask
            from PIL import ImageDraw
            img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(img)
            pts = [(int(p[0]), int(p[1])) for p in img_points]
            draw.polygon(pts, fill=255)
            mask = np.array(img) > 0

        return mask

    def _update_roi_combos(self):
        """更新 ROI 运算下拉框的选项"""
        items = []
        for i, roi in enumerate(self.roi_list):
            area = int(roi["mask"].sum()) if roi.get("mask") is not None else 0
            items.append(f"#{i+1} {roi['name']} ({area:,}px)")

        self.roi_a_combo['values'] = items
        self.roi_b_combo['values'] = items

        # 保持选中项有效
        if items:
            if not self.roi_a_var.get() or self.roi_a_combo.current() < 0:
                self.roi_a_combo.current(0)
            if not self.roi_b_var.get() or self.roi_b_combo.current() < 0:
                idx = min(1, len(items) - 1)
                self.roi_b_combo.current(idx)
        else:
            self.roi_a_var.set("")
            self.roi_b_var.set("")

    def _update_roi_op_label(self):
        """根据 ROI A/B 选择和运算类型，自动更新结果名"""
        a_idx = self.roi_a_combo.current()
        b_idx = self.roi_b_combo.current()
        op = self.roi_op_var.get()

        # 获取 A/B 的名字
        a_name = self.roi_list[a_idx]["name"] if 0 <= a_idx < len(self.roi_list) else "A"
        b_name = self.roi_list[b_idx]["name"] if 0 <= b_idx < len(self.roi_list) else "B"

        # 根据运算类型生成结果名
        op_map = {
            "Union": f"{a_name}_Union_{b_name}",
            "Intersection": f"{a_name}_Intersect_{b_name}",
            "Minus": f"{a_name}_Minus_{b_name}",
        }
        label = op_map.get(op, f"{a_name}_{op}_{b_name}")
        self.roi_op_label_var.set(label)

    def _get_selected_roi_indices(self) -> tuple:
        """获取下拉框选中的 ROI 索引"""
        a_idx = self.roi_a_combo.current()
        b_idx = self.roi_b_combo.current()

        if a_idx < 0 or b_idx < 0:
            messagebox.showwarning("提示", "请至少有 2 个 ROI")
            return None, None

        if a_idx >= len(self.roi_list) or b_idx >= len(self.roi_list):
            messagebox.showwarning("提示", "选中的 ROI 索引无效")
            return None, None

        if a_idx == b_idx:
            a_name = self.roi_list[a_idx]["name"]
            messagebox.showwarning("提示", f"ROI A 和 ROI B 不能相同！\n当前都选了 #{a_idx+1} {a_name}\n请选择不同的 ROI")
            return None, None

        return a_idx, b_idx

    def _preview_roi_operation(self):
        """预览 ROI 运算结果（不添加到 ROI 列表）"""
        a_idx, b_idx = self._get_selected_roi_indices()
        if a_idx is None:
            return

        op = self.roi_op_var.get()
        mask_a = self.roi_list[a_idx]["mask"]
        mask_b = self.roi_list[b_idx]["mask"]
        
        # Debug: 检查 mask 数据
        logger = logging.getLogger("SAM3App")
        logger.info(f"[ROI运算] A#{a_idx+1} mask shape={mask_a.shape}, dtype={mask_a.dtype}, sum={mask_a.sum()}")
        logger.info(f"[ROI运算] B#{b_idx+1} mask shape={mask_b.shape}, dtype={mask_b.dtype}, sum={mask_b.sum()}")
        logger.info(f"[ROI运算] op={op}, shapes match={mask_a.shape == mask_b.shape}")

        try:
            result_mask = SAM3Model.compute_mask_operation(mask_a, mask_b, op)
        except ValueError as e:
            messagebox.showerror("运算错误", str(e))
            return

        area = int(result_mask.sum())

        # 在画布上预览运算结果（用橙色高亮叠加到原图）
        if self.image_np is None:
            return

        overlay = self.image_np.copy().astype(np.float64)
        orange = (255, 140, 0)
        alpha = 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(
                result_mask,
                (1 - alpha) * overlay[:, :, c] + alpha * orange[c],
                overlay[:, :, c],
            )

        self._display_image(overlay_np=overlay.astype(np.uint8))

        a_area = int(mask_a.sum())
        b_area = int(mask_b.sum())
        self._set_status(
            f"👁️ 预览 ROI {op}: A#{a_idx+1}({a_area:,}px) {op} B#{b_idx+1}({b_area:,}px) "
            f"= {area:,}px"
        )

    def _execute_roi_operation(self):
        """执行 ROI 运算并将结果添加到 ROI 列表"""
        a_idx, b_idx = self._get_selected_roi_indices()
        if a_idx is None:
            return

        op = self.roi_op_var.get()
        mask_a = self.roi_list[a_idx]["mask"]
        mask_b = self.roi_list[b_idx]["mask"]

        try:
            result_mask = SAM3Model.compute_mask_operation(mask_a, mask_b, op)
        except ValueError as e:
            messagebox.showerror("运算错误", str(e))
            return

        area = int(result_mask.sum())
        if area == 0:
            messagebox.showinfo("运算结果", "运算结果为空（没有重叠/剩余区域）")
            return

        # 计算 bbox
        rows = np.any(result_mask, axis=1)
        cols = np.any(result_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = [float(cmin), float(rmin), float(cmax), float(rmax)]

        label = self.roi_op_label_var.get() or f"ROI_{op}"

        # 运算结果总是 polygon 类型（因为形状可能不规则）
        self.roi_counter += 1
        roi_color = self._get_roi_color(self.roi_counter - 1)

        # 恢复原图（清除预览叠加），然后绘制 ROI
        self._display_image()

        # 在画布上绘制（用 mask 轮廓虚线）
        canvas_ids = self._draw_roi_mask_on_canvas(result_mask, roi_color)

        roi = {
            "name": label,
            "shape": "mask",
            "bbox": bbox,
            "img_points": None,
            "canvas_ids": canvas_ids,
            "color": roi_color,
            "mask": result_mask,
            "prompt_type": f"op_{op}",
        }
        self.roi_list.append(roi)

        # 更新 ROI 可见性列表 — 新 ROI 可见，其他全部隐藏
        for i in range(len(self.roi_visible)):
            self.roi_visible[i] = False
        self.roi_visible.append(True)

        # 更新 ROI 列表 UI
        self._update_roi_tree()
        new_idx = len(self.roi_list) - 1
        self.roi_tree.selection_set(str(new_idx))
        self.roi_tree.see(str(new_idx))
        self._highlight_rois([new_idx])

        self._update_roi_combos()

        a_area = int(mask_a.sum())
        b_area = int(mask_b.sum())
        self._set_status(
            f"⚡ ROI {op} 完成: A#{a_idx+1}({a_area:,}px) {op} B#{b_idx+1}({b_area:,}px) "
            f"→ {label} ({area:,}px) 已添加"
        )

    def _draw_roi_mask_on_canvas(self, mask, color, width=2, dash=(8, 4)):
        """在画布上绘制 mask 类型的 ROI（虚线轮廓，无填充）"""
        canvas_ids = []

        if self.image_np is None:
            return canvas_ids

        h, w = self.image_np.shape[:2]

        # 用轮廓检测获取多边形顶点
        try:
            import cv2
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # 取最大轮廓
                contour = max(contours, key=cv2.contourArea)
                # 简化轮廓点数
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
            else:
                points = []
        except ImportError:
            # 没有 cv2，用 bbox 四角代替
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            points = [[cmin, rmin], [cmax, rmin], [cmax, rmax], [cmin, rmax]]

        if len(points) >= 3:
            canvas_pts = []
            for pt in points:
                cx, cy = self._image_to_canvas_coords(pt[0], pt[1])
                canvas_pts.extend([cx, cy])

            pid = self.canvas.create_polygon(
                canvas_pts,
                outline=color, width=width, fill="", dash=dash,
                tags="roi_shape",
            )
            canvas_ids.append(pid)

        return canvas_ids

    # ================================================================
    #  分割执行（累积模式）
    # ================================================================

    def _segment_points(self):
        """执行点提示分割"""
        if not self._check_ready():
            return

        if not self.click_point_list:
            messagebox.showwarning("提示", "请先在图片上点击添加提示点")
            return

        # 只使用可见的点进行分割
        visible_points = [p for p in self.click_point_list if p["visible"]]
        if not visible_points:
            messagebox.showwarning("提示", "没有可见的提示点，请先显示至少一个点")
            return

        points = [p["coords"] for p in visible_points]
        labels = [p["label"] for p in visible_points]

        self._run_segmentation(
            lambda: self.engine.segment_by_points(
                self.image_path, points, labels
            )
        )

    def _segment_box(self):
        """执行框选分割（使用选中的 ROI）"""
        if not self._check_ready():
            return

        if not self.roi_list or self.selected_roi_idx < 0:
            messagebox.showwarning("提示", "请先画 ROI 并在列表中选中一个")
            return

        roi = self.roi_list[self.selected_roi_idx]
        box = roi["bbox"]  # [x1, y1, x2, y2]

        self._run_segmentation(
            lambda: self.engine.segment_by_box(self.image_path, box)
        )

    def _check_ready(self) -> bool:
        """检查是否准备好执行分割"""
        if not self.engine.is_loaded():
            messagebox.showwarning("提示", "请先加载模型")
            return False
        if self.image_np is None:
            messagebox.showwarning("提示", "请先打开图片")
            return False
        return True

    def _run_segmentation(self, segment_fn):
        """在后台线程执行分割"""
        self._set_status("正在分割，请稍候...")
        self.root.config(cursor="watch")

        def worker():
            try:
                new_results = segment_fn()
                self.root.after(0, lambda: self._on_segmentation_done(new_results))
            except Exception as e:
                logging.getLogger("SAM3App").error(f"分割失败: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: messagebox.showerror("分割失败", str(e)))
                self.root.after(0, lambda: self._set_status("分割失败"))
            finally:
                self.root.after(0, lambda: self.root.config(cursor=""))

        threading.Thread(target=worker, daemon=True).start()

    def _on_segmentation_done(self, new_results):
        """分割完成回调 — 累积结果而非替换"""
        if not new_results:
            self._set_status("本次分割未检测到新区域")
            return

        # 累积添加新结果
        prev_count = len(self.segmentation_results)
        self.segmentation_results.extend(new_results)

        # 新增的 mask 默认可见
        self.mask_visible.extend([True] * len(new_results))

        # 隐藏所有 ROI，避免视觉混乱
        for i in range(len(self.roi_visible)):
            self.roi_visible[i] = False
        self.canvas.delete("roi_shape")
        self._update_roi_tree()

        # 隐藏所有提示点，避免视觉混乱
        for pt_data in self.click_point_list:
            pt_data["visible"] = False
        self.canvas.delete("prompt_point")
        self._update_point_tree()

        self._update_result_tree()
        self._update_overlay()
        self._set_status(
            f"本次新增 {len(new_results)} 个区域 — "
            f"累计共 {len(self.segmentation_results)} 个区域"
        )

        # 自动保存分割结果到会话
        self.session.save_masks_session(
            self.image_path, self.segmentation_results, self.mask_visible
        )

    # ================================================================
    #  结果显示与交互
    # ================================================================

    def _update_result_tree(self):
        """更新结果列表"""
        self.result_tree.delete(*self.result_tree.get_children())
        for i, r in enumerate(self.segmentation_results):
            vis_mark = "✅" if self.mask_visible[i] else "❌"
            name = r.get("name", f"Mask_{i+1}")
            area_str = self._format_calibrated_area(r['area'])
            self.result_tree.insert("", tk.END, iid=str(i), values=(
                f"#{i + 1}",
                name,
                f"{r['score']:.2f}",
                area_str,
                vis_mark,
            ))
        self.result_count_var.set(f"累计分割: {len(self.segmentation_results)} 个区域")
        # 更新 Mask 运算下拉框
        self._update_mask_combos()

    def _update_overlay(self):
        """更新叠加显示 — 只显示可见的 mask"""
        if self.image_np is None:
            return

        if not self.segmentation_results:
            self._display_image()
            return

        # 过滤出可见的 mask
        visible_indices = [i for i, v in enumerate(self.mask_visible) if v]
        if not visible_indices:
            self._display_image()
            return

        overlay = SAM3Model.overlay_masks(
            self.image_np, self.segmentation_results,
            alpha=self.alpha_var.get(),
            visible_indices=visible_indices,
        )
        self.overlay_image = overlay
        self._display_image(overlay_np=overlay)

    def _on_result_select(self, event):
        """结果列表选中事件 — 高亮所有选中的 mask（支持多选）"""
        selected = self.result_tree.selection()
        if not selected or not self.segmentation_results:
            return

        # 收集所有选中的可见 mask
        selected_results = []
        for item_id in selected:
            idx = int(item_id)
            if 0 <= idx < len(self.segmentation_results) and self.mask_visible[idx]:
                selected_results.append(self.segmentation_results[idx])

        if not selected_results:
            return

        # 高亮所有选中的 mask
        overlay = SAM3Model.overlay_masks(
            self.image_np, selected_results, alpha=0.6,
        )
        self._display_image(overlay_np=overlay)

    def _on_result_click(self, event):
        """点击结果列表 — 点击可见列切换可见性"""
        region = self.result_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.result_tree.identify_column(event.x)
        # column "#5" = vis column (1-based: #1=label_id, #2=name, #3=score, #4=area, #5=vis)
        if column != "#5":
            return
        selected = self.result_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        self.mask_visible[idx] = not self.mask_visible[idx]
        self._update_result_tree()
        self._update_overlay()
        # 重新选中该行
        self.result_tree.selection_set(str(idx))

    def _on_result_double_click(self, event):
        """双击结果列表 — 编辑名字"""
        region = self.result_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.result_tree.identify_column(event.x)
        # column "#2" = name column (1-based: #1=label_id, #2=name)
        if column != "#2":
            return
        selected = self.result_tree.selection()
        if not selected:
            return
        item_id = selected[0]
        idx = int(item_id)
        # 获取当前名字
        current_values = self.result_tree.item(item_id, "values")
        current_name = current_values[1] if len(current_values) > 1 else f"Mask_{idx+1}"
        # 弹出输入框
        new_name = simpledialog.askstring(
            "修改名字", f"输入新名字（当前: {current_name}）:",
            initialvalue=current_name,
            parent=self.root,
        )
        if new_name is not None and new_name.strip():
            new_name = new_name.strip()
            self.segmentation_results[idx]["name"] = new_name
            self._update_result_tree()
            self._update_overlay()
            # 重新选中该行
            self.result_tree.selection_set(str(idx))
            self._set_status(f"已将 #{idx+1} 名字改为: {new_name}")
            # 保存会话
            self.session.save_masks_session(
                self.image_path, self.segmentation_results, self.mask_visible
            )

    def _toggle_selected_visibility(self):
        """切换选中 mask 的可见性（支持多选，每个 mask 反转当前状态）"""
        selected = self.result_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在结果列表中选择一个或多个区域")
            return

        for item_id in selected:
            idx = int(item_id)
            self.mask_visible[idx] = not self.mask_visible[idx]

        self._update_result_tree()
        self._update_overlay()

        # 重新选中之前的行
        for item_id in selected:
            if self.result_tree.exists(item_id):
                self.result_tree.selection_add(item_id)

    def _delete_selected_result(self):
        """删除选中的分割结果（支持多选）"""
        selected = self.result_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在结果列表中选择要删除的区域")
            return

        # 收集所有选中项的索引（从大到小排序，避免删除后索引偏移）
        indices = sorted([int(s) for s in selected], reverse=True)
        labels = [f"#{idx+1} ({self.segmentation_results[idx]['label']})" for idx in indices]

        if not messagebox.askyesno("确认删除", f"删除以下 {len(indices)} 个分割结果？\n\n" + "\n".join(labels)):
            return

        for idx in indices:
            self.segmentation_results.pop(idx)
            self.mask_visible.pop(idx)

        self._update_result_tree()
        self._update_overlay()
        self._set_status(f"已删除 {len(indices)} 个区域 — 剩余 {len(self.segmentation_results)} 个")

        # 更新保存的分割结果
        self.session.save_masks_session(
            self.image_path, self.segmentation_results, self.mask_visible
        )

    # ================================================================
    #  Mask 布尔运算
    # ================================================================

    def _update_mask_combos(self):
        """更新 Mask 运算下拉框的选项"""
        items = []
        for i, r in enumerate(self.segmentation_results):
            vis = "✅" if self.mask_visible[i] else "❌"
            items.append(f"#{i+1} {r.get('name', r['label'])} ({r['area']:,}px) {vis}")

        self.mask_a_combo['values'] = items
        self.mask_b_combo['values'] = items

        # 保持选中项有效
        if items:
            if not self.mask_a_var.get() or self.mask_a_combo.current() < 0:
                self.mask_a_combo.current(0)
            if not self.mask_b_var.get() or self.mask_b_combo.current() < 0:
                idx = min(1, len(items) - 1)
                self.mask_b_combo.current(idx)
        else:
            self.mask_a_var.set("")
            self.mask_b_var.set("")

        # 自动更新结果名
        self._update_mask_op_label()

    def _update_mask_op_label(self):
        """根据选中的 Mask A/B 和运算类型自动更新结果名（与 ROI 运算逻辑一致）"""
        a_idx = self.mask_a_combo.current()
        b_idx = self.mask_b_combo.current()
        op = self.mask_op_var.get()

        if a_idx < 0 or b_idx < 0:
            self.mask_op_label_var.set("选择 Mask 和运算后自动生成")
            return

        a_name = self.segmentation_results[a_idx].get("name", self.segmentation_results[a_idx]["label"]) if a_idx < len(self.segmentation_results) else "A"
        b_name = self.segmentation_results[b_idx].get("name", self.segmentation_results[b_idx]["label"]) if b_idx < len(self.segmentation_results) else "B"

        op_map = {
            "Union": f"{a_name}_Union_{b_name}",
            "Intersection": f"{a_name}_Intersect_{b_name}",
            "Minus": f"{a_name}_Minus_{b_name}",
        }
        label = op_map.get(op, f"{a_name}_{op}_{b_name}")
        self.mask_op_label_var.set(label)

    def _get_selected_mask_indices(self) -> tuple:
        """获取下拉框选中的 mask 索引"""
        a_idx = self.mask_a_combo.current()
        b_idx = self.mask_b_combo.current()

        if a_idx < 0 or b_idx < 0:
            messagebox.showwarning("提示", "请在结果列表中至少有 2 个 mask")
            return None, None

        if a_idx >= len(self.segmentation_results) or b_idx >= len(self.segmentation_results):
            messagebox.showwarning("提示", "选中的 mask 索引无效")
            return None, None

        if a_idx == b_idx:
            name = self.segmentation_results[a_idx].get("name", self.segmentation_results[a_idx]["label"])
            messagebox.showwarning(
                "⚠️ 提示",
                f"Mask A 和 Mask B 不能相同！\n\n当前都选了 #{a_idx+1} {name}\n请选择不同的 Mask"
            )
            return None, None

        return a_idx, b_idx

    def _preview_mask_operation(self):
        """预览 mask 运算结果（不添加到结果列表）"""
        a_idx, b_idx = self._get_selected_mask_indices()
        if a_idx is None:
            return

        op = self.mask_op_var.get()
        mask_a = self.segmentation_results[a_idx]["mask"]
        mask_b = self.segmentation_results[b_idx]["mask"]

        try:
            result_mask = SAM3Model.compute_mask_operation(mask_a, mask_b, op)
        except ValueError as e:
            messagebox.showerror("运算错误", str(e))
            return

        area = int(result_mask.sum())
        a_area = int(mask_a.sum())
        b_area = int(mask_b.sum())

        # 在画布上预览运算结果（用橙色高亮）
        if self.image_np is None:
            return

        overlay = self.image_np.copy().astype(np.float64)
        # 用橙色叠加运算结果
        orange = (255, 140, 0)
        alpha = 0.6
        for c in range(3):
            overlay[:, :, c] = np.where(
                result_mask,
                (1 - alpha) * overlay[:, :, c] + alpha * orange[c],
                overlay[:, :, c],
            )

        self._display_image(overlay_np=overlay.astype(np.uint8))

        self._set_status(
            f"👁️ 预览 {op}: A#{a_idx}({a_area:,}px) {op} B#{b_idx}({b_area:,}px) "
            f"= {area:,}px"
        )

    def _execute_mask_operation(self):
        """执行 mask 运算并将结果添加到分割结果列表"""
        a_idx, b_idx = self._get_selected_mask_indices()
        if a_idx is None:
            return

        op = self.mask_op_var.get()
        mask_a = self.segmentation_results[a_idx]["mask"]
        mask_b = self.segmentation_results[b_idx]["mask"]

        try:
            result_mask = SAM3Model.compute_mask_operation(mask_a, mask_b, op)
        except ValueError as e:
            messagebox.showerror("运算错误", str(e))
            return

        area = int(result_mask.sum())
        if area == 0:
            messagebox.showinfo("运算结果", "运算结果为空（没有重叠/剩余区域）")
            return

        # 计算 bbox
        rows = np.any(result_mask, axis=1)
        cols = np.any(result_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = [float(cmin), float(rmin), float(cmax), float(rmax)]

        label = self.mask_op_label_var.get() or f"{op}_result"

        # 恢复原图（清除预览叠加）
        self._display_image()
        self._redraw_all_rois()

        # 添加到结果列表
        new_result = {
            "mask": result_mask,
            "bbox": bbox,
            "score": 1.0,
            "label": label,
            "name": label,
            "area": area,
            "prompt_type": f"op_{op}",
        }

        self.segmentation_results.append(new_result)

        # 隐藏 A 和 B 的 mask，只显示运算结果
        self.mask_visible[a_idx] = False
        self.mask_visible[b_idx] = False
        self.mask_visible.append(True)

        # 选中运算结果
        self._update_result_tree()
        self._update_overlay()
        new_idx = len(self.segmentation_results) - 1
        self.result_tree.selection_set(str(new_idx))
        self.result_tree.see(str(new_idx))

        a_area = int(mask_a.sum())
        b_area = int(mask_b.sum())
        self._set_status(
            f"⚡ {op} 运算完成: A#{a_idx}({a_area:,}px) {op} B#{b_idx}({b_area:,}px) "
            f"→ #{len(self.segmentation_results)-1} {label} ({area:,}px) 已添加到结果列表"
        )

        # 自动保存
        self.session.save_masks_session(
            self.image_path, self.segmentation_results, self.mask_visible
        )

    # ================================================================
    #  合并 Mask
    # ================================================================

    def _combine_and_preview(self):
        """合并所有可见 mask → 生成新的 combined mask → 自动选中 → 其他不可见"""
        if not self.segmentation_results:
            messagebox.showwarning("提示", "暂无分割结果可合并")
            return

        if self.image_np is None:
            return

        # 合并所有可见的 mask
        visible_results = [r for r, v in zip(self.segmentation_results, self.mask_visible) if v]
        if not visible_results:
            messagebox.showwarning("提示", "所有 mask 均已隐藏，请至少显示一个")
            return

        combined = SAM3Model.combine_masks(visible_results, self.image_np.shape[:2])

        # 计算面积
        area = int(combined.sum())

        # 生成新的 combined mask 结果
        combined_result = {
            "mask": combined,
            "bbox": [0, 0, self.image_np.shape[1], self.image_np.shape[0]],
            "score": 1.0,
            "label": f"combined_{len(self.segmentation_results) + 1}",
            "area": area,
            "name": f"combined_{len(self.segmentation_results) + 1}",
        }

        # 将所有现有 mask 设为不可见
        self.mask_visible = [False] * len(self.mask_visible)

        # 添加新的 combined mask
        self.segmentation_results.append(combined_result)
        self.mask_visible.append(True)

        # 更新显示
        self._update_result_tree()
        self._update_overlay()

        # 自动选中新添加的 combined mask
        new_idx = len(self.segmentation_results) - 1
        self.result_tree.selection_set(str(new_idx))
        self.result_tree.see(str(new_idx))

        self._set_status(
            f"🔗 已合并 {len(visible_results)} 个 mask → combined_{new_idx + 1} "
            f"(面积 {area:,} px)，其他 mask 已隐藏"
        )

        # 保存会话
        self.session.save_masks_session(
            self.image_path, self.segmentation_results, self.mask_visible
        )

    # ================================================================
    #  校准功能（两点标定）
    # ================================================================

    def _start_calibration(self):
        """开始校准 — 进入直线拖拽标定模式"""
        if self.image_np is None:
            messagebox.showwarning("提示", "请先打开一张图片")
            return

        if self.calibration_mode:
            # 取消校准
            self.calibration_mode = False
            self.calib_start_img = None
            self.calib_end_img = None
            self.calib_is_dragging = False
            if self.calib_line_id:
                self.canvas.delete(self.calib_line_id)
                self.calib_line_id = None
            self.canvas.delete("calibration")
            self.calib_canvas_ids = []
            self.calib_btn.config(text="📏 Scale Bar校准")
            self._set_status("校准已取消")
            return

        # 清除旧的校准线
        self.canvas.delete("calibration")
        self.calib_canvas_ids = []
        self.calib_start_img = None
        self.calib_end_img = None
        self.calib_is_dragging = False

        # 进入校准模式
        self.calibration_mode = True

        # 如果当前是 Unlock 模式，自动切换为 Lock（校准需要在 Lock 模式下画直线）
        if not self.image_locked:
            self.image_locked = True
            self.lock_btn.config(text="🔒 Lock", fg="#4CAF50")
            self._update_cursor()
        self.calib_btn.config(text="📏 取消Scale Bar")
        self._set_status("📏 校准模式: 在图片上拖拽画一条直线（对应 scale bar）")

    def _manual_mapping(self):
        """手动输入 Mapping（单位/px）— 显示当前校准数据，需 Unlock 才能修改，修改后自动 Lock"""
        dialog = tk.Toplevel(self.root)
        dialog.title("手动 Mapping")
        dialog.geometry("420x420")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="手动 Mapping", font=("", 12, "bold")).pack(pady=(10, 4))
        ttk.Label(dialog, text="输入每个像素对应的物理尺寸（如 0.32 μm/px）").pack(pady=2)
        ttk.Label(dialog, text="⚠️ 请先点击 🔓 Unlock 后再输入 Mapping 数据", foreground="orange", font=("", 10)).pack(pady=(2, 0))

        ttk.Separator(dialog).pack(fill=tk.X, pady=8)

        # ── 当前校准数据显示 ──
        current_frame = ttk.LabelFrame(dialog, text="当前校准数据")
        current_frame.pack(fill=tk.X, padx=12, pady=4)

        if self.calibration_data:
            scale = self.calibration_data["scale"]
            calib_unit = self.calibration_data["unit"]
            unit_to_mm = {"mm": 1.0, "um": 0.001, "μm": 0.001, "cm": 10.0, "in": 25.4}
            unit_display = {"mm": "mm", "um": "μm", "μm": "μm", "cm": "cm", "in": "in"}
            u = unit_display.get(calib_unit, calib_unit)
            um_per_px = 1.0 / scale  # unit/px
            source = "Scale Bar" if not self.calibration_data.get("manual") else "手动 Mapping"
            ttk.Label(current_frame, text=f"来源: {source}").pack(anchor=tk.W, padx=8, pady=2)
            ttk.Label(current_frame, text=f"Mapping: {um_per_px:.4f} {u}/px").pack(anchor=tk.W, padx=8, pady=2)
            ttk.Label(current_frame, text=f"Scale: {scale:.2f} px/{u}").pack(anchor=tk.W, padx=8, pady=2)
            # 初始值同步到当前校准数据
            init_mapping_val = um_per_px
            init_unit = calib_unit
        else:
            ttk.Label(current_frame, text="来源: 无校准数据").pack(anchor=tk.W, padx=8, pady=2)
            ttk.Label(current_frame, text="请在下方输入 Mapping 值").pack(anchor=tk.W, padx=8, pady=2)
            init_mapping_val = 0.0
            init_unit = "um"

        # ── Unlock 按钮 + 编辑区 ──
        edit_frame = ttk.LabelFrame(dialog, text="编辑 Mapping")
        edit_frame.pack(fill=tk.X, padx=12, pady=4)

        # Unlock/Lock 按钮
        lock_unlock_frame = ttk.Frame(edit_frame)
        lock_unlock_frame.pack(fill=tk.X, padx=8, pady=4)

        is_locked = tk.BooleanVar(value=True)  # 默认 Lock（不可编辑）

        lock_btn = tk.Button(
            lock_unlock_frame, text="🔒 Lock", font=("", 10),
            bg="#3c3c3c", fg="#4CAF50", relief=tk.FLAT, padx=8, pady=2, cursor="hand2", bd=0
        )
        lock_btn.pack(side=tk.LEFT, padx=4)

        # mapping 值输入（unit/px）
        map_frame = ttk.Frame(edit_frame)
        map_frame.pack(pady=4)
        ttk.Label(map_frame, text="Mapping (unit/px):").pack(side=tk.LEFT, padx=4)
        map_var = tk.StringVar(value=f"{init_mapping_val:.4f}")
        map_entry = ttk.Entry(map_frame, textvariable=map_var, width=12, state="disabled")
        map_entry.pack(side=tk.LEFT, padx=4)

        # 单位选择
        unit_frame = ttk.Frame(edit_frame)
        unit_frame.pack(pady=4)
        ttk.Label(unit_frame, text="单位:").pack(side=tk.LEFT, padx=4)
        unit_var = tk.StringVar(value=init_unit)
        unit_radios = []
        for text, val in [("μm", "um"), ("mm", "mm"), ("cm", "cm"), ("in", "in")]:
            rb = ttk.Radiobutton(unit_frame, text=text, variable=unit_var, value=val, state="disabled")
            rb.pack(side=tk.LEFT, padx=4)
            unit_radios.append(rb)

        # 预览提示
        preview_var = tk.StringVar(value="")
        ttk.Label(edit_frame, textvariable=preview_var, foreground="blue").pack(pady=4)

        def toggle_lock_unlock():
            if is_locked.get():
                # Unlock → 允许编辑
                is_locked.set(False)
                lock_btn.config(text="🔓 Unlock", fg="#FF9800")
                map_entry.config(state="normal")
                map_entry.select_range(0, tk.END)
                map_entry.focus()
                for rb in unit_radios:
                    rb.config(state="normal")
            else:
                # Lock → 禁止编辑（自动 Lock）
                is_locked.set(True)
                lock_btn.config(text="🔒 Lock", fg="#4CAF50")
                map_entry.config(state="disabled")
                for rb in unit_radios:
                    rb.config(state="disabled")

        lock_btn.config(command=toggle_lock_unlock)

        def update_preview(*args):
            try:
                val = float(map_var.get())
                unit_display = {"mm": "mm", "um": "μm", "cm": "cm", "in": "in"}
                u = unit_display.get(unit_var.get(), unit_var.get())
                preview_var.set(f"预览: 1 px = {val} {u}，scale = {1/val:.2f} px/{u}")
            except (ValueError, ZeroDivisionError):
                preview_var.set("请输入有效数字")

        map_var.trace_add("write", update_preview)
        unit_var.trace_add("write", update_preview)
        update_preview()

        def confirm():
            try:
                mapping_val = float(map_var.get())
                if mapping_val <= 0:
                    messagebox.showwarning("提示", "Mapping 值必须大于 0")
                    return
            except ValueError:
                messagebox.showwarning("提示", "请输入有效的数字")
                return

            unit = unit_var.get()
            scale = 1.0 / mapping_val  # pixels per unit

            # 清除旧的校准线
            self._clear_calibration()

            self.calibration_data = {
                "point1": None,
                "point2": None,
                "pixel_dist": None,
                "real_dist": None,
                "unit": unit,
                "scale": scale,
                "manual": True,
            }

            # 同步 display_unit 到校准单位
            self.display_unit.set(unit)

            # 更新校准信息显示（外部自动同步）
            self._on_unit_change()

            unit_display = {"mm": "mm", "um": "μm", "cm": "cm", "in": "in"}
            u = unit_display.get(unit, unit)
            self._set_status(f"✅ 手动 Mapping: {mapping_val:.4f} {u}/px")

            dialog.destroy()

        def cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="✅ 确认", command=confirm).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="❌ 取消", command=cancel).pack(side=tk.LEFT, padx=8)

    def _finish_calibration(self, pixel_dist: float):
        """完成校准 — 弹出对话框输入实际距离"""
        dialog = tk.Toplevel(self.root)
        dialog.title("输入校准距离")
        dialog.geometry("360x220")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Scale Bar 校准", font=("", 12, "bold")).pack(pady=(10, 4))
        ttk.Label(dialog, text=f"像素距离: {pixel_dist:.1f} px").pack(pady=2)

        ttk.Separator(dialog).pack(fill=tk.X, pady=8)

        # 实际距离输入
        dist_frame = ttk.Frame(dialog)
        dist_frame.pack(pady=4)
        ttk.Label(dist_frame, text="实际距离:").pack(side=tk.LEFT, padx=4)
        dist_var = tk.StringVar(value="10")
        dist_entry = ttk.Entry(dist_frame, textvariable=dist_var, width=12)
        dist_entry.pack(side=tk.LEFT, padx=4)
        dist_entry.select_range(0, tk.END)
        dist_entry.focus()

        # 单位选择
        unit_frame = ttk.Frame(dialog)
        unit_frame.pack(pady=4)
        ttk.Label(unit_frame, text="单位:").pack(side=tk.LEFT, padx=4)
        unit_var = tk.StringVar(value="mm")
        for text, val in [("μm", "um"), ("mm", "mm"), ("cm", "cm"), ("in", "in")]:
            ttk.Radiobutton(unit_frame, text=text, variable=unit_var, value=val).pack(side=tk.LEFT, padx=4)

        def confirm():
            try:
                real_dist = float(dist_var.get())
                if real_dist <= 0:
                    messagebox.showwarning("提示", "实际距离必须大于 0")
                    return
            except ValueError:
                messagebox.showwarning("提示", "请输入有效的数字")
                return

            unit = unit_var.get()
            scale = pixel_dist / real_dist  # pixels per unit

            self.calibration_data = {
                "point1": self.calib_start_img,
                "point2": self.calib_end_img,
                "pixel_dist": pixel_dist,
                "real_dist": real_dist,
                "unit": unit,
                "scale": scale,
                "manual": False,  # Scale Bar 校准
            }

            # 同步 display_unit 到校准单位
            self.display_unit.set(unit)

            # 更新校准信息显示（使用 _on_unit_change 统一处理）
            self._on_unit_change()
            unit_display = {"mm": "mm", "um": "μm", "cm": "cm", "in": "in"}
            u = unit_display.get(unit, unit)
            # 显示 unit/px 格式（如 0.32 μm/px），而非 px/unit
            um_per_px = real_dist / pixel_dist  # unit/px（校准单位下）
            self._set_status(f"✅ Scale Bar 校准完成: {um_per_px:.4f} {u}/px，1px = {um_per_px:.4f} {u}")

            dialog.destroy()

        def cancel():
            # 取消校准
            self.calib_start_img = None
            self.calib_end_img = None
            self.canvas.delete("calibration")
            self.calib_canvas_ids = []
            self.calibration_data = None
            self.calib_info_var.set("⚠️ 未校准")
            self._set_status("校准已取消")
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="✅ 确认校准", command=confirm).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="❌ 取消", command=cancel).pack(side=tk.LEFT, padx=8)

        dist_entry.bind("<Return>", lambda e: confirm())

    def _add_area_watermark(self, img: Image.Image, area_text: str) -> Image.Image:
        """在图片左上角添加 Area: XXXX 水印（黑色背景白色字体，字体大小按图片尺寸缩放）

        字体大小 = 图片高度的 1.5%，最小 24px，确保大图片水印清晰可见。

        Args:
            img: PIL Image 对象（RGB 或 L 模式）
            area_text: 要显示的面积文本，如 "Area: 349,278 (3.52 mm²)"

        Returns:
            添加水印后的 PIL Image（RGB 模式）
        """
        from PIL import ImageDraw, ImageFont

        # 确保 RGB 模式
        if img.mode != "RGB":
            img = img.convert("RGB")

        draw = ImageDraw.Draw(img)

        # 字体大小按图片高度缩放：1.5%，最小 24px
        font_size = max(24, int(img.height * 0.015))
        font = None
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    continue
        if font is None:
            try:
                font = ImageFont.truetype("Helvetica", font_size)
            except Exception:
                font = ImageFont.load_default()

        # 计算文字尺寸
        text_bbox = draw.textbbox((0, 0), area_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 黑色背景矩形（左上角，padding 4px）
        padding = 4
        bg_rect = [0, 0, text_w + 2 * padding, text_h + 2 * padding]
        draw.rectangle(bg_rect, fill="black")

        # 白色文字
        draw.text((padding, padding), area_text, fill="white", font=font)

        return img

    def _format_calibrated_area(self, area_px: int) -> str:
        """格式化校准后的面积显示（仅显示校准面积，无校准返回空字符串）

        Returns:
            校准面积文本（如 "3.52 mm²"），无校准时返回空字符串
        """
        if self.calibration_data is None:
            return ""

        scale = self.calibration_data["scale"]  # pixels per calibration unit
        calib_unit = self.calibration_data["unit"]  # 校准时输入的单位
        display_unit = self.display_unit.get()  # 用户选择的显示单位

        # 面积 = (area_px / scale^2) calibration_unit^2
        real_area_calib = area_px / (scale * scale)

        # 单位换算系数（从校准单位到显示单位）
        # 同时支持 "um" 和 "μm" 键（Combobox 值为 "μm"，校准存储为 "um"）
        unit_to_mm = {"mm": 1.0, "um": 0.001, "μm": 0.001, "cm": 10.0, "in": 25.4}
        calib_mm_factor = unit_to_mm.get(calib_unit, 1.0)
        display_mm_factor = unit_to_mm.get(display_unit, 1.0)

        # 面积换算: real_area_display = real_area_calib * (calib_mm_factor / display_mm_factor)^2
        linear_ratio = calib_mm_factor / display_mm_factor
        real_area_display = real_area_calib * (linear_ratio ** 2)

        # 单位显示（同时支持 "um" 和 "μm"）
        unit_sq = {"mm": "mm²", "um": "μm²", "μm": "μm²", "cm": "cm²", "in": "in²"}
        u2 = unit_sq.get(display_unit, f"{display_unit}²")

        if real_area_display >= 100:
            return f"{real_area_display:.1f} {u2}"
        elif real_area_display >= 1:
            return f"{real_area_display:.2f} {u2}"
        elif real_area_display >= 0.01:
            return f"{real_area_display:.4f} {u2}"
        else:
            return f"{real_area_display:.6f} {u2}"

    def _get_calib_tooltip_text(self):
        """返回校准信息的 tooltip 文本（动态，每次 hover 时调用）"""
        if self.calibration_data is None:
            return "⚠️ 图片未校准\n请使用 📏 Scale Bar 校准或 🔢 手动 Mapping"
        scale = self.calibration_data["scale"]
        calib_unit = self.calibration_data["unit"]
        unit_display = {"mm": "mm", "um": "μm", "μm": "μm", "cm": "cm", "in": "in"}
        u = unit_display.get(calib_unit, calib_unit)
        um_per_px = 1.0 / scale
        source = "Scale Bar" if not self.calibration_data.get("manual") else "手动 Mapping"
        lines = [f"来源: {source}", f"Mapping: {um_per_px:.4f} {u}/px", f"Scale: {scale:.2f} px/{u}"]
        if not self.calibration_data.get("manual"):
            pixel_dist = self.calibration_data.get("pixel_dist")
            real_dist = self.calibration_data.get("real_dist")
            if pixel_dist and real_dist:
                lines.append(f"校准线: {pixel_dist:.0f}px = {real_dist:.2f}{u}")
        return "\n".join(lines)

    def _on_unit_change(self):
        """切换显示单位时刷新面积显示"""
        # 更新校准信息标签
        if self.calibration_data:
            scale = self.calibration_data["scale"]
            pixel_dist = self.calibration_data.get("pixel_dist")
            real_dist = self.calibration_data.get("real_dist")
            calib_unit = self.calibration_data["unit"]
            display_unit = self.display_unit.get()

            # 同时支持 "um" 和 "μm" 键（Combobox 值为 "μm"，校准存储为 "um"）
            unit_to_mm = {"mm": 1.0, "um": 0.001, "μm": 0.001, "cm": 10.0, "in": 25.4}
            calib_mm_factor = unit_to_mm.get(calib_unit, 1.0)
            display_mm_factor = unit_to_mm.get(display_unit, 1.0)
            linear_ratio = calib_mm_factor / display_mm_factor

            # 换算 scale 到显示单位
            scale_display = scale / linear_ratio  # pixels per display_unit

            unit_display = {"mm": "mm", "um": "μm", "μm": "μm", "cm": "cm", "in": "in"}
            u = unit_display.get(display_unit, display_unit)
            # 显示 unit/px 格式（如 0.32 μm/px）
            um_per_px_display = 1.0 / scale_display  # display_unit/px

            # 手动 Mapping 没有 pixel_dist 和 real_dist，只显示 mapping 值
            if pixel_dist is not None and real_dist is not None:
                real_dist_display = real_dist * linear_ratio
                self.calib_info_var.set(f"📏 {um_per_px_display:.4f} {u}/px ({pixel_dist:.0f}px = {real_dist_display:.2f}{u})")
            else:
                self.calib_info_var.set(f"📏 {um_per_px_display:.4f} {u}/px (手动)")
        else:
            self.calib_info_var.set("⚠️ 未校准")

        # 更新结果列表和 ROI 列表
        self._update_result_tree()
        self._update_roi_tree()

    def _clear_calibration(self):
        """清除校准数据"""
        self.calibration_data = None
        self.calib_start_img = None
        self.calib_end_img = None
        self.calib_is_dragging = False
        if self.calib_line_id:
            self.canvas.delete(self.calib_line_id)
            self.calib_line_id = None
        self.canvas.delete("calibration")
        self.calib_canvas_ids = []
        self.calib_info_var.set("⚠️ 未校准")
        self._update_result_tree()
        self._update_roi_tree()

    # ================================================================
    #  提示点管理
    # ================================================================

    def _add_click_point(self, coords, label=1):
        """添加一个点击提示点"""
        self.click_point_counter += 1
        tag = "正" if label == 1 else "负"
        pt_data = {
            "id": self.click_point_counter,
            "name": f"Point_{self.click_point_counter}",
            "coords": coords,
            "label": label,  # 1=正 0=负
            "visible": True,
        }
        self.click_point_list.append(pt_data)
        self._update_point_tree()
        self._redraw_points()

    def _update_point_tree(self):
        """更新提示点 Treeview"""
        self.point_tree.delete(*self.point_tree.get_children())
        for pt_data in self.click_point_list:
            tag = "正(前景)" if pt_data["label"] == 1 else "负(背景)"
            vis = "✅" if pt_data["visible"] else "❌"
            self.point_tree.insert("", tk.END, iid=str(pt_data["id"]), values=(
                f"#{pt_data['id']}", pt_data["name"], tag, vis
            ))

    def _on_point_tree_double_click(self, event=None):
        """双击提示点 Treeview 编辑名字"""
        sel = self.point_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        # 找到对应的点数据
        pt_data = None
        for p in self.click_point_list:
            if str(p["id"]) == item_id:
                pt_data = p
                break
        if not pt_data:
            return

        # 弹出编辑对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("编辑名字")
        dialog.geometry("300x100")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=f"编辑 {pt_data['name']} 的名字:").pack(pady=(10, 4))
        name_var = tk.StringVar(value=pt_data["name"])
        entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        entry.pack(pady=4)
        entry.select_range(0, tk.END)
        entry.focus()

        def confirm():
            pt_data["name"] = name_var.get().strip() or pt_data["name"]
            self._update_point_tree()
            dialog.destroy()

        entry.bind("<Return>", lambda e: confirm())
        ttk.Button(dialog, text="确定", command=confirm).pack(pady=4)

    def _on_point_tree_click(self, event):
        """点击提示点列表 — 点击可见列切换可见性（与框选ROI模式行为一致）"""
        region = self.point_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.point_tree.identify_column(event.x)
        # column "#4" = visible column (1-based: #1=id, #2=name, #3=type, #4=visible)
        if column != "#4":
            return
        row = self.point_tree.identify_row(event.y)
        if not row:
            return
        # 找到对应的点数据并切换可见性
        for p in self.click_point_list:
            if str(p["id"]) == row:
                p["visible"] = not p["visible"]
                self._update_point_tree()
                self._redraw_points()
                # 重新选中该行
                self.point_tree.selection_set(row)
                break

    def _toggle_selected_point_visibility(self):
        """切换选中提示点的可见性"""
        sel = self.point_tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先选择要切换可见性的提示点")
            return
        for item_id in sel:
            for pt_data in self.click_point_list:
                if str(pt_data["id"]) == item_id:
                    pt_data["visible"] = not pt_data["visible"]
                    break
        self._update_point_tree()
        self._redraw_points()

    def _delete_selected_point(self):
        """删除选中的提示点"""
        sel = self.point_tree.selection()
        if not sel:
            return
        # 从大到小删除避免索引偏移
        ids_to_delete = [int(item_id) for item_id in sel]
        self.click_point_list = [p for p in self.click_point_list if p["id"] not in ids_to_delete]
        self._update_point_tree()
        self._redraw_points()

    def _save_project(self):
        """保存项目文件（.sam3proj）— 包含图片、模型路径、所有 mask 数据"""
        if not self.image_path:
            messagebox.showwarning("提示", "请先打开一张图片")
            return

        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + ".sam3proj"
        proj_path = filedialog.asksaveasfilename(
            title="保存项目文件",
            defaultextension=".sam3proj",
            initialfile=default_name,
            filetypes=[("SAM3 项目文件", "*.sam3proj"), ("所有文件", "*.*")],
        )
        if not proj_path:
            return

        try:
            import zipfile
            import tempfile

            with zipfile.ZipFile(proj_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. 保存项目元数据
                meta = {
                    "version": self.APP_VERSION,
                    "image_path": self.image_path,
                    "model_path": self._model_path_var.get(),
                    "image_filename": os.path.basename(self.image_path),
                    "mask_count": len(self.segmentation_results),
                }
                zf.writestr("project.json", json.dumps(meta, ensure_ascii=False, indent=2))

                # 2. 保存原图（拷贝到 zip）
                if os.path.isfile(self.image_path):
                    zf.write(self.image_path, "image/" + os.path.basename(self.image_path))

                # 3. 保存所有 mask 数据
                masks_meta = []
                for i, r in enumerate(self.segmentation_results):
                    # mask 保存为 numpy .npy
                    mask_filename = f"masks/mask_{i:03d}.npy"
                    buf_path = os.path.join(tempfile.gettempdir(), f"sam3_mask_{i:03d}.npy")
                    np.save(buf_path, r["mask"])
                    zf.write(buf_path, mask_filename)
                    os.remove(buf_path)

                    masks_meta.append({
                        "mask_file": mask_filename,
                        "bbox": r["bbox"],
                        "score": r["score"],
                        "label": r["label"],
                        "name": r.get("name", f"Mask_{i+1}"),
                        "area": r["area"],
                        "prompt_type": r.get("prompt_type", "point"),
                        "visible": self.mask_visible[i] if i < len(self.mask_visible) else True,
                    })

                zf.writestr("masks/masks_meta.json", json.dumps(masks_meta, ensure_ascii=False, indent=2))

                # 4. 保存 ROI 数据（框选ROI模式的选区）
                rois_meta = []
                for i, roi in enumerate(self.roi_list):
                    # ROI mask 保存为 numpy .npy
                    roi_mask_filename = f"rois/roi_mask_{i:03d}.npy"
                    buf_path = os.path.join(tempfile.gettempdir(), f"sam3_roi_mask_{i:03d}.npy")
                    np.save(buf_path, roi["mask"])
                    zf.write(buf_path, roi_mask_filename)
                    os.remove(buf_path)

                    rois_meta.append({
                        "mask_file": roi_mask_filename,
                        "name": roi["name"],
                        "shape": roi["shape"],
                        "bbox": roi["bbox"],
                        "img_points": roi.get("img_points"),
                        "color": roi["color"],
                        "prompt_type": roi.get("prompt_type", "box"),
                        "visible": self.roi_visible[i] if i < len(self.roi_visible) else True,
                    })

                zf.writestr("rois/rois_meta.json", json.dumps(rois_meta, ensure_ascii=False, indent=2))

                # 5. 保存点击ROI数据
                points_meta = {
                    "click_points": [p["coords"] for p in self.click_point_list],
                    "click_labels": [p["label"] for p in self.click_point_list],
                    "click_point_list": [
                        {
                            "id": p["id"],
                            "name": p["name"],
                            "coords": p["coords"],
                            "label": p["label"],
                            "visible": p["visible"],
                        }
                        for p in self.click_point_list
                    ],
                }
                zf.writestr("points/points_meta.json", json.dumps(points_meta, ensure_ascii=False, indent=2))

                # 6. 保存校准数据
                if self.calibration_data:
                    calib_meta = {
                        "point1": self.calibration_data["point1"],
                        "point2": self.calibration_data["point2"],
                        "pixel_dist": self.calibration_data["pixel_dist"],
                        "real_dist": self.calibration_data["real_dist"],
                        "unit": self.calibration_data["unit"],
                        "scale": self.calibration_data["scale"],
                        "display_unit": self.display_unit.get(),
                    }
                    zf.writestr("calibration/calibration.json", json.dumps(calib_meta, ensure_ascii=False, indent=2))

            total_items = len(self.segmentation_results) + len(self.roi_list)
            self._set_status(f"项目已保存: {proj_path}")
            messagebox.showinfo(
                "保存成功",
                f"项目已保存到:\n{proj_path}\n\n"
                f"包含 {len(self.segmentation_results)} 个 mask\n"
                f"包含 {len(self.roi_list)} 个 ROI\n"
                f"包含 {len(self.click_point_list)} 个提示点\n\n"
                f"📦 原始图片已打包进项目文件，删除原图不影响打开项目"
            )

        except Exception as e:
            messagebox.showerror("保存失败", f"保存项目时出错:\n{e}")

    def _open_project(self):
        """打开项目文件（.sam3proj）— 恢复图片、模型路径、所有 mask 数据"""
        proj_path = filedialog.askopenfilename(
            title="打开项目文件",
            filetypes=[("SAM3 项目文件", "*.sam3proj"), ("所有文件", "*.*")],
        )
        if not proj_path:
            return

        try:
            import zipfile
            import tempfile

            with zipfile.ZipFile(proj_path, 'r') as zf:
                # 1. 读取项目元数据
                meta = json.loads(zf.read("project.json"))

                # 2. 解压图片到临时目录
                temp_dir = tempfile.mkdtemp(prefix="sam3_proj_")
                image_filename = meta.get("image_filename", "")
                image_entries = [n for n in zf.namelist() if n.startswith("image/")]

                # 先解压图片
                loaded_from_proj = False
                if image_entries:
                    for entry in image_entries:
                        zf.extract(entry, temp_dir)
                    extracted_img_path = os.path.join(temp_dir, "image", image_filename)
                    if os.path.isfile(extracted_img_path):
                        img_path_to_load = extracted_img_path
                        loaded_from_proj = True

                if not loaded_from_proj:
                    orig_path = meta.get("image_path", "")
                    if orig_path and os.path.isfile(orig_path):
                        img_path_to_load = orig_path
                    else:
                        messagebox.showwarning("提示", "无法找到项目中的图片文件")
                        return

                # 清空旧数据再加载图片（_load_image_from_path 会 _clear_prompts + _display_image）
                self.segmentation_results = []
                self.mask_visible = []
                self.roi_list = []
                self.roi_visible = []
                self.roi_counter = 0
                self.click_point_list = []
                self.click_point_counter = 0
                self._load_image_from_path(img_path_to_load)

                # 3. 恢复模型路径
                model_path = meta.get("model_path", "")
                if model_path:
                    self._model_path_var.set(model_path)
                    try:
                        self.model_btn.config(text=f"🧠 加载模型 ({os.path.basename(model_path)})")
                    except Exception:
                        pass

                # 4. 恢复 mask 数据
                if "masks/masks_meta.json" in [n for n in zf.namelist()]:
                    masks_meta = json.loads(zf.read("masks/masks_meta.json"))
                    masks = []
                    visible = []

                    for m in masks_meta:
                        mask_file = m["mask_file"]
                        buf_path = os.path.join(temp_dir, mask_file)
                        zf.extract(mask_file, temp_dir)

                        if os.path.isfile(buf_path):
                            mask = np.load(buf_path)
                            masks.append({
                                "mask": mask,
                                "bbox": m["bbox"],
                                "score": m["score"],
                                "label": m["label"],
                                "name": m.get("name", "Mask_1"),
                                "area": m["area"],
                                "prompt_type": m["prompt_type"],
                            })
                            visible.append(m.get("visible", True))

                    if masks:
                        self.segmentation_results = masks
                        self.mask_visible = visible
                        self._update_result_tree()
                        self._update_overlay()
                        self.session.save_masks_session(
                            self.image_path, self.segmentation_results, self.mask_visible
                        )

                # 5. 恢复 ROI 数据（框选ROI模式的选区）
                if "rois/rois_meta.json" in zf.namelist():
                    rois_meta = json.loads(zf.read("rois/rois_meta.json"))
                    self.roi_list = []
                    self.roi_visible = []
                    self.roi_counter = 0

                    for r in rois_meta:
                        mask_file = r["mask_file"]
                        buf_path = os.path.join(temp_dir, mask_file)
                        zf.extract(mask_file, temp_dir)

                        roi_mask = None
                        if os.path.isfile(buf_path):
                            roi_mask = np.load(buf_path)

                        # 在画布上绘制 ROI
                        roi_color = r.get("color", self._get_roi_color(self.roi_counter))
                        canvas_ids = self._draw_roi_on_canvas(
                            r["shape"], r["bbox"],
                            color=roi_color,
                            img_points=r.get("img_points"),
                        )
                        # 如果是 mask 类型（运算结果），用轮廓绘制
                        if r["shape"] == "mask" and roi_mask is not None:
                            self.canvas.delete("roi_shape")
                            canvas_ids = self._draw_roi_mask_on_canvas(roi_mask, roi_color)

                        self.roi_counter += 1
                        self.roi_list.append({
                            "name": r["name"],
                            "shape": r["shape"],
                            "bbox": r["bbox"],
                            "img_points": r.get("img_points"),
                            "canvas_ids": canvas_ids,
                            "color": roi_color,
                            "mask": roi_mask,
                            "prompt_type": r.get("prompt_type", "box"),
                        })
                        self.roi_visible.append(r.get("visible", True))

                    # 隐藏不可见的 ROI
                    for i, vis in enumerate(self.roi_visible):
                        if not vis and i < len(self.roi_list):
                            for cid in self.roi_list[i]["canvas_ids"]:
                                try:
                                    self.canvas.delete(cid)
                                except Exception:
                                    pass
                            self.roi_list[i]["canvas_ids"] = []

                    self._update_roi_tree()
                    self._update_roi_combos()

                # 6. 恢复点击ROI数据
                if "points/points_meta.json" in zf.namelist():
                    points_meta = json.loads(zf.read("points/points_meta.json"))
                    # 优先使用新格式 click_point_list
                    if "click_point_list" in points_meta and points_meta["click_point_list"]:
                        self.click_point_list = points_meta["click_point_list"]
                        self.click_point_counter = max(p["id"] for p in self.click_point_list)
                    else:
                        # 兼容旧格式
                        old_points = points_meta.get("click_points", [])
                        old_labels = points_meta.get("click_labels", [])
                        self.click_point_list = []
                        for i, (coords, label) in enumerate(zip(old_points, old_labels)):
                            self.click_point_counter += 1
                            self.click_point_list.append({
                                "id": self.click_point_counter,
                                "name": f"Point_{self.click_point_counter}",
                                "coords": coords,
                                "label": label,
                                "visible": True,
                            })
                    self._update_point_tree()
                    self._redraw_points()

                # 7. 恢复校准数据
                if "calibration/calibration.json" in zf.namelist():
                    calib_meta = json.loads(zf.read("calibration/calibration.json"))
                    self.calibration_data = calib_meta
                    # 恢复校准线显示
                    self.calib_start_img = calib_meta["point1"]
                    self.calib_end_img = calib_meta["point2"]
                    # 同步 display_unit（优先使用保存的 display_unit，兼容旧项目）
                    saved_display_unit = calib_meta.get("display_unit", calib_meta["unit"])
                    self.display_unit.set(saved_display_unit)
                    # 重绘校准线（使用统一方法）
                    self._redraw_calibration_line()
                    # 更新校准信息显示（使用 _on_unit_change 统一处理）
                    self._on_unit_change()
                else:
                    # 项目没有校准数据，确保显示 ⚠️ 未校准
                    self.calib_info_var.set("⚠️ 未校准")
                    self._on_unit_change()

            roi_count = len(self.roi_list) if hasattr(self, 'roi_list') else 0
            point_count = len(self.click_point_list) if hasattr(self, 'click_point_list') else 0
            self._set_status(
                f"项目已打开: {os.path.basename(proj_path)} "
                f"({len(self.segmentation_results)} mask, {roi_count} ROI, {point_count} 提示点)"
            )

        except Exception as e:
            messagebox.showerror("打开失败", f"打开项目时出错:\n{e}")

    def _open_project_from_path(self, proj_path):
        """从指定路径打开项目文件（拖拽传入路径，无需弹窗选文件）"""
        try:
            import zipfile
            import tempfile

            with zipfile.ZipFile(proj_path, 'r') as zf:
                # 1. 读取项目元数据
                meta = json.loads(zf.read("project.json"))

                # 2. 解压图片到临时目录
                temp_dir = tempfile.mkdtemp(prefix="sam3_proj_")
                image_filename = meta.get("image_filename", "")
                image_entries = [n for n in zf.namelist() if n.startswith("image/")]

                # 先解压图片
                loaded_from_proj = False
                if image_entries:
                    for entry in image_entries:
                        zf.extract(entry, temp_dir)
                    extracted_img_path = os.path.join(temp_dir, "image", image_filename)
                    if os.path.isfile(extracted_img_path):
                        img_path_to_load = extracted_img_path
                        loaded_from_proj = True

                if not loaded_from_proj:
                    orig_path = meta.get("image_path", "")
                    if orig_path and os.path.isfile(orig_path):
                        img_path_to_load = orig_path
                    else:
                        messagebox.showwarning("提示", "无法找到项目中的图片文件")
                        return

                # 清空旧数据再加载图片
                self.segmentation_results = []
                self.mask_visible = []
                self.roi_list = []
                self.roi_visible = []
                self.roi_counter = 0
                self.click_point_list = []
                self.click_point_counter = 0
                self._load_image_from_path(img_path_to_load)

                # 3. 恢复模型路径
                model_path = meta.get("model_path", "")
                if model_path:
                    self._model_path_var.set(model_path)
                    try:
                        self.model_btn.config(text=f"🧠 加载模型 ({os.path.basename(model_path)})")
                    except Exception:
                        pass

                # 4. 恢复 mask 数据
                if "masks/masks_meta.json" in [n for n in zf.namelist()]:
                    masks_meta = json.loads(zf.read("masks/masks_meta.json"))
                    masks = []
                    visible = []

                    for m in masks_meta:
                        mask_file = m["mask_file"]
                        buf_path = os.path.join(temp_dir, mask_file)
                        zf.extract(mask_file, temp_dir)

                        if os.path.isfile(buf_path):
                            mask = np.load(buf_path)
                            masks.append({
                                "mask": mask,
                                "bbox": m["bbox"],
                                "score": m["score"],
                                "label": m["label"],
                                "name": m.get("name", "Mask_1"),
                                "area": m["area"],
                                "prompt_type": m["prompt_type"],
                            })
                            visible.append(m.get("visible", True))

                    if masks:
                        self.segmentation_results = masks
                        self.mask_visible = visible
                        self._update_result_tree()
                        self._update_overlay()
                        self.session.save_masks_session(
                            self.image_path, self.segmentation_results, self.mask_visible
                        )

                # 5. 恢复 ROI 数据
                if "rois/rois_meta.json" in zf.namelist():
                    rois_meta = json.loads(zf.read("rois/rois_meta.json"))
                    self.roi_list = []
                    self.roi_visible = []
                    self.roi_counter = 0

                    for r in rois_meta:
                        mask_file = r["mask_file"]
                        buf_path = os.path.join(temp_dir, mask_file)
                        zf.extract(mask_file, temp_dir)

                        roi_mask = None
                        if os.path.isfile(buf_path):
                            roi_mask = np.load(buf_path)

                        roi_color = r.get("color", self._get_roi_color(self.roi_counter))
                        canvas_ids = self._draw_roi_on_canvas(
                            r["shape"], r["bbox"],
                            color=roi_color,
                            img_points=r.get("img_points"),
                        )
                        if r["shape"] == "mask" and roi_mask is not None:
                            self.canvas.delete("roi_shape")
                            canvas_ids = self._draw_roi_mask_on_canvas(roi_mask, roi_color)

                        self.roi_counter += 1
                        self.roi_list.append({
                            "name": r["name"],
                            "shape": r["shape"],
                            "bbox": r["bbox"],
                            "img_points": r.get("img_points"),
                            "canvas_ids": canvas_ids,
                            "color": roi_color,
                            "mask": roi_mask,
                            "prompt_type": r.get("prompt_type", "box"),
                        })
                        self.roi_visible.append(r.get("visible", True))

                    for i, vis in enumerate(self.roi_visible):
                        if not vis and i < len(self.roi_list):
                            for cid in self.roi_list[i]["canvas_ids"]:
                                try:
                                    self.canvas.delete(cid)
                                except Exception:
                                    pass
                            self.roi_list[i]["canvas_ids"] = []

                    self._update_roi_tree()
                    self._update_roi_combos()

                # 6. 恢复点击ROI数据
                if "points/points_meta.json" in zf.namelist():
                    points_meta = json.loads(zf.read("points/points_meta.json"))
                    if "click_point_list" in points_meta and points_meta["click_point_list"]:
                        self.click_point_list = points_meta["click_point_list"]
                        self.click_point_counter = max(p["id"] for p in self.click_point_list)
                    else:
                        old_points = points_meta.get("click_points", [])
                        old_labels = points_meta.get("click_labels", [])
                        self.click_point_list = []
                        for i, (coords, label) in enumerate(zip(old_points, old_labels)):
                            self.click_point_counter += 1
                            self.click_point_list.append({
                                "id": self.click_point_counter,
                                "name": f"Point_{self.click_point_counter}",
                                "coords": coords,
                                "label": label,
                                "visible": True,
                            })
                    self._update_point_tree()
                    self._redraw_points()

                # 7. 恢复校准数据
                if "calibration/calibration.json" in zf.namelist():
                    calib_meta = json.loads(zf.read("calibration/calibration.json"))
                    self.calibration_data = calib_meta
                    self.calib_start_img = calib_meta["point1"]
                    self.calib_end_img = calib_meta["point2"]
                    saved_display_unit = calib_meta.get("display_unit", calib_meta["unit"])
                    self.display_unit.set(saved_display_unit)
                    self._redraw_calibration_line()
                    self._on_unit_change()
                else:
                    # 项目没有校准数据，确保显示 ⚠️ 未校准
                    self.calib_info_var.set("⚠️ 未校准")
                    self._on_unit_change()

            roi_count = len(self.roi_list) if hasattr(self, 'roi_list') else 0
            point_count = len(self.click_point_list) if hasattr(self, 'click_point_list') else 0
            self._set_status(
                f"项目已打开: {os.path.basename(proj_path)} "
                f"({len(self.segmentation_results)} mask, {roi_count} ROI, {point_count} 提示点)"
            )

        except Exception as e:
            messagebox.showerror("打开失败", f"打开项目时出错:\n{e}")

    def _clear_prompts_with_confirm(self):
        """清除所有提示点、框选和 ROI（带确认弹窗）"""
        point_count = len(self.click_point_list) if hasattr(self, 'click_point_list') else 0
        roi_count = len(self.roi_list) if hasattr(self, 'roi_list') else 0
        total = point_count + roi_count
        if total == 0:
            messagebox.showinfo("提示", "没有需要清除的内容")
            return
        msg = f"确定要清除所有内容吗？\n\n• 提示点: {point_count} 个\n• 框选 ROI: {roi_count} 个"
        if not messagebox.askyesno("确认清除", msg):
            return
        self._clear_prompts()

    def _clear_prompts(self):
        """清除所有提示点、框选和 ROI（静默清除，不弹确认框）"""
        self.click_point_list = []
        self.click_point_counter = 0
        self.is_drawing_box = False
        self.box_start = None
        self.box_end = None
        self.box_start_img = None
        self.box_end_img = None
        if self.box_rect_id:
            self.canvas.delete(self.box_rect_id)
            self.box_rect_id = None
        self._clear_polygon()
        if hasattr(self, 'roi_list'):
            # 静默清除所有 ROI（不弹确认框）
            for roi in self.roi_list:
                for cid in roi.get("canvas_ids", []):
                    try:
                        self.canvas.delete(cid)
                    except Exception:
                        pass
            self.roi_list = []
            self.roi_visible = []
            self.roi_counter = 0
            self.selected_roi_idx = -1
            self._update_roi_tree()
            self._update_roi_combos()
        self._update_point_tree()
        self._redraw_points()
    # ================================================================
    #  保存结果（简化版：只保存 binary mask + overlay 图）
    # ================================================================

    def _save_selected_result(self):
        """保存选中的分割结果 — overlay + binary（支持多选）"""
        selected = self.result_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在结果列表中选择要保存的区域")
            return

        # 收集所有选中项的索引
        indices = sorted([int(s) for s in selected])

        # 检查是否有隐藏的 mask
        hidden = [idx for idx in indices if not self.mask_visible[idx]]
        if hidden:
            hidden_labels = [f"#{idx+1}" for idx in hidden]
            messagebox.showwarning("提示", f"以下 mask 已隐藏，请先设为可见再保存:\n{', '.join(hidden_labels)}")
            return

        image_dir = os.path.dirname(self.image_path) if self.image_path else os.path.expanduser("~")
        save_dir = filedialog.askdirectory(title="选择保存目录", initialdir=image_dir)
        if not save_dir:
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "result"

        saved_files = []

        # 是否需要添加面积水印（图片已校准时）
        has_calibration = self.calibration_data is not None

        for idx in indices:
            result = self.segmentation_results[idx]
            label = result["label"]
            name = result.get("name", f"Mask_{idx+1}")
            area_px = result["area"]

            # 1. 保存 overlay 图（原图 + 该 mask 叠加）
            overlay = SAM3Model.overlay_masks(self.image_np, [result], alpha=self.alpha_var.get())
            overlay_img = Image.fromarray(overlay)
            if has_calibration:
                area_text = f"Area: {self._format_calibrated_area(area_px)}"
                overlay_img = self._add_area_watermark(overlay_img, area_text)
            overlay_path = os.path.join(save_dir, f"{base_name}_{name}_overlay.png")
            overlay_img.save(overlay_path)
            saved_files.append(f"  📷 {base_name}_{name}_overlay.png")

            # 2. 保存 binary mask
            mask = result["mask"]
            binary_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            if has_calibration:
                area_text = f"Area: {self._format_calibrated_area(area_px)}"
                binary_img = self._add_area_watermark(binary_img, area_text)
            binary_path = os.path.join(save_dir, f"{base_name}_{name}_binary.png")
            binary_img.save(binary_path)
            saved_files.append(f"  ⬛ {base_name}_{name}_binary.png")

        # 3. 如果多选，额外保存合并后的 overlay（所有选中 mask 叠加在一起）
        if len(indices) > 1:
            selected_results = [self.segmentation_results[idx] for idx in indices]
            combined_overlay = SAM3Model.overlay_masks(self.image_np, selected_results, alpha=self.alpha_var.get())
            combined_overlay_img = Image.fromarray(combined_overlay)
            combined_mask = np.zeros_like(self.image_np[:, :, 0], dtype=bool)
            combined_area_px = 0
            for idx in indices:
                combined_mask |= self.segmentation_results[idx]["mask"]
                combined_area_px += self.segmentation_results[idx]["area"]
            if has_calibration:
                area_text = f"Area: {self._format_calibrated_area(combined_area_px)}"
                if area_text:
                    combined_overlay_img = self._add_area_watermark(combined_overlay_img, area_text)
            combined_path = os.path.join(save_dir, f"{base_name}_combined_overlay.png")
            combined_overlay_img.save(combined_path)
            saved_files.append(f"  🖼️ {base_name}_combined_overlay.png")

            # 合并 binary mask
            combined_binary = Image.fromarray((combined_mask.astype(np.uint8) * 255), mode="L")
            if has_calibration:
                area_text = f"Area: {self._format_calibrated_area(combined_area_px)}"
                if area_text:
                    combined_binary = self._add_area_watermark(combined_binary, area_text)
            combined_binary_path = os.path.join(save_dir, f"{base_name}_combined_binary.png")
            combined_binary.save(combined_binary_path)
            saved_files.append(f"  ⬛ {base_name}_combined_binary.png")

        messagebox.showinfo(
            "保存完成",
            f"已保存 {len(indices)} 个选中 mask 到:\n{save_dir}\n\n已保存文件:\n" + "\n".join(saved_files)
        )
        self._set_status(f"已保存 {len(indices)} 个选中 mask 到 {save_dir}")



    # ================================================================
    #  工具方法
    # ================================================================

    def _set_status(self, msg: str):
        """更新状态栏 + 输出到 log 面板"""
        self.status_var.set(msg)
        # 判断颜色标签
        if "✅" in msg or "完成" in msg:
            tag = "success"
        elif "❌" in msg or "失败" in msg or "错误" in msg:
            tag = "error"
        else:
            tag = "info"
        self._log(msg, tag)

    def _log(self, msg: str, tag: str = "info"):
        """在底部 log 面板追加一行信息

        Args:
            msg: 日志消息
            tag: 颜色标签 ("info" / "success" / "error" / "progress")
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, f"[{timestamp}] ", "time")
        self._log_text.insert(tk.END, f"{msg}\n", tag)
        self._log_text.config(state=tk.DISABLED)
        # 自动滚动到底部
        self._log_text.see(tk.END)


# ================================================================
#  启动入口（带全局异常捕获，防止闪退）
# ================================================================

def main():
    # 初始化日志
    logger = setup_logging()
    logger.info(f"=== {SAM3App.APP_TITLE} v{SAM3App.APP_VERSION} 启动 ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"工作目录: {app_base_dir()}")


    # ── Monkey-patch tk.Misc._substitute for Python 3.11 DND %# bug ──
    # cpython #94861: _substitute fails on %# (DND event serial)
    # Fixed in Python 3.12+ but not backported to 3.11
    # _substitute is on tk.Misc class (not tk module)
    if sys.version_info < (3, 12) and HAS_DND:
        _orig_substitute = tk.Misc._substitute

        def _safe_substitute(self, *args):
            try:
                return _orig_substitute(self, *args)
            except tk.TclError as exc:
                if "expected integer" in str(exc):
                    # DND event with %# — construct minimal Event
                    e = tk.Event()
                    e.serial = 0
                    e.num = 0
                    e.focus = 0
                    e.width = 0
                    e.height = 0
                    e.keycode = 0
                    e.state = 0
                    e.time = 0
                    e.x = 0
                    e.y = 0
                    e.x_root = 0
                    e.y_root = 0
                    e.char = ''
                    e.send_event = 0
                    e.type = 0
                    # Extract file path from args
                    for arg in args:
                        if isinstance(arg, str) and ('/' in arg or '\\' in arg or arg.startswith('file://')):
                            e.data = arg
                            break
                    if not hasattr(e, 'data'):
                        e.data = ''
                    return (e,)
                raise

        tk.Misc._substitute = _safe_substitute
        logger.info("已应用 tk.Misc._substitute 补丁（Python 3.11 DND 兼容）")

    try:
        if HAS_DND:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()

        # 设置主题
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")

        # macOS 原生菜单栏适配
        if sys.platform == "darwin":
            try:
                root.createcommand("::tk::mac::ShowPreferences", lambda: None)
            except Exception:
                pass

        # 全局异常处理（防止闪退）
        def handle_exception(exc_type, exc_value, exc_tb):
            """全局异常处理器 — 记录日志并显示错误对话框，而不是直接崩溃"""
            logger.error(
                f"未捕获的异常:\n"
                f"类型: {exc_type.__name__}\n"
                f"信息: {exc_value}\n"
                f"堆栈:\n{''.join(traceback.format_tb(exc_tb))}"
            )
            # 尝试显示错误对话框
            try:
                messagebox.showerror(
                    "程序错误",
                    f"发生了一个错误:\n\n{exc_type.__name__}: {exc_value}\n\n"
                    f"详细信息已记录到日志文件。\n"
                    f"日志路径: {os.path.join(app_base_dir(), 'sam3_segmenter.log')}"
                )
            except Exception:
                pass  # 如果连对话框都弹不出来，只能写日志

        sys.excepthook = handle_exception

        # Tkinter 内部异常处理
        # report_callback_exception 的回调签名是 (exc, val, tb) 三个参数
        def tk_error_handler(exc, val, tb):
            """处理 Tkinter 回调中的异常"""
            logger.error(f"Tkinter 回调异常: {val}\n{''.join(traceback.format_exception(exc, val, tb))}")

        root.report_callback_exception = tk_error_handler

        app = SAM3App(root)
        logger.info("应用初始化完成，进入主循环")
        root.mainloop()

    except Exception as e:
        logger.critical(f"启动失败: {e}\n{traceback.format_exc()}")
        # 尝试弹出错误提示
        try:
            import tkinter.messagebox as mb
            mb.showerror(
                "启动失败",
                f"应用无法启动:\n\n{e}\n\n"
                f"详细信息请查看日志:\n"
                f"{os.path.join(app_base_dir(), 'sam3_segmenter.log')}"
            )
        except Exception:
            print(f"CRITICAL: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("应用已退出")


if __name__ == "__main__":
    main()
