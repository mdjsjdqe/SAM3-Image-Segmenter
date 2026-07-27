"""
SAM3 Image Segmenter - 核心推理模块
基于 Ultralytics SAM3 实现，支持文字提示和交互式点/框提示的图像分割
支持多次分割结果累积、合并 mask 导出
"""

import os
import sys
import logging
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger("SAM3")


# ── PyInstaller 打包适配 ──
def resource_path(relative_path: str) -> str:
    """获取资源文件绝对路径（兼容 PyInstaller 打包和开发环境）"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)


class SAM3Model:
    """SAM3 模型加载与推理封装"""

    def __init__(self):
        self.model = None
        self.device = None
        self.model_path = None
        # 延迟检测设备：不在 __init__ 时 import torch（torch 加载很慢）
        # 改为在 load_model() 时才检测

    def _detect_device(self):
        """自动检测最佳计算设备"""
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if os.environ.get("SAM3_FORCE_CPU") == "1":
                logger.info("环境变量 SAM3_FORCE_CPU=1，强制使用 CPU")
                self.device = "cpu"
            else:
                self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"检测到设备: {self.device}")

    def load_model(self, model_path: str = "sam3.pt", progress_callback=None):
        """
        加载 SAM3 模型
        model_path: 模型权重文件路径（sam3.pt 或 sam3.1_hiera_l.pt 等）
        progress_callback: 回调函数 callback(step, total, message)
        """
        total_steps = 5
        try:
            logger.info(f"=== SAM3 模型加载开始 ===")
            logger.info(f"模型路径: {model_path}")
            logger.info(f"文件存在: {os.path.isfile(model_path)}")
            if os.path.isfile(model_path):
                logger.info(f"文件大小: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
            
            # Step 1: 导入 torch
            if progress_callback:
                progress_callback(1, total_steps, "导入 PyTorch...")
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"Python: {sys.executable}")

            # Step 2: 检测设备
            if progress_callback:
                progress_callback(2, total_steps, "检测计算设备...")
            if self.device is None:
                self._detect_device()
            logger.info(f"设备: {self.device}")

            # Step 3: 导入 SAM
            if progress_callback:
                progress_callback(3, total_steps, "导入 SAM 模块...")
            
            # 冻结 ultralytics 的 auto-update（frozen 环境下不能 pip install）
            if getattr(sys, 'frozen', False):
                try:
                    import ultralytics.utils.checks as _checks
                    # 禁用 checks.check_requirements 的自动安装功能
                    _checks.check_requirements = lambda *args, **kwargs: True
                    logger.info("已冻结 ultralytics auto-update")
                except Exception as e:
                    logger.warning(f"冻结 auto-update 失败: {e}")
                # 同时冻结 Ultralytics 的全局 SETTINGS
                try:
                    from ultralytics.utils import SETTINGS
                    SETTINGS['auto_update'] = False
                except Exception:
                    pass
            
            from ultralytics import SAM
            logger.info("ultralytics.SAM 导入成功")

            # Step 4: 加载模型权重
            if progress_callback:
                progress_callback(4, total_steps, "加载模型权重（可能需要几十秒）...")
            
            logger.info(f"即将加载模型，设备: {self.device}")
            
            try:
                self.model = SAM(model_path)
                logger.info("SAM() 构造成功")
            except Exception as load_err:
                logger.error(f"SAM 加载异常: {type(load_err).__name__}: {load_err}")
                import traceback as _tb
                logger.error(f"异常堆栈:\n{_tb.format_exc()}")
                if self.device == "mps":
                    logger.warning("尝试回退到 CPU...")
                    self.device = "cpu"
                    try:
                        self.model = SAM(model_path)
                        logger.info("CPU 回退加载成功")
                    except Exception as e2:
                        raise RuntimeError(
                            f"模型加载失败（MPS 错误: {load_err}，CPU 错误: {e2}）\n"
                            f"模型路径: {model_path}\n"
                            f"设备: {self.device}\n"
                            f"PyTorch: {torch.__version__}"
                        )
                raise
            self.model_path = model_path
            logger.info("模型加载成功！")

            # Step 5: 完成
            msg = f"模型加载完成 (设备: {self.device})"
            if progress_callback:
                progress_callback(5, total_steps, msg)
            return True
        except Exception as e:
            error_msg = str(e)
            if "No such file" in error_msg or "not found" in error_msg:
                error_msg = (
                    f"模型文件未找到: {model_path}\n"
                    "请先从 HuggingFace 下载 sam3.pt 并放置到正确路径。\n"
                    "下载地址: https://huggingface.co/facebook/sam3"
                )
            elif "clip" in error_msg.lower():
                error_msg = (
                    f"CLIP 依赖错误: {error_msg}\n"
                    "请运行以下命令修复:\n"
                    "  pip uninstall clip -y\n"
                    "  pip install git+https://github.com/ultralytics/CLIP.git"
                )
            print(f"[SAM3] 模型加载失败: {error_msg}")
            if progress_callback:
                progress_callback(-1, total_steps, f"加载失败: {error_msg}")
            raise RuntimeError(error_msg)

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def segment_by_text(
        self,
        image_path: str,
        texts: List[str],
        conf: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        文字提示分割（Promptable Concept Segmentation）
        输入图片路径和文字描述，返回所有匹配的分割结果

        Args:
            image_path: 图片文件路径
            texts: 文字提示列表，如 ["cat", "dog"]
            conf: 置信度阈值

        Returns:
            分割结果列表，每项包含 mask, bbox, score, label
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        print(f"[SAM3] 文字提示分割: texts={texts}, conf={conf}")
        results = self.model(image_path, texts=texts, conf=conf)

        return self._parse_results(results, prompt_type="text")

    def segment_by_points(
        self,
        image_path: str,
        points: List[List[float]],
        labels: List[int],
    ) -> List[Dict[str, Any]]:
        """
        点提示分割（Promptable Visual Segmentation - SAM2 风格）
        在图片上点击正/负提示点进行分割

        Args:
            image_path: 图片文件路径
            points: 点坐标列表 [[x1,y1], [x2,y2], ...]
            labels: 点标签列表，1=正提示(前景), 0=负提示(背景)

        Returns:
            分割结果列表
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        points_np = np.array(points)
        labels_np = np.array(labels)

        print(f"[SAM3] 点提示分割: {len(points)} 个点")
        results = self.model(image_path, points=points_np, labels=labels_np)

        return self._parse_results(results, prompt_type="point")

    def segment_by_box(
        self,
        image_path: str,
        box: List[float],
    ) -> List[Dict[str, Any]]:
        """
        框提示分割（Promptable Visual Segmentation - SAM2 风格）
        用矩形框选目标区域进行分割

        Args:
            image_path: 图片文件路径
            box: 框坐标 [x1, y1, x2, y2]

        Returns:
            分割结果列表
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        print(f"[SAM3] 框提示分割: box={box}")
        results = self.model(image_path, bboxes=[box])

        return self._parse_results(results, prompt_type="box")

    def segment_all(
        self,
        image_path: str,
    ) -> List[Dict[str, Any]]:
        """
        自动分割所有对象（无需提示）
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        print(f"[SAM3] 自动全图分割")
        results = self.model(image_path)

        return self._parse_results(results, prompt_type="auto")

    def _parse_results(
        self, results, prompt_type: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        解析 Ultralytics SAM 推理结果为统一格式

        Returns:
            [{
                "mask": np.ndarray (H, W) bool,
                "bbox": [x1, y1, x2, y2],
                "score": float,
                "label": str,
                "area": int,
                "prompt_type": str,
            }, ...]
        """
        parsed = []

        if results is None or len(results) == 0:
            print("[SAM3] 无分割结果")
            return parsed

        result = results[0]  # 取第一张图的结果

        if result.masks is None:
            print("[SAM3] 无 mask 输出")
            return parsed

        masks_data = result.masks.data  # (N, H, W)
        boxes_data = result.boxes

        num_masks = masks_data.shape[0]
        print(f"[SAM3] 检测到 {num_masks} 个分割区域")

        for i in range(num_masks):
            mask = masks_data[i].cpu().numpy().astype(bool)

            # 获取边界框
            if boxes_data is not None and i < len(boxes_data):
                bbox = boxes_data[i].xyxy[0].cpu().numpy().tolist()
                score = float(boxes_data[i].conf[0].cpu().numpy())
                cls_id = int(boxes_data[i].cls[0].cpu().numpy())
                # 尝试获取类别名
                if result.names and cls_id in result.names:
                    label = result.names[cls_id]
                else:
                    label = f"object_{i+1}"
            else:
                # 从 mask 计算 bbox
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox = [float(cmin), float(rmin), float(cmax), float(rmax)]
                score = 1.0
                label = f"object_{i+1}"

            area = int(mask.sum())

            parsed.append({
                "mask": mask,
                "bbox": bbox,
                "score": score,
                "label": label,
                "area": area,
                "prompt_type": prompt_type,
            })

        # 按面积降序排列
        parsed.sort(key=lambda x: x["area"], reverse=True)
        return parsed

    @staticmethod
    def combine_masks(masks: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        合并所有 mask 为一个 combined mask

        Args:
            masks: 分割结果列表
            image_shape: (H, W) 原图尺寸

        Returns:
            combined_mask: np.ndarray (H, W) uint8
                0 = 背景
                1, 2, 3... = 各个 mask 的编号（1-indexed）
        """
        h, w = image_shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)

        for i, m in enumerate(masks):
            mask = m["mask"]
            # 后添加的 mask 覆盖前面的（也可改为只填充空白区域）
            combined[mask] = i + 1  # 1-indexed

        return combined

    @staticmethod
    def overlay_masks(
        image: np.ndarray,
        masks: List[Dict[str, Any]],
        alpha: float = 0.5,
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        visible_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        将分割 mask 叠加到原图上

        Args:
            image: 原图 (H, W, 3) RGB
            masks: 分割结果列表
            alpha: 叠加透明度
            color_map: 自定义颜色映射 {index: (R, G, B)}
            visible_indices: 仅显示指定索引的 mask，None 表示全部显示

        Returns:
            叠加后的图像 (H, W, 3) RGB
        """
        if not masks:
            return image.copy()

        overlay = image.copy()

        # 默认颜色调色板（高对比度）
        default_colors = [
            (255, 0, 0),     # 红
            (0, 255, 0),     # 绿
            (0, 0, 255),     # 蓝
            (255, 255, 0),   # 黄
            (255, 0, 255),   # 品红
            (0, 255, 255),   # 青
            (255, 128, 0),   # 橙
            (128, 0, 255),   # 紫
            (0, 128, 255),   # 天蓝
            (255, 128, 128), # 粉
        ]

        for i, m in enumerate(masks):
            # 如果指定了可见索引，跳过不可见的
            if visible_indices is not None and i not in visible_indices:
                continue

            if color_map and i in color_map:
                color = color_map[i]
            else:
                color = default_colors[i % len(default_colors)]

            mask = m["mask"]
            # 叠加颜色
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask,
                    (1 - alpha) * overlay[:, :, c] + alpha * color[c],
                    overlay[:, :, c],
                )

        return overlay.astype(np.uint8)

    @staticmethod
    def save_mask(mask: np.ndarray, save_path: str):
        """保存单个 mask 为黑白图片"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(save_path)
        print(f"[SAM3] Mask 已保存: {save_path}")

    @staticmethod
    def save_combined_mask(combined: np.ndarray, save_path: str):
        """
        保存 combined mask
        combined 中 0=背景, 1,2,3...=各实例编号
        保存为彩色图以便区分不同实例
        """
        # 生成彩色 combined mask
        h, w = combined.shape
        color_combined = np.zeros((h, w, 3), dtype=np.uint8)

        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 128, 255),
            (255, 128, 128), (128, 255, 0), (0, 128, 128),
        ]

        for i in range(1, combined.max() + 1):
            color = palette[(i - 1) % len(palette)]
            for c in range(3):
                color_combined[:, :, c] = np.where(
                    combined == i, color[c], color_combined[:, :, c]
                )

        img = Image.fromarray(color_combined)
        img.save(save_path)
        print(f"[SAM3] Combined mask 已保存: {save_path}")

    @staticmethod
    def save_combined_mask_binary(combined: np.ndarray, save_path: str):
        """
        保存 combined mask 为二值图（所有实例合并为白色）
        """
        binary = (combined > 0).astype(np.uint8) * 255
        img = Image.fromarray(binary)
        img.save(save_path)
        print(f"[SAM3] Combined binary mask 已保存: {save_path}")

    @staticmethod
    def save_combined_mask_instances(combined: np.ndarray, save_path: str):
        """
        保存 combined mask 为实例编号灰度图
        每个实例有不同的灰度值（1, 2, 3...），方便后续程序读取
        """
        img = Image.fromarray(combined)
        img.save(save_path)
        print(f"[SAM3] Combined instances mask 已保存: {save_path}")

    @staticmethod
    def save_overlay(image: np.ndarray, overlay: np.ndarray, save_path: str):
        """保存叠加结果"""
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(save_path)
        print(f"[SAM3] 叠加图已保存: {save_path}")

    @staticmethod
    def mask_and(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """
        两个 mask 的交集（AND 运算）
        只保留两个 mask 都为 True 的区域

        Args:
            mask_a: 第一个 mask (H, W) bool
            mask_b: 第二个 mask (H, W) bool

        Returns:
            交集 mask (H, W) bool
        """
        return np.logical_and(mask_a, mask_b)

    @staticmethod
    def mask_or(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """
        两个 mask 的并集（OR 运算）
        保留两个 mask 中任意一个为 True 的区域

        Args:
            mask_a: 第一个 mask (H, W) bool
            mask_b: 第二个 mask (H, W) bool

        Returns:
            并集 mask (H, W) bool
        """
        return np.logical_or(mask_a, mask_b)

    @staticmethod
    def mask_sub(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """
        从 mask_a 中减去 mask_b（Minus / 差集运算）
        保留 mask_a 中不被 mask_b 覆盖的区域

        Args:
            mask_a: 被减 mask (H, W) bool
            mask_b: 要减去的 mask (H, W) bool

        Returns:
            差集 mask (H, W) bool = mask_a AND (NOT mask_b)
        """
        return np.logical_and(mask_a, np.logical_not(mask_b))

    @staticmethod
    def compute_mask_operation(
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """
        执行 mask 布尔运算

        Args:
            mask_a: 第一个 mask (H, W) bool
            mask_b: 第二个 mask (H, W) bool
            operation: 运算类型 "Union" / "Intersection" / "Minus"

        Returns:
            运算结果 mask (H, W) bool
        """
        ops = {
            "Union": SAM3Model.mask_or,
            "Intersection": SAM3Model.mask_and,
            "Minus": SAM3Model.mask_sub,
        }
        if operation not in ops:
            raise ValueError(f"不支持的运算: {operation}，可选: {list(ops.keys())}")
        return ops[operation](mask_a, mask_b)

    @staticmethod
    def _combined_to_color_preview(combined: np.ndarray) -> np.ndarray:
        """
        将 combined mask 转为彩色预览图（用于内部预览，不保存）
        """
        h, w = combined.shape
        color_combined = np.zeros((h, w, 3), dtype=np.uint8)

        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 128, 255),
            (255, 128, 128), (128, 255, 0), (0, 128, 128),
        ]

        for i in range(1, combined.max() + 1):
            color = palette[(i - 1) % len(palette)]
            for c in range(3):
                color_combined[:, :, c] = np.where(
                    combined == i, color[c], color_combined[:, :, c]
                )

        return color_combined
