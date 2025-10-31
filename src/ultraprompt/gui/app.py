# src/ultraprompt/gui/app.py
#!/usr/bin/env python3
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import json

# ---- import the core module (no predictor usage inside) ----
from ultraprompt.core.sam_yolo_annotation import (
    UltraSAM2, load_image_rgb, colorize_masks_rgba, mask_to_polygon, write_yolo_seg
)

#from PyQt5.QtCore import Qt, QRectF, QPointF, QEvent
#from PyQt5.QtGui import QPixmap, QPen, QBrush, QImage, QColor
#from PyQt5.QtWidgets import (
#    QApplication, QMainWindow, QFileDialog, QAction, QToolBar, QLabel,
#    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
#    QGraphicsRectItem, QMessageBox, QStatusBar, QComboBox
#)
from PySide6.QtCore import Qt, QRectF, QPointF, QEvent
from PySide6.QtGui import QAction, QPixmap, QPainter, QPen, QBrush, QImage, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QToolBar, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
    QGraphicsRectItem, QMessageBox, QStatusBar, QComboBox
)


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def clamp(v, a, b): return max(a, min(b, v))


def np_to_qimage_rgba(img_rgba: np.ndarray) -> QImage:
    h, w, c = img_rgba.shape
    assert c == 4 and img_rgba.dtype == np.uint8
    return QImage(img_rgba.data, w, h, 4*w, QImage.Format_RGBA8888)


class GraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHints(self.renderHints() | self.renderHints())
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._hand_drag_active = False

    def wheelEvent(self, event):
        zf = 1.25 if event.angleDelta().y() > 0 else 1/1.25
        old = self.mapToScene(event.pos())
        self.scale(zf, zf)
        new = self.mapToScene(event.pos())
        delta = new - old
        self.translate(delta.x(), delta.y())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not self._hand_drag_active:
            self._hand_drag_active = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and self._hand_drag_active:
            self._hand_drag_active = False
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(event)


class SamPromptAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Prompt Annotator (Ultralytics)")
        self.resize(1280, 860)

        # dataset state
        self.image_dir: Optional[Path] = None
        self.out_dir: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.idx: int = -1

        # UI state
        self.mode: str = "points"  # or "boxes"
        self.scene = QGraphicsScene(self)
        self.view = GraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.seg_item: Optional[QGraphicsPixmapItem] = None

        # annotations for current image
        self.point_items: List[QGraphicsEllipseItem] = []
        self.point_data: List[Tuple[float, float, int]] = []  # (x,y,label: 1 fg, 0 bg)
        self.box_items: List[QGraphicsRectItem] = []
        self.box_data: List[Tuple[float, float, float, float]] = []  # (x0,y0,x1,y1)
        self.box_classes: List[int] = []
        self._drawing_box = False
        self._box_start_scene: Optional[QPointF] = None
        self._current_box_item: Optional[QGraphicsRectItem] = None

        # SAM2 core wrapper
        self.sam = UltraSAM2()
        self.sam2_weights: Optional[Path] = None
        self.device_pref: str = "Auto"  # Auto / CUDA / CPU

        # Classes + export state
        self.class_names: List[str] = ["object"]
        self.current_class_id: int = 0
        self.last_masks: List[np.ndarray] = []
        self.last_mask_classes: List[int] = []

        self._build_ui()
        self._install_event_filters()
        self._update_status()

    # ---------- UI ----------
    def _build_ui(self):
        tb = QToolBar("Main", self); tb.setMovable(False); self.addToolBar(tb)

        a_open = QAction("Open Folder…", self); a_open.triggered.connect(self.open_folder); tb.addAction(a_open)
        a_out  = QAction("Set Output…", self); a_out.triggered.connect(self.set_output_dir); tb.addAction(a_out)

        tb.addSeparator()
        self.act_prev = QAction("Prev", self); self.act_prev.triggered.connect(self.prev_image); tb.addAction(self.act_prev)
        self.act_next = QAction("Next", self); self.act_next.triggered.connect(self.next_image); tb.addAction(self.act_next)

        tb.addSeparator()
        self.act_mode_points = QAction("Points Mode", self, checkable=True)
        self.act_mode_boxes  = QAction("Boxes Mode",  self, checkable=True)
        self.act_mode_points.setChecked(True)
        self.act_mode_points.triggered.connect(lambda: self.set_mode("points"))
        self.act_mode_boxes.triggered.connect(lambda: self.set_mode("boxes"))
        tb.addAction(self.act_mode_points); tb.addAction(self.act_mode_boxes)

        tb.addSeparator()
        a_undo = QAction("Undo", self); a_undo.triggered.connect(self.undo); tb.addAction(a_undo)
        a_clrp = QAction("Clear Points", self); a_clrp.triggered.connect(self.clear_points); tb.addAction(a_clrp)
        a_clrb = QAction("Clear Boxes",  self); a_clrb.triggered.connect(self.clear_boxes);  tb.addAction(a_clrb)

        tb.addSeparator()
        a_save = QAction("Save JSON", self); a_save.triggered.connect(self.save_current_json); tb.addAction(a_save)
        a_exp  = QAction("Export All", self); a_exp.triggered.connect(self.export_all_json); tb.addAction(a_exp)

        tb.addSeparator()
        a_w = QAction("Load SAM2 Weights…", self); a_w.triggered.connect(self.load_sam2_weights); tb.addAction(a_w)

        self.cmb_device = QComboBox(self)
        self.cmb_device.addItems(["Auto", "CUDA", "CPU"])
        self.cmb_device.currentTextChanged.connect(self._on_device_changed)
        tb.addWidget(QLabel(" Device: ")); tb.addWidget(self.cmb_device)

        # --- Classes UI ---
        self.cmb_class = QComboBox(self)
        self.cmb_class.addItems(self.class_names)
        self.cmb_class.currentIndexChanged.connect(self._on_class_changed)
        tb.addWidget(QLabel(" Class: "))
        tb.addWidget(self.cmb_class)

        act_load_classes = QAction("Load Classes…", self)
        act_load_classes.triggered.connect(self.load_classes_txt)
        tb.addAction(act_load_classes)

        # Export labels
        act_save_yolo = QAction("Save YOLO Labels", self)
        act_save_yolo.triggered.connect(self.save_yolo_seg_labels)
        tb.addAction(act_save_yolo)

        a_run = QAction("Run SAM2", self); a_run.triggered.connect(self.run_sam2); tb.addAction(a_run)
        a_clrseg = QAction("Clear Segmentation", self); a_clrseg.triggered.connect(self.clear_segmentation); tb.addAction(a_clrseg)

        self.status = QStatusBar(self); self.setStatusBar(self.status)
        self.info = QLabel(""); self.status.addWidget(self.info, 1)

    def _on_class_changed(self, idx: int):
        self.current_class_id = int(idx)
        self._update_status(f"class={self.class_names[idx]} ({idx})")

    def load_classes_txt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select classes.txt", "", "Text files (*.txt);;All files (*)")
        if not f:
            return
        try:
            names = [ln.strip() for ln in open(f, "r", encoding="utf-8").read().splitlines() if ln.strip()]
            if not names:
                QMessageBox.warning(self, "Empty classes.txt", "No class names found.")
                return
            self.class_names = names
            self.cmb_class.clear()
            self.cmb_class.addItems(self.class_names)
            self.current_class_id = 0
            self.cmb_class.setCurrentIndex(0)
            self._update_status(f"Loaded {len(self.class_names)} classes")
        except Exception as e:
            QMessageBox.critical(self, "classes.txt error", str(e))

    def _install_event_filters(self):
        self.view.viewport().installEventFilter(self)

    def _update_status(self, extra: str = ""):
        parts = []
        parts.append(f"Mode: {self.mode.upper()}")
        if self.image_paths: parts.append(f"Image {self.idx+1}/{len(self.image_paths)}")
        if self.sam2_weights: parts.append(f"SAM2: {self.sam2_weights.name} @ {self._effective_device()}")
        if extra: parts.append(f"— {extra}")
        self.info.setText(" | ".join(parts))

    # ---------- File ops ----------
    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not d: return
        self.image_dir = Path(d)
        self.image_paths = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "No images found in that folder."); return
        self.idx = 0
        self.load_image()
        self._update_status("Folder loaded")

    def set_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not d: return
        self.out_dir = Path(d)
        self._update_status(f"Output → {self.out_dir}")

    def current_image_path(self) -> Optional[Path]:
        if 0 <= self.idx < len(self.image_paths):
            return self.image_paths[self.idx]
        return None

    # ---------- Image/Scene ----------
    def load_image(self):
        self.scene.clear(); self.seg_item = None
        self.point_items.clear(); self.point_data.clear()
        self.box_items.clear();   self.box_data.clear(); self.box_classes.clear()

        ip = self.current_image_path()
        if not ip: return

        pm = QPixmap(str(ip))
        if pm.isNull():
            QMessageBox.critical(self, "Load error", f"Failed to load: {ip}"); return

        self.pixmap_item = QGraphicsPixmapItem(pm); self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)
        self.view.resetTransform()
        self.scene.setSceneRect(QRectF(pm.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # load prior annotation if present
        if self.out_dir:
            js = self.out_dir / f"{ip.stem}.json"
            if js.exists():
                try:
                    with open(js, "r", encoding="utf-8") as f: data = json.load(f)
                    self._load_from_json(data)
                except Exception as e:
                    print(f"Failed to parse {js}: {e}")

        self._update_status(f"{ip.name} loaded")

    def _load_from_json(self, data: dict):
        W = self.pixmap_item.pixmap().width(); H = self.pixmap_item.pixmap().height()
        # points
        for (x, y), lab in zip(data.get("point_coords", []), data.get("point_labels", [])):
            if 0 <= x < W and 0 <= y < H:
                self._add_point_visual(QPointF(x, y), int(lab))
                self.point_data.append((float(x), float(y), int(lab)))
        # boxes + classes
        boxes = data.get("boxes", [])
        box_classes = data.get("box_classes", [])
        for i, b in enumerate(boxes):
            if len(b) != 4: continue
            x0,y0,x1,y1 = b
            rect = QRectF(QPointF(max(0,min(x0,W-1)), max(0,min(y0,H-1))),
                          QPointF(max(0,min(x1,W-1)), max(0,min(y1,H-1)))).normalized()
            cls_id = int(box_classes[i]) if i < len(box_classes) else 0
            item = self._make_box_item(rect, cls_id)
            self.scene.addItem(item)
            self.box_items.append(item)
            self.box_data.append((rect.left(), rect.top(), rect.right(), rect.bottom()))
            self.box_classes.append(cls_id)

    # ---------- Navigate ----------
    def prev_image(self):
        if not self.image_paths: return
        self.idx = (self.idx - 1) % len(self.image_paths)
        self.load_image()

    def next_image(self):
        if not self.image_paths: return
        self.idx = (self.idx + 1) % len(self.image_paths)
        self.load_image()

    # ---------- Modes & Events ----------
    def set_mode(self, mode: str):
        self.mode = mode
        self.act_mode_points.setChecked(mode == "points")
        self.act_mode_boxes.setChecked(mode == "boxes")
        self._update_status()

    def eventFilter(self, obj, event):
        if obj is self.view.viewport():
            if self.view.dragMode() == QGraphicsView.ScrollHandDrag:
                return False
            if event.type() == QEvent.MouseButtonPress:
                return self._on_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                return self._on_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self._on_mouse_release(event)
        return super().eventFilter(obj, event)

    def _view_to_image_xy(self, pos) -> Tuple[float, float]:
        if not self.pixmap_item: return (None, None)
        scene_pt = self.view.mapToScene(pos)
        W = self.pixmap_item.pixmap().width(); H = self.pixmap_item.pixmap().height()
        x = max(0, min(scene_pt.x(), W-1)); y = max(0, min(scene_pt.y(), H-1))
        return x, y

    def _on_mouse_press(self, event):
        if not self.pixmap_item: return False
        if self.mode == "points":
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                x, y = self._view_to_image_xy(event.pos())
                lab = 1 if event.button() == Qt.LeftButton else 0  # 1=fg, 0=bg
                self.point_data.append((x, y, lab))
                self._add_point_visual(QPointF(x, y), lab)
                self._update_status(f"Point ({int(x)},{int(y)}) label={lab}")
                return True
        elif self.mode == "boxes" and event.button() == Qt.LeftButton:
            self._drawing_box = True
            self._box_start_scene = self.view.mapToScene(event.pos())
            self._current_box_item = self._make_box_item(QRectF(self._box_start_scene, self._box_start_scene),
                                                         self.current_class_id)
            self.scene.addItem(self._current_box_item)
            return True
        return False

    def _on_mouse_move(self, event):
        if self.mode == "boxes" and self._drawing_box and self._current_box_item is not None:
            now = self.view.mapToScene(event.pos())
            rect = QRectF(self._box_start_scene, now).normalized()
            rect = rect.intersected(self.pixmap_item.boundingRect())
            self._current_box_item.setRect(rect)
            return True
        return False

    def _on_mouse_release(self, event):
        if self.mode == "boxes" and self._drawing_box and event.button() == Qt.LeftButton:
            self._drawing_box = False
            if self._current_box_item:
                r = self._current_box_item.rect().normalized()
                if r.width() >= 2 and r.height() >= 2:
                    cls_id = int(self.current_class_id)
                    self._apply_box_style(self._current_box_item, cls_id)
                    self.box_items.append(self._current_box_item)
                    self.box_data.append((r.left(), r.top(), r.right(), r.bottom()))
                    self.box_classes.append(cls_id)
                    self._update_status(
                        f"Box: ({int(r.left())},{int(r.top())})-({int(r.right())},{int(r.bottom())}) class={self.class_names[cls_id]}"
                    )
                else:
                    self.scene.removeItem(self._current_box_item)
                self._current_box_item = None
            return True
        return False

    def _add_point_visual(self, pt: QPointF, lab: int):
        r = 1
        color = Qt.green if lab == 1 else Qt.red
        pen = QPen(color); pen.setWidth(1)
        item = QGraphicsEllipseItem(pt.x()-r, pt.y()-r, 2*r, 2*r)
        item.setPen(pen); item.setBrush(QBrush(color)); item.setZValue(10)
        self.scene.addItem(item)
        self.point_items.append(item)

    # ---------- Box styling ----------
    def _class_qcolor(self, cls_id: int) -> QColor:
        rng = np.random.default_rng(12345 + int(cls_id))
        r, g, b = [int(x) for x in rng.integers(60, 230, size=3)]
        return QColor(r, g, b)

    def _apply_box_style(self, item: QGraphicsRectItem, cls_id: int):
        pen = QPen(self._class_qcolor(cls_id))
        pen.setWidth(2)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.transparent))
        item.setZValue(5)

    def _make_box_item(self, rect: QRectF, cls_id: int) -> QGraphicsRectItem:
        item = QGraphicsRectItem(rect)
        self._apply_box_style(item, cls_id)
        return item

    # ---------- Edit ----------
    def undo(self):
        if self.mode == "points" and self.point_items:
            it = self.point_items.pop()
            self.scene.removeItem(it)
            self.point_data.pop()
            self._update_status("Undid point")
        elif self.mode == "boxes" and self.box_items:
            it = self.box_items.pop()
            self.scene.removeItem(it)
            if self.box_data: self.box_data.pop()
            if self.box_classes: self.box_classes.pop()
            self._update_status("Undid box")

    def clear_points(self):
        for it in self.point_items: self.scene.removeItem(it)
        self.point_items.clear(); self.point_data.clear()
        self._update_status("Cleared points")

    def clear_boxes(self):
        for it in self.box_items: self.scene.removeItem(it)
        self.box_items.clear(); self.box_data.clear(); self.box_classes.clear()
        self._update_status("Cleared boxes")

    def clear_segmentation(self):
        if self.seg_item is not None:
            self.scene.removeItem(self.seg_item)
            self.seg_item = None
            self._update_status("Cleared segmentation")

    # ---------- Save/Export ----------
    def _annotation_dict(self) -> dict:
        ip = self.current_image_path()
        if not ip or not self.pixmap_item: return {}
        W = self.pixmap_item.pixmap().width(); H = self.pixmap_item.pixmap().height()
        pts  = [[float(x), float(y)] for (x,y,_) in self.point_data]
        labs = [int(l) for (_,_,l) in self.point_data]
        boxes = [[float(x0),float(y0),float(x1),float(y1)] for (x0,y0,x1,y1) in self.box_data]
        return {
            "image": ip.name, "image_size": [int(W), int(H)],
            "point_coords": pts, "point_labels": labs,
            "boxes": boxes,
            "box_classes": [int(c) for c in self.box_classes],
            "class_names": self.class_names
        }

    def save_current_json(self):
        ip = self.current_image_path()
        if not ip: return
        if not self.out_dir:
            QMessageBox.warning(self, "Output", "Set an output folder first."); return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        outp = self.out_dir / f"{ip.stem}.json"
        with open(outp, "w", encoding="utf-8") as f: json.dump(self._annotation_dict(), f, indent=2)
        self._update_status(f"Saved {outp.name}")

    def export_all_json(self):
        if not self.out_dir:
            QMessageBox.warning(self, "Output", "Set an output folder first."); return
        if not self.image_paths: return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.pixmap_item: self.save_current_json()
        saved = 0
        for i, p in enumerate(self.image_paths):
            jp = self.out_dir / f"{p.stem}.json"
            if jp.exists(): saved += 1; continue
            pm = QPixmap(str(p))
            data = {
                "image": p.name,
                "image_size": [pm.width(), pm.height()],
                "point_coords": [], "point_labels": [],
                "boxes": [], "box_classes": [],
                "class_names": self.class_names
            }
            with open(jp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            saved += 1
        self._update_status(f"Exported JSON for {saved} images")

    # ---------- YOLO export helpers ----------
    def save_yolo_seg_labels(self):
        ip = self.current_image_path()
        if not ip or not self.pixmap_item:
            QMessageBox.information(self, "No image", "Load an image first."); return
        if not self.last_masks:
            QMessageBox.information(self, "No masks", "Run SAM2 first to generate masks."); return

        labels_dir = (self.out_dir / "labels") if self.out_dir else None
        if labels_dir is None:
            d = QFileDialog.getExistingDirectory(self, "Select labels output folder")
            if not d: return
            labels_dir = Path(d)
        labels_dir.mkdir(parents=True, exist_ok=True)

        W = self.pixmap_item.pixmap().width()
        H = self.pixmap_item.pixmap().height()
        label_path = labels_dir / f"{ip.stem}.txt"

        polys = [mask_to_polygon(m, simplify_eps=2.0) for m in self.last_masks]

        if self.last_mask_classes and len(self.last_mask_classes) == len(polys):
            class_ids = [int(c) for c in self.last_mask_classes]
        else:
            class_ids = [self.current_class_id] * len(polys)

        write_yolo_seg(label_path, polys, class_ids, W, H)

        try:
            if self.out_dir:
                json_path = self.out_dir / f"{ip.stem}.json"
                data = self._annotation_dict()
                data["pred_instances"] = []
                for cls_id, poly in zip(class_ids, polys):
                    if poly is None: continue
                    data["pred_instances"].append({
                        "class_id": int(cls_id),
                        "polygon": [[float(x), float(y)] for x, y in poly]
                    })
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print("Note: failed to write pred_instances into JSON:", e)

        self._update_status(f"Saved YOLO labels → {label_path}")

    # ---------- SAM2 (Ultralytics) ----------
    def _effective_device(self) -> str:
        # Use the core wrapper device for display; GUI keeps a pref
        try:
            import torch
            if self.device_pref == "CUDA":
                return "cuda"
            if self.device_pref == "CPU":
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _on_device_changed(self, txt: str):
        self.device_pref = txt
        self._update_status("Device set")

    def load_sam2_weights(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select SAM2 weights (.pt)", "", "Model files (*.pt)")
        if not f:
            return
        self.sam2_weights = Path(f)
        try:
            self.sam.load(self.sam2_weights, device=self.device_pref.lower())
            self._update_status(f"Loaded {self.sam2_weights.name}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self.sam2_weights = None

    def run_sam2(self):
        ip = self.current_image_path()
        if not ip:
            return
        try:
            image = load_image_rgb(ip)
        except Exception as e:
            QMessageBox.critical(self, "Image load error", str(e)); return

        masks_to_draw: List[np.ndarray] = []
        mask_classes: List[int] = []
        try:
            # Bind image once (no predictor usage)
            self.sam.bind_image(image)

            has_points = len(self.point_data) > 0
            has_boxes  = len(self.box_data) > 0

            pts  = [[x, y] for (x, y, _) in self.point_data] if has_points else None
            labs = [int(l) for (_, _, l) in self.point_data] if has_points else None

            if has_boxes:
                masks_to_draw = self.sam.infer(points=pts, labels=labs, boxes=self.box_data, multimask_output=True)
                mask_classes  = list(self.box_classes)
            elif has_points:
                masks_to_draw = self.sam.infer(points=pts, labels=labs, boxes=None, multimask_output=True)
                mask_classes  = [int(self.current_class_id)] * len(masks_to_draw)
            else:
                masks_to_draw = self.sam.segment_everything(image, top_n=20)
                mask_classes  = [int(self.current_class_id)] * len(masks_to_draw)

            if not masks_to_draw:
                QMessageBox.information(self, "No masks", "SAM2 returned no masks."); return

            overlay = colorize_masks_rgba(masks_to_draw, alpha=0.45)
            qimg = np_to_qimage_rgba(overlay)
            pm = QPixmap.fromImage(qimg)
            if self.seg_item is not None:
                self.scene.removeItem(self.seg_item); self.seg_item = None
            self.seg_item = QGraphicsPixmapItem(pm)
            self.seg_item.setZValue(7.5)
            self.scene.addItem(self.seg_item)
            self.last_masks = masks_to_draw
            self.last_mask_classes = mask_classes
            self._update_status("Segmentation overlay updated (SAM2).")

        except Exception as e:
            QMessageBox.critical(self, "SAM2 error", str(e))


def main():
    app = QApplication(sys.argv)
    win = SamPromptAnnotator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
