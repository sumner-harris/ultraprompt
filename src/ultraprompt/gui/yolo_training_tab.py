# src/ultraprompt/gui/yolo_training_tab.py
"""
YOLO Training Tab for ultraprompt.
PySide6 — universal, works on both Windows PC and Linux VM.

Platform differences handled automatically:
  Windows PC : workers=0, CPU default, UTF-8 encoding forced
  Linux VM   : workers=4, GPU auto-detected

Features:
  - Dataset builder with train/val/test split
  - Device selection (CPU / GPU auto-detected)
  - Model weights: auto-download from Ultralytics OR browse custom .pt
  - Live training output in coloured log terminal
  - Live loss + mAP plots updated every 30s
  - Confusion matrix display after training
  - Metrics summary table
  - Load existing results.csv for offline review
"""

import os
import sys
import platform
import shutil
import random
import threading
import subprocess
from pathlib import Path

from PySide6.QtCore    import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QSplitter, QTabWidget,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QScrollArea, QSpinBox,
)
from PySide6.QtGui import QFont

try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    import matplotlib.image as mpimg
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Platform detection ─────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX   = platform.system() == 'Linux'


# ══════════════════════════════════════════════════════════════════════════════
# Signals — allow background threads to update GUI safely
# ══════════════════════════════════════════════════════════════════════════════
class TrainingSignals(QObject):
    update_log     = Signal(str)
    update_epoch   = Signal(str)
    training_done  = Signal()
    training_error = Signal(str)
    dataset_done   = Signal(str)
    dataset_error  = Signal(str)


# ══════════════════════════════════════════════════════════════════════════════
# Main YOLO Training Tab
# ══════════════════════════════════════════════════════════════════════════════
class YoloTrainingTab(QWidget):

    MODEL_OPTIONS = [
        'yolo26n-seg', 'yolo26s-seg', 'yolo26m-seg',
        'yolo26l-seg', 'yolo26x-seg',
        'yolov8n-seg',  'yolov8m-seg',  'yolov8x-seg',
    ]

    # Best defaults from Ray Tune (yolo26n, 150 trials, RHEED dataset)
    # workers auto-set to 0 on Windows, 4 on Linux
    DEFAULTS = {
        'epochs'     : '200',
        'imgsz'      : '1280',
        'batch'      : '8',
        'patience'   : '50',
        'cls'        : '0.391',
        'conf'       : '0.276',
        'dropout'    : '0.248',
        'mask_ratio' : '4',
        'lr0'        : '0.000503',
        'box'        : '11.779',
        'workers'    : '0' if IS_WINDOWS else '4',
        'seed'       : '42',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process      = None
        self._training     = False
        self._results_path = None
        self._signals      = TrainingSignals()
        self._timer        = QTimer(self)

        self._signals.update_log.connect(self._append_log)
        self._signals.update_epoch.connect(self._update_epoch_label)
        self._signals.training_done.connect(self._on_training_complete)
        self._signals.training_error.connect(self._on_training_error)
        self._signals.dataset_done.connect(self._on_dataset_done)
        self._signals.dataset_error.connect(self._on_dataset_error)
        self._timer.timeout.connect(self._refresh_plots)

        self._build_ui()
        self._detect_devices()

    # ══════════════════════════════════════════════════════════════════════
    # PLATFORM HELPERS
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _platform_label():
        return 'Windows PC' if IS_WINDOWS else 'Linux VM'

    @staticmethod
    def _safe_workers(requested):
        """Force workers=0 on Windows to prevent multiprocessing crash."""
        return 0 if IS_WINDOWS else requested

    # ══════════════════════════════════════════════════════════════════════
    # DEVICE DETECTION
    # ══════════════════════════════════════════════════════════════════════
    def _detect_devices(self):
        devices = ['cpu']
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    devices.append(f'cuda:{i}  ({name})')
        except Exception:
            pass
        self._device_combo.clear()
        self._device_combo.addItems(devices)
        self._device_combo.setCurrentIndex(0)
        if len(devices) > 1:
            self._device_status.setText(
                f'GPU detected: {devices[1]}  [{self._platform_label()}]')
            self._device_status.setStyleSheet('color: green;')
        else:
            self._device_status.setText(
                f'No GPU — CPU mode  [{self._platform_label()}]')
            self._device_status.setStyleSheet('color: gray;')

    def _get_device(self):
        raw = self._device_combo.currentText().strip()
        if raw.startswith('cuda:'):
            return raw.split()[0].replace('cuda:', '')
        return 'cpu'

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400)
        left = QWidget()
        self._left_layout = QVBoxLayout(left)
        self._left_layout.setSpacing(6)
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._build_dataset_section()
        self._build_device_section()
        self._build_model_section()
        self._build_hyperparams_section()
        self._build_flags_section()
        self._build_buttons_section()
        self._left_layout.addStretch()
        self._build_plots(right_layout)

    def _group(self, title):
        gb = QGroupBox(title)
        gb.setStyleSheet('QGroupBox { font-weight: bold; }')
        return gb

    # ── DATASET BUILDER ───────────────────────────────────────────────────
    def _build_dataset_section(self):
        gb  = self._group('Dataset Builder')
        lay = QGridLayout(gb)
        lay.setSpacing(4)
        row = 0

        for label, attr, placeholder, browse_fn in [
            ('Images folder:', '_img_folder_edit',
             'Folder containing all images...',
             self._browse_img_folder),
            ('Labels folder:', '_lbl_folder_edit',
             'Folder containing YOLO .txt labels...',
             self._browse_lbl_folder),
            ('Dataset output:', '_dataset_out_edit',
             'Where to create the dataset folder...',
             self._browse_dataset_out),
        ]:
            lay.addWidget(QLabel(label), row, 0)
            edit = QLineEdit('')
            edit.setPlaceholderText(placeholder)
            setattr(self, attr, edit)
            lay.addWidget(edit, row, 1)
            btn = QPushButton('Browse')
            btn.setFixedWidth(60)
            btn.clicked.connect(browse_fn)
            lay.addWidget(btn, row, 2)
            row += 1

        lay.addWidget(QLabel('Class names:'), row, 0)
        self._classes_edit = QLineEdit(
            'direct_beam, spot, streak, kikuchi')
        self._classes_edit.setPlaceholderText(
            'comma separated: dog, cat, ...')
        lay.addWidget(self._classes_edit, row, 1, 1, 2)
        row += 1

        # Split ratios
        ratio_row = QHBoxLayout()
        self._train_ratio = QSpinBox()
        self._train_ratio.setRange(50, 90)
        self._train_ratio.setValue(80)
        self._train_ratio.setSuffix('%')
        self._val_ratio = QSpinBox()
        self._val_ratio.setRange(5, 30)
        self._val_ratio.setValue(10)
        self._val_ratio.setSuffix('%')
        self._test_ratio_label = QLabel('10%')
        self._test_ratio_label.setStyleSheet('color: gray;')
        self._train_ratio.valueChanged.connect(self._update_test_label)
        self._val_ratio.valueChanged.connect(self._update_test_label)
        for w in [QLabel('Train:'), self._train_ratio,
                  QLabel('Val:'),   self._val_ratio,
                  QLabel('Test:'),  self._test_ratio_label]:
            ratio_row.addWidget(w)
        ratio_row.addStretch()
        lay.addWidget(QLabel('Split ratios:'), row, 0)
        rw = QWidget()
        rw.setLayout(ratio_row)
        lay.addWidget(rw, row, 1, 1, 2)
        row += 1

        lay.addWidget(QLabel('Random seed:'), row, 0)
        self._dataset_seed = QSpinBox()
        self._dataset_seed.setRange(0, 9999)
        self._dataset_seed.setValue(42)
        lay.addWidget(self._dataset_seed, row, 1)
        row += 1

        self._btn_build = QPushButton(
            '🔨  Build Dataset + Write data.yaml')
        self._btn_build.setStyleSheet(
            'QPushButton{background:#3498db;color:white;'
            'font-weight:bold;padding:5px;border-radius:4px;}'
            'QPushButton:hover{background:#2980b9;}')
        self._btn_build.clicked.connect(self._build_dataset)
        lay.addWidget(self._btn_build, row, 0, 1, 3)
        row += 1

        self._dataset_status = QLabel('No dataset built yet')
        self._dataset_status.setStyleSheet(
            'color:gray;font-size:11px;')
        self._dataset_status.setWordWrap(True)
        lay.addWidget(self._dataset_status, row, 0, 1, 3)
        lay.setColumnStretch(1, 1)
        self._left_layout.addWidget(gb)

    def _update_test_label(self):
        t = max(0, 100 - self._train_ratio.value()
                - self._val_ratio.value())
        self._test_ratio_label.setText(f'{t}%')
        self._test_ratio_label.setStyleSheet(
            'color:red;' if t < 5 else 'color:gray;')

    # ── DEVICE SECTION ────────────────────────────────────────────────────
    def _build_device_section(self):
        gb  = self._group(' Device')
        lay = QVBoxLayout(gb)
        row = QHBoxLayout()
        row.addWidget(QLabel('Device:'))
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(200)
        row.addWidget(self._device_combo)
        rb = QPushButton('🔄')
        rb.setFixedWidth(30)
        rb.setToolTip('Refresh device list')
        rb.clicked.connect(self._detect_devices)
        row.addWidget(rb)
        row.addStretch()
        lay.addLayout(row)
        self._device_status = QLabel('Detecting...')
        self._device_status.setStyleSheet('color:gray;font-size:11px;')
        lay.addWidget(self._device_status)
        hint = QLabel(
            ' Windows PC: use CPU for testing\n'
            '   Linux VM: switch to GPU:0 for A100 training')
        hint.setStyleSheet('color:#888;font-size:10px;')
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self._left_layout.addWidget(gb)

    # ── MODEL SECTION ─────────────────────────────────────────────────────
    def _build_model_section(self):
        gb  = self._group(' Model Weights')
        lay = QGridLayout(gb)
        lay.setSpacing(4)

        lay.addWidget(QLabel('Model:'), 0, 0)
        self._model_combo = QComboBox()
        self._model_combo.addItems(self.MODEL_OPTIONS)
        self._model_combo.setCurrentText('yolo26n-seg')
        lay.addWidget(self._model_combo, 0, 1, 1, 2)

        lay.addWidget(QLabel('Weights:'), 1, 0)
        self._weights_auto = QCheckBox(
            'Auto-download from Ultralytics ')
        self._weights_auto.setChecked(True)
        self._weights_auto.stateChanged.connect(
            self._toggle_weights_browse)
        lay.addWidget(self._weights_auto, 1, 1, 1, 2)

        lay.addWidget(QLabel('Custom .pt:'), 2, 0)
        self._weights_edit = QLineEdit('')
        self._weights_edit.setPlaceholderText(
            'Browse to custom .pt file...')
        self._weights_edit.setEnabled(False)
        lay.addWidget(self._weights_edit, 2, 1)
        self._weights_btn = QPushButton('Browse')
        self._weights_btn.setFixedWidth(60)
        self._weights_btn.setEnabled(False)
        self._weights_btn.clicked.connect(self._browse_weights)
        lay.addWidget(self._weights_btn, 2, 2)

        self._weights_info = QLabel(
            'Downloads official pretrained weights automatically')
        self._weights_info.setStyleSheet(
            'color:green;font-size:10px;')
        self._weights_info.setWordWrap(True)
        lay.addWidget(self._weights_info, 3, 0, 1, 3)

        lay.addWidget(QLabel('Output dir:'), 4, 0)
        self._outdir_edit = QLineEdit('')
        self._outdir_edit.setPlaceholderText(
            'Browse to output folder...')
        lay.addWidget(self._outdir_edit, 4, 1)
        bo = QPushButton('Browse')
        bo.setFixedWidth(60)
        bo.clicked.connect(self._browse_outdir)
        lay.addWidget(bo, 4, 2)

        lay.addWidget(QLabel('Run name:'), 5, 0)
        self._runname_edit = QLineEdit('yolo_run_1')
        lay.addWidget(self._runname_edit, 5, 1, 1, 2)

        lay.addWidget(QLabel('data.yaml:'), 6, 0)
        self._yaml_edit = QLineEdit('')
        self._yaml_edit.setPlaceholderText(
            'Auto-filled after Dataset Builder...')
        lay.addWidget(self._yaml_edit, 6, 1)
        by = QPushButton('Browse')
        by.setFixedWidth(60)
        by.clicked.connect(self._browse_yaml)
        lay.addWidget(by, 6, 2)

        lay.setColumnStretch(1, 1)
        self._left_layout.addWidget(gb)

    def _toggle_weights_browse(self):
        custom = not self._weights_auto.isChecked()
        self._weights_edit.setEnabled(custom)
        self._weights_btn.setEnabled(custom)
        if custom:
            self._weights_info.setText(
                'Using custom weights — ensure architecture matches')
            self._weights_info.setStyleSheet(
                'color:orange;font-size:10px;')
        else:
            self._weights_info.setText(
                'Downloads official pretrained weights automatically')
            self._weights_info.setStyleSheet(
                'color:green;font-size:10px;')

    # ── HYPERPARAMETERS ───────────────────────────────────────────────────
    def _build_hyperparams_section(self):
        gb  = self._group('Hyperparameters')
        lay = QGridLayout(gb)
        lay.setSpacing(3)

        rows = [
            ('Epochs',      'epochs',     'int'),
            ('Image size',  'imgsz',      'int'),
            ('Batch size',  'batch',      'int'),
            ('Patience',    'patience',   'int'),
            ('cls weight',  'cls',        'float'),
            ('conf thresh', 'conf',       'float'),
            ('dropout',     'dropout',    'float'),
            ('mask_ratio',  'mask_ratio', 'int'),
            ('lr0',         'lr0',        'float'),
            ('box weight',  'box',        'float'),
            ('workers',     'workers',    'int'),
            ('seed',        'seed',       'int'),
        ]
        hints = {
            'cls'        : 'class loss weight',
            'conf'       : 'confidence threshold',
            'dropout'    : 'regularization',
            'mask_ratio' : '1/2/4 mask resolution',
            'lr0'        : 'initial learning rate',
            'box'        : 'box loss weight',
            'batch'      : 'reduce if OOM',
            'patience'   : 'early stop epochs',
            'workers'    : '0=Windows  4=VM (auto-enforced)',
        }

        self._param_edits = {}
        for r, (label, key, dtype) in enumerate(rows):
            lbl = QLabel(f'{label}:')
            if key == 'workers':
                lbl.setStyleSheet('color:#e67e22;')
            lay.addWidget(lbl, r, 0)
            edit = QLineEdit(self.DEFAULTS[key])
            edit.setFixedWidth(90)
            if key in hints:
                edit.setToolTip(hints[key])
            lay.addWidget(edit, r, 1)
            if key in hints:
                hl = QLabel(f'  {hints[key]}')
                hl.setStyleSheet('color:#888;font-size:10px;')
                lay.addWidget(hl, r, 2)
            self._param_edits[key] = (edit, dtype)

        lay.setColumnStretch(2, 1)
        self._left_layout.addWidget(gb)

    # ── FLAGS ─────────────────────────────────────────────────────────────
    def _build_flags_section(self):
        gb  = self._group('🏳️  Options')
        lay = QVBoxLayout(gb)
        lay.setSpacing(3)
        self._flag_checks = {}
        for key, label, default in [
            ('save',        'Save best.pt & last.pt',       True),
            ('plots',       'Save YOLO training plots',     True),
            ('amp',         'Mixed precision AMP',           True),
            ('verbose',     'Verbose output',                True),
            ('save_period', 'Checkpoint every 25 epochs',    True),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(default)
            lay.addWidget(cb)
            self._flag_checks[key] = cb

        if IS_WINDOWS:
            note = QLabel(
                'ℹ️  On Windows: workers auto-set to 0\n'
                '   Uncheck AMP if training crashes immediately')
            note.setStyleSheet('color:#888;font-size:10px;')
            note.setWordWrap(True)
            lay.addWidget(note)

        self._left_layout.addWidget(gb)

    # ── BUTTONS ───────────────────────────────────────────────────────────
    def _build_buttons_section(self):
        gb  = self._group(' Run Training')
        lay = QVBoxLayout(gb)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton('▶  Start Training')
        self._btn_run.setStyleSheet(
            'QPushButton{background:#2ecc71;color:white;'
            'font-weight:bold;padding:7px;border-radius:4px;}'
            'QPushButton:hover{background:#27ae60;}'
            'QPushButton:disabled{background:#ccc;color:#666;}')
        self._btn_run.clicked.connect(self._start_training)
        btn_row.addWidget(self._btn_run)

        self._btn_stop = QPushButton('⏹  Stop')
        self._btn_stop.setStyleSheet(
            'QPushButton{background:#e74c3c;color:white;'
            'font-weight:bold;padding:7px;border-radius:4px;}'
            'QPushButton:disabled{background:#ccc;color:#666;}')
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self._btn_stop)
        lay.addLayout(btn_row)

        bl = QPushButton('📂  Load Existing results.csv')
        bl.clicked.connect(self._load_results)
        lay.addWidget(bl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._status_label = QLabel(
            'Ready — build dataset first, then start training')
        self._status_label.setStyleSheet(
            'color:gray;font-size:11px;')
        self._status_label.setWordWrap(True)
        lay.addWidget(self._status_label)

        self._epoch_label = QLabel('Epoch: —')
        self._epoch_label.setStyleSheet(
            'font-weight:bold;font-size:13px;')
        lay.addWidget(self._epoch_label)

        self._left_layout.addWidget(gb)

    # ── RIGHT PANEL — PLOTS + LOG ─────────────────────────────────────────
    def _build_plots(self, parent_layout):
        tabs = QTabWidget()
        parent_layout.addWidget(tabs)

        if HAS_PLOT:
            for attr_fig, attr_ax, attr_canvas, tab_label in [
                ('_fig_loss', '_ax_loss', '_canvas_loss', 'Loss Curves'),
                ('_fig_map',  '_ax_map',  '_canvas_map',  'mAP Curves'),
                ('_fig_cm',   '_ax_cm',   '_canvas_cm',   'Confusion Matrix'),
            ]:
                w   = QWidget()
                lay = QVBoxLayout(w)
                fig = Figure(figsize=(6, 4), dpi=90)
                ax  = fig.add_subplot(111)
                cnv = FigureCanvasQTAgg(fig)
                lay.addWidget(cnv)
                setattr(self, attr_fig,    fig)
                setattr(self, attr_ax,     ax)
                setattr(self, attr_canvas, cnv)
                tabs.addTab(w, tab_label)
            self._init_empty_plots()
        else:
            tabs.addTab(
                QLabel('  Install matplotlib:\n  uv add matplotlib'),
                'Plots')

        # Metrics tab
        self._metrics_text = QTextEdit()
        self._metrics_text.setReadOnly(True)
        self._metrics_text.setFont(QFont('Courier New', 10))
        self._metrics_text.setPlaceholderText(
            'Metrics will appear here during training...')
        tabs.addTab(self._metrics_text, 'Metrics')

        # Training Log tab — dark terminal style
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont('Courier New', 9))
        self._log_text.setStyleSheet(
            'QTextEdit{'
            'background-color:#1e1e1e;'
            'color:#ffffff;'
            'border:1px solid #444;}')
        self._log_text.setPlaceholderText(
            'Training output will stream here live...')
        tabs.addTab(self._log_text, 'Training Log')

        # default to log tab so user sees output immediately
        tabs.setCurrentIndex(tabs.count() - 1)

    def _init_empty_plots(self):
        for ax, canvas, title in [
            (self._ax_loss, self._canvas_loss,
             'Training & Validation Loss'),
            (self._ax_map,  self._canvas_map,
             'mAP50 and mAP50-95'),
        ]:
            ax.clear()
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Epoch')
            ax.text(0.5, 0.5,
                    'Start training to see live curves',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=11, color='gray')
            ax.grid(alpha=0.3)
            canvas.draw()

    # ══════════════════════════════════════════════════════════════════════
    # DATASET BUILDER LOGIC
    # ══════════════════════════════════════════════════════════════════════
    def _build_dataset(self):
        img_dir     = self._img_folder_edit.text().strip()
        lbl_dir     = self._lbl_folder_edit.text().strip()
        out_dir     = self._dataset_out_edit.text().strip()
        classes_raw = self._classes_edit.text().strip()
        train_pct   = self._train_ratio.value() / 100
        val_pct     = self._val_ratio.value() / 100
        seed        = self._dataset_seed.value()

        for path, name in [(img_dir, 'Images'), (lbl_dir, 'Labels')]:
            if not path or not Path(path).exists():
                QMessageBox.critical(self, 'Error',
                    f'{name} folder not found.\n'
                    'Please use Browse to select it.')
                return
        if not out_dir:
            QMessageBox.critical(self, 'Error',
                'Please select an output folder.')
            return
        if not classes_raw:
            QMessageBox.critical(self, 'Error',
                'Please enter class names.')
            return
        if 1.0 - train_pct - val_pct < 0.02:
            QMessageBox.warning(self, 'Warning',
                'Test split too small. Reduce Train % or Val %.')
            return

        class_names = [c.strip() for c in classes_raw.split(',')
                       if c.strip()]
        self._btn_build.setEnabled(False)
        self._dataset_status.setText('Building dataset...')
        self._dataset_status.setStyleSheet('color:#e67e22;')

        threading.Thread(
            target=self._run_dataset_builder,
            args=(img_dir, lbl_dir, out_dir, class_names,
                  train_pct, val_pct,
                  1.0 - train_pct - val_pct, seed),
            daemon=True
        ).start()

    def _run_dataset_builder(self, img_dir, lbl_dir, out_dir,
                              class_names, train_pct, val_pct,
                              test_pct, seed):
        try:
            img_dir = Path(img_dir)
            lbl_dir = Path(lbl_dir)
            out_dir = Path(out_dir)

            IMG_EXTS    = {'.png', '.jpg', '.jpeg',
                           '.tif', '.tiff', '.bmp'}
            label_files = list(lbl_dir.glob('*.txt'))

            if not label_files:
                self._signals.dataset_error.emit(
                    f'No .txt label files found in:\n{lbl_dir}\n\n'
                    'The labels folder must contain one .txt file '
                    'per image.\ne.g. frame_001.txt')
                return

            matched, missing = [], []
            for lf in label_files:
                found = False
                for ext in IMG_EXTS:
                    img = img_dir / (lf.stem + ext)
                    if img.exists():
                        matched.append((img, lf))
                        found = True
                        break
                if not found:
                    missing.append(lf.stem)

            if not matched:
                self._signals.dataset_error.emit(
                    'No matching image+label pairs found!\n\n'
                    f'Images : {img_dir}\n'
                    f'Labels : {lbl_dir}\n\n'
                    'Filenames must match:\n'
                    '  frame_001.png  <->  frame_001.txt')
                return

            random.seed(seed)
            random.shuffle(matched)
            n       = len(matched)
            n_train = int(n * train_pct)
            n_val   = int(n * val_pct)
            splits  = {
                'train' : matched[:n_train],
                'val'   : matched[n_train:n_train + n_val],
                'test'  : matched[n_train + n_val:],
            }

            for split in ['train', 'val', 'test']:
                (out_dir / 'images' / split).mkdir(
                    parents=True, exist_ok=True)
                (out_dir / 'labels' / split).mkdir(
                    parents=True, exist_ok=True)

            counts = {}
            for split, pairs in splits.items():
                for ip, lp in pairs:
                    shutil.copy(str(ip),
                        str(out_dir / 'images' / split / ip.name))
                    shutil.copy(str(lp),
                        str(out_dir / 'labels' / split / lp.name))
                counts[split] = len(pairs)

            # forward slashes in data.yaml work on both OS
            out_str   = str(out_dir).replace('\\', '/')
            names_str = str(class_names).replace('"', "'")
            yaml_path = out_dir / 'data.yaml'
            yaml_path.write_text(
                f'path: {out_str}\n'
                f'train: images/train\n'
                f'val:   images/val\n'
                f'test:  images/test\n\n'
                f'nc: {len(class_names)}\n'
                f'names: {names_str}\n',
                encoding='utf-8')

            summary = (
                f'Dataset built successfully!\n'
                f'  Total:{n}  '
                f'Train:{counts["train"]}  '
                f'Val:{counts["val"]}  '
                f'Test:{counts["test"]}\n'
                f'  Classes: {class_names}\n'
                f'  data.yaml -> {yaml_path}'
                + (f'\n  Warning: {len(missing)} labels had no image'
                   if missing else '')
            )
            self._signals.dataset_done.emit(
                f'{summary}|||{str(yaml_path)}')

        except Exception as e:
            self._signals.dataset_error.emit(str(e))

    def _on_dataset_done(self, msg):
        parts = msg.split('|||')
        self._dataset_status.setText(parts[0])
        self._dataset_status.setStyleSheet(
            'color:green;font-size:11px;')
        self._btn_build.setEnabled(True)
        if len(parts) > 1 and parts[1]:
            self._yaml_edit.setText(parts[1])
            self._status_label.setText(
                'Dataset ready — data.yaml auto-filled. '
                'Click Start Training.')
            self._status_label.setStyleSheet('color:green;')

    def _on_dataset_error(self, msg):
        self._dataset_status.setText(f'Error: {msg[:100]}')
        self._dataset_status.setStyleSheet('color:red;')
        self._btn_build.setEnabled(True)
        QMessageBox.critical(self, 'Dataset Builder Error', msg)

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING LOGIC
    # ══════════════════════════════════════════════════════════════════════
    def _get_params(self):
        params = {}
        for key, (edit, dtype) in self._param_edits.items():
            val = edit.text().strip()
            try:
                params[key] = (int(val) if dtype == 'int'
                               else float(val))
            except ValueError:
                raise ValueError(
                    f'Invalid value for {key}: "{val}"')
        return params

    def _get_weights_arg(self):
        if self._weights_auto.isChecked():
            return f"{self._model_combo.currentText()}.pt"
        custom = self._weights_edit.text().strip()
        if not custom or not Path(custom).exists():
            raise ValueError(
                'Custom weights file not found.\n'
                'Browse to select a valid .pt file.')
        return custom

    def _start_training(self):
        try:
            params  = self._get_params()
            weights = self._get_weights_arg()
        except ValueError as e:
            QMessageBox.critical(self, 'Invalid Parameters', str(e))
            return

        yaml_path = self._yaml_edit.text().strip()
        out_dir   = self._outdir_edit.text().strip()
        run_name  = self._runname_edit.text().strip()
        device    = self._get_device()

        if not yaml_path or not Path(yaml_path).exists():
            QMessageBox.critical(self, 'Error',
                'data.yaml not found.\n'
                'Build dataset first OR browse to existing data.yaml.')
            return
        if not out_dir:
            QMessageBox.critical(self, 'Error',
                'Please select an output directory.')
            return
        if not run_name:
            QMessageBox.critical(self, 'Error',
                'Please enter a run name.')
            return

        # enforce platform-safe workers
        params['workers'] = self._safe_workers(params['workers'])

        self._results_path = Path(out_dir) / run_name / 'results.csv'
        self._training     = True
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)
        self._status_label.setText(
            f'Training on {device} [{self._platform_label()}]...')
        self._status_label.setStyleSheet('color:#e67e22;')
        self._log_text.clear()
        if HAS_PLOT:
            self._init_empty_plots()

        threading.Thread(
            target=self._run_script,
            args=(self._make_script(
                weights, yaml_path, out_dir,
                run_name, device, params),),
            daemon=True
        ).start()
        self._timer.start(30_000)

    def _make_script(self, weights, yaml_path,
                     out_dir, run_name, device, params):
        flags = self._flag_checks
        sp    = 25 if flags['save_period'].isChecked() else -1
        # forward slashes safe on both OS
        yp = str(yaml_path).replace('\\', '/')
        od = str(out_dir).replace('\\', '/')
        wt = str(weights).replace('\\', '/')

        return f"""
import os, gc, sys, torch, platform
from ultralytics import YOLO
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONUTF8']       = '1'
try:
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
except Exception:
    pass

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect()
torch.cuda.empty_cache()
Path(r'{od}').mkdir(parents=True, exist_ok=True)

print('=' * 55)
print('YOLO Training Started')
print('=' * 55)
print(f'Platform   : {{platform.system()}}')
print(f'Model      : {wt}')
print(f'Data       : {yp}')
print(f'Device     : {device}')
print(f'Epochs     : {params["epochs"]}')
print(f'Batch      : {params["batch"]}')
print(f'Image size : {params["imgsz"]}')
print(f'Workers    : {params["workers"]}')
print(f'Output     : {od}/{run_name}')
print('=' * 55)
sys.stdout.flush()

model = YOLO(r'{wt}')
print('Weights loaded starting training...')
sys.stdout.flush()

model.train(
    data        = r'{yp}',
    epochs      = {params['epochs']},
    imgsz       = {params['imgsz']},
    batch       = {params['batch']},
    device      = '{device}',
    project     = r'{od}',
    name        = '{run_name}',
    exist_ok    = True,
    cls         = {params['cls']},
    conf        = {params['conf']},
    dropout     = {params['dropout']},
    mask_ratio  = {params['mask_ratio']},
    lr0         = {params['lr0']},
    box         = {params['box']},
    patience    = {params['patience']},
    workers     = {params['workers']},
    seed        = {params['seed']},
    optimizer   = 'auto',
    amp         = {flags['amp'].isChecked()},
    save        = {flags['save'].isChecked()},
    plots       = {flags['plots'].isChecked()},
    save_period = {sp},
    verbose     = True,
)
print('=' * 55)
print('__TRAINING_COMPLETE__')
sys.stdout.flush()
"""

    def _run_script(self, script):
        """
        Stream subprocess output line by line to the log tab.
        Fixes applied:
          encoding='utf-8'  -> prevents charmap error on Windows
          errors='replace'  -> unknown chars become ? not crash
          PYTHONUTF8=1      -> forces subprocess to use UTF-8
          CREATE_NO_WINDOW  -> no flash console on Windows
        """
        try:
            env = {
                **os.environ,
                'PYTHONUNBUFFERED' : '1',
                'PYTHONUTF8'       : '1',   # ← fixes charmap error
                'ULTRALYTICS_QUIET': '0',
            }
            kwargs = {}
            if IS_WINDOWS:
                kwargs['creationflags'] = 0x08000000  # CREATE_NO_WINDOW

            proc = subprocess.Popen(
                [sys.executable, '-c', script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',   # ← fixes charmap error on Windows
                errors='replace',   # ← unknown chars become ? not crash
                env=env,
                **kwargs
            )
            self._process = proc

            for line in iter(proc.stdout.readline, ''):
                line = line.rstrip()
                if not line:
                    continue
                self._signals.update_log.emit(line)
                if '__TRAINING_COMPLETE__' in line:
                    self._signals.training_done.emit()
                    return
                # parse epoch from YOLO output: "      1/200    16.1G ..."
                for p in line.strip().split():
                    if '/' in p:
                        s = p.split('/')
                        if (len(s) == 2
                                and s[0].isdigit()
                                and s[1].isdigit()
                                and 1 <= int(s[0]) <= 10000
                                and 1 <= int(s[1]) <= 10000):
                            self._signals.update_epoch.emit(
                                f'Epoch: {s[0]}/{s[1]}')
                            break

            proc.wait()
            if proc.returncode not in (0, None):
                self._signals.training_error.emit(
                    f'Process exited with code {proc.returncode}\n\n'
                    'Check Training Log tab.\n\n'
                    'Common fixes:\n'
                    '  Reduce batch size\n'
                    '  Set workers=0 (Windows)\n'
                    '  Uncheck AMP if crash is immediate\n'
                    '  Check data.yaml paths are correct')
        except Exception as e:
            self._signals.training_error.emit(str(e))
        finally:
            self._process = None

    def _stop_training(self):
        if self._process:
            self._process.terminate()
        self._timer.stop()
        self._signals.update_log.emit('Training stopped by user')
        self._status_label.setText('Stopped by user')
        self._status_label.setStyleSheet('color:gray;')
        self._done_ui()

    # ══════════════════════════════════════════════════════════════════════
    # LOG — colour coded terminal output
    # ══════════════════════════════════════════════════════════════════════
    def _append_log(self, line):
        """Append line with colour coding to dark terminal log."""
        # sanitise for HTML
        safe = (line
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace(' ', '&nbsp;'))

        if any(x in line for x in [
                'Error', 'error', 'ERROR',
                'Failed', 'failed', 'Traceback', 'exception']):
            color = '#ff6b6b'           # red

        elif any(x in line for x in [
                'WARNING', 'Warning', 'warning', 'UserWarning']):
            color = '#ffa94d'           # orange

        elif any(x in line for x in [
                '__TRAINING_COMPLETE__', 'Training Started',
                'Weights loaded', 'Results saved', 'Best']):
            color = '#69db7c'           # bright green

        elif line.strip().startswith('='):
            color = '#69db7c'           # green dividers

        elif any(x in line for x in [
                'loss', 'Loss', 'mAP', 'Precision', 'Recall',
                'GPU_mem', 'Epoch', 'epoch']):
            color = '#74c0fc'           # blue — training metrics

        elif any(x in line for x in [
                'Platform', 'Model', 'Data :', 'Device',
                'Epochs', 'Batch', 'Output', 'Workers']):
            color = '#da77f2'           # purple — config info

        elif any(x in line for x in [
                'Downloading', 'downloaded',
                'Loading', 'loaded', 'Ultralytics']):
            color = '#ffd43b'           # yellow — downloads

        else:
            color = '#ced4da'           # light grey — default

        self._log_text.insertHtml(
            f'<span style="color:{color};'
            f'font-family:Courier New;font-size:11px;">'
            f'{safe}</span><br>')
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ══════════════════════════════════════════════════════════════════════
    # PLOT REFRESH — every 30s while training
    # ══════════════════════════════════════════════════════════════════════
    def _refresh_plots(self):
        if not self._results_path:
            return
        path = Path(self._results_path)
        if not path.exists() or not HAS_PANDAS:
            return
        try:
            df         = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            if len(df) < 2:
                return
            epochs = range(1, len(df) + 1)

            if HAS_PLOT:
                # ── Loss curves ────────────────────────────────────────
                self._ax_loss.clear()
                for col, lbl, clr, ls in [
                    ('train/box_loss', 'Train Box', '#3498db', '-'),
                    ('train/seg_loss', 'Train Seg', '#2ecc71', '-'),
                    ('val/box_loss',   'Val Box',   '#e74c3c', '--'),
                    ('val/seg_loss',   'Val Seg',   '#e67e22', '--'),
                ]:
                    if col in df.columns:
                        self._ax_loss.plot(
                            epochs, df[col], label=lbl,
                            color=clr, linestyle=ls, linewidth=1.5)
                self._ax_loss.set_title(
                    'Training & Validation Loss', fontsize=10)
                self._ax_loss.set_xlabel('Epoch')
                self._ax_loss.set_ylabel('Loss')
                self._ax_loss.legend(fontsize=8)
                self._ax_loss.grid(alpha=0.3)
                self._canvas_loss.draw()

                # ── mAP curves — no baseline lines ─────────────────────
                self._ax_map.clear()
                for col, lbl, clr, ls in [
                    ('metrics/mAP50(M)',
                     'Mask mAP50',    '#2ecc71', '-'),
                    ('metrics/mAP50-95(M)',
                     'Mask mAP50-95', '#27ae60', '--'),
                    ('metrics/mAP50(B)',
                     'Box mAP50',     '#3498db', '-'),
                    ('metrics/mAP50-95(B)',
                     'Box mAP50-95',  '#2980b9', '--'),
                ]:
                    if col in df.columns:
                        self._ax_map.plot(
                            epochs, df[col], label=lbl,
                            color=clr, linestyle=ls, linewidth=1.5)
                self._ax_map.set_title(
                    'mAP50 and mAP50-95', fontsize=10)
                self._ax_map.set_xlabel('Epoch')
                self._ax_map.set_ylabel('mAP')
                self._ax_map.legend(fontsize=8)
                self._ax_map.grid(alpha=0.3)
                self._canvas_map.draw()

            # ── Metrics text ───────────────────────────────────────────
            last    = df.iloc[-1]
            ep_done = len(df)
            lines   = [
                f'Platform         : {self._platform_label()}',
                f'Epochs completed : {ep_done}',
                '-' * 42,
            ]
            for col, lbl in [
                ('metrics/mAP50(M)',    'Mask mAP50    '),
                ('metrics/mAP50-95(M)','Mask mAP50-95 '),
                ('metrics/mAP50(B)',    'Box  mAP50    '),
                ('metrics/mAP50-95(B)','Box  mAP50-95 '),
                ('metrics/precision(B)','Precision     '),
                ('metrics/recall(B)',   'Recall        '),
            ]:
                if col in df.columns:
                    lines.append(f'{lbl}: {last[col]:.4f}')
            self._metrics_text.setPlainText('\n'.join(lines))
            self._epoch_label.setText(f'Epoch: {ep_done}')

        except Exception as e:
            self._status_label.setText(f'Plot error: {e}')

    def _load_confusion_matrix(self):
        if not self._results_path or not HAS_PLOT:
            return
        run_dir = Path(self._results_path).parent
        for name in ['confusion_matrix_normalized.png',
                     'confusion_matrix.png']:
            cm = run_dir / name
            if cm.exists():
                try:
                    self._ax_cm.clear()
                    self._ax_cm.imshow(mpimg.imread(str(cm)))
                    self._ax_cm.axis('off')
                    self._ax_cm.set_title(
                        'Confusion Matrix', fontsize=10)
                    self._canvas_cm.draw()
                except Exception as e:
                    print(f'CM error: {e}')
                break

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════
    def _update_epoch_label(self, text):
        self._epoch_label.setText(text)

    def _on_training_complete(self):
        self._timer.stop()
        self._refresh_plots()
        self._load_confusion_matrix()
        self._status_label.setText('Training complete!')
        self._status_label.setStyleSheet('color:green;')
        self._done_ui()
        weights = (Path(self._outdir_edit.text())
                   / self._runname_edit.text()
                   / 'weights' / 'best.pt')
        QMessageBox.information(
            self, 'Training Complete',
            f'Training finished!\n\nBest weights:\n{weights}')

    def _on_training_error(self, msg):
        self._timer.stop()
        self._status_label.setText('Training error — check Log tab')
        self._status_label.setStyleSheet('color:red;')
        self._done_ui()
        QMessageBox.critical(self, 'Training Error', msg)

    def _done_ui(self):
        self._training = False
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)

    def _load_results(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select results.csv', '',
            'CSV files (*.csv);;All files (*.*)')
        if not path:
            return
        self._results_path = path
        self._refresh_plots()
        cm = Path(path).parent / 'confusion_matrix_normalized.png'
        if cm.exists():
            self._load_confusion_matrix()
        self._status_label.setText(
            f'Loaded: {Path(path).parent.name}')

    # ── BROWSE HELPERS ────────────────────────────────────────────────────
    def _browse_yaml(self):
        p, _ = QFileDialog.getOpenFileName(
            self, 'Select data.yaml', '',
            'YAML files (*.yaml *.yml);;All files (*.*)')
        if p:
            self._yaml_edit.setText(p)

    def _browse_outdir(self):
        p = QFileDialog.getExistingDirectory(
            self, 'Select output directory')
        if p:
            self._outdir_edit.setText(p)

    def _browse_weights(self):
        p, _ = QFileDialog.getOpenFileName(
            self, 'Select weights file', '',
            'PyTorch weights (*.pt);;All files (*.*)')
        if p:
            self._weights_edit.setText(p)

    def _browse_img_folder(self):
        p = QFileDialog.getExistingDirectory(
            self, 'Select images folder')
        if p:
            self._img_folder_edit.setText(p)

    def _browse_lbl_folder(self):
        p = QFileDialog.getExistingDirectory(
            self, 'Select labels folder')
        if p:
            self._lbl_folder_edit.setText(p)

    def _browse_dataset_out(self):
        p = QFileDialog.getExistingDirectory(
            self, 'Select dataset output folder')
        if p:
            self._dataset_out_edit.setText(p)