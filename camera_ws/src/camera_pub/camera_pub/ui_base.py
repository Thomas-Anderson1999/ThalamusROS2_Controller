import json
import os
import numpy as np
from enum import Enum

import pyqtgraph as pg

pg.setConfigOptions(antialias=False)

from PyQt5.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QComboBox, QListWidget, QSpacerItem, QSizePolicy,
    QScrollArea, QWidget, QVBoxLayout
)
from PyQt5.QtCore import QSize, pyqtSignal


_PG_COLORS = {
    "r": (220,  50,  50),
    "g": ( 50, 200,  50),
    "b": ( 50, 100, 255),
    "c": (  0, 200, 200),
    "m": (200,   0, 200),
    "y": (200, 200,   0),
    "k": ( 30,  30,  30),
    "w": (200, 200, 200),
}


# ==============================================================
# PyQtGraph Canvas
# ==============================================================

class PgCanvas(QWidget):
    def __init__(self, parent=None, ui_type=None):
        super().__init__(parent)
        self._ui_type = ui_type
        self._lines   = []   # list of (PlotDataItem, axis_side)
        self._vb_r    = None # right-axis ViewBox

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._plot = pg.PlotWidget(background='w')
        self._plot.showGrid(x=True, y=True, alpha=0.4)
        layout.addWidget(self._plot)

    def sizeHint(self):
        if self._ui_type == UIElementType.GRAPH_2D_TRAJECTORY:
            return QSize(700, 700)
        return QSize(700, 400)

    def is_valid_size(self):
        return self.width() > 1 and self.height() > 1

    def _setup_right_axis(self):
        p = self._plot
        p.showAxis('right')
        vb = pg.ViewBox()
        p.scene().addItem(vb)
        p.getAxis('right').linkToView(vb)
        vb.setXLink(p)

        def _sync():
            vb.setGeometry(p.getViewBox().sceneBoundingRect())

        p.getViewBox().sigResized.connect(_sync)
        self._vb_r = vb


# ==============================================================
# UI Element Definitions
# ==============================================================

class UIElementType(Enum):
    NEW_LINE = 0
    EMPTY_CULUMN = 1
    EDIT = 2
    BUTTON = 3
    CHECK = 4
    LISTBOX = 5
    COMBO = 6
    GRAPH_1D = 7
    GRAPH_2D_TRAJECTORY = 8


class UIElement:
    def __init__(self, _type=None, _caption="NoCaption", _default_val=None, _defalt_culumn=1, _defalt_row=1):
        self.type = _type
        self.caption = _caption
        self.default_val = _default_val
        self.culumn = _defalt_culumn
        self.row = _defalt_row


class DoubleClickButton(QPushButton):
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


# ==============================================================
# UIBase
# ==============================================================

class UIBase:
    def __init__(self, fname="preset.txt"):
        self.fname = fname

    # ------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------

    def createGroupBox(self, gbxName, UI_ele_list):
        groupBox = QGroupBox(gbxName)
        grid = QGridLayout()
        groupBox.setLayout(grid)

        tmp_dic = {}
        pos_i, pos_j = 0, 0

        for element in UI_ele_list:
            if element.type == UIElementType.NEW_LINE:
                pos_j += 1
                pos_i = 0
                continue

            if element.type == UIElementType.EMPTY_CULUMN:
                spacer = QSpacerItem(500, 500, QSizePolicy.Expanding, QSizePolicy.Minimum)
                grid.addItem(spacer, pos_j, pos_i, element.row, element.culumn)
                pos_i += element.culumn
                continue

            if element.type == UIElementType.EDIT:
                label = QLabel(element.caption + ":", self)
                grid.addWidget(label, pos_j, pos_i)
                pos_i += 1

                edit = QLineEdit(self)
                if element.default_val is not None:
                    edit.setText(element.default_val)
                grid.addWidget(edit, pos_j, pos_i, element.row, element.culumn)
                pos_i += element.culumn
                tmp_dic[element.caption] = edit
                continue

            if element.type == UIElementType.BUTTON:
                btn = QPushButton(element.caption, self)
                if element.default_val is not None:
                    btn.clicked.connect(element.default_val)
                grid.addWidget(btn, pos_j, pos_i)
                pos_i += element.culumn
                continue

            if element.type == UIElementType.CHECK:
                cb = QCheckBox(element.caption, self)
                if element.default_val is not None:
                    cb.setChecked(element.default_val)
                grid.addWidget(cb, pos_j, pos_i)
                pos_i += element.culumn
                tmp_dic[element.caption] = cb
                continue

            if element.type == UIElementType.COMBO:
                combo = QComboBox(self)
                if element.default_val is not None:
                    combo.addItems(element.default_val)
                grid.addWidget(combo, pos_j, pos_i)
                pos_i += element.culumn
                tmp_dic[element.caption] = combo
                continue

            if element.type == UIElementType.LISTBOX:
                lb = QListWidget()
                grid.addWidget(lb, pos_j, pos_i, element.row, element.culumn)
                pos_i += element.culumn
                tmp_dic[element.caption] = lb
                continue

            if element.type == UIElementType.GRAPH_1D:
                canvas = PgCanvas(self, ui_type=UIElementType.GRAPH_1D)
                grid.addWidget(canvas, pos_j, pos_i, element.row, element.culumn)
                pos_i += element.culumn
                tmp_dic[element.caption] = canvas
                continue

            if element.type == UIElementType.GRAPH_2D_TRAJECTORY:
                canvas = PgCanvas(self, ui_type=UIElementType.GRAPH_2D_TRAJECTORY)
                grid.addWidget(canvas, pos_j, pos_i, element.row, element.culumn)
                pos_i += element.culumn
                tmp_dic[element.caption] = canvas
                continue

        return groupBox, tmp_dic

    # ------------------------------------------------------------
    # UI access
    # ------------------------------------------------------------

    def getUIVal(self, key):
        res = []
        for ui_dic in self.ui_dic_list:
            ctl = ui_dic.get(key)
            if isinstance(ctl, QLineEdit):
                res.append(ctl.text())
            elif isinstance(ctl, QCheckBox):
                res.append(ctl.isChecked())
            elif isinstance(ctl, QComboBox):
                res.append(ctl.currentText())
            elif isinstance(ctl, QListWidget):
                res = [i.text() for i in ctl.selectedItems()]
        return res

    def bind_listbox_changed(self, callback):
        for ui_dic in self.ui_dic_list:
            for key, ctl in ui_dic.items():
                if isinstance(ctl, QListWidget):
                    ctl.itemSelectionChanged.connect(lambda k=key: callback(k))

    # ------------------------------------------------------------
    # UI set
    # ------------------------------------------------------------

    def get_ui_val(self, key, _type=str):
        try:
            res = self.getUIVal(key)
        except Exception:
            return None

        if not res:
            return None

        val = res[0]

        try:
            return _type(val)
        except (ValueError, TypeError):
            return None

    def set_ui_val(self, key, val):
        self.setUIVal(key, val)

    def setUIVal(self, key, val):
        for ui_dic in self.ui_dic_list:
            ctl = ui_dic.get(key)
            if ctl is None:
                continue

            if isinstance(ctl, QLineEdit):
                ctl.setText(str(val))
                continue
            if isinstance(ctl, QCheckBox):
                ctl.setChecked(val)
                continue
            if isinstance(ctl, QComboBox):
                ctl.clear()
                ctl.addItems(val)
                continue
            if isinstance(ctl, QListWidget):
                ctl.clear()
                ctl.addItems(val)
                continue

            # ---------------------------
            # 1D Graph
            # ---------------------------
            if isinstance(ctl, PgCanvas) and ctl._ui_type == UIElementType.GRAPH_1D:
                need_rebuild = (not ctl._lines) or (len(ctl._lines) != len(val))

                if need_rebuild:
                    ctl._plot.clear()
                    if ctl._vb_r is not None:
                        ctl._vb_r.clear()
                        ctl._vb_r = None
                    ctl._lines = []

                    legend = ctl._plot.addLegend(offset=(10, 10))

                    for dataset in val:
                        axis_info  = dataset[2] if len(dataset) > 2 else None
                        label      = dataset[3] if len(dataset) > 3 else None
                        color_str  = dataset[4] if len(dataset) > 4 else "k"
                        axis_side  = axis_info[0] if isinstance(axis_info, (list, tuple)) and len(axis_info) >= 1 else "l"

                        color = _PG_COLORS.get(color_str, (100, 100, 100))
                        pen   = pg.mkPen(color=color, width=1)

                        if axis_side == "r":
                            if ctl._vb_r is None:
                                ctl._setup_right_axis()
                            item = pg.PlotDataItem(pen=pen, name=label)
                            ctl._vb_r.addItem(item)
                            if label:
                                legend.addItem(item, label)
                        else:
                            item = ctl._plot.plot(pen=pen, name=label)

                        ctl._lines.append((item, axis_side))

                # 데이터 교체
                x_last = None
                for i, dataset in enumerate(val):
                    if i >= len(ctl._lines):
                        break
                    x, y      = dataset[0], dataset[1]
                    axis_info = dataset[2] if len(dataset) > 2 else None
                    item, axis_side = ctl._lines[i]
                    item.setData(x=x, y=y)
                    x_last = x

                    if isinstance(axis_info, (list, tuple)) and len(axis_info) >= 3:
                        _, ymin, ymax = axis_info
                        if ymin != ymax:
                            if axis_side == "r" and ctl._vb_r:
                                ctl._vb_r.setYRange(ymin, ymax, padding=0)
                            else:
                                ctl._plot.setYRange(ymin, ymax, padding=0)

                if x_last is not None and len(x_last) > 0:
                    ctl._plot.setXRange(float(x_last[0]), float(x_last[-1]), padding=0)

                continue

            # ---------------------------
            # 2D Trajectory
            # ---------------------------
            if isinstance(ctl, PgCanvas) and ctl._ui_type == UIElementType.GRAPH_2D_TRAJECTORY:
                if len(val) < 3:
                    continue

                x   = np.asarray(val[0], float)
                y   = np.asarray(val[1], float)
                yaw = np.asarray(val[2], float)
                idx = val[3] if len(val) >= 4 else None

                if len(x) == 0 or len(y) == 0:
                    continue

                # 최초 1회 아이템 생성
                if not hasattr(ctl, '_traj_line'):
                    ctl._traj_line  = ctl._plot.plot(pen=pg.mkPen('b', width=2))
                    ctl._traj_arrow = pg.ArrowItem(angle=0, headLen=20, tipAngle=30,
                                                   brush=(220, 50, 50), pen=None)
                    ctl._plot.addItem(ctl._traj_arrow)
                    ctl._traj_text = pg.TextItem(color=(200, 0, 0),
                                                 fill=pg.mkBrush(255, 255, 255, 180))
                    ctl._plot.addItem(ctl._traj_text)

                ctl._traj_line.setData(x=x, y=y)

                if idx is not None and 0 <= idx < len(x):
                    px, py, pyaw = float(x[idx]), float(y[idx]), float(yaw[idx])
                    # pyqtgraph ArrowItem angle: 0=오른쪽, CCW 양수
                    ctl._traj_arrow.setPos(px, py)
                    ctl._traj_arrow.setStyle(angle=180 - pyaw)
                    ctl._traj_text.setPos(px + 0.05, py + 0.05)
                    ctl._traj_text.setText(f"x={px:.2f}  y={py:.2f}  yaw={pyaw:.1f}°")

                # 정사각형 뷰 유지
                cx = (x.min() + x.max()) / 2
                cy = (y.min() + y.max()) / 2
                half = max((x.max() - x.min()) / 2, (y.max() - y.min()) / 2, 1.0)
                ctl._plot.setXRange(cx - half, cx + half, padding=0)
                ctl._plot.setYRange(cy - half, cy + half, padding=0)

                continue

    # ------------------------------------------------------------
    # Preset
    # ------------------------------------------------------------

    def SavePreset(self, _filename=None):
        fname = _filename or self.fname
        with open(fname, "w") as f:
            geo = self.geometry()
            f.write(json.dumps({
                "x": geo.x(), "y": geo.y(),
                "width": geo.width(), "height": geo.height()
            }) + "\n")
            for dic in self.ui_dic_list:
                out = {}
                for k, v in dic.items():
                    if isinstance(v, QLineEdit):
                        out[k] = v.text()
                    elif isinstance(v, QCheckBox):
                        out[k] = v.isChecked()
                    elif isinstance(v, QComboBox):
                        out[k] = v.currentText()
                f.write(json.dumps(out) + "\n")

    def LoadPreset(self, _filename=None):
        fname = _filename or self.fname
        if not os.path.isfile(fname):
            return
        with open(fname, "r") as f:
            lines = f.readlines()

        idx_offset = 0
        for i, line in enumerate(lines):
            if i == 0:
                try:
                    d = json.loads(line)
                    self.setGeometry(d["x"], d["y"], d["width"], d["height"])
                    idx_offset = -1
                    continue
                except Exception:
                    pass
            data = json.loads(line)

            if len(self.ui_dic_list) <= i + idx_offset:
                break
            dic = self.ui_dic_list[i + idx_offset]
            for k, v in data.items():
                ctl = dic.get(k)
                if isinstance(ctl, QLineEdit):
                    ctl.setText(v)
                elif isinstance(ctl, QCheckBox):
                    ctl.setChecked(v)
                elif isinstance(ctl, QComboBox):
                    idx = ctl.findText(v)
                    if idx >= 0:
                        ctl.setCurrentIndex(idx)
