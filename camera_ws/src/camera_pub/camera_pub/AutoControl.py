from PyQt5.QtWidgets import QWidget, QGridLayout

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from ThalamusEngine.Interface import LoadThalamusInterface, InitEngine


_ASM_FILE = "ScriptFreeModel.txt".encode("UTF-8")
_thalamus_ready = False


def _init_thalamus():
    global _thalamus_ready
    if _thalamus_ready:
        return True
    if LoadThalamusInterface():
        InitEngine(_ASM_FILE, 1280, 720, 0)
        _thalamus_ready = True
        print("[thalamus] InitEngine OK")
        return True
    print("[thalamus] LoadThalamusInterface FAILED")
    return False


class PresetCamWindow(QWidget, UIBase):
    def __init__(self, parent_window):
        super(PresetCamWindow, self).__init__()
        UIBase.__init__(self)

        self.parent_window = parent_window
        self.ui_dic_list   = parent_window.ui_dic_list  # share ui state

        tilt     = ["UP", "MD", "DN"]
        tilt_def = [0,    20,   65  ]
        pan      = ["LLL", "LL", "L", "N", "R", "RR", "RRR"]
        pan_def  = [-127,  -82, -37,   8,  53,   98,   135 ]

        # ── pan angle header ───────────────────────────────────
        ele_list = [
            UIElement(_type=UIElementType.BUTTON, _caption="Quit", _default_val=self.parent_window.quit_app),
            UIElement(_type=UIElementType.NEW_LINE),
        ]
        for p, d in zip(pan, pan_def):
            ele_list.append(UIElement(_type=UIElementType.EDIT, _caption=p, _default_val=str(d)))
        ele_list.append(UIElement(_type=UIElementType.NEW_LINE))

        # ── tilt rows × pan buttons ────────────────────────────
        for t, td in zip(tilt, tilt_def):
            ele_list.append(UIElement(_type=UIElementType.EDIT, _caption=t, _default_val=str(td)))
            for p in pan:
                colspan = 6 if p == "N" else 1
                ele_list.append(UIElement(
                    _type=UIElementType.BUTTON,
                    _caption=f"{p} {t}",
                    _default_val=lambda state, _t=t, _p=p: self.parent_window.preset_control(_t, _p),
                    _defalt_culumn=colspan,
                ))
            ele_list.append(UIElement(_type=UIElementType.NEW_LINE))

        mainGrid = QGridLayout()
        subgrid, dic = self.createGroupBox("Preset Cam Control", ele_list)
        self.ui_dic_list.append(dic)
        mainGrid.addWidget(subgrid, 0, 0)

        # ── Auto-Control info ──────────────────────────────────
        info_ele_list = [
            UIElement(_type=UIElementType.EDIT,   _caption="ClickPosX",       _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="ClickPosY",       _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="TgtPan",          _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="TgtTilt",         _default_val="0"),
            UIElement(_type=UIElementType.BUTTON, _caption="set_clicked_pos", _default_val=self.parent_window.set_clicked_pos),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_X",      _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_Y",      _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_Yaw",    _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="local_action_yaw", _default_val="0"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.COMBO, _caption="Depth_Src", _default_val=["Virtual_Plane", "Depth_Map"]),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosX", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosY", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosZ", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtBearing", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtDist", _default_val="0"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.COMBO,  _caption="CamTrackMode", _default_val=["None", "fix_cam_rotation", "fix_on_target"]),
            UIElement(_type=UIElementType.BUTTON, _caption="excute_first_action", _default_val=self.parent_window.excute_first_action),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.LISTBOX, _caption="SeqCmd", _default_val=None, _defalt_row=3, _defalt_culumn=8),
        ]
        subgrid2, dic2 = self.createGroupBox("Semi-Auto-Control", info_ele_list)
        self.ui_dic_list.append(dic2)
        self.bind_listbox_changed(self.parent_window.on_listbox_changed)
        mainGrid.addWidget(subgrid2, 1, 0)

        self.setLayout(mainGrid)
        self.setWindowTitle("Auto Control")
        self.resize(1200, 400)
