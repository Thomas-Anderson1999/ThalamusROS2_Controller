import os
import time
import sys
import json
from datetime import datetime
import numpy as np
import cv2

from ultralytics import YOLO

_PKG_DIR       = os.path.dirname(os.path.abspath(__file__))
_LIGHT_WEIGHT  = os.path.join(_PKG_DIR, "cnn_weight", "light_best.pt")
_PERSON_WEIGHT = os.path.join(_PKG_DIR, "cnn_weight", "person_best.pt")

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import Qt, QTimer
from camera_pub.img_proc_main import img_proc_odometry, get_last_optflow_pts, get_last_optflow_ids, get_optflow_reset_count, reset_optflow_pts, reset_vio, reset_optical_flow, get_last_vio_yaw
from camera_pub.recon_util import pixel_to_vec
from camera_pub.depth_recon import depthmap_floor_ransac, make_birdview

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from ThalamusEngine.Interface import LoadThalamusInterface, InitEngine, Pixelto3D, Convert3DtoPixel


meter_per_enc = 11600
rotate_pi_per_enc = 4819
TRACK_MAX_PAN_STEP_DEG = 3.0
TRACK_MAX_TILT_STEP_DEG = 3.0
TRACK_PAN_DEADZONE_DEG = 2.0
TRACK_TILT_DEADZONE_DEG = 2.0

_ASM_FILE = os.path.join(os.path.expanduser("~/ros_workplace"), "ScriptFreeModel.txt").encode("UTF-8")
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


def get_tilt_servo_angle(cam_angle):
    _CAM_ANGLES   = np.array([20, 10, 0, -13, -27, -40, -54, -90], dtype=float)
    _SERVO_ANGLES = np.array([ 0,  9, 19,  29,  39,  49,  59,  69], dtype=float)
    cam_angle = np.clip(cam_angle, _CAM_ANGLES.min(), _CAM_ANGLES.max())
    servo = np.interp(cam_angle, _CAM_ANGLES[::-1], _SERVO_ANGLES[::-1])
    return int(np.clip(servo, 0, 90))


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
            UIElement(_type=UIElementType.BUTTON, _caption="Quit",          _default_val=self.parent_window.quit_app),
            UIElement(_type=UIElementType.BUTTON, _caption="Basic control", _default_val=self.focus_basic_window),
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
                    _default_val=lambda state, _t=t, _p=p: self.preset_control(_t, _p),
                    _defalt_culumn=colspan,
                ))
            ele_list.append(UIElement(_type=UIElementType.NEW_LINE))

        mainGrid = QGridLayout()
        subgrid, dic = self.createGroupBox("Preset Cam Control", ele_list)
        self.ui_dic_list.append(dic)
        mainGrid.addWidget(subgrid, 0, 0)

        # ── Auto-Control info ──────────────────────────────────
        info_ele_list = [

            UIElement(_type=UIElementType.EDIT, _caption="MaxRotate", _default_val="30"),
            UIElement(_type=UIElementType.EDIT, _caption="MaxRotMargin", _default_val="6"),
            UIElement(_type=UIElementType.EDIT, _caption="InPosRotAngle", _default_val="1"),
            UIElement(_type=UIElementType.EDIT, _caption="VioRotAngle", _default_val="90"),
            UIElement(_type=UIElementType.BUTTON, _caption="vio_rot", _default_val=self.vio_rotation),
            UIElement(_type=UIElementType.BUTTON, _caption="vio_estop", _default_val=self.vio_estop),
            UIElement(_type=UIElementType.NEW_LINE),


            UIElement(_type=UIElementType.EDIT,   _caption="ClickPosX",       _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="ClickPosY",       _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="TgtPan",          _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="TgtTilt",         _default_val="0"),
            UIElement(_type=UIElementType.BUTTON, _caption="set_clicked_pos", _default_val=self.set_clicked_pos),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_X",      _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_Y",      _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_Yaw",    _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="local_action_yaw", _default_val="0"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.COMBO, _caption="Depth_Src", _default_val=["Depth_Map", "Virtual_Plane"]),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosX", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosY", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosZ", _default_val="0"),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT, _caption="TgtBearing", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtDist",    _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtRelDist", _default_val="-1"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.COMBO,  _caption="CamTrackMode", _default_val=["None", "fix_cam_rotation", "fix_on_target"]),
            UIElement(_type=UIElementType.BUTTON, _caption="excute_first_action", _default_val=self.excute_first_action),
            UIElement(_type=UIElementType.EDIT,   _caption="local_pos",      _default_val="0,0,0"),
            UIElement(_type=UIElementType.EDIT,   _caption="local_cam_pose", _default_val="0,0,0"),
            UIElement(_type=UIElementType.BUTTON, _caption="init_local",     _default_val=self.init_local),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.LISTBOX, _caption="SeqCmd", _default_val=None, _defalt_row=3, _defalt_culumn=8),




        ]
        subgrid2, dic2 = self.createGroupBox("Semi-Auto-Control", info_ele_list)
        self.ui_dic_list.append(dic2)
        self.bind_listbox_changed(self.on_listbox_changed)
        mainGrid.addWidget(subgrid2, 1, 0)

        # ── Image Processing ──────────────────────────────────
        info_ele_list = [
            UIElement(_type=UIElementType.EDIT, _caption="conf", _default_val="0.25"),
            UIElement(_type=UIElementType.EDIT, _caption="iou",  _default_val="0.5"),
            UIElement(_type=UIElementType.BUTTON, _caption="predict_light", _default_val=self.predict_light),
            UIElement(_type=UIElementType.BUTTON, _caption="predict_person", _default_val=self.predict_person),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="dist_th",    _default_val="50.0"),
            UIElement(_type=UIElementType.EDIT,   _caption="close_iter", _default_val="3"),
            UIElement(_type=UIElementType.EDIT,   _caption="forced_px",  _default_val="128,230"),
            UIElement(_type=UIElementType.EDIT,   _caption="normal_th",  _default_val="0.7"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.BUTTON, _caption="predict_floor_ransac",   _default_val=self.predict_floor_ransac),
            UIElement(_type=UIElementType.BUTTON, _caption="clr_acc_bird_view",      _default_val=self.clr_acc_bird_view),
            UIElement(_type=UIElementType.EDIT, _caption="stable_time(ms)", _default_val="500"),
            UIElement(_type=UIElementType.BUTTON, _caption="act_get_local_birdview", _default_val=self.act_get_local_birdview),
            UIElement(_type=UIElementType.BUTTON, _caption="get_global_birdview",    _default_val=self.get_global_birdview),
            UIElement(_type=UIElementType.BUTTON, _caption="clear_tmp_localmap",     _default_val=self.clear_tmp_localmap),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT, _caption="gmap_ox", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="gmap_oy", _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="scale_per1pix", _default_val="20"),
            UIElement(_type=UIElementType.BUTTON, _caption="set_gram_param", _default_val=self.set_gram_param),
            UIElement(_type=UIElementType.BUTTON, _caption="merge_on_globalmap", _default_val=self.merge_on_globalmap),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.BUTTON, _caption="image_proc_odometry", _default_val=self.image_proc_odometry),
            UIElement(_type=UIElementType.BUTTON, _caption="reset_vio",          _default_val=self.on_reset_vio),
            UIElement(_type=UIElementType.BUTTON, _caption="reset_optical_flow", _default_val=self.on_reset_optical_flow),
        ]
        subgrid2, dic2 = self.createGroupBox("Image Process", info_ele_list)
        self.ui_dic_list.append(dic2)
        self.bind_listbox_changed(self.on_listbox_changed)
        mainGrid.addWidget(subgrid2, 2, 0)

        self.setLayout(mainGrid)
        self.setWindowTitle("Auto Control")
        self.resize(1200, 400)

    # ── Image Processing ──────────────────────────────────
    def _get_camera_frame(self):
        if not self.parent_window._streaming:
            print("[predict] 스트리밍이 열려있지 않습니다. open_streaming 먼저 실행하세요.")
            return None
        img = self.parent_window.ros_node.rgb_image
        if img is None:
            print("[predict] 카메라 이미지가 없습니다.")
            return None
        return img.copy()

    def _draw_and_show(self, img, results, names, window_title):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow(window_title, img)
        cv2.waitKey(1)

    def predict_light(self):
        img = self._get_camera_frame()
        if img is None:
            return
        if not hasattr(self, '_model_light') or self._model_light is None:
            print(f"[predict_light] 모델 로딩: {_LIGHT_WEIGHT}")
            self._model_light = YOLO(_LIGHT_WEIGHT)
        conf = float(self.get_ui_val("conf") or 0.25)
        iou  = float(self.get_ui_val("iou")  or 0.5)
        results = self._model_light(img, conf=conf, iou=iou)[0]
        self._draw_and_show(img, results, self._model_light.names, "Light Detection")

    def predict_person(self):
        img = self._get_camera_frame()
        if img is None:
            return
        if not hasattr(self, '_model_person') or self._model_person is None:
            print(f"[predict_person] 모델 로딩: {_PERSON_WEIGHT}")
            self._model_person = YOLO(_PERSON_WEIGHT)
        conf = float(self.get_ui_val("conf") or 0.25)
        iou  = float(self.get_ui_val("iou")  or 0.5)
        results = self._model_person(img, conf=conf, iou=iou)[0]
        self._draw_and_show(img, results, self._model_person.names, "Person Detection")


    def set_gram_param(self):
        if not hasattr(self, '_globalmap'):
            self._globalmap = np.zeros((512, 512, 3), dtype=np.uint8)

        mm_per_pix = float(self.get_ui_val("scale_per1pix") or 20.0)
        scale      = 1.0 / mm_per_pix
        img_size   = 512
        cx_g       = img_size // 2
        cy_g       = img_size // 2

        ox_mm = float(self.get_ui_val("gmap_ox") or 0.0) * 1000.0
        oy_mm = float(self.get_ui_val("gmap_oy") or 0.0) * 1000.0

        # 세계 좌표 (ox_mm, oy_mm) 위치를 맵 픽셀로 변환
        cx_cross = int(cx_g + ox_mm * scale)
        cy_cross = int(cy_g - oy_mm * scale)

        # 십자가 팔 길이: 1m = 1000mm / mm_per_pix pixels
        arm = int(1000.0 * scale)

        display = self._globalmap.copy()
        cv2.line(display, (cx_cross - arm, cy_cross), (cx_cross + arm, cy_cross), (0, 255, 0), 2)
        cv2.line(display, (cx_cross, cy_cross - arm), (cx_cross, cy_cross + arm), (0, 255, 0), 2)

        cv2.imshow("Global Map", display)
        cv2.waitKey(1)



    def merge_on_globalmap(self):
        if not hasattr(self, '_last_floor_pts') or self._last_floor_pts is None:
            print("[merge_on_globalmap] predict_floor_ransac 먼저 실행하세요.")
            return

        if not hasattr(self, '_globalmap'):
            self._globalmap = np.zeros((512, 512, 3), dtype=np.uint8)

        img_size   = 512
        mm_per_pix = float(self.get_ui_val("scale_per1pix") or 20.0)
        scale      = 1.0 / mm_per_pix
        ox_mm      = float(self.get_ui_val("gmap_ox") or 0.0) * 1000.0
        oy_mm      = float(self.get_ui_val("gmap_oy") or 0.0) * 1000.0
        cx_g       = img_size // 2
        cy_g       = img_size // 2

        odom_x   = self.parent_window.odom_x[-1]   if self.parent_window.odom_x   else 0.0
        odom_y   = self.parent_window.odom_y[-1]   if self.parent_window.odom_y   else 0.0
        odom_yaw = self.parent_window.odom_yaw[-1] if self.parent_window.odom_yaw else 90.0
        yaw_rad  = np.deg2rad(odom_yaw)
        print(f"[merge_on_globalmap] odom_x={odom_x:.3f}m  odom_y={odom_y:.3f}m  odom_yaw={odom_yaw:.1f}deg")

        # 바닥 포인트만 추출
        pts = self._last_floor_pts.reshape(-1, 3)
        pts = pts[pts[:, 2] > 0]
        if len(pts) == 0:
            return

        # 카메라 프레임 → 로봇 프레임 (roll + pitch + heading 일괄 회전)
        r = np.deg2rad(self._last_cam_roll)
        p = np.deg2rad(self._last_cam_pitch)
        h = np.deg2rad(self._last_cam_heading)
        Rz = np.array([[np.cos(r), -np.sin(r), 0],
                       [np.sin(r),  np.cos(r), 0],
                       [0,          0,          1]])
        Rx = np.array([[1, 0,          0         ],
                       [0, np.cos(p), -np.sin(p) ],
                       [0, np.sin(p),  np.cos(p) ]])
        Ry = np.array([[np.cos(h),  0, np.sin(h)],
                       [0,          1, 0         ],
                       [-np.sin(h), 0, np.cos(h)]])
        R = Ry @ Rx @ Rz
        pts_robot = (R @ pts.T).T  # (N, 3)

        x_r = pts_robot[:, 0]
        z_r = pts_robot[:, 2] + 100.0  # cam_axis_pos=100mm 보정 → 로봇 회전축 기준

        # 로봇 프레임 → 월드 프레임 (mm)
        # forward 방향: (cos(yaw), sin(yaw)) in (odom_x, odom_y)
        # right   방향: (sin(yaw), -cos(yaw))
        odom_x_mm = odom_x * 1000.0
        odom_y_mm = odom_y * 1000.0
        world_x = odom_x_mm + z_r * np.cos(yaw_rad) + x_r * np.sin(yaw_rad)
        world_y = odom_y_mm + z_r * np.sin(yaw_rad) - x_r * np.cos(yaw_rad)

        # 현재 프레임 임시 레이어에 포인트 찍기 (세계 0,0 = 맵 중앙)
        gx = (cx_g + world_x * scale).astype(int)
        gy = (cy_g - world_y * scale).astype(int)
        valid = (gx >= 0) & (gx < img_size) & (gy >= 0) & (gy < img_size)
        layer = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        layer[gy[valid], gx[valid]] = 255

        # MORPH_CLOSE로 구멍 메운 뒤 글로벌맵에 병합
        close_iter = int(self.get_ui_val("close_iter") or 3)
        if close_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        self._globalmap[layer[:, :, 0] > 0] = 255

        cv2.imshow("Global Map", self._globalmap)
        cv2.waitKey(1)

    def predict_floor_ransac(self):
        if not self.parent_window._streaming:
            print("[predict_floor_ransac] 스트리밍이 열려있지 않습니다.")
            return
        depth = self.parent_window.ros_node.depth_image
        if depth is None:
            print("[predict_floor_ransac] depth 이미지가 없습니다.")
            return

        dist_th   = float(self.get_ui_val("dist_th")  or 50.0)
        normal_th = float(self.get_ui_val("normal_th") or 0.7)

        forced_pixel = None
        forced_px_str = (self.get_ui_val("forced_px") or "").strip()
        if forced_px_str:
            try:
                fx, fy = [int(v.strip()) for v in forced_px_str.split(",")]
                forced_pixel = (fx, fy)
            except Exception:
                print(f"[predict_floor_ransac] forced_px 파싱 실패: '{forced_px_str}' (예: 100,200)")

        att = self.parent_window.ros_node.attitude
        cam_pitch = att.y if att is not None else 0.0
        cam_roll  = att.x if att is not None else 0.0

        depth_mm = depth * 1000.0  # m → mm
        img_h, img_w = depth_mm.shape
        print(f"[predict_floor_ransac] depth image size: {img_w}x{img_h}")

        display_mask, floor_pts = depthmap_floor_ransac(
            depth_mm, (720, 1280), dist_th, cam_pitch, cam_roll,
            forced_pixel=forced_pixel, normal_th=normal_th,
        )

        rgb = self.parent_window.ros_node.rgb_image
        vis = cv2.resize(rgb, (img_w, img_h)) if rgb is not None else np.zeros((img_h, img_w, 3), dtype=np.uint8)
        vis[display_mask[:, :, 0] > 0, 1] = 255

        close_iter  = int(self.get_ui_val("close_iter") or 3)
        cam_heading = (self.get_ui_val("PanningAngle", float) or 0.0) - (self.get_ui_val("N", float) or 0.0)
        birdview = make_birdview(floor_pts, cam_pitch, cam_roll, close_iter=close_iter,
                                 cam_heading=cam_heading, cam_axis_pos=100.0)

        # 다음 merge_on_globalmap을 위해 저장
        self._last_floor_pts   = floor_pts
        self._last_cam_pitch   = cam_pitch
        self._last_cam_roll    = cam_roll
        self._last_cam_heading = cam_heading

        # Accumulated Bird View: 현재 Bird View와 동일한 사이즈로 선언 후 누적
        if not hasattr(self, '_acc_birdview') or self._acc_birdview.shape != birdview.shape:
            self._acc_birdview = np.zeros_like(birdview)
        self._acc_birdview = cv2.bitwise_or(self._acc_birdview, birdview)

        cv2.imshow("Floor RANSAC", vis)
        cv2.imshow("Floor Mask", display_mask)
        cv2.imshow("Bird View", birdview)
        cv2.imshow("Accumulated Bird View", self._acc_birdview)
        cv2.waitKey(1)

    def clr_acc_bird_view(self):
        if not hasattr(self, '_acc_birdview'):
            print("[clr_acc_bird_view] 누적된 Bird View가 없습니다.")
            return
        self._acc_birdview[:] = 0
        cv2.imshow("Accumulated Bird View", self._acc_birdview)
        cv2.waitKey(1)

    def act_get_local_birdview(self):
        """카메라를 LL→L→N→R→RR (틸트 DN 고정) 순서로 돌리며 매 위치마다
        stable_time(ms) 만큼 정착 대기 후 predict_floor_ransac을 실행,
        Accumulated Bird View에 누적한다."""
        self.clr_acc_bird_view()

        stable_ms  = self.get_ui_val("stable_time(ms)", float)
        stable_sec = (stable_ms if stable_ms is not None else 500.0) / 1000.0

        presets = [("DN", "LL"), ("DN", "L"), ("DN", "N"), ("MD", "N"),  ("DN", "R"), ("DN", "RR")]
        for tilt_key, pan_key in presets:
            print(f"[act_get_local_birdview] preset tilt={tilt_key} pan={pan_key}, "
                  f"stable_time={stable_sec:.2f}s 대기 후 predict_floor_ransac")
            self.preset_control(tilt_key, pan_key)
            time.sleep(stable_sec)
            self.predict_floor_ransac()

        if hasattr(self, '_acc_birdview'):
            cv2.imshow("Accumulated Bird View", self._acc_birdview)
            cv2.waitKey(1)

        self._save_local_birdview()
        print("[act_get_local_birdview] 완료")

    def _save_local_birdview(self):
        """tmp_local_view/birdviewNN.png + birdviewNN.json 으로
        현재 Accumulated Bird View, 오도메트리 위치, 카메라 자세를 저장."""
        if not hasattr(self, '_acc_birdview'):
            print("[act_get_local_birdview] 저장할 Accumulated Bird View가 없습니다.")
            return

        save_dir = os.path.expanduser('~/ros_workplace/tmp_local_view')
        os.makedirs(save_dir, exist_ok=True)

        idx = 0
        while os.path.exists(os.path.join(save_dir, f'birdview{idx:02d}.png')):
            idx += 1
        png_name  = f'birdview{idx:02d}.png'
        json_name = f'birdview{idx:02d}.json'

        pw = self.parent_window
        odom_x   = pw.odom_x[-1]   if pw.odom_x   else 0.0
        odom_y   = pw.odom_y[-1]   if pw.odom_y   else 0.0
        odom_yaw = pw.odom_yaw[-1] if pw.odom_yaw else 0.0

        att = pw.ros_node.attitude
        cam_roll  = float(att.x) if att is not None else 0.0
        cam_pitch = float(att.y) if att is not None else 0.0
        cam_yaw   = float(att.z) if att is not None else 0.0

        data = {
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "odometry":  {"x": odom_x, "y": odom_y, "yaw": odom_yaw},
            "attitude":  {"roll": cam_roll, "pitch": cam_pitch, "yaw": cam_yaw},
            "birdview_png": png_name,
        }

        cv2.imwrite(os.path.join(save_dir, png_name), self._acc_birdview)
        with open(os.path.join(save_dir, json_name), 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[act_get_local_birdview] 저장 완료: {save_dir}/{png_name}, {json_name}")

    def get_global_birdview(self):
        """tmp_local_view에 저장된 모든 birdview*.json/png를 각 스냅샷의
        오도메트리(x,y,yaw)를 이용해 world 프레임으로 정합, 하나의 큰
        Global Bird View로 합쳐서 저장/표시한다."""
        load_dir = os.path.expanduser('~/ros_workplace/tmp_local_view')
        if not os.path.isdir(load_dir):
            print(f"[get_global_birdview] {load_dir} 없음")
            return

        json_files = sorted(f for f in os.listdir(load_dir) if f.endswith('.json'))
        snapshots = []
        for jf in json_files:
            try:
                with open(os.path.join(load_dir, jf)) as f:
                    meta = json.load(f)
                png_path = os.path.join(load_dir, meta.get('birdview_png', ''))
                img = cv2.imread(png_path)
                if img is None:
                    print(f"[get_global_birdview] {png_path} 로드 실패, 스킵")
                    continue
                snapshots.append((meta, img))
            except Exception as e:
                print(f"[get_global_birdview] {jf} 파싱 실패: {e}")

        if not snapshots:
            print("[get_global_birdview] tmp_local_view에 병합할 birdview가 없습니다.")
            return

        mm_per_px = 20.0          # local birdview / merge_on_globalmap 기본값과 동일 축척 (2cm/px)
        scale     = 1.0 / mm_per_px

        # world(mm) 범위 계산 → 캔버스 크기/원점 결정 (각 스냅샷의 로컬 맵 반경만큼 여유)
        odom_xs_mm = [meta["odometry"]["x"] * 1000.0 for meta, _ in snapshots]
        odom_ys_mm = [meta["odometry"]["y"] * 1000.0 for meta, _ in snapshots]
        margin_mm  = max(max(img.shape[0], img.shape[1]) for _, img in snapshots) / scale
        min_x, max_x = min(odom_xs_mm) - margin_mm, max(odom_xs_mm) + margin_mm
        min_y, max_y = min(odom_ys_mm) - margin_mm, max(odom_ys_mm) + margin_mm

        g_w = max(int((max_x - min_x) * scale), 1)
        g_h = max(int((max_y - min_y) * scale), 1)
        global_map = np.zeros((g_h, g_w, 3), dtype=np.uint8)

        for meta, local_img in snapshots:
            odom_x_mm = meta["odometry"]["x"] * 1000.0
            odom_y_mm = meta["odometry"]["y"] * 1000.0
            yaw_rad   = np.deg2rad(meta["odometry"]["yaw"])

            h, w = local_img.shape[:2]
            lcx, lcy = w // 2, int(h * 0.8)   # make_birdview과 동일한 로봇 축 픽셀

            floor_mask = np.all(local_img == 255, axis=2)   # 마커(빨강/파랑) 제외, 순백=바닥만
            py_idx, px_idx = np.nonzero(floor_mask)
            if py_idx.size == 0:
                continue

            z_r = (lcy - py_idx) / scale   # 로봇 축 기준 전방 거리 (mm)
            x_r = (px_idx - lcx) / scale   # 로봇 축 기준 우측 거리 (mm)
            world_x = odom_x_mm + z_r * np.cos(yaw_rad) + x_r * np.sin(yaw_rad)
            world_y = odom_y_mm + z_r * np.sin(yaw_rad) - x_r * np.cos(yaw_rad)

            gx = ((world_x - min_x) * scale).astype(int)
            gy = (g_h - (world_y - min_y) * scale).astype(int)
            valid = (gx >= 0) & (gx < g_w) & (gy >= 0) & (gy < g_h)
            global_map[gy[valid], gx[valid]] = 255

        # 각 스냅샷의 오도메트리 위치를 로봇 축으로 표시
        for meta, _ in snapshots:
            odom_x_mm = meta["odometry"]["x"] * 1000.0
            odom_y_mm = meta["odometry"]["y"] * 1000.0
            gx = int((odom_x_mm - min_x) * scale)
            gy = int(g_h - (odom_y_mm - min_y) * scale)
            if 0 <= gx < g_w and 0 <= gy < g_h:
                cv2.circle(global_map, (gx, gy), 3, (0, 0, 255), -1)

        save_path = os.path.join(load_dir, 'global_birdview.png')
        cv2.imwrite(save_path, global_map)
        cv2.imshow("Global Bird View", global_map)
        cv2.waitKey(1)
        print(f"[get_global_birdview] 스냅샷 {len(snapshots)}개 병합 완료 "
              f"({g_w}x{g_h}, {mm_per_px:.0f}mm/px) → {save_path}")

    def clear_tmp_localmap(self):
        """tmp_local_view 디렉토리 안의 birdview*.png/json, global_birdview.png를 모두 삭제."""
        load_dir = os.path.expanduser('~/ros_workplace/tmp_local_view')
        if not os.path.isdir(load_dir):
            print(f"[clear_tmp_localmap] {load_dir} 없음")
            return

        removed = 0
        for fname in os.listdir(load_dir):
            if fname.endswith('.png') or fname.endswith('.json'):
                try:
                    os.remove(os.path.join(load_dir, fname))
                    removed += 1
                except OSError as e:
                    print(f"[clear_tmp_localmap] {fname} 삭제 실패: {e}")

        print(f"[clear_tmp_localmap] {load_dir} 안 파일 {removed}개 삭제 완료")

    def qt_imshow(self, title, img):
        cv2.imshow(title, img)
        cv2.waitKey(1)

    def image_proc_odometry(self):
        if not hasattr(self, '_proc_odom_timer'):
            self._proc_odom_local_idx = 0
            self._proc_odom_timer = QTimer(self)
            self._proc_odom_timer.timeout.connect(self._proc_odom_tick)

        if self._proc_odom_timer.isActive():
            self._proc_odom_timer.stop()
            print("[image_proc_odometry] 중지")
        else:
            if not self.parent_window._streaming:
                self.parent_window.open_streaming()
            self._proc_odom_local_idx = 0
            self._proc_odom_timer.start(50)
            print("[image_proc_odometry] 시작")

    def reset_proc_odom_idx(self):
        """MoveFoward/MoveBackward/RotateLeft/RotateRight 등 로버 이동 명령 시 호출.
        img_proc_odometry의 local_idx 카운터를 0으로 되돌려 reset_img_proc_odometry()의
        전역 플래그 리셋과 실제 추적 재초기화 주기를 동기화한다."""
        if hasattr(self, '_proc_odom_local_idx'):
            self._proc_odom_local_idx = 0
            print("[reset_proc_odom_idx] _proc_odom_local_idx=0")

    def on_reset_vio(self):
        """vio_yaw / vio_dist만 0으로 리셋 (옵티컬 플로우 기준점은 그대로)."""
        reset_vio()
        print("[reset_vio] vio_yaw, vio_dist = 0")

    def on_reset_optical_flow(self):
        """longterm/shortterm 옵티컬 플로우 기준점만 다시 잡음 (vio_yaw/vio_dist는 그대로)."""
        reset_optical_flow()
        print("[reset_optical_flow] longterm/shortterm 기준점 재설정 예약됨")

    # ── VIO 피드백 기반 점진적 회전 ──────────────────────────────
    def vio_rotation(self):
        """VioRotAngle만큼 카메라(패닝)를 회전시키되, 한 번에 MaxRotate(deg)씩만
        나눠서 명령하고 그 결과(vio_yaw, 누적치)를 보며 다음 스텝 각도를 보정한다.

        예) VioRotAngle=90, MaxRotate=30, MaxRotMargin=6
            1) 30 명령 -> 실제 28 달성 (vio_yaw 누적)
            2) 다음 이상적 누적 목표 = min(2*30, 90) = 60, 60-28=32 -> 32 명령
               (32 <= MaxRotate+MaxRotMargin=36 이므로 허용)
            3) |남은 각도| <= InPosRotAngle 이 될 때까지 반복

        reset_vio/reset_optical_flow는 시작할 때 딱 한 번만 하고, 이후 스텝들은
        하지 않아서 vio_yaw가 전체 시퀀스에 걸쳐 누적되도록 한다.
        vio_estop()으로 언제든 중단 가능하도록 QTimer 기반 논블로킹으로 동작한다.
        """
        if getattr(self, '_vio_rot_active', False):
            print("[vio_rotation] 이미 진행 중입니다. 먼저 vio_estop으로 멈춰주세요.")
            return

        target = self.get_ui_val("VioRotAngle", float)
        if not target:
            print("[vio_rotation] VioRotAngle 값이 없거나 0입니다.")
            return
        max_rot = self.get_ui_val("MaxRotate", float) or 30.0
        if max_rot <= 0:
            print("[vio_rotation] MaxRotate는 0보다 커야 합니다.")
            return

        self._vio_rot_active      = True
        self._vio_rot_stop_req    = False
        self._vio_rot_direction   = 1.0 if target >= 0 else -1.0
        self._vio_rot_target_abs  = abs(target)
        self._vio_rot_max_rot     = max_rot
        self._vio_rot_margin      = self.get_ui_val("MaxRotMargin", float) or 0.0
        self._vio_rot_tol         = self.get_ui_val("InPosRotAngle", float) or 1.0
        self._vio_rot_checkpoint  = 0.0
        self._vio_rot_step_count  = 0
        self._vio_rot_max_steps   = int(np.ceil(self._vio_rot_target_abs / self._vio_rot_max_rot)) + 10
        self._vio_rot_seen_moving = False
        self._vio_rot_cmd_issued  = False  # 이번 시퀀스에서 RT 커맨드를 한 번이라도 냈는지
        self._vio_rot_cmd_time    = None   # 마지막 커맨드를 낸 시각 (stat 타임아웃 판단용)

        print(f"[vio_rotation] 시작: target={target}deg MaxRotate={max_rot} "
              f"MaxRotMargin={self._vio_rot_margin} InPosRotAngle={self._vio_rot_tol}")

        # 1) 카메라 뉴트럴(N/MD)
        self.preset_control("MD", "N")
        QTimer.singleShot(300, self._vio_rot_setup)

    def _vio_rot_setup(self):
        if not self._vio_rot_active or self._vio_rot_stop_req:
            self._vio_rot_finish("시작 전 중단됨")
            return

        # 2) image_proc_odometry 추적 타이머 확인 후 안 켜져 있으면 켬
        if not hasattr(self, '_proc_odom_timer') or not self._proc_odom_timer.isActive():
            self.image_proc_odometry()

        # 3) VIO / 옵티컬 플로우 기준점 리셋 (시퀀스 전체에서 딱 한 번)
        reset_vio()
        reset_optical_flow()

        if not hasattr(self, '_vio_rot_timer'):
            self._vio_rot_timer = QTimer(self)
            self._vio_rot_timer.timeout.connect(self._vio_rot_tick)
        self._vio_rot_timer.start(150)

    _VIO_ROT_STEP_TIMEOUT = 4.0  # stat(모션 시작/종료) 신호를 못 받을 때 강제로 다음 단계로 넘어가기까지의 대기시간(s)

    def _vio_rot_tick(self):
        if self._vio_rot_stop_req:
            self._vio_rot_finish("vio_estop으로 중단")
            return

        if self._vio_rot_cmd_issued:
            # 이전에 낸 커맨드가 있으면, 그게 끝났는지(움직임 시작 -> 정지 엣지) 확인.
            # stat1/stat2가 기대만큼 0으로 안 돌아오거나(혹은 애초에 3으로 안 올라가거나)
            # 하는 하드웨어 특성이 있을 수 있으므로, 타임아웃 안에서는 정상 엣지 감지로
            # 기다리고 타임아웃을 넘기면 stat 상태와 무관하게 강제로 다음 단계로 넘어간다.
            md = self.parent_window.ros_node.motion_data
            is_moving = (md is not None and md[4] != 0 and md[5] != 0)
            elapsed = time.time() - self._vio_rot_cmd_time

            if elapsed < self._VIO_ROT_STEP_TIMEOUT:
                if is_moving:
                    self._vio_rot_seen_moving = True
                    return
                if not self._vio_rot_seen_moving:
                    return  # 아직 이전 명령이 모터에 반영되기 전 - 다음 틱에 재확인
                # else: 정상적으로 움직였다가 멈춘 것 확인됨 -> 아래로 진행
            else:
                print(f"[vio_rotation] {self._VIO_ROT_STEP_TIMEOUT:.1f}s 타임아웃 -> "
                      f"stat 상태(is_moving={is_moving})와 무관하게 완료로 간주하고 진행")

            self._vio_rot_seen_moving = False  # 이번 스텝의 모션 종료 처리
        # else: 아직 커맨드를 한 번도 안 냈으면(첫 스텝) 모션 대기 없이 바로 아래로 진행

        achieved  = get_last_vio_yaw() * self._vio_rot_direction  # 목표 방향 기준으로 부호 정리
        remaining = self._vio_rot_target_abs - achieved

        if abs(remaining) <= self._vio_rot_tol:
            self._vio_rot_finish(f"목표 도달 (achieved={achieved:.2f}deg)")
            return

        self._vio_rot_step_count += 1
        if self._vio_rot_step_count > self._vio_rot_max_steps:
            self._vio_rot_finish(f"최대 스텝({self._vio_rot_max_steps}) 초과, 수렴 실패 "
                                 f"(achieved={achieved:.2f}deg)")
            return

        if remaining > 0:
            # 아직 목표에 못 미침 -> 원래 방향으로, 이상적 누적 스케줄 대비 부족분만큼 전진
            self._vio_rot_checkpoint = min(self._vio_rot_checkpoint + self._vio_rot_max_rot,
                                            self._vio_rot_target_abs)
            cmd = self._vio_rot_checkpoint - achieved
            cmd = max(0.0, min(cmd, self._vio_rot_max_rot + self._vio_rot_margin))
            cmd = min(cmd, remaining)  # 최종 목표를 넘어가지 않도록
            step_dir = self._vio_rot_direction
        else:
            # 목표를 지나침(오버슈트) -> 반대 방향으로 되돌아오는 보정 커맨드
            cmd = min(-remaining, self._vio_rot_max_rot + self._vio_rot_margin)
            step_dir = -self._vio_rot_direction
            self._vio_rot_checkpoint = achieved  # 다음 스케줄 기준점을 실제 위치로 재조정

        if cmd <= 1e-6:
            self._vio_rot_finish(f"진행 불가(cmd<=0), achieved={achieved:.2f}deg")
            return

        print(f"[vio_rotation] step{self._vio_rot_step_count}: achieved={achieved:.2f} "
              f"remaining={remaining:.2f} -> cmd={cmd:.2f} ({'정' if step_dir > 0 else '역'}방향)")

        self.set_ui_val("RotAngle", f"{cmd:.2f}")
        if step_dir > 0:
            self.parent_window.RotateRight()
        else:
            self.parent_window.RotateLeft()

        self._vio_rot_cmd_issued = True
        self._vio_rot_cmd_time   = time.time()

    def vio_estop(self):
        """vio_rotation 진행 중이면 즉시 정지."""
        if not getattr(self, '_vio_rot_active', False):
            print("[vio_estop] 진행 중인 vio_rotation이 없습니다.")
            return
        self._vio_rot_stop_req = True
        self.parent_window.MotCancel()
        print("[vio_estop] 정지 요청")

    def _vio_rot_finish(self, msg):
        if hasattr(self, '_vio_rot_timer'):
            self._vio_rot_timer.stop()
        self._vio_rot_active = False
        # MotCancel()이 stop_vio()도 호출해서 _vio_cmd_type을 꺼줌 -> 시퀀스가 끝난 뒤
        # 백그라운드 image_proc_odometry 트래커가 계속 calcul_yaw()를 재계산하지 않게 함
        self.parent_window.MotCancel()
        print(f"[vio_rotation] 종료: {msg}")

    def _proc_odom_tick(self):
        if not self.parent_window._streaming:
            return
        rgb   = self.parent_window.ros_node.rgb_image
        depth = self.parent_window.ros_node.depth_image
        if rgb is None or depth is None:
            return

        odom_x   = self.parent_window.odom_x[-1]   if self.parent_window.odom_x   else 0.0
        odom_y   = self.parent_window.odom_y[-1]   if self.parent_window.odom_y   else 0.0
        odom_yaw = (self.parent_window.odom_yaw[-1] - self.parent_window.odom_yaw[0]) \
                   if len(self.parent_window.odom_yaw) > 1 else 0.0
        local_odometry = np.array([odom_x, odom_y, odom_yaw])

        att = self.parent_window.ros_node.attitude
        pose_ars = [att.z, att.x, att.y, att.z] if att is not None else [0.0, 0.0, 0.0, 0.0]

        img_proc_odometry(self,
                          image=rgb.copy(),
                          depthmap=depth.copy(),
                          local_idx=self._proc_odom_local_idx,
                          local_odometry=local_odometry,
                          pose_ars=pose_ars,
                          cmd_type=None)
        self._proc_odom_local_idx += 1

    def focus_basic_window(self):
        self.parent_window.hide()
        self.parent_window.show()

    # ── preset cam control ─────────────────────────────────────
    def preset_control(self, tilt, pan):
        self.set_ui_val("TiltAngle",    self.get_ui_val(tilt))
        self.set_ui_val("PanningAngle", self.get_ui_val(pan))
        self.parent_window.SetTilt()
        time.sleep(0.2)
        self.parent_window.SetPanning()

    def set_clicked_pos(self):
        self.set_ui_val("TiltAngle", self.get_ui_val("TgtTilt"))
        self.parent_window.SetTilt()

        pan = (self.get_ui_val("PanningAngle", float) or 0.0) + (self.get_ui_val("TgtPan", float) or 0.0)
        self.set_ui_val("PanningAngle", str(int(pan)))
        self.parent_window.SetPanning()

    def init_local(self):
        pw = self.parent_window
        self._local_ref_x   = pw.odom_x[-1]   if pw.odom_x   else 0.0
        self._local_ref_y   = pw.odom_y[-1]   if pw.odom_y   else 0.0
        self._local_ref_yaw = pw.odom_yaw[-1] if pw.odom_yaw else 0.0
        # Ctrl+click 시점 카메라 기준값 저장(이후에는 현재값-기준값 델타만 사용)
        # UI edit 값이 아닌 attitude 원본값을 사용해 수동 입력 영향 제거
        att = pw.ros_node.attitude
        self._local_ref_cam_yaw = float(att.z) if att is not None else 0.0
        self._local_ref_pitch = float(att.y) if att is not None else 0.0
        self._local_ref_roll  = float(att.x) if att is not None else 0.0
        self.set_ui_val("local_pos", "0,0,0")
        self.set_ui_val("local_cam_pose", "0.0,0.0,0.0")

    def update_local(self, odom_x, odom_y, odom_yaw):
        if not hasattr(self, '_local_ref_x') or self._local_ref_x is None:
            return
        dx = odom_x - self._local_ref_x
        dy = odom_y - self._local_ref_y
        self.set_ui_val("local_pos", f"{dx:.3f},{dy:.3f},0.0")
        pw    = self.parent_window
        att = pw.ros_node.attitude
        pitch = float(att.y) if att is not None else 0.0
        roll  = float(att.x) if att is not None else 0.0
        cam_yaw = float(att.z) if att is not None else 0.0
        ref_cam_yaw = getattr(self, '_local_ref_cam_yaw', cam_yaw)
        ref_pitch = getattr(self, '_local_ref_pitch', pitch)
        ref_roll  = getattr(self, '_local_ref_roll', roll)
        # yaw는 실제 카메라 yaw 피드백의 변화량으로 계산
        yaw = cam_yaw - ref_cam_yaw
        pitch_delta = pitch - ref_pitch
        roll_delta = roll - ref_roll
        self.set_ui_val("local_cam_pose", f"{pitch_delta:.1f},{roll_delta:.1f},{yaw:.1f}")

    def update_marker(self):
        """매 프레임 마커를 갱신한다.

        1) 클릭 시 저장한 3D 카메라 좌표를 기준으로 차량 이동/회전을 역보정
        2) 보정된 3D 좌표를 현재 영상 픽셀로 재투영해 빨간 마커 표시
        3) optflow로 초록 마커를 추적하고 pan/tilt 목표를 갱신
        """
        if not hasattr(self, '_marker_cam_pos') or self._marker_cam_pos is None:
            return
        sw = getattr(self.parent_window, '_streaming_window', None)
        if sw is None:
            return
        if not _init_thalamus():
            return

        # 최신 odom + pan 상태를 local 상태값(local_pos/local_cam_pose)에 반영
        pw = self.parent_window
        self.update_local(
            pw.odom_x[-1]   if pw.odom_x   else 0.0,
            pw.odom_y[-1]   if pw.odom_y   else 0.0,
            pw.odom_yaw[-1] if pw.odom_yaw else 0.0,
        )

        # local_cam_pose 형식: "pitch_delta,roll_delta,yaw_delta" (deg)
        local_cam_pose = self.get_ui_val("local_cam_pose") or "0,0,0"
        vals = [s.strip() for s in local_cam_pose.split(",")]
        try:
            local_pitch = float(vals[0]) if len(vals) >= 1 else 0.0
            local_roll = float(vals[1]) if len(vals) >= 2 else 0.0
            local_yaw = float(vals[2]) if len(vals) >= 3 else 0.0
        except ValueError:
            local_pitch = 0.0
            local_roll = 0.0
            local_yaw = 0.0

        # local_pos 형식: "dx,dy,0" (odom 기준 이동량, m)
        # ref_yaw 기준으로 odom 이동량(dx,dy)을 카메라 프레임(X=우, Z=전방)으로 변환
        # 변환된 차량 이동량을 marker 3D 좌표에서 빼서 역병진 적용
        local_pos_str = self.get_ui_val("local_pos") or "0,0,0"
        pvals = [s.strip() for s in local_pos_str.split(",")]
        try:
            dx = float(pvals[0]) if len(pvals) >= 1 else 0.0
            dy = float(pvals[1]) if len(pvals) >= 2 else 0.0
        except ValueError:
            dx, dy = 0.0, 0.0

        ref_rad  = np.deg2rad(getattr(self, '_local_ref_yaw', 0.0))
        veh_cam_z = dx * np.cos(ref_rad) + dy * np.sin(ref_rad)   # 전방(Z)
        veh_cam_x = dx * np.sin(ref_rad) - dy * np.cos(ref_rad)   # 우(X)
        vehicle_cam = np.array([veh_cam_x, 0.0, veh_cam_z])

        cam_pos_adjusted = self._marker_cam_pos - vehicle_cam

        # 클릭 시점 대비 카메라 자세 변화량을 역회전으로 보정
        px = -np.deg2rad(local_pitch)
        rz = -np.deg2rad(local_roll)
        ry = -np.deg2rad(local_yaw)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(px), -np.sin(px)],
                       [0, np.sin(px),  np.cos(px)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0,           1, 0         ],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0,           0,          1]])
        cam_pos_now = (Ry @ Rx @ Rz) @ cam_pos_adjusted

        # 보정된 3D 카메라 좌표를 현재 영상 픽셀로 투영
        pw = self.parent_window
        rgb = pw.ros_node.rgb_image
        if rgb is None:
            return
        img_h, img_w = rgb.shape[:2]

        if cam_pos_now[2] <= 0:
            sw.clear_marker()
            return
        try:
            px_1280, py_720 = Convert3DtoPixel(
                float(cam_pos_now[0]), float(cam_pos_now[1]), float(cam_pos_now[2]))
        except Exception as e:
            print(f"[update_marker] Convert3DtoPixel error: {e}")
            sw.clear_marker()
            return

        px = int(px_1280 * img_w / 1280)
        py = int(py_720  * img_h / 720)
        if 0 <= px < img_w and 0 <= py < img_h:
            sw.set_marker(px, py, self._marker_cam_text,
                          pos_3d=getattr(self, '_marker_pnt_world', None))
        else:
            sw.clear_marker()

        # 초록 마커: optflow 점 중 이전 마커 위치와 가장 가까운 점으로 추적
        if not hasattr(self, '_marker_flow_512_pos') or self._marker_flow_512_pos is None:
            sw.clear_rover_line()
            return
        optflow_pts = get_last_optflow_pts()
        optflow_ids = get_last_optflow_ids()
        if not optflow_pts:
            sw.clear_rover_line()
            return
        # initOptflow 세대가 바뀌면 저장된 ID는 다른 피쳐를 가리키므로 무효화
        current_gen = get_optflow_reset_count()
        if getattr(self, '_marker_flow_reset_gen', None) != current_gen:
            self._marker_flow_feature_id = None
            self._marker_flow_reset_gen  = current_gen

        # 고정된 피쳐 ID로 추적 — ID가 없으면 nearest-neighbor 폴백
        feature_id = getattr(self, '_marker_flow_feature_id', None)
        cx_512, cy_512 = None, None
        if feature_id is not None and feature_id in optflow_ids:
            idx = optflow_ids.index(feature_id)
            cx_512, cy_512 = optflow_pts[idx]
        else:
            # 피쳐가 사라진 경우 현재 위치 기준 nearest-neighbor 로 재포착
            fx, fy = self._marker_flow_512_pos
            best_i = min(range(len(optflow_pts)),
                         key=lambda i: (optflow_pts[i][0] - fx)**2 + (optflow_pts[i][1] - fy)**2)
            cx_512, cy_512 = optflow_pts[best_i]
            self._marker_flow_feature_id = optflow_ids[best_i] if best_i < len(optflow_ids) else None
        self._marker_flow_512_pos = [cx_512, cy_512]
        flow_px = int(cx_512 * img_w / 512)
        flow_py = int(cy_512 * img_h / 512)
        if not (0 <= flow_px < img_w and 0 <= flow_py < img_h):
            return
        # 초록 마커 3D 위치 매 프레임 재계산 (로버 이동에 따른 거리 갱신)
        flow_pnt_world_now = None
        try:
            _, _, flow_pnt_world_now, _ = self._compute_3d(flow_px, flow_py, img_w, img_h)
            self._marker_flow_cam_text = self._world_text(flow_pnt_world_now)
            sw.set_marker2(flow_px, flow_py, self._marker_flow_cam_text,
                           pos_3d=flow_pnt_world_now)
        except Exception:
            sw.set_marker2(flow_px, flow_py, self._marker_flow_cam_text)

        # 마젠타 로버 라인: 초록 마커 → 로버 월드 좌표 (0,0.8,0) 투영
        rover_world = np.array([0.0, 0.8, 0.0])
        odom_yaw_now  = pw.odom_yaw[-1] if pw.odom_yaw else 90.0
        cam_pitch_now = self.get_ui_val("camera_pitch", float) or 0.0
        cam_roll_now  = self.get_ui_val("camera_roll",  float) or 0.0
        cam_pan_now   = (self.get_ui_val("PanningAngle", float) or 0.0) - (self.get_ui_val("N", float) or 0.0)
        rover_cam = self._rotate_vector_inv(
            rover_world, cam_pitch_now, cam_roll_now, odom_yaw_now + cam_pan_now)
        if rover_cam[2] > 0:
            try:
                rvx_1280, rvy_720 = Convert3DtoPixel(
                    float(rover_cam[0]), float(rover_cam[1]), float(rover_cam[2]))
                rvx = int(rvx_1280 * img_w / 1280)
                rvy = int(rvy_720  * img_h / 720)
                if flow_pnt_world_now is not None:
                    rover_dist = float(np.linalg.norm(flow_pnt_world_now - rover_world))
                elif sw._marker2_3d is not None:
                    rover_dist = float(np.linalg.norm(
                        np.asarray(sw._marker2_3d) - rover_world))
                else:
                    rover_dist = None
                sw.set_rover_line(rvx, rvy, rover_dist)
            except Exception as e:
                print(f"[update_marker] rover projection error: {e}")
                sw.clear_rover_line()
        else:
            sw.clear_rover_line()

        # 과이동 감지: 초록-로버 거리가 출발 시 TgtRelDist보다 짧아지면 MotCancel
        tgt_rel_dist = self.get_ui_val("TgtRelDist", float) or -1.0
        if tgt_rel_dist > 0 and sw._rover_dist is not None and sw._rover_dist < tgt_rel_dist:
            self.set_ui_val("TgtRelDist", "-1")
            pw.MotCancel()

        # 초록 마커를 화면 중앙으로 유도하기 위한 pan/tilt 목표 계산
        # 명령 충돌을 줄이기 위해 틱 기반으로 tilt/pan을 교대로 전송
        _TRACK_N    = 10    # 명령 스케줄 주기
        _TILT_TICK  = 1    # tilt 전송 틱
        _PAN_TICK   = 6    # pan 전송 틱

        self.set_ui_val("ClickPosX", str(flow_px))
        self.set_ui_val("ClickPosY", str(flow_py))

        flow_tx = int(flow_px * 1280 / img_w)
        flow_ty = int(flow_py * 720  / img_h)
        try:
            fnorm = np.array(Pixelto3D(flow_tx, flow_ty, 1))
        except Exception:
            return
        fnx, fny, fnz = fnorm
        pan_angle    = np.degrees(np.arctan2(fnx, fnz))
        tilt_deg     = np.degrees(np.arctan2(-fny, np.sqrt(fnx*fnx + fnz*fnz)))
        current_tilt = self.get_ui_val("camera_pitch", float) or 0.0
        tilt_servo   = get_tilt_servo_angle(current_tilt + tilt_deg)
        pan_servo    = pan_angle  # 증분 스텝: 화면 중앙 기준 편차각만 사용 (N 오프셋 불필요)

        pan_step = 0.0 if abs(pan_servo) < TRACK_PAN_DEADZONE_DEG else float(np.clip(pan_servo, -TRACK_MAX_PAN_STEP_DEG, TRACK_MAX_PAN_STEP_DEG))
        cur_tilt_servo = float(self.get_ui_val("TiltAngle", float) or 0.0)
        tilt_delta = float(tilt_servo) - cur_tilt_servo
        tilt_step = 0.0 if abs(tilt_delta) < TRACK_TILT_DEADZONE_DEG else float(np.clip(tilt_delta, -TRACK_MAX_TILT_STEP_DEG, TRACK_MAX_TILT_STEP_DEG))
        next_tilt_servo = int(np.clip(cur_tilt_servo + tilt_step, 0.0, 90.0))

        self.set_ui_val("TgtPan",  str(int(pan_step)))
        self.set_ui_val("TgtTilt", str(next_tilt_servo))

        tick = getattr(self, '_track_tick', 0)
        self._track_tick = (tick + 1) % _TRACK_N

        if tick == _TILT_TICK:
            self.set_ui_val("TiltAngle", str(next_tilt_servo))
            pw.SetTilt()
        elif tick == _PAN_TICK:
            new_pan = float(np.clip(
                (self.get_ui_val("PanningAngle", float) or 0.0) + pan_step,
                -135.0, 135.0))
            self.set_ui_val("PanningAngle", str(int(new_pan)))
            pw.SetPanning()

    def excute_first_action(self):
        if not self.parent_window.sequential_cmd_list:
            return
        first_cmd = self.parent_window.sequential_cmd_list[0]
        parts = first_cmd.split()
        cmd_type, spd, val = parts[0], parts[1], parts[2]
        pw = self.parent_window
        sw = getattr(pw, '_streaming_window', None)
        m2_3d = sw._marker2_3d if sw is not None else None
        m_3d  = sw._marker_3d  if sw is not None else None
        if m2_3d is not None and m_3d is not None:
            rel_dist = float(np.linalg.norm(np.asarray(m2_3d) - np.asarray(m_3d)))
            self.set_ui_val("TgtRelDist", f"{rel_dist:.4f}")
        else:
            self.set_ui_val("TgtRelDist", "-1")

        if cmd_type == "MV_FWD":
            pw.set_ui_val("MVSpd", spd)
            pw.set_ui_val("Distance", val)
            pw.MoveFoward()
        elif cmd_type == "MV_BACK":
            pw.set_ui_val("MVSpd", spd)
            pw.set_ui_val("Distance", val)
            pw.MoveBackward()
        elif cmd_type == "RT_LEFT":
            pw.set_ui_val("RotSpd", spd)
            pw.set_ui_val("RotAngle", val)
            pw.RotateLeft()
        elif cmd_type == "RT_RIGHT":
            pw.set_ui_val("RotSpd", spd)
            pw.set_ui_val("RotAngle", val)
            pw.RotateRight()
        self.parent_window.sequential_cmd_list = self.parent_window.sequential_cmd_list[1:]
        self.setUIVal("SeqCmd", self.parent_window.sequential_cmd_list)

    def cancel_tracking(self):
        """초록 마커 트래킹을 중단하고 관련 표시를 모두 제거한다."""
        self._marker_flow_512_pos    = None
        self._marker_flow_cam_text   = ""
        self._marker_flow_feature_id = None
        self._marker_flow_reset_gen  = None
        reset_optflow_pts()
        sw = getattr(self.parent_window, '_streaming_window', None)
        if sw is not None:
            sw.set_marker2(0, 0, "")
            sw.clear_rover_line()

    def on_listbox_changed(self, key):
        selected = self.getUIVal(key)
        if selected:
            self.set_ui_val("SendStr", selected[0])

    # ── 3D 계산 공통 헬퍼 ──────────────────────────────────────

    @staticmethod
    def _rotate_vector(v, pitch, roll, yaw):
        pitch, roll, yaw = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw)
        Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        Rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
        return Rz @ Ry @ Rx @ v

    @staticmethod
    def _rotate_vector_inv(v, pitch, roll, yaw):
        """_rotate_vector의 역변환 (월드 프레임 → 카메라 프레임)"""
        pitch, roll, yaw = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw)
        Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        Rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
        return Rx.T @ Ry.T @ Rz.T @ v

    @staticmethod
    def _get_crosspnt(v, pos_m=0.7):
        """방향벡터 v 와 y=pos_m 수평면의 교차점 (m 단위)"""
        v = np.asarray(v, dtype=float)
        ny = v[1]
        if abs(ny) < 1e-8 or pos_m / ny <= 0:
            return np.array([0., 0., 0.])
        return v * (pos_m / ny)

    @staticmethod
    def _world_text(pnt_m):
        """월드 프레임 3D 좌표를 마커 텍스트로 변환 (TgtPosX/Y/Z 축 순서)"""
        return f"({pnt_m[2]:.2f},{pnt_m[1]:.2f},{pnt_m[0]:.2f})m"

    def _compute_3d(self, px, py, img_w, img_h):
        """픽셀 (px, py) → 카메라 pose 반영 월드 프레임 3D 좌표.

        depth_src UI 설정(Depth_Map / Virtual_Plane)을 따른다.
        5×5 패치 median depth 사용.

        Returns:
            norm_vec   (np.array): 카메라 프레임 방향벡터 (pan/tilt·트래킹 계산용)
            depth_m    (float)   : 깊이 (m), Virtual_Plane 또는 depth 무효면 0
            pnt_world  (np.array): 월드 프레임 3D (m), pose 반영
            cam_pos    (np.array|None): 카메라 프레임 3D (m), update_marker 트래킹용
        """
        tx = int(1280 * px / img_w)
        ty = int(720  * py / img_h)
        norm_vec = np.array(Pixelto3D(tx, ty, 1))

        odom_yaw  = self.parent_window.odom_yaw[-1] if self.parent_window.odom_yaw else 90.0
        cam_pitch = self.get_ui_val("camera_pitch", float) or 0.0
        cam_roll  = self.get_ui_val("camera_roll",  float) or 0.0
        cam_pan   = (self.get_ui_val("PanningAngle", float) or 0.0) - (self.get_ui_val("N", float) or 0.0)
        norm_rot  = self._rotate_vector(norm_vec, cam_pitch, cam_roll, odom_yaw + cam_pan)

        depth_m = 0.0
        if (self.get_ui_val("Depth_Src") or "Virtual_Plane") == "Depth_Map":
            depth_img = self.parent_window.ros_node.depth_image
            if depth_img is not None:
                dh, dw = depth_img.shape
                ddx = max(0, min(int(px * dw / img_w), dw - 1))
                ddy = max(0, min(int(py * dh / img_h), dh - 1))
                r = 2
                patch = depth_img[max(0, ddy-r):ddy+r+1, max(0, ddx-r):ddx+r+1]
                valid = patch[np.isfinite(patch) & (patch > 0)]
                if valid.size > 0:
                    depth_m = float(np.median(valid))

        if depth_m > 0:
            pnt_world = norm_rot * depth_m      # 월드 프레임, m
            cam_pos   = norm_vec * depth_m      # 카메라 프레임, m (트래킹용)
        else:
            pnt_world = self._get_crosspnt(norm_rot)   # 가상 바닥면, m
            cam_pos   = None

        return norm_vec, depth_m, pnt_world, cam_pos

    # ── 클릭 이벤트 ────────────────────────────────────────────

    def _on_image_click(self, name, x, y):
        """영상 클릭 이벤트를 처리한다.

        모든 마커의 3D 좌표는 _compute_3d 로 통일 계산한다.
        (카메라 pitch/roll/pan + odom_yaw 반영 월드 프레임, m 단위)

        - Alt+클릭  : 빨간 마커 (3D 좌표 표시)
        - Ctrl+클릭 : 주행 시퀀스 커맨드 + 노랑/초록 마커
        """
        self.set_ui_val("ClickPosX", str(x))
        self.set_ui_val("ClickPosY", str(y))
        print(f"[click] name={name} x={x} y={y}")

        rgb = self.parent_window.ros_node.rgb_image
        if rgb is None:
            print("[click] rgb is None, skip")
            return
        img_h, img_w = rgb.shape[:2]

        if not _init_thalamus():
            print("[click] thalamus not ready")
            return

        try:
            norm_vec, depth_m, pnt_world, cam_pos = self._compute_3d(x, y, img_w, img_h)
        except Exception as e:
            print(f"[click] _compute_3d error: {e}")
            return

        nx, ny, nz = tuple(norm_vec)
        print(f"[click] norm_vec={norm_vec}  depth={depth_m:.3f}m  pnt_world={pnt_world}")

        # pan/tilt 목표 계산 (카메라 프레임 방향벡터 기준)
        pan_angle    = np.degrees(np.arctan2(nx, nz))
        tilt_deg     = np.degrees(np.arctan2(-ny, np.sqrt(nx*nx + nz*nz)))
        current_tilt = self.get_ui_val("camera_pitch", float) or 0.0
        tilt_servo   = get_tilt_servo_angle(current_tilt + tilt_deg)
        pan_servo    = int(pan_angle - (self.get_ui_val("N", int) or 0))
        print(f"[click] pan_servo={pan_servo}  tilt_servo={tilt_servo}")
        self.set_ui_val("TgtPan",  str(pan_servo))
        self.set_ui_val("TgtTilt", str(tilt_servo))

        # 월드 좌표 UI 갱신 (X=forward, Y=vertical, Z=lateral)
        self.set_ui_val("TgtPosX", f"{pnt_world[2]:.3f}")
        self.set_ui_val("TgtPosY", f"{pnt_world[1]:.3f}")
        self.set_ui_val("TgtPosZ", f"{pnt_world[0]:.3f}")

        # 주행용 수평 거리 (mm, 최대 3000) — world frame X/Z 수평면 거리
        if depth_m > 0:
            veh_dist = int(min(np.sqrt(pnt_world[0]**2 + pnt_world[2]**2) * 1000, 3000))
        else:
            cp = self._get_crosspnt(norm_vec)
            veh_dist = int(min(np.sqrt(cp[0]**2 + cp[2]**2) * 1000, 3000))
        self.set_ui_val("TgtDist", f"{veh_dist}")

        # 절대 방위각
        bearing_deg = int(pan_angle
                          - (self.get_ui_val("N", int) or 0)
                          + (self.get_ui_val("PanningAngle", int) or 0))
        self.set_ui_val("TgtBearing", f"{bearing_deg}")

        # ── Alt+클릭: 빨간 마커 ──────────────────────────────────
        if QApplication.keyboardModifiers() & Qt.AltModifier:
            sw = getattr(self.parent_window, '_streaming_window', None)
            if sw is not None:
                sw.set_marker3(x, y, self._world_text(pnt_world))
                sw.setFocus()
            return

        # ── Ctrl+클릭: 시퀀스 커맨드 + 노랑/초록 마커 ───────────
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.init_local()
            self.parent_window.sequential_cmd_list = []
            if abs(bearing_deg) > 1:
                spd = self.getUIVal("RotSpd")[0]
                cmd_type = "RT_RIGHT" if bearing_deg > 0 else "RT_LEFT"
                self.parent_window.sequential_cmd_list.append(
                    f"{cmd_type} {spd} {abs(bearing_deg)}")
            if veh_dist > 100:
                spd = self.getUIVal("MVSpd")[0]
                self.parent_window.sequential_cmd_list.append(
                    f"MV_FWD {spd} {int(veh_dist)}")
            self.setUIVal("SeqCmd", self.parent_window.sequential_cmd_list)

            sw = getattr(self.parent_window, '_streaming_window', None)
            if sw is not None:
                # 노랑 마커: 월드 3D 텍스트, cam_pos 는 update_marker 트래킹용
                self._marker_cam_pos   = cam_pos.copy() if cam_pos is not None else None
                self._marker_cam_text  = self._world_text(pnt_world)
                self._marker_pnt_world = pnt_world.copy()   # 클릭 시점 월드좌표 보관
                sw.set_marker(x, y, self._marker_cam_text, pos_3d=pnt_world)

                # 초록 마커: 가장 가까운 optflow 피처를 _compute_3d 로 동일 계산
                self._marker_flow_512_pos    = None
                self._marker_flow_cam_text   = ""
                self._marker_flow_feature_id = None  # 고정 피쳐 ID
                self._marker_flow_reset_gen  = get_optflow_reset_count()  # 클릭 시 세대
                if hasattr(self, '_proc_odom_timer') and self._proc_odom_timer.isActive():
                    optflow_pts = get_last_optflow_pts()
                    optflow_ids = get_last_optflow_ids()
                    if optflow_pts:
                        click_512_x = x * 512.0 / img_w
                        click_512_y = y * 512.0 / img_h
                        best_i  = min(range(len(optflow_pts)),
                                      key=lambda i: (optflow_pts[i][0] - click_512_x)**2
                                                   + (optflow_pts[i][1] - click_512_y)**2)
                        cx_512, cy_512 = optflow_pts[best_i]
                        # 클릭 시점 피쳐 ID + 세대 고정
                        self._marker_flow_feature_id = optflow_ids[best_i] if best_i < len(optflow_ids) else None
                        self._marker_flow_reset_gen  = get_optflow_reset_count()
                        flow_px = cx_512 * img_w / 512
                        flow_py = cy_512 * img_h / 512
                        try:
                            _, flow_depth_m, flow_pnt_world, _ = \
                                self._compute_3d(flow_px, flow_py, img_w, img_h)
                        except Exception:
                            flow_pnt_world = pnt_world
                            flow_depth_m   = depth_m
                        self._marker_flow_512_pos  = [cx_512, cy_512]
                        self._marker_flow_cam_text = self._world_text(flow_pnt_world)
                        sw.set_marker2(int(flow_px), int(flow_py),
                                       self._marker_flow_cam_text,
                                       pos_3d=flow_pnt_world)
                    else:
                        sw.set_marker2(0, 0, "")
