import os
import time
import sys
import numpy as np
import cv2

from ultralytics import YOLO

_PKG_DIR       = os.path.dirname(os.path.abspath(__file__))
_LIGHT_WEIGHT  = os.path.join(_PKG_DIR, "cnn_weight", "light_best.pt")
_PERSON_WEIGHT = os.path.join(_PKG_DIR, "cnn_weight", "person_best.pt")

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import Qt, QTimer
from camera_pub.img_proc_main import img_proc_odometry
from camera_pub.recon_util import pixel_to_vec
from camera_pub.depth_recon import depthmap_floor_ransac, make_birdview

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from ThalamusEngine.Interface import LoadThalamusInterface, InitEngine, Pixelto3D, Convert3DtoPixel


meter_per_enc = 11600
rotate_pi_per_enc = 4819

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
            UIElement(_type=UIElementType.COMBO, _caption="Depth_Src", _default_val=["Virtual_Plane", "Depth_Map"]),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosX", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosY", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosZ", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtBearing", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtDist", _default_val="0"),

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
            UIElement(_type=UIElementType.BUTTON, _caption="predict_floor_ransac", _default_val=self.predict_floor_ransac),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT, _caption="gmap_ox", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="gmap_oy", _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="scale_per1pix", _default_val="20"),
            UIElement(_type=UIElementType.BUTTON, _caption="set_gram_param", _default_val=self.set_gram_param),
            UIElement(_type=UIElementType.BUTTON, _caption="merge_on_globalmap", _default_val=self.merge_on_globalmap),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.BUTTON, _caption="image_proc_odometry", _default_val=self.image_proc_odometry),
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

        cv2.imshow("Floor RANSAC", vis)
        cv2.imshow("Floor Mask", display_mask)
        cv2.imshow("Bird View", birdview)
        cv2.waitKey(1)


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
        time.sleep(0.2)
        pan = (self.get_ui_val("PanningAngle", float) or 0.0) + (self.get_ui_val("TgtPan", float) or 0.0)
        self.set_ui_val("PanningAngle", str(int(pan)))
        self.parent_window.SetPanning()

    def init_local(self):
        pw = self.parent_window
        self._local_ref_x   = pw.odom_x[-1]   if pw.odom_x   else 0.0
        self._local_ref_y   = pw.odom_y[-1]   if pw.odom_y   else 0.0
        self._local_ref_yaw = pw.odom_yaw[-1] if pw.odom_yaw else 0.0
        self.set_ui_val("local_pos", "0,0,0")
        pitch = pw.get_ui_val("camera_pitch") or "0"
        roll  = pw.get_ui_val("camera_roll")  or "0"
        pan   = pw.get_ui_val("PanningAngle") or "0"
        n     = pw.get_ui_val("N") or "0"
        yaw   = pw.odom_yaw[-1] - self._local_ref_yaw + (float(pan) - float(n))
        self.set_ui_val("local_cam_pose", f"{pitch},{roll},{yaw:.1f}")

    def update_local(self, odom_x, odom_y, odom_yaw):
        if not hasattr(self, '_local_ref_x') or self._local_ref_x is None:
            return
        dx = odom_x - self._local_ref_x
        dy = odom_y - self._local_ref_y
        dz = 0.0
        self.set_ui_val("local_pos", f"{dx:.3f},{dy:.3f},{dz:.3f}")
        pw    = self.parent_window
        pitch = pw.get_ui_val("camera_pitch") or "0"
        roll  = pw.get_ui_val("camera_roll")  or "0"
        pan   = pw.get_ui_val("PanningAngle") or "0"
        n     = pw.get_ui_val("N") or "0"
        yaw   = odom_yaw - self._local_ref_yaw + (float(pan) - float(n))
        self.set_ui_val("local_cam_pose", f"{pitch},{roll},{yaw:.1f}")

    def update_marker(self):
        """매 프레임 호출: 클릭 시점 cam_pos를 local_yaw만큼 역회전해 현재 픽셀로 재투영"""
        if not hasattr(self, '_marker_cam_pos') or self._marker_cam_pos is None:
            return
        sw = getattr(self.parent_window, '_streaming_window', None)
        if sw is None:
            return
        if not _init_thalamus():
            return

        # local_cam_pose = "pitch,roll,yaw"
        # yaw = 클릭 이후 카메라 전체 회전량
        #     = (odom_yaw - ref_yaw) + (PanningAngle - N)
        local_cam_pose = self.get_ui_val("local_cam_pose") or "0,0,0"
        vals = [s.strip() for s in local_cam_pose.split(",")]
        try:
            local_yaw = float(vals[2]) if len(vals) >= 3 else 0.0
        except ValueError:
            local_yaw = 0.0

        # ── [C'] 역병진: 차량 이동분을 클릭 시점 카메라 프레임으로 변환 후 차감 ─
        # local_pos = "dx,dy,0" (odom 좌표, 단위: m)
        # ref_yaw 기준으로 odom(dx,dy) → 카메라 프레임(X=우, Z=전방)으로 투영
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

        # ── [B'] 역회전: local_yaw만큼 Ry(local_yaw) 적용 ───────────────
        _y = np.deg2rad(local_yaw)
        Ry_mat = np.array([[ np.cos(_y), 0, np.sin(_y)],
                           [0,           1, 0          ],
                           [-np.sin(_y), 0, np.cos(_y)]])
        cam_pos_now = Ry_mat @ cam_pos_adjusted

        # ── [A'] 3D (camera frame) → pixel ───────────────────────────────
        if cam_pos_now[2] <= 0:
            return
        try:
            px_1280, py_720 = Convert3DtoPixel(
                float(cam_pos_now[0]), float(cam_pos_now[1]), float(cam_pos_now[2]))
        except Exception as e:
            print(f"[update_marker] Convert3DtoPixel error: {e}")
            return

        pw = self.parent_window
        rgb = pw.ros_node.rgb_image
        if rgb is None:
            return
        img_h, img_w = rgb.shape[:2]
        px = int(px_1280 * img_w / 1280)
        py = int(py_720  * img_h / 720)
        if 0 <= px < img_w and 0 <= py < img_h:
            sw.set_marker(px, py, self._marker_cam_text)

    def excute_first_action(self):
        if not self.parent_window.sequential_cmd_list:
            return
        first_cmd = self.parent_window.sequential_cmd_list[0]
        parts = first_cmd.split()
        cmd_type, spd, val = parts[0], parts[1], parts[2]
        pw = self.parent_window
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

    def on_listbox_changed(self, key):
        selected = self.getUIVal(key)
        if selected:
            self.set_ui_val("SendStr", selected[0])

    # ── image click handler ────────────────────────────────────
    def _on_image_click(self, name, x, y):
        self.set_ui_val("ClickPosX", str(x))
        self.set_ui_val("ClickPosY", str(y))
        print(f"[click] name={name} x={x} y={y}")

        # ── TgtPan / TgtTilt 계산 ──────────────────────────────
        rgb = self.parent_window.ros_node.rgb_image
        if rgb is None:
            print("[click] rgb is None, skip")
            return
        img_h, img_w = rgb.shape[:2]
        tx = int(1280 * x / img_w)
        ty = int(720  * y / img_h)
        print(f"[click] tx={tx} ty={ty}")

        if not _init_thalamus():
            print("[click] thalamus not ready")
            return
        try:
            norm_vec = np.array(Pixelto3D(tx, ty, 1))
            print(f"[click] norm_vec={norm_vec}")
        except Exception as e:
            print(f"[click] Pixelto3D error: {e}")
            return

        nx, ny, nz = tuple(norm_vec)

        pan_angle  = np.degrees(np.arctan2(nx, nz))
        tilt_deg   = np.degrees(np.arctan2(-ny, np.sqrt(nx*nx + nz*nz)))

        current_tilt = self.get_ui_val("camera_pitch", float) or 0.0
        tilt_servo   = get_tilt_servo_angle(current_tilt + tilt_deg)
        pan_servo    = int(pan_angle - (self.get_ui_val("N", int) or 0))
        print(f"[click] pan_servo={pan_servo} tilt_servo={tilt_servo}")

        self.set_ui_val("TgtPan",  str(pan_servo))
        self.set_ui_val("TgtTilt", str(tilt_servo))

        # ── TgtPos / TgtDist / TgtBearing 계산 ────────────────
        def rotate_vector(v, pitch, roll, yaw):
            pitch, roll, yaw = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw)
            Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
            Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
            Rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
            return Rz @ Ry @ Rx @ v

        def get_crosspnt(v, pos=700.0):
            v = np.asarray(v, dtype=float)
            ny_ = v[1]
            if abs(ny_) < 1e-8 or pos / ny_ <= 0:
                return np.array([0., 0., 0.])
            return v * (pos / ny_)

        odom_x   = self.parent_window.odom_x[-1]   if self.parent_window.odom_x   else 0.0
        odom_y   = self.parent_window.odom_y[-1]   if self.parent_window.odom_y   else 0.0
        odom_yaw = self.parent_window.odom_yaw[-1] if self.parent_window.odom_yaw else 90.0

        cam_pitch = self.get_ui_val("camera_pitch", float) or 0.0
        cam_pan   = (self.get_ui_val("PanningAngle", float) or 0.0) - (self.get_ui_val("N", float) or 0.0)
        norm_rot  = rotate_vector(norm_vec, cam_pitch, 0, odom_yaw + cam_pan)

        tgt_depth_m = 0.0
        depth_src = self.get_ui_val("Depth_Src") or "Virtual_Plane"
        if depth_src == "Depth_Map":
            depth = self.parent_window.ros_node.depth_image
            if depth is not None:
                dh, dw = depth.shape
                dx_ = int(dw * x / img_w)
                dy_ = int(dh * y / img_h)
                _d = float(depth[dy_, dx_])
                if np.isfinite(_d) and _d > 0:
                    tgt_depth_m = _d
                    pnt_3d = norm_rot * tgt_depth_m * 1000.0  # m → mm
                else:
                    pnt_3d = get_crosspnt(norm_rot)
            else:
                pnt_3d = get_crosspnt(norm_rot)
        else:
            pnt_3d = get_crosspnt(norm_rot)

        self.set_ui_val("TgtPosX", f"{odom_y + pnt_3d[2] / 1000:.3f}")
        self.set_ui_val("TgtPosY", f"{pnt_3d[1] / 1000:.3f}")
        self.set_ui_val("TgtPosZ", f"{odom_x + pnt_3d[0] / 1000:.3f}")

        if tgt_depth_m > 0:
            ny = norm_vec[1]
            horiz_m = np.sqrt(max(tgt_depth_m**2 - (ny * tgt_depth_m)**2, 0))
            veh_dist = float(min(horiz_m * 1000, 3000))
        else:
            cam_pnt = get_crosspnt(norm_vec)  # camera frame, 0,0,0 기준
            veh_dist = float(min(np.sqrt(cam_pnt[0]**2 + cam_pnt[2]**2), 3000))
        self.set_ui_val("TgtDist", f"{int(veh_dist)}")

        bearing_deg = int(pan_angle - (self.get_ui_val("N", int) or 0) + (self.get_ui_val("PanningAngle", int) or 0))
        self.set_ui_val("TgtBearing", f"{bearing_deg}")

        # ── Ctrl+클릭: SeqCmd 자동 생성 + 마커 표시 ───────────
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.init_local()
            self.parent_window.sequential_cmd_list = []
            if abs(bearing_deg) > 1:
                spd = self.getUIVal("RotSpd")[0]
                cmd_type = "RT_RIGHT" if bearing_deg > 0 else "RT_LEFT"
                self.parent_window.sequential_cmd_list.append(f"{cmd_type} {spd} {abs(bearing_deg)}")
            if veh_dist > 100:
                spd = self.getUIVal("MVSpd")[0]
                self.parent_window.sequential_cmd_list.append(f"MV_FWD {spd} {int(veh_dist)}")
            self.setUIVal("SeqCmd", self.parent_window.sequential_cmd_list)
            sw = getattr(self.parent_window, '_streaming_window', None)
            if sw is not None:
                depth_img = self.parent_window.ros_node.depth_image
                marker_text = "depth N/A"
                self._marker_local_pos  = None
                self._marker_local_text = None
                if depth_img is not None:
                    dh, dw = depth_img.shape
                    ddx = max(0, min(int(x * dw / img_w), dw - 1))
                    ddy = max(0, min(int(y * dh / img_h), dh - 1))
                    r = 2
                    patch = depth_img[max(0, ddy-r):ddy+r+1, max(0, ddx-r):ddx+r+1]
                    valid = patch[np.isfinite(patch) & (patch > 0)]
                    if valid.size > 0:
                        depth_m = float(np.median(valid))

                        # ── [A] pixel → 3D (camera frame, 클릭 시점) ──────
                        # cam_pos: 카메라 프레임 기준 3D 위치 (단위: m)
                        cam_pos = norm_vec * depth_m

                        # ── 저장: 클릭 시점 카메라 프레임 벡터 그대로 보관 ─
                        # 재투영은 update_marker에서 local_cam_pose yaw로 처리
                        self._marker_cam_pos  = cam_pos.copy()
                        self._marker_cam_text = f"({cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f},{depth_m:.1f})"
                        marker_text = self._marker_cam_text
                    else:
                        marker_text = "depth 0"
                        self._marker_cam_pos  = None
                sw.set_marker(x, y, marker_text)
