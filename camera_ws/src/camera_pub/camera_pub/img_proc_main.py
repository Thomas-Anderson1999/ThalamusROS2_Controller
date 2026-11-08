import copy

import cv2

from camera_pub.ThalamusCV.ThalamusCV.cvContext import *
from camera_pub.ThalamusCV.ThalamusCV.cvUtil import *
from camera_pub.recon_util import *
from ThalamusEngine.Interface import *
try:
    from cnn_predict import *
except ImportError:
    pass

#+Image Processing : Odometry
#Parameter
cutOffbboxOpt1 = [20, 20, 10, 10]
featureParams = [featureOpt.corner_GOOD2TRACK,
                 dict(maxCorners=100, qualityLevel=0.1, minDistance=5, blockSize=15)]
# featureParams = [featureOpt.corner_FAST, 30]
trackingCriteria = dict(winSize=(31, 31), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#longterm : 오도메트리용
context_opt1 = [(contextOpt.context_optflow, cutOffbboxOpt1, featureParams, trackingCriteria)
                # left, right, top, bottom, cut off on boder
                ]
longterm_imgContext = cimgContext(context_opt=context_opt1)


#shorterm : 돌발물체확인용
context_opt2 = [(contextOpt.context_diff, 1),
                (contextOpt.context_optflow, cutOffbboxOpt1, featureParams, trackingCriteria)
                # left, right, top, bottom, cut off on boder
                ]
shrotterm_imgContext = cimgContext(context_opt=context_opt2)


img_sz = 512
preprocess1_opt = [
    (adjustOpt.size_Resize, img_sz, img_sz),  # 0
    (adjustOpt.color_adj, cv2.COLOR_BGR2GRAY)  # 1
]

#+ui util
def get_text_height(text_scale):
    (_, text_h), baseline = cv2.getTextSize("I", cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
    return text_h
#-ui util

#+Image Processing : Odometry Reset / VIO cmd
_odometry_reset_flag = False
_vio_reset_flag      = False  # reset_vio() 전용 — vio_yaw/vio_dist만 리셋
_optflow_reset_flag  = False  # reset_optical_flow() 전용 — longterm/shortterm 기준점만 리셋
_imu_yaw_bias        = 0.0    # reset_vio() 시점의 raw imu_yaw — 표시용 imu_yaw에서 빼서 0으로 맞춤
_imu_yaw_reset_flag  = False  # 다음 img_proc_odometry 프레임에서 _imu_yaw_bias를 새로 캡처하라는 신호
_vio_cmd_type = None  # "MOVE" | "ROT" | None
_last_vio_yaw  = 0.0
_last_vio_dist = 0.0
_last_optflow_pts = []    # list of (x, y) in img_sz(512) space
_last_optflow_ids = []    # list of int feature IDs (idx_optKey), parallel to _last_optflow_pts
_optflow_reset_count = 0  # initOptflow 호출 횟수 — ID 세대 구분용

def get_last_optflow_pts():
    return _last_optflow_pts

def get_last_optflow_ids():
    return _last_optflow_ids

def get_optflow_reset_count():
    return _optflow_reset_count

def reset_optflow_pts():
    global _last_optflow_pts, _last_optflow_ids, _optflow_reset_count
    _last_optflow_pts = []
    _last_optflow_ids = []
    _optflow_reset_count += 1

def reset_img_proc_odometry():
    global _odometry_reset_flag, _last_vio_yaw, _last_vio_dist
    _odometry_reset_flag = True
    _last_vio_yaw  = 0.0
    _last_vio_dist = 0.0

def reset_vio():
    """vio_yaw / vio_dist만 0으로 리셋 (다음 img_proc_odometry 프레임에 반영).
    표시용 imu_yaw도 그 시점 값을 바이어스로 잡아서 같이 0부터 시작하게 한다."""
    global _vio_reset_flag, _last_vio_yaw, _last_vio_dist, _imu_yaw_reset_flag
    _vio_reset_flag = True
    _imu_yaw_reset_flag = True
    _last_vio_yaw  = 0.0
    _last_vio_dist = 0.0

def reset_optical_flow():
    """longterm/shortterm 옵티컬 플로우 기준점만 다시 잡음 (다음 img_proc_odometry 프레임에 반영)."""
    global _optflow_reset_flag
    _optflow_reset_flag = True

def stop_vio():
    global _vio_cmd_type
    _vio_cmd_type = None

def set_vio_cmd_type(cmd_type):
    global _vio_cmd_type
    _vio_cmd_type = cmd_type

def get_last_vio_yaw():
    return _last_vio_yaw
#-Image Processing : Odometry Reset / VIO cmd

#+vio/icp
class VIO:
    def __init__(self, _imgsz):
        self.imgsz = _imgsz
        self.target_sz = (720, 1280)

        self.reset()

    def reset(self):
        self.acc_yaw = 0
        self._last_computed_yaw = 0

    def save_yaw(self, _yaw):
        self.acc_yaw = _yaw

    def get_rot_sign(self, vec1, vec2):
        cross = np.cross(vec1, vec2)  # 3D cross
        sin_val = cross[1]  # yaw 축 성분
        return 1 if sin_val > 0 else -1

    def calcul_yaw(self, base, cur):
        pnt_num = min(len(base), len(cur))
        tar_szy, tar_szx = self.target_sz
        y_thresh = self.imgsz * 0.05

        yaw_val = []
        yaw_sign = []
        for _idx in range(pnt_num):
            if abs(base[_idx][1] - cur[_idx][1]) > y_thresh:
                continue

            bx = base[_idx][0] * tar_szx / self.imgsz
            by = base[_idx][1] * tar_szy / self.imgsz
            cx = cur[_idx][0] * tar_szx / self.imgsz
            cy = cur[_idx][1] * tar_szy / self.imgsz

            vec1 = np.array(Pixelto3D(bx, by, 1.))
            vec2 = np.array(Pixelto3D(cx, cy, 1.))

            # 정규화
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)

            # dot → angle
            dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            yaw_val.append(np.arccos(dot))
            yaw_sign.append(self.get_rot_sign(vec2, vec1))

        if not yaw_val:
            return self._last_computed_yaw

        _yaw = np.average(np.array(yaw_val))

        # 부호는 다수결로 결정
        pos = np.count_nonzero(np.array(yaw_sign) > 0)
        neg = np.count_nonzero(np.array(yaw_sign) < 0)
        _yaw_sign = 1 if pos >= neg else -1

        result = np.degrees(_yaw)*_yaw_sign + self.acc_yaw
        self._last_computed_yaw = result
        return result

    def regist_depthmap(self, depthmap):
        self.base_depthmap = copy.deepcopy(depthmap)

    def calcul_dist(self, base, cur, depthmap):
        depthmap_h, depthmap_w = depthmap.shape

        valid_diffs = []
        pnt_num = min(len(base), len(cur))
        for _idx in range(pnt_num):
            bx = int(base[_idx][0] * depthmap_w / self.imgsz)
            by = int(base[_idx][1] * depthmap_h / self.imgsz)
            cx = int(cur[_idx][0] * depthmap_w / self.imgsz)
            cy = int(cur[_idx][1] * depthmap_h / self.imgsz)

            b_d = self.base_depthmap[by][bx]
            c_d = depthmap[cy][cx]

            if b_d > 0.1 and c_d > 0.1:
                valid_diffs.append(b_d - c_d)

        if not valid_diffs:
            return 0.0
        return float(np.mean(valid_diffs))

_vio = VIO(img_sz)
#+vio

def img_proc_odometry(self, image, depthmap, local_idx, local_odometry, pose_ars, cmd_type):
    global _odometry_reset_flag, _vio_cmd_type, _vio_reset_flag, _optflow_reset_flag
    if _odometry_reset_flag:
        local_idx = 0
        _odometry_reset_flag = False

    # local_idx==0(로버 이동/회전 시 리셋)이면 옵티컬 플로우/VIO 둘 다 리셋,
    # reset_vio()/reset_optical_flow()로 세팅된 플래그면 해당하는 것만 리셋
    do_optflow_reset = (local_idx == 0) or _optflow_reset_flag
    do_vio_reset      = (local_idx == 0) or _vio_reset_flag
    _optflow_reset_flag = False
    _vio_reset_flag      = False

    global _imu_yaw_bias, _imu_yaw_reset_flag
    if _imu_yaw_reset_flag:
        _imu_yaw_bias = pose_ars[0]
        _imu_yaw_reset_flag = False
    imu_yaw = pose_ars[0] - _imu_yaw_bias

    effective_cmd = _vio_cmd_type

    text_scale = 0.7
    text_h = get_text_height(text_scale)

    # +image process
    preprocess1 = getAdjustedImg(image, preprocess1_opt)
    resizedImg = preprocess1[0]
    grayImg = preprocess1[1]
    showOnceParam = [(resizedImg, None, "resized Img")]#, (grayImg, cv2.COLOR_GRAY2BGR, "gray Img")]

    lt_contextRes = longterm_imgContext.inputImg(grayImg)
    st_contextRes = shrotterm_imgContext.inputImg(grayImg)

    if do_optflow_reset: #initialize
        global _optflow_reset_count, _last_vio_yaw
        _optflow_reset_count += 1
        if not do_vio_reset:
            # VIO는 그대로 유지하면서 옵티컬 플로우 기준점만 새로 잡는 경우:
            # base/cur가 리셋되면 상대각이 0으로 끊기므로, 지금까지 누적된 yaw를
            # _vio.acc_yaw에 먼저 저장해서 리셋 이후에도 값이 끊기지 않게 한다.
            _vio.save_yaw(_last_vio_yaw)
        longterm_imgContext.initOptflow(cutOffbboxOpt1, featureParams)
        shrotterm_imgContext.initOptflow(cutOffbboxOpt1, featureParams)

        # inputImg()가 리셋 전에 이미 계산해둔 stale 결과를, 방금 재초기화된
        # 키포인트로 갱신 (show_once 화면 / _last_optflow_pts 모두 즉시 반영되도록)
        lt_contextRes[0] = [longterm_imgContext.base_optKey,
                            longterm_imgContext.cur_optKey,
                            longterm_imgContext.idx_optKey]
        if len(st_contextRes) > 1:
            st_contextRes[1] = [shrotterm_imgContext.base_optKey,
                                shrotterm_imgContext.cur_optKey,
                                shrotterm_imgContext.idx_optKey]

    if do_vio_reset:
        _vio.reset()
        _vio.regist_depthmap(depthmap)

    if lt_contextRes[0] is not None:
        base_optKey, cur_optKey, idx_opt = lt_contextRes[0]

        optRes = show_optflow(resizedImg.copy(), base_optKey, cur_optKey, idx_opt)

        lens = cur_optKey.shape[0]
        base_list = []
        cur_list = []
        for i in range(lens):
            bx = int(base_optKey[i][0][0])
            by = int(base_optKey[i][0][1])
            base_list.append((bx,by))

            cx = int(cur_optKey[i][0][0])
            cy = int(cur_optKey[i][0][1])
            cur_list.append((cx,cy))

        global _last_optflow_pts, _last_optflow_ids
        _last_optflow_pts = cur_list
        _last_optflow_ids = [int(v) for v in idx_opt]

        global _last_vio_dist  # _last_vio_yaw는 위(do_optflow_reset 블록)에서 이미 global 선언됨
        if not do_vio_reset:
            if effective_cmd == "ROT":
                _last_vio_yaw = _vio.calcul_yaw(base_list, cur_list)
            elif effective_cmd == "MOVE":
                _last_vio_dist = _vio.calcul_dist(base_list, cur_list, depthmap)
        vio_yaw  = _last_vio_yaw
        vio_dist = _last_vio_dist

        if lens < 4: #re-initialize
            _optflow_reset_count += 1
            longterm_imgContext.initOptflow(cutOffbboxOpt1, featureParams)
            _vio.save_yaw(vio_yaw)

        # +show text
        wheel_odometry_dist = np.sqrt(local_odometry[0]**2 + local_odometry[1]**2)
        text = f"{effective_cmd}: oDist:{wheel_odometry_dist:.2f} oPos:{local_odometry[0]:.1f},{local_odometry[1]:.1f} oAtt:{local_odometry[2]:.1f}"
        cv2.putText(optRes, text, (10, img_sz - int(text_h * 3)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 1, cv2.LINE_AA)
        text = f"imu_yaw:{imu_yaw:.1f} vio_yaw:{vio_yaw:.1f}, vio_dist:{vio_dist:.2f}"
        cv2.putText(optRes, text, (10, img_sz - int(text_h * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 1, cv2.LINE_AA)
        # -show text
        showOnceParam.append([optRes, None, "longterm optical flow Img"])



    if st_contextRes[1] is not None:
        base_optKey, cur_optKey, idx_opt = st_contextRes[1]

        optRes = show_optflow(resizedImg.copy(), base_optKey, cur_optKey, idx_opt)
        showOnceParam.append([optRes, None, "shorterm optical flow Img"])

        lens = cur_optKey.shape[0]
        for i in range(lens):
            bx = int(base_optKey[i][0][0])
            by = int(base_optKey[i][0][1])
            cx = int(cur_optKey[i][0][0])
            cy = int(cur_optKey[i][0][1])

        if (lens < 4) or (local_idx % 15 == 14): #re-initialize
            shrotterm_imgContext.initOptflow(cutOffbboxOpt1, featureParams)



    show_once_res = showOnce(imgList=showOnceParam, restitle="title", show_window=False)
    self.qt_imshow("show_once", show_once_res)
# -Image Processing


#################################################################
#+Image Processing : super pixel

try:
    from skimage.segmentation import slic, mark_boundaries
    from skimage.color import label2rgb
    import skimage
except ImportError:
    pass
import numpy as np
import cv2


def preprocess_for_lab(labels, preprocessed_img):
    label_max = np.max(labels)

    # allocate stats
    label_cnt = np.zeros(label_max + 1, np.int32)
    label_L_sum = np.zeros(label_max + 1, np.float64)
    label_A_sum = np.zeros(label_max + 1, np.float64)
    label_B_sum = np.zeros(label_max + 1, np.float64)
    label_L_sqsum = np.zeros(label_max + 1, np.float64)
    label_A_sqsum = np.zeros(label_max + 1, np.float64)
    label_B_sqsum = np.zeros(label_max + 1, np.float64)

    # result: avg, std
    label_L_avg = np.zeros(label_max + 1, np.float64)
    label_A_avg = np.zeros(label_max + 1, np.float64)
    label_B_avg = np.zeros(label_max + 1, np.float64)
    label_L_std = np.zeros(label_max + 1, np.float64)
    label_A_std = np.zeros(label_max + 1, np.float64)
    label_B_std = np.zeros(label_max + 1, np.float64)

    # result: adjacency graph
    label_adj = {i: set() for i in range(label_max + 1)}

    h, w = labels.shape
    for y in range(h):
        for x in range(w):
            idx = labels[y, x]

            L, A, B = preprocessed_img[y, x]

            label_cnt[idx] += 1
            label_L_sum[idx] += L
            label_A_sum[idx] += A
            label_B_sum[idx] += B
            label_L_sqsum[idx] += L * L
            label_A_sqsum[idx] += A * A
            label_B_sqsum[idx] += B * B

            # right neighbor
            if x + 1 < w:
                r = labels[y, x + 1]
                if r != idx:
                    label_adj[idx].add(r)
                    label_adj[r].add(idx)

            # down neighbor
            if y + 1 < h:
                d = labels[y + 1, x]
                if d != idx:
                    label_adj[idx].add(d)
                    label_adj[d].add(idx)

    for idx in range(label_max + 1):
        if label_cnt[idx] == 0:
            continue
        label_L_avg[idx] = label_L_sum[idx] / label_cnt[idx]
        label_A_avg[idx] = label_A_sum[idx] / label_cnt[idx]
        label_B_avg[idx] = label_B_sum[idx] / label_cnt[idx]

        var_L = label_L_sqsum[idx] / label_cnt[idx] - label_L_avg[idx] ** 2
        var_A = label_A_sqsum[idx] / label_cnt[idx] - label_A_avg[idx] ** 2
        var_B = label_B_sqsum[idx] / label_cnt[idx] - label_B_avg[idx] ** 2

        label_L_std[idx] = np.sqrt(max(var_L, 0.0))
        label_A_std[idx] = np.sqrt(max(var_A, 0.0))
        label_B_std[idx] = np.sqrt(max(var_B, 0.0))

    return (label_max, label_cnt,
            label_L_avg, label_A_avg, label_B_avg,
            label_L_std, label_A_std, label_B_std, label_adj)


def show_SLIC(self, image, depthmap, local_idx,
              segment_num, compactness,
              start_pos="128,250",
              lab_merge_param="1,1,1,4000,20,5.0", show_boundary=True):
    # 1️⃣ 전처리 : resize
    resized_img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    # -----(중요) 정확한 CIE Lab 변환----- #
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) / 255.0
    lab_img = skimage.color.rgb2lab(rgb)

    # SLIC (CIE Lab 기반)
    labels = slic(lab_img, n_segments=segment_num,
                  compactness=compactness, start_label=0)

    merged_labels = labels

    # 시각화
    resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    seg_vis = label2rgb(merged_labels, resized_rgb, kind='avg')  # merged_labels에서  resized_rgb의 색을 평균내서 seg_vis를 만든다.

    if show_boundary:
        seg_vis = mark_boundaries(seg_vis, merged_labels,
                                  color=(1, 0, 0), mode='outer')

    vis_bgr = cv2.cvtColor((seg_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    print("super pixel num:", np.max(merged_labels))
    self.qt_imshow("Original (resized)", resized_img)
    self.qt_imshow("Segmented", vis_bgr)






def img_proc_segmentation(self, image, depthmap, local_idx,
                          segment_num, compactness,
                          start_pos="128,250",
                          lab_merge_param="1,1,1,4000,20,5.0", show_boundary=True):

    # 1️⃣ 전처리 : resize
    resized_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # -----(중요) 정확한 CIE Lab 변환----- #
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) / 255.0
    lab_img = skimage.color.rgb2lab(rgb)

    # SLIC (CIE Lab 기반)
    labels = slic(lab_img, n_segments=segment_num,
                  compactness=compactness, start_label=0)

    merged_labels = labels

    preprocessed_img = lab_img.astype(np.float64)

    # 라벨별 Lab 통계 및 adjacency
    (label_max, label_cnt, #라벨의 최댓값(슈퍼픽셀의 갯수), 각 슈퍼픽셀의 크기
     label_L_avg, label_A_avg, label_B_avg, # 평균
     label_L_std, label_A_std, label_B_std, # 표준편차
     label_adj #이웃들의 리스트: dictionary 안에 set 구조  , dictionary key는 슈퍼픽셀 라벨의 인덱스, 안쪽의 리스트는 key의 이웃라벨
     ) = preprocess_for_lab(labels, preprocessed_img)

    ###########################################################################
    # ★ Superpixel 대표값(outlier 제거 + center pixel 기반) 업데이트
    ###########################################################################

    outlier_th = 4.0  # 표준편차 threshold

    h, w = labels.shape

    # superpixel별 픽셀 좌표 저장
    sp_pixels = {u: [] for u in range(label_max + 1)}
    for y in range(h):
        for x in range(w):
            sp_pixels[labels[y, x]].append((y, x))

    # 새로운 평균 저장할 배열
    new_L_avg = np.copy(label_L_avg)
    new_A_avg = np.copy(label_A_avg)
    new_B_avg = np.copy(label_B_avg)

    for u in range(label_max + 1):
        cnt = label_cnt[u]
        if cnt == 0:
            continue

        # outlier 검출: std가 outlier_th 넘으면 outlier로 판단, L은 제외, 색이 다른 것만
        if (label_A_std[u] > outlier_th or
            label_B_std[u] > outlier_th): #label_L_std[u] > outlier_th or

            # superpixel 내부 픽셀들의 좌표
            coords = sp_pixels[u]
            ys = np.array([p[0] for p in coords])
            xs = np.array([p[1] for p in coords])

            # superpixel 중심(centroid) 픽셀 좌표
            cy = int(np.clip(np.round(ys.mean()), 0, h - 1))
            cx = int(np.clip(np.round(xs.mean()), 0, w - 1))

            # center pixel의 Lab 값을 대표값으로 사용
            Lc, Ac, Bc = preprocessed_img[cy, cx]

            new_L_avg[u] = Lc
            new_A_avg[u] = Ac
            new_B_avg[u] = Bc

    # 대표값 교체
    label_L_avg = new_L_avg
    label_A_avg = new_A_avg
    label_B_avg = new_B_avg
    ###########################################################################


    """
    #
    outlier_th = 4.0  # 표준편차 임계값

    # outlier 슈퍼픽셀 median 기반으로 보정
    for label in range(label_max + 1):
        if label_cnt[label] == 0:
            continue

        # 해당 label의 모든 픽셀 위치 가져오기 (mask)
        mask = (labels == label)

        # 해당 SP의 Lab 배열 추출
        L_vals = preprocessed_img[..., 0][mask]
        A_vals = preprocessed_img[..., 1][mask]
        B_vals = preprocessed_img[..., 2][mask]

        # L 채널 보정
        if label_L_std[label] > outlier_th:
            label_L_avg[label] = np.median(L_vals)

        # A 채널 보정
        if label_A_std[label] > outlier_th:
            label_A_avg[label] = np.median(A_vals)

        # B 채널 보정
        if label_B_std[label] > outlier_th:
            label_B_avg[label] = np.median(B_vals)
    """

    """
    #아웃라이어 빼고 다시 평균 
    outlier_th = 2.0  # 표준편차 threshold (예: 3σ)

    # 새로운 평균 저장 배열
    label_L_new = np.zeros_like(label_L_avg)
    label_A_new = np.zeros_like(label_A_avg)
    label_B_new = np.zeros_like(label_B_avg)

    for sp in range(label_max + 1):

        # 해당 superpixel의 mask
        mask = (labels == sp)
        if not np.any(mask):
            continue

        # superpixel 내 픽셀들의 Lab 값
        L_vals = preprocessed_img[..., 0][mask]
        A_vals = preprocessed_img[..., 1][mask]
        B_vals = preprocessed_img[..., 2][mask]

        # 기존 평균/표준편차
        L_avg, A_avg, B_avg = label_L_avg[sp], label_A_avg[sp], label_B_avg[sp]
        L_std, A_std, B_std = label_L_std[sp], label_A_std[sp], label_B_std[sp]

        # outlier 제거 마스크
        L_mask = np.abs(L_vals - L_avg) <= outlier_th * max(L_std, 1e-6)
        A_mask = np.abs(A_vals - A_avg) <= outlier_th * max(A_std, 1e-6)
        B_mask = np.abs(B_vals - B_avg) <= outlier_th * max(B_std, 1e-6)

        valid = L_mask & A_mask & B_mask

        # 만약 outlier 제거로 너무 작아지면 전체 사용
        if np.sum(valid) < 3:
            valid = np.ones_like(valid, dtype=bool)

        # 새로운 평균 계산
        label_L_new[sp] = np.mean(L_vals[valid])
        label_A_new[sp] = np.mean(A_vals[valid])
        label_B_new[sp] = np.mean(B_vals[valid])

    # 기존 평균을 새 평균으로 교체
    label_L_avg = label_L_new
    label_A_avg = label_A_new
    label_B_avg = label_B_new
    """
    #######################################################################


    #####################################################################################
    #+Merge (Felzenszwalb-style union-find on superpixel graph)
    #+파라메터
    _lab_merge = lab_merge_param.split(',')
    L_th = float(_lab_merge[0])
    A_th = float(_lab_merge[1])
    B_th = float(_lab_merge[2])
    k = float(_lab_merge[3])
    min_size = float(_lab_merge[4])
    global_merge_th = float(_lab_merge[5])
    #-파라메터

    num_sp = label_max + 1

    #+ 엣지 웨이트 계산 및 정랼
    edges = []
    for u in range(num_sp):
        for v in label_adj.get(u, []):
            if v <= u: #같은 엣지를 두번 처리하지 않기 위해
                continue
            dL = label_L_avg[u] - label_L_avg[v]
            dA = label_A_avg[u] - label_A_avg[v]
            dB = label_B_avg[u] - label_B_avg[v]
            w = np.sqrt(L_th*dL*dL + A_th*dA*dA + B_th*dB*dB)
            edges.append((w, u, v))

    edges.sort(key=lambda t: t[0]) #edge들을 w(경계강도)가 약한 순으로 정렬
    # - 엣지 웨이트 계산 및 정랼

    parent = np.arange(num_sp, dtype=np.int32) #최초의 paraent구조는 0에서 num_sp으로 초기화 되어있으므로, 10번의 부모는 parent[10] = 9로 세팅되어있다.
    size = label_cnt.copy().astype(np.int32)
    Int = np.zeros(num_sp, dtype=np.float64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, w):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        Int[ra] = max(Int[ra], Int[rb], w) #세그먼트 내부에서 등장한 가장 큰 w값(가장 강한 내부 경계)
        return ra

    def threshold(r):
        return k / max(1, size[r]) # 큰 사이즈의 세그먼트들은 함부로 병합되지 않도록

    # 3) 라벨 병합
    for w, u, v in edges: #엣지간 웨이트차이가 가장 약하 것끼리
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if w <= min(Int[ru] + threshold(ru), Int[rv] + threshold(rv)):
            union(ru, rv, w)

    # 4) 작은 컴포넌트 강제 병합
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if size[ru] < min_size or size[rv] < min_size:
            union(ru, rv, w)

    # 5) 1차 결과 라벨 remap
    #root_map = np.array([find(i) for i in range(num_sp)])
    #merged_labels = root_map[labels]

    ##############################################################################################################################################################
    # ★ 6) Global Color Merge (후처리) — 고립된 superpixel 제거 (정렬 기반)
    ###############################################################################

    global_color_th = global_merge_th   # ex: 5.0

    # 1) 글로벌 색 기반 edge 생성
    color_edges = []
    for u in range(num_sp):
        ru = find(u)
        for v in label_adj.get(u, []):
            rv = find(v)
            if ru == rv:
                continue

            dL = label_L_avg[ru] - label_L_avg[rv]
            dA = label_A_avg[ru] - label_A_avg[rv]
            dB = label_B_avg[ru] - label_B_avg[rv]
            w_color = np.sqrt(dL*dL + dA*dA + dB*dB)

            # 글로벌 컬러 threshold 안쪽인 edge만 대상
            if w_color < global_color_th:
                color_edges.append((w_color, ru, rv))

    # 2) 색 거리 순으로 정렬 (안정성 핵심!)
    color_edges.sort(key=lambda t: t[0])

    # 3) 약한 색 경계부터 안정적으로 병합
    for w_color, u, v in color_edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            union(ru, rv, w_color)


    # 4) 후처리 라벨 재매핑
    root_map = np.array([find(i) for i in range(num_sp)])
    merged_labels = root_map[labels]

    #############################################################################

    ################################################################################
    # 🔥 Superpixel 기반 Gradient Extraction (Segment-level)
    ################################################################################

    H, W = merged_labels.shape
    L_map = preprocessed_img[..., 0]  # LAB L channel (0~100)

    # SLIC superpixel → 중심 좌표 계산 (superpixel centroid)
    sp_centroids = {}  # u -> (cy, cx)
    for sp_id in range(label_max + 1):
        ys, xs = np.where(labels == sp_id)
        if len(xs) == 0:
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        sp_centroids[sp_id] = (cy, cx)

    # 세그먼트 단위 extreme superpixel 찾기
    segment_dark_sps = {}
    segment_bright_sps = {}

    unique_segments = np.unique(merged_labels)

    for seg_id in unique_segments:
        # 세그먼트 내부 pixel mask
        mask = (merged_labels == seg_id)
        ys, xs = np.where(mask)

        # 이 세그먼트에 포함된 superpixel들 찾기
        sp_ids = np.unique(labels[ys, xs])

        # 각 superpixel의 L평균 리스트
        sp_Ls = [(sp_id, label_L_avg[sp_id]) for sp_id in sp_ids]

        # 가장 밝은 / 어두운 SP 리스트
        min_L = min(l for (_, l) in sp_Ls)
        max_L = max(l for (_, l) in sp_Ls)
        dark_sp_list = [sp for sp, l in sp_Ls if l == min_L]
        bright_sp_list = [sp for sp, l in sp_Ls if l == max_L]

        segment_dark_sps[seg_id] = dark_sp_list
        segment_bright_sps[seg_id] = bright_sp_list

    ################################################################################
    # 🔥 가장 먼 SP쌍 선택 후 화살표 그리기
    ################################################################################

    gradient_vis = resized_img.copy()

    for seg_id in unique_segments:
        dark_list = segment_dark_sps[seg_id]
        bright_list = segment_bright_sps[seg_id]

        best_dist = -1
        best_pair = None

        for d in dark_list:
            dy, dx = sp_centroids[d]
            for b in bright_list:
                by, bx = sp_centroids[b]

                dist = (dy - by) ** 2 + (dx - bx) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best_pair = ((dx, dy), (bx, by))

        if best_pair is None:
            continue

        p_dark, p_bright = best_pair

        cv2.arrowedLine(
            gradient_vis,
            p_dark, p_bright,
            (0, 0, 255),
            thickness=2,
            tipLength=0.3
        )

    # QT 출력
    self.qt_imshow("gradient", gradient_vis)

    ################################################################################
    # 🔥 텍스트 출력 (optional)
    ################################################################################

    print("===== Segment Extremes =====")
    for seg_id in unique_segments:
        print(f"[Segment {seg_id}]")
        print("  Dark SPs:", segment_dark_sps[seg_id])
        print("  Bright SPs:", segment_bright_sps[seg_id])

    ###############################################################################

    # 시각화
    resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    seg_vis = label2rgb(merged_labels, resized_rgb, kind='avg') #merged_labels에서  resized_rgb의 색을 평균내서 seg_vis를 만든다.

    if show_boundary:
        seg_vis = mark_boundaries(seg_vis, merged_labels,
                                  color=(1, 0, 0), mode='outer')

    vis_bgr = cv2.cvtColor((seg_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    print("super pixel num:", np.max(merged_labels))
    self.qt_imshow("Original (resized)", resized_img)
    self.qt_imshow("Segmented", vis_bgr)

#-Image Processing : super pixel



#+show normal vector
def get_3pixel(item):
    """
    item = {
        "min_x": (x,y),
        "max_x": (x,y),
        "min_y": (x,y),
        "max_y": (x,y)
    }
    """
    # 후보 리스트 뽑기
    pts = []
    for key in ["min_x", "max_x", "min_y", "max_y"]:
        if item[key] is not None:
            pts.append(item[key])

    # 유효한 점이 3개 미만이면 그냥 종료
    if len(pts) < 3:
        return []

    # ----------------------------------------
    # (1) p1, p2 : 가장 먼 두 점 찾기
    # ----------------------------------------
    max_d = -1
    p1 = p2 = None

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (x1, y1) = pts[i]
            (x2, y2) = pts[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx * dx + dy * dy  # 거리 제곱

            if d2 > max_d:
                max_d = d2
                p1 = pts[i]
                p2 = pts[j]

    # ----------------------------------------
    # (2) p3 : p1-p2 직선에서 가장 멀리 떨어진 점
    # ----------------------------------------
    best_dist = -1
    p3 = None

    (x1, y1) = p1
    (x2, y2) = p2
    vx = x2 - x1
    vy = y2 - y1
    v_norm = (vx * vx + vy * vy) ** 0.5
    if v_norm < 1e-6:
        return []

    for (x, y) in pts:
        if (x, y) == p1 or (x, y) == p2:
            continue

        wx = x - x1
        wy = y - y1

        # cross product magnitude / |v|
        dist = abs(vx * wy - vy * wx) / v_norm

        if dist > best_dist:
            best_dist = dist
            p3 = (x, y)

    if p3 is None:
        return []

    # ----------------------------------------
    # 반환: (p1, p2, p3)
    # ----------------------------------------
    return [p1, p2, p3]

def get_labeled_normal_vector(labels, depthmap):
    label_h, label_w = labels.shape
    lab_max = np.max(labels)

    # 법선좌표의 후보를 뽑는다
    normal_pixels = [
        {
            "min_x": None,
            "max_x": None,
            "min_y": None,
            "max_y": None
        }
        for _ in range(lab_max + 1)
    ]

    for y in range(label_h):
        for x in range(label_w):
            lb = labels[y, x]
            item = normal_pixels[lb]

            # min_x
            if item["min_x"] is None or x < item["min_x"][0]:
                item["min_x"] = (x, y)

            # max_x
            if item["max_x"] is None or x > item["max_x"][0]:
                item["max_x"] = (x, y)

            # min_y
            if item["min_y"] is None or y < item["min_y"][1]:
                item["min_y"] = (x, y)

            # max_y
            if item["max_y"] is None or y > item["max_y"][1]:
                item["max_y"] = (x, y)

    normal_vectors = [None for _ in range(lab_max + 1)]
    normal_centers = [None for _ in range(lab_max + 1)]

    target_shape = (720, 1280)

    for lab_id in range(lab_max + 1):
        # normal_pixels 중, 가장 먼 세개를 뽑는다.
        norm_pixels = get_3pixel(normal_pixels[lab_id])

        if len(norm_pixels) < 3:
            normal_vectors[lab_id] = None
        else:
            vec0 = pixel_to_vec(*norm_pixels[0], target_shape, depthmap)
            vec1 = pixel_to_vec(*norm_pixels[1], target_shape, depthmap)
            vec2 = pixel_to_vec(*norm_pixels[2], target_shape, depthmap)

            normal_vectors[lab_id] = np.cross(vec1 - vec0, vec2 - vec0)
            normal_centers[lab_id] = (vec0 + vec1 + vec2) / 3

    return normal_vectors, normal_centers

# 우선 앞의 SLIC으로 수퍼 픽셀 받아서, 노말벡터 베이스로 머지 해보려고 시도하던 것중간 과정
def show_normal_vector(self, image, depthmap, local_idx,
                          segment_num, compactness,
                          start_pos="128,250",
                          lab_merge_param="1,5,5,4000,20", show_boundary=True):

    # 1) Resize
    resized_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # 2) SLIC 입력 → OpenCV BGR → LAB
    lab_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)

    # 3) SLIC Superpixel
    labels = slic(lab_img, n_segments=segment_num, compactness=compactness, start_label=0)

    # superpixel 개수
    print("superpixel (raw SLIC) num:", np.unique(labels).size)

    # 4) Visualization (색상 공간 해결)
    resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    seg_vis = label2rgb(labels, resized_rgb, kind='avg')

    if show_boundary:
        seg_vis = mark_boundaries(seg_vis, labels, color=(1, 0, 0), mode='outer')

    vis_bgr = cv2.cvtColor((seg_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 5) Display
    self.qt_imshow("Original (resized)", resized_img)
    self.qt_imshow("Segmented (SLIC only)", vis_bgr)

    norm_Depth_Map = cv2.normalize(depthmap, None, alpha=0, beta=255., norm_type=cv2.NORM_MINMAX)
    norm_Depth_Map = norm_Depth_Map.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(norm_Depth_Map, cv2.COLORMAP_JET)
    self.qt_imshow("depth_colormap", depth_colormap)

    normal_vectors, normal_centers =  get_labeled_normal_vector(labels, depthmap)

    # ===========================================================
    # 6) Normal Visualization on 256×256 image
    # ===========================================================

    debug_vis = resized_img.copy()  # BGR 256×256

    arrow_len = 20  # 화살표 길이 (픽셀)
    lab_max = np.max(labels)

    for lab_id in range(lab_max + 1):

        n = normal_vectors[lab_id]
        c = normal_centers[lab_id]

        if n is None or c is None:
            continue

        cx, cy = Convert3DtoPixel(c[0], c[1], c[2])
        cx = int(cx * 256 / 1280)
        cy = int(cy * 256 / 720)

        # normal length 0 체크
        norm_len = np.linalg.norm(n)
        if norm_len < 1e-6:
            continue
        n_c = c + -(n/norm_len) * 100


        ex, ey = Convert3DtoPixel(n_c[0], n_c[1], n_c[2])
        ex = int(ex * 256 / 1280)
        ey = int(ey * 256 / 720)




        start_pt = (cx, cy)
        end_pt = (ex, ey)

        # 범위 체크
        if 0 <= cx < 256 and 0 <= cy < 256:
            cv2.arrowedLine(debug_vis, start_pt, end_pt,
                            (0, 255, 0), 1, tipLength=0.3)

    self.qt_imshow("Normal Vectors", debug_vis)
#-Image Processing : super pixel



#+ refactor segmentation

def preprocess_for_lab2(labels, preprocessed_img):
    label_max = np.max(labels)

    # allocate stats
    label_cnt = np.zeros(label_max + 1, np.int32)
    label_L_sum = np.zeros(label_max + 1, np.float64)
    label_A_sum = np.zeros(label_max + 1, np.float64)
    label_B_sum = np.zeros(label_max + 1, np.float64)
    label_L_sqsum = np.zeros(label_max + 1, np.float64)
    label_A_sqsum = np.zeros(label_max + 1, np.float64)
    label_B_sqsum = np.zeros(label_max + 1, np.float64)

    # centroid
    label_X_sum = np.zeros(label_max + 1, np.float64)
    label_Y_sum = np.zeros(label_max + 1, np.float64)
    label_cx = np.zeros(label_max + 1, np.float64)
    label_cy = np.zeros(label_max + 1, np.float64)

    # result: avg, std
    label_L_avg = np.zeros(label_max + 1, np.float64)
    label_A_avg = np.zeros(label_max + 1, np.float64)
    label_B_avg = np.zeros(label_max + 1, np.float64)
    label_L_std = np.zeros(label_max + 1, np.float64)
    label_A_std = np.zeros(label_max + 1, np.float64)
    label_B_std = np.zeros(label_max + 1, np.float64)

    # result: adjacency graph
    label_adj = {i: set() for i in range(label_max + 1)}

    h, w = labels.shape
    for y in range(h):
        for x in range(w):
            idx = labels[y, x]

            L, A, B = preprocessed_img[y, x]

            label_cnt[idx] += 1

            # centroid accumulate
            label_X_sum[idx] += x
            label_Y_sum[idx] += y

            label_L_sum[idx] += L
            label_A_sum[idx] += A
            label_B_sum[idx] += B
            label_L_sqsum[idx] += L * L
            label_A_sqsum[idx] += A * A
            label_B_sqsum[idx] += B * B

            # right neighbor
            if x + 1 < w:
                r = labels[y, x + 1]
                if r != idx:
                    label_adj[idx].add(r)
                    label_adj[r].add(idx)

            # down neighbor
            if y + 1 < h:
                d = labels[y + 1, x]
                if d != idx:
                    label_adj[idx].add(d)
                    label_adj[d].add(idx)

    for idx in range(label_max + 1):
        if label_cnt[idx] == 0:
            continue

        # centroid
        label_cx[idx] = label_X_sum[idx] / label_cnt[idx]
        label_cy[idx] = label_Y_sum[idx] / label_cnt[idx]

        label_L_avg[idx] = label_L_sum[idx] / label_cnt[idx]
        label_A_avg[idx] = label_A_sum[idx] / label_cnt[idx]
        label_B_avg[idx] = label_B_sum[idx] / label_cnt[idx]

        var_L = label_L_sqsum[idx] / label_cnt[idx] - label_L_avg[idx] ** 2
        var_A = label_A_sqsum[idx] / label_cnt[idx] - label_A_avg[idx] ** 2
        var_B = label_B_sqsum[idx] / label_cnt[idx] - label_B_avg[idx] ** 2

        label_L_std[idx] = np.sqrt(max(var_L, 0.0))
        label_A_std[idx] = np.sqrt(max(var_A, 0.0))
        label_B_std[idx] = np.sqrt(max(var_B, 0.0))

    return (label_max, label_cnt,
            label_L_avg, label_A_avg, label_B_avg,
            label_L_std, label_A_std, label_B_std,
            label_cx, label_cy, label_adj)

class Adj_rel_analysis:
    def __init__(self):
        self.is_edge = False
        self.is_depth_jump = False
        self.is_convex = False
        self.is_concave = False


def trans_pnt(cx, cy, current_sz, target_sz):
    return cx * target_sz[1] / current_sz[1], cy * target_sz[0] / current_sz[0]


def refactor_segmentation(self, image, depthmap, local_idx,
              segment_num, compactness,
              start_pos="128,250",
              lab_merge_param="1,1,1,4000,20,5.0", show_boundary=True):

    camsz = image.shape
    depth_sz = depthmap.shape

    gnd_pos = (128,200)

    #+preprocess
    slic_img_sz = (512, 512)
    resized_img = cv2.resize(image, slic_img_sz, interpolation=cv2.INTER_AREA)
    #-preprocess

    cam_att_pitch = -27.
    cam_att_roll = 0


    #+CIE Lab SLIC#
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) / 255.0
    lab_img = skimage.color.rgb2lab(rgb)

    labels = slic(lab_img, n_segments=segment_num, compactness=compactness, start_label=0)
    merged_labels = labels
    #-CIE Lab SLIC#


    #+canny edge 시각화
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, threshold1=20, threshold2=40)
    #-canny edge 시각화

    #+depth map
    norm_Depth_Map = cv2.normalize(depthmap, None, alpha=0, beta=255., norm_type=cv2.NORM_MINMAX)
    norm_Depth_Map = norm_Depth_Map.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(norm_Depth_Map, cv2.COLORMAP_JET)
    self.qt_imshow("depth_colormap", depth_colormap)
    #-depth map


    # 라벨별 Lab 통계 및 adjacency
    (label_max, label_cnt,  # 라벨의 최댓값(슈퍼픽셀의 갯수), 각 슈퍼픽셀의 크기
     label_L_avg, label_A_avg, label_B_avg,  # 평균
     label_L_std, label_A_std, label_B_std,  # 표준편차
     label_cx, label_cy, label_adj  # 이웃들의 리스트: dictionary 안에 set 구조  , dictionary key는 슈퍼픽셀 라벨의 인덱스, 안쪽의 리스트는 key의 이웃라벨
     ) = preprocess_for_lab2(labels, lab_img.astype(np.float64))



    floor_y_min = 750
    floor_y_max = 900

    floor_labid = []
    for lab_id in range(label_max + 1):
        sx, sy = label_cx[lab_id], label_cy[lab_id]

        dx, dy = trans_pnt(sx, sy, slic_img_sz, depth_sz)
        _depth = depthmap[int(dy)][int(dx)]

        cx, cy = trans_pnt(sx, sy, slic_img_sz, camsz)
        #px, py, pz = Pixelto3D(cx, cy, _depth)
        nvector = Pixelto3D(cx, cy, 1)
        nvector = rotate_rpy(nvector, pitch=cam_att_pitch)
        px, py, pz = nvector * _depth


        if floor_y_min < py :
            floor_labid.append(lab_id)

    #+floor 시각화
    floor_vis = resized_img.copy()
    for lab_id in floor_labid:
        ys, xs = np.where(labels == lab_id)
        floor_vis[ys, xs, 1] = 255  # G채널 255
    self.qt_imshow("floor", floor_vis)
    #-floor 시각화

    label_h, label_w = labels.shape
    depthmap_resized = cv2.resize(depthmap, (label_w, label_h), interpolation=cv2.INTER_NEAREST)
    normal_vectors, normal_centers = get_labeled_normal_vector(labels, depthmap_resized)

    #+normal Y 우세 floor 시각화
    normal_floor_vis = resized_img.copy()
    normal_walls_hsv = np.zeros((label_h, label_w, 3), dtype=np.uint8)
    for lab_id in range(label_max + 1):
        n = normal_vectors[lab_id]
        if n is None:
            continue

        n = np.array(rotate_rpy(n, pitch=cam_att_pitch))
        norm_len = np.linalg.norm(n)
        if norm_len < 1e-6:
            continue

            
        nx, ny, nz = n / norm_len
        ys, xs = np.where(labels == lab_id)
        if abs(ny) > abs(nx) and abs(ny) > abs(nz) and 0.6 < abs(ny):
            normal_floor_vis[ys, xs, 1] = 255
        else:
            yaw = np.arctan2(nx, nz)  # -π ~ π
            hue = int(np.degrees(yaw) % 360 / 360 * 180)  # 0=red 기준
            normal_walls_hsv[ys, xs, 0] = hue
            normal_walls_hsv[ys, xs, 1] = 255
            normal_walls_hsv[ys, xs, 2] = 255

    normal_walls_vis = cv2.cvtColor(normal_walls_hsv, cv2.COLOR_HSV2BGR)
    self.qt_imshow("normal_floor", normal_floor_vis)
    self.qt_imshow("normal_walls_vis", normal_walls_vis)
    #-normal Y 우세 floor 시각화


    #+이웃간 관계라인 긋기
    slic_adj = resized_img.copy()
    adj_pixels = {}  # {(u,v): [(x,y), ...]}
    adj_relation = {}

    for u, neighbors in label_adj.items():
        for v in neighbors:
            if v <= u:
                continue
            x1, y1 = int(label_cx[u]), int(label_cy[u])
            x2, y2 = int(label_cx[v]), int(label_cy[v])
            n = max(abs(x2 - x1), abs(y2 - y1), 1)
            pts = [(int(x1 + (x2-x1)*t/n), int(y1 + (y2-y1)*t/n)) for t in range(n+1)]
            adj_pixels[(u, v)] = pts

    #+슬릭 센터 사이 analysis
    for u, neighbors in label_adj.items():
        for v in neighbors:
            if v <= u:
                continue

            prior_depth = None
            depth_line = []
            for x, y in adj_pixels[(u, v)]:
                if (u, v) not in adj_relation:
                    adj_relation[(u, v)] = Adj_rel_analysis()

                if 100 < canny[y][x]:
                    adj_relation[(u, v)].is_edge = True

                """
                if prior_depth is not None:
                    if 300 < abs(prior_depth - depthmap_resized[y][x]):
                        adj_relation[(u, v)].is_depth_jump = True
                        
                if 400 < depthmap_resized[y][x]:
                    prior_depth = depthmap_resized[y][x]
                """
                depth_line.append(depthmap_resized[y][x])

            depth_line = np.array(depth_line)
            if 3 < len(depth_line) and 500 < abs(depth_line[0] - depth_line[-1]):
                adj_relation[(u, v)].is_depth_jump = True

            if 3 < len(depth_line):
                depth_st = depth_line[0]
                depth_ed =  depth_line[-1]
                depth_max = np.max(depth_line)
                depth_min = np.min(depth_line)
                depth_std = np.std(depth_line)
                depth_avg = np.average(depth_line)

                #if depth_st < depth_max - 2*depth_std or depth_ed < depth_max - 2*depth_std:

                #if depth_st < depth_avg and depth_ed < depth_avg and depth_avg < depth_max:
                #if depth_st > depth_avg and depth_ed > depth_avg and depth_avg > depth_min:

                if depth_st > depth_avg and depth_ed > depth_avg:
                    adj_relation[(u, v)].is_convex = True

                if depth_st < depth_avg and depth_ed < depth_avg:
                    adj_relation[(u, v)].is_concave = True

    #-슬릭 센터 사이 analysis

    for u, neighbors in label_adj.items():
        for v in neighbors:
            if v <= u:
                continue

            _color = [0,0,0]
            if adj_relation[(u,v)].is_edge:
                _color[2] = 255
            if adj_relation[(u, v)].is_convex:
                _color[1] = 255
            if adj_relation[(u, v)].is_concave:
                _color[0] = 255

            color = tuple(_color)
            for x, y in adj_pixels[(u, v)]:
                if 0 <= y < label_h and 0 <= x < label_w:
                    slic_adj[y, x] = color

    self.qt_imshow("slic_adj", slic_adj)
    #-이웃간 관계라인 긋기



    #+지정된 위치 좌표확인
    _depth = depthmap[gnd_pos[1]][gnd_pos[0]]
    cx, cy = trans_pnt(gnd_pos[0], gnd_pos[1], depth_sz, camsz)
    print("gndpos ", gnd_pos, Pixelto3D(cx, cy, _depth))
    #-지정된 위치 좌표확인


    #+시각화
    resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    seg_vis = label2rgb(merged_labels, resized_rgb, kind='avg')  # merged_labels에서  resized_rgb의 색을 평균내서 seg_vis를 만든다.

    if show_boundary:
        seg_vis = mark_boundaries(seg_vis, merged_labels, color=(1, 0, 0), mode='outer')

    vis_bgr = cv2.cvtColor((seg_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    print("super pixel num:", np.max(merged_labels))
    self.qt_imshow("Original (resized)", resized_img)
    self.qt_imshow("Segmented", vis_bgr)


    self.qt_imshow("canny_edge", canny)
    #-시각화


#- refactor segmentation
