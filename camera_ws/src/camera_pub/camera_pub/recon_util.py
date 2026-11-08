
##+engoine
from ThalamusEngine.Interface import *

AsmFileName = "ScriptFreeModel.txt".encode("UTF-8")
SimWindowText = "Modeling Debug".encode("UTF-8")

def StartOpenGL(delay_sec=0.5):
    StartExt3DEngine(AsmFileName, SimWindowText)
    time.sleep(delay_sec)
    return True

def init_engine(use_gl=False):
    if not LoadThalamusInterface():
        raise RuntimeError("Failed to load Thalamus interface")

    #errCode = InitEngine(AsmFileName,1280, 720, 0)
    #errCode = InitEngine(AsmFileName, 1280, 720, 1)

    #if errCode != 0:
    #    raise RuntimeError(f"InitEngine error code: {errCode}")
    if use_gl:
        StartOpenGL()

##-engoine


def fill_invalid_depth(depth_map, valid_min=100, valid_max=6000):
    """
    Depth map의 비어있는 부분을 인접 유효값으로 채워 넣음.
    (좌→우, 우→좌, 상→하, 하→상 순으로 스캔)

    Parameters:
        depth_map (np.ndarray): 2D 뎁스맵 (H x W)
        valid_min (int): 유효 뎁스 최소값
        valid_max (int): 유효 뎁스 최대값

    Returns:
        np.ndarray: hole-filling 된 뎁스맵
    """
    h, w = depth_map.shape

    def scan_and_fill_line(line):
        """한 줄(1D array)을 양방향으로 스캔하며 빈값을 채움"""
        last_val = -1
        # 왼쪽 → 오른쪽
        for i in range(len(line)):
            if valid_min <= line[i] <= valid_max:
                last_val = line[i]
            elif last_val > 0:
                line[i] = float(last_val)

        last_val = -1
        # 오른쪽 → 왼쪽
        for i in range(len(line) - 1, -1, -1):
            if valid_min <= line[i] <= valid_max:
                last_val = line[i]
            elif last_val > 0:
                line[i] = float(last_val)
        return line

    # 행 단위 보정
    for y in range(h):
        depth_map[y, :] = scan_and_fill_line(depth_map[y, :])

    # 열 단위 보정
    for x in range(w):
        depth_map[:, x] = scan_and_fill_line(depth_map[:, x])

    return depth_map


#+reconstruction utility
def get_src_pixelpos(target_shape, depth_shape, x, y):
    H, W = depth_shape
    th, tw = target_shape
    return int(tw * x / W), int(th * y / H)


def pixel_to_vec(px, py, target_shape, Depth_Map, prev=None, depth_range=(-1,-1)):
    if depth_range[0] == -1 and depth_range[1] == -1:
        tx, ty = get_src_pixelpos(target_shape, Depth_Map.shape, px, py)
        return np.array(Pixelto3D(tx, ty, Depth_Map[py, px]))
    else:
        tx, ty = get_src_pixelpos(target_shape, Depth_Map.shape, px, py)
        if depth_range[0] < Depth_Map[py, px] < depth_range[1]:
            return np.array(Pixelto3D(tx, ty, prev))
        return np.array(Pixelto3D(tx, ty, Depth_Map[py, px]))


def pixels_center(pixel_list, target_shape, Depth_Map):
    ctr_x = 0
    ctr_y = 0
    for pixel in pixel_list:
        tx, ty = get_src_pixelpos(target_shape, Depth_Map.shape, pixel[0], pixel[1])
        ctr_x += tx
        ctr_y += ty

    return int(ctr_x / len(pixel_list)), int(ctr_y / len(pixel_list)),


def rotate_rpy(vec, roll=0.0, pitch=0.0, yaw=0.0):
    """
    좌표계: [좌우(X), 위아래(Y), 앞뒤(Z)]
    회전 정의:
        - roll: Z축 (앞뒤축) 회전  → 좌우 기울기
        - pitch: X축 (좌우축) 회전 → 위아래 끄덕임
        - yaw: Y축 (위아래축) 회전 → 좌우 회전
    회전 순서: yaw(Y) → pitch(X) → roll(Z)
    """
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)

    # Roll (Z축 회전)
    Rz = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r), np.cos(r), 0],
        [0, 0, 1]
    ])

    # Pitch (X축 회전)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)]
    ])

    # Yaw (Y축 회전)
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    # 순서: R = Ry * Rx * Rz
    R = Ry @ Rx @ Rz
    return R @ np.array(vec)


def is_inside(vec, x_th_min, x_th_max, y_th_min, y_th_max, z_th_min, z_th_max):
    if x_th_min <= vec[0] <= x_th_max and y_th_min <= vec[1] <= y_th_max and z_th_min <= vec[1] <= z_th_max:
        return True
    return False
#+reconstruction utility

from collections import deque
class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.reset()

    def update(self, value):
        if len(self.window) == self.window.maxlen:
            self.sum -= self.window[0]
        self.window.append(value)
        self.sum += value
        return self.sum / len(self.window)

    def reset(self):
        self.window = deque(maxlen=self.window_size)
        self.sum = 0.0






