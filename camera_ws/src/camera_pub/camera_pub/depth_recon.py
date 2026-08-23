import cv2
import numpy as np
from camera_pub.recon_util import pixel_to_vec, rotate_rpy


def depthmap_floor_ransac(depth_mm, target_shape, dist_th, cam_pitch=0.0, cam_roll=0.0,
                           forced_pixel=None, normal_th=0.7):
    """
    return:
        display_mask : (h, w, 3) uint8  - 바닥=흰색, 나머지=검정 (표시용)
        floor_pts    : (h, w, 3) float32 - 바닥 픽셀의 카메라 프레임 3D 좌표(mm),
                       비바닥 픽셀은 [0,0,0]  (bird view용 mask)
    forced_pixel : (px, py) 정수 튜플 - 해당 픽셀의 3D 점을 RANSAC 샘플에 항상 포함
    normal_th    : 법선 방향 일치 임계값 (0~1, 클수록 엄격)
    """
    img_h, img_w = depth_mm.shape

    point_map = np.zeros((img_h, img_w, 3), dtype=np.float32)
    for py in range(img_h):
        for px in range(img_w):
            point_map[py, px] = pixel_to_vec(px, py, target_shape, depth_mm)

    p = np.deg2rad(cam_pitch)
    r = np.deg2rad(cam_roll)
    Rx_T = np.array([[1, 0,          0         ],
                     [0, np.cos(p),  np.sin(p) ],
                     [0, -np.sin(p), np.cos(p) ]])
    Rz_T = np.array([[np.cos(r),  np.sin(r), 0],
                     [-np.sin(r), np.cos(r), 0],
                     [0,          0,          1]])
    n_floor_cam = Rz_T @ Rx_T @ np.array([0.0, -1.0, 0.0])

    pts = point_map.reshape(-1, 3)
    pts = pts[pts[:, 2] > 0]

    # forced point 설정
    forced_pt   = None
    forced_fpx  = -1
    forced_fpy  = -1
    if forced_pixel is not None:
        fpx, fpy = int(forced_pixel[0]), int(forced_pixel[1])
        if 0 <= fpy < img_h and 0 <= fpx < img_w:
            fp = point_map[fpy, fpx].astype(np.float64)  # float64로 복사 (부동소수점 오차 방지)
            if fp[2] > 0:
                forced_pt  = fp
                forced_fpx = fpx
                forced_fpy = fpy
                print(f"[depthmap_floor_ransac] forced_pt=({fpx},{fpy}) 3D={fp}")
            else:
                print(f"[depthmap_floor_ransac] forced_pixel ({fpx},{fpy}) depth 없음, 무시")

    best_normal     = None
    best_d          = 0
    best_inlier_cnt = 0
    ransac_iter     = 200

    rng = np.random.default_rng()
    for _ in range(ransac_iter):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-6:
            continue
        normal = normal / n_len
        if abs(np.dot(normal, n_floor_cam)) < normal_th:
            continue
        # forced_pt가 있으면 d를 forced_pt 기준으로 덮어씀 → 평면이 반드시 그 점을 지남
        d = np.dot(normal, forced_pt) if forced_pt is not None else np.dot(normal, p0)
        dists = np.abs(pts @ normal - d)
        inlier_cnt = int(np.sum(dists < dist_th))
        if inlier_cnt > best_inlier_cnt:
            best_inlier_cnt = inlier_cnt
            best_normal = normal
            best_d = d

    print(f"[depthmap_floor_ransac] normal={best_normal}, d={best_d:.1f}, inliers={best_inlier_cnt}/{len(pts)}")

    empty = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if best_normal is None:
        return empty, np.zeros((img_h, img_w, 3), dtype=np.float32)

    flat = point_map.reshape(-1, 3)
    dists = np.abs(flat @ best_normal - best_d)
    inlier = ((flat[:, 2] > 0) & (dists < dist_th)).reshape(img_h, img_w)

    display_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    display_mask[inlier] = 255

    floor_pts = np.zeros((img_h, img_w, 3), dtype=np.float32)
    floor_pts[inlier] = point_map[inlier]

    # forced pixel 강제 포함 (부동소수점 오차 등으로 빠졌을 경우 보장)
    if forced_fpy >= 0 and forced_fpx >= 0:
        display_mask[forced_fpy, forced_fpx] = 255
        floor_pts[forced_fpy, forced_fpx]    = point_map[forced_fpy, forced_fpx]
        if not inlier[forced_fpy, forced_fpx]:
            print(f"[depthmap_floor_ransac] forced_pixel ({forced_fpx},{forced_fpy}) 강제 포함됨")

    return display_mask, floor_pts


def make_birdview(mask, cam_pitch=0.0, cam_roll=0.0, img_size=512, scale=0.05, close_iter=3,
                  cam_heading=0.0, cam_axis_pos=0.0):
    """
    mask         : (h, w, 3) float32 - 바닥 픽셀의 카메라 프레임 3D 좌표(mm),
                   비바닥=0  (depthmap_floor_ransac의 floor_pts)
    close_iter   : closing(팽창→침식) 반복 횟수 - 구멍 메우기
    cam_heading  : 카메라 패닝 각도 (deg) - 로봇 전방 기준 좌우 회전 (PanningAngle - N)
    cam_axis_pos : 카메라가 로봇 회전축에서 Z축 방향으로 앞으로 나아간 거리 (mm)
    return       : (img_size, img_size, 3) uint8 bird view
                   바닥=흰색, 로봇축=빨간 점, 카메라=파란 점
    """
    bv = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cx = img_size // 2
    cy = int(img_size * 0.8)  # 로봇 회전축 기준

    h, w = mask.shape[:2]
    for py in range(h):
        for px in range(w):
            pt_cam = mask[py, px]
            if pt_cam[2] <= 0:
                continue
            # roll + pitch + heading 한번에 회전 → 로봇 프레임
            pt_world = rotate_rpy(pt_cam, cam_roll, cam_pitch, cam_heading)
            # 카메라가 로봇 축에서 cam_axis_pos mm 전방에 있으므로 Z 보정
            bv_x = int(cx + pt_world[0] * scale)
            bv_y = int(cy - (pt_world[2] + cam_axis_pos) * scale)
            if 0 <= bv_x < img_size and 0 <= bv_y < img_size:
                bv[bv_y, bv_x] = (255, 255, 255)

    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        bv = cv2.morphologyEx(bv, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # 로봇 회전축 - 빨간 점
    cv2.circle(bv, (cx, cy), 4, (0, 0, 255), -1)
    # 카메라 위치 (축에서 cam_axis_pos mm 전방) - 파란 점
    cam_bv_y = cy - int(cam_axis_pos * scale)
    if 0 <= cam_bv_y < img_size:
        cv2.circle(bv, (cx, cam_bv_y), 2, (255, 0, 0), -1)
    return bv
